import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import numpy as np

    # Database and embedding imports
    import duckdb
    from sentence_transformers import SentenceTransformer
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv()

    return SentenceTransformer, duckdb, mo, np, os, genai, types


@app.cell
def _(SentenceTransformer, duckdb):
    # Initialize the same model and database as in extract_embeddings.py
    model = SentenceTransformer(
        "nomic-ai/modernbert-embed-base", trust_remote_code=True
    )

    # Connect to the existing DuckDB vector database
    db = duckdb.connect("dof_db/db.duckdb")

    return db, model


@app.cell
def _(genai, types, os):
    """Inicializa el cliente Gemini y configuración de generación."""

    # Verificar que la API key esté disponible
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ GEMINI_API_KEY no encontrada en variables de entorno")

        client = None
        model_id = "gemini-2.5-flash"
    else:
        try:
            client = genai.Client(api_key=api_key)
            model_id = "gemini-2.5-flash"
            print("✅ Cliente Gemini inicializado correctamente")
        except Exception as e:
            print(f"❌ Error al inicializar cliente Gemini: {e}")
            client = None
            model_id = "gemini-2.5-flash"

    return client, model_id


@app.cell
def _(db, model, np):
    def search_similar_chunks(query: str, limit: int = 5):
        """
        Search for similar chunks in the DuckDB vector database using semantic similarity.
        """
        try:
            # Generate embedding for the query
            query_embedding = model.encode(query)

            # Convert numpy array to list for DuckDB compatibility
            query_embedding_list = query_embedding.tolist()

            # Búsqueda vectorial usando distancia coseno
            search_sql = """
            SELECT 
                c.id,
                c.text,
                c.header,
                d.title,
                d.url,
                d.file_path,
                array_cosine_distance(c.embedding, CAST(? AS FLOAT[768])) as distance_score
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
            ORDER BY distance_score ASC
            LIMIT ?
            """

            results = db.execute(search_sql, [query_embedding_list, limit]).fetchall()

            # Convert results to list of dictionaries
            search_results = []
            for row in results:
                # Convertir distancia coseno a similitud intuitiva
                similarity_score = 1 - (row[6] / 2)  # Normalizar distancia a similitud

                search_results.append(
                    {
                        "id": row[0],
                        "text": row[1],
                        "header": row[2],
                        "document_title": row[3],
                        "url": row[4],
                        "file_path": row[5],
                        "similarity_score": max(0, similarity_score),
                        "distance_score": row[6],
                    }
                )

            return search_results

        except Exception as e:
            print(f"Error during search: {e}")
            return []

    return (search_similar_chunks,)


@app.cell
def _(mo):
    # Display title and description
    mo.md(
        """
        # Chat con Base de Datos del DOF

        **Asistente Inteligente para Consultas del Diario Oficial de la Federación**

        **Instrucciones:**
        1. Escribe tu pregunta en el chat
        2. El sistema buscará en la base de datos vectorial
        3. Recibirás respuestas con fuentes y enlaces relevantes
        """
    )


@app.cell
def _(search_similar_chunks, client, model_id, mo, types):
    def rag_model(messages, config) -> str:
        """Modelo RAG con Gemini 2.5-flash y fuentes plegables.

        Args:
            messages: Lista de mensajes del chat
            config: Configuración del chat de marimo (contiene temperature, max_tokens,
                   top_p, top_k, frequency_penalty, presence_penalty, layout, etc.)

        Returns:
            str: Respuesta del modelo con fuentes en formato Markdown
        """

        if not messages:
            return "¡Hola! Soy tu asistente para consultar documentos del Diario Oficial de la Federación (DOF). ¿En qué puedo ayudarte?"

        # Obtener el último mensaje y validar que tenga contenido
        last_message = messages[-1]

        latest_message = last_message.content.strip()

        search_results = search_similar_chunks(latest_message, limit=3)

        if not search_results:
            return (
                f"No encontré información relevante sobre '{latest_message}' en la base de datos del DOF. "
                "¿Podrías reformular tu pregunta o ser más específico?"
            )

        context_chunks, fuentes_md = [], []
        for i, res in enumerate(search_results, 1):
            sim_pct = res["similarity_score"] * 100

            # Validar y sanitizar campos que pueden ser None
            doc_title = res.get("document_title") or "Sin título"
            header = res.get("header") or "Sin sección"
            text_content = res.get("text") or "Sin contenido"
            url = res.get("url") or "Sin URL"

            context_chunks.append(
                f"Documento: {doc_title}\nSección: {header}\nContenido: {text_content}"
            )
            fuentes_md.append(
                f"**Fuente {i}** (Similitud: {sim_pct:.1f}%)  \n"
                f"📄 **Documento:** {doc_title}  \n"
                f"📋 **Sección:** {header}  \n"
                f"🔗 **URL:** {url}"
            )

        context_text = "\n\n".join(context_chunks)

        # Verificar si el cliente Gemini está disponible
        if client is None:
            # Modo fallback: respuesta manual sin LLM
            answer = (
                f"**Información encontrada sobre '{latest_message}':**\n\n"
                f"He encontrado {len(search_results)} documentos relevantes en la base de datos del DOF. "
                f"Aquí tienes un resumen basado en los documentos más similares:\n\n"
                f"**Contexto principal:**\n{context_text[:500]}...\n\n"
                f"⚠️ **Nota:** Respuesta generada automáticamente. Para respuestas más elaboradas, "
                f"configura GEMINI_API_KEY en tu archivo .env"
            )
        else:
            # Modo normal: usar Gemini para generar respuesta
            prompt = (
                "Responde a la pregunta usando exclusivamente la información provista en el Contexto. "
                "Si la respuesta no se encuentra allí, indícalo explícitamente. "
                "Responde en español y en formato Markdown.\n\n"
                f"Contexto:\n{context_text}\n\n"
                f"Pregunta: {latest_message}\n\nRespuesta:"
            )

            try:
                # Extraer configuración del chat de marimo
                chat_config = {}
                if config:
                    # Mapear parámetros de marimo a Gemini (con múltiples nombres posibles)
                    # Temperature
                    temp_value = getattr(config, "temperature", None) or getattr(
                        config, "temp", None
                    )
                    if temp_value is not None:
                        chat_config["temperature"] = max(
                            0.0, min(2.0, float(temp_value))
                        )

                    # Max Tokens (puede venir como max_tokens o max_output_tokens)
                    max_tokens_value = (
                        getattr(config, "max_tokens", None)
                        or getattr(config, "max_output_tokens", None)
                        or getattr(config, "maxTokens", None)
                    )
                    if max_tokens_value is not None:
                        chat_config["max_output_tokens"] = max(
                            1, min(8192, int(max_tokens_value))
                        )

                    # Top P
                    top_p_value = getattr(config, "top_p", None) or getattr(
                        config, "topP", None
                    )
                    if top_p_value is not None:
                        chat_config["top_p"] = max(0.0, min(1.0, float(top_p_value)))

                    # Top K
                    top_k_value = getattr(config, "top_k", None) or getattr(
                        config, "topK", None
                    )
                    if top_k_value is not None:
                        chat_config["top_k"] = max(1, min(40, int(top_k_value)))

                    # Frequency Penalty
                    freq_penalty_value = getattr(
                        config, "frequency_penalty", None
                    ) or getattr(config, "frequencyPenalty", None)
                    if freq_penalty_value is not None:
                        chat_config["frequency_penalty"] = max(
                            -2.0, min(2.0, float(freq_penalty_value))
                        )

                    # Presence Penalty
                    pres_penalty_value = getattr(
                        config, "presence_penalty", None
                    ) or getattr(config, "presencePenalty", None)
                    if pres_penalty_value is not None:
                        chat_config["presence_penalty"] = max(
                            -2.0, min(2.0, float(pres_penalty_value))
                        )

                    # Debug: mostrar configuración aplicada (opcional)
                    if chat_config:
                        print(f"🔧 Configuración aplicada a Gemini: {chat_config}")

                # Crear configuración de Gemini con parámetros del chat
                gemini_config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=-1),
                    response_mime_type="text/plain",
                    **chat_config,
                )

                contents = [
                    types.Content(
                        role="user", parts=[types.Part.from_text(text=prompt)]
                    )
                ]
                resp = client.models.generate_content(
                    model=model_id, contents=contents, config=gemini_config
                )

                # Validar respuesta de Gemini
                if hasattr(resp, "text") and resp.text:
                    answer = resp.text
                elif hasattr(resp, "candidates") and resp.candidates:
                    # Intentar extraer texto del primer candidato
                    candidate = resp.candidates[0]
                    if (
                        hasattr(candidate, "content")
                        and candidate.content
                        and candidate.content.parts
                    ):
                        answer = candidate.content.parts[0].text
                    else:
                        answer = f"⚠️ Respuesta de Gemini sin contenido de texto.\n\n**Información encontrada:**\n{context_text[:500]}..."
                else:
                    answer = f"⚠️ Respuesta inesperada de Gemini.\n\n**Información encontrada:**\n{context_text[:500]}..."

            except ImportError as e:
                answer = f"⚠️ Error de importación: {e}\n\n**Información encontrada (modo fallback):**\n{context_text[:500]}..."
            except ValueError as e:
                answer = f"⚠️ Error de configuración: {e}\n\n**Información encontrada (modo fallback):**\n{context_text[:500]}..."
            except Exception as e:
                # Manejo específico de errores de la API de Gemini
                error_message = str(e)
                if "API_KEY" in error_message.upper():
                    answer = f"⚠️ Error de API Key de Gemini. Verifica tu configuración.\n\n**Información encontrada (modo fallback):**\n{context_text[:500]}..."
                elif (
                    "QUOTA" in error_message.upper()
                    or "RATE_LIMIT" in error_message.upper()
                ):
                    answer = f"⚠️ Límite de API alcanzado. Intenta nuevamente en unos minutos.\n\n**Información encontrada (modo fallback):**\n{context_text[:500]}..."
                else:
                    answer = f"⚠️ Error al consultar Gemini: {error_message}\n\n**Información encontrada (modo fallback):**\n{context_text[:500]}..."

        fuentes_md_block = "\n\n".join(fuentes_md)

        # Asegurar que answer sea string
        answer = str(answer)

        # Obtener layout del config o usar valor por defecto
        layout = getattr(config, "layout", "details") if config else "details"

        if layout == "accordion":
            fuentes_widget = mo.accordion(
                {f"Fuentes ({len(search_results)})": mo.md(fuentes_md_block)}
            )
            return mo.vstack([mo.md(answer), fuentes_widget])

        # Layout por defecto: details
        details_block = (
            f"\n/// details | Fuentes ({len(search_results)})\n{fuentes_md_block}\n///"
        )
        return mo.md(answer + details_block)

    return (rag_model,)


@app.cell
def _(mo, rag_model):
    # Create and display the chat interface
    mo.ui.chat(
        rag_model,
        prompts=[
            "¿Qué información hay sobre regulaciones ambientales?",
            "Buscar decretos sobre {{tema}}",
            "¿Cuáles son las últimas modificaciones en {{área_legal}}?",
            "Información sobre impuestos y contribuciones",
            "Regulaciones sobre salud pública",
            "Normativas de educación",
            "¿Qué dice sobre {{concepto_específico}}?",
        ],
        show_configuration_controls=True,
    )


if __name__ == "__main__":
    app.run()
