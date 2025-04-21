"""
Lógica principal del sistema RAG.

Este módulo contiene las funciones para:
1. Procesar documentos (ingestar)
2. Procesar consultas (query)

Orquesta todos los componentes del sistema RAG para trabajar juntos.
"""
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

# Configurar logging
logger = logging.getLogger(__name__)

# Importación de lo necesario para colorama
from colorama import init, Fore, Style, Back

# Inicializar colorama para que funcione en todas las plataformas
init(autoreset=True)

# Definir colores y estilos para mejorar la legibilidad
C_TITLE = Back.BLUE + Fore.WHITE + Style.BRIGHT
C_SUBTITLE = Fore.BLUE + Style.BRIGHT
C_SUCCESS = Fore.GREEN + Style.BRIGHT
C_ERROR = Fore.RED + Style.BRIGHT
C_WARNING = Fore.YELLOW
C_HIGHLIGHT = Fore.MAGENTA + Style.BRIGHT
C_COMMAND = Fore.YELLOW
C_INFO = Style.RESET_ALL
C_VALUE = Fore.CYAN + Style.BRIGHT
C_PROMPT = Style.BRIGHT + Fore.GREEN
C_RESET = Style.RESET_ALL
C_SEPARATOR = Style.DIM + Fore.BLUE

# Reducir verbosidad de logging en algunos módulos
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('modulos.databases').setLevel(logging.ERROR)
logging.getLogger('modulos.databases.implementaciones').setLevel(logging.ERROR)
logging.getLogger('modulos.databases.implementaciones.sqlite').setLevel(logging.ERROR)
logging.getLogger('modulos.databases.implementaciones.duckdb').setLevel(logging.ERROR)
logging.getLogger('modulos.session_manager').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('filelock').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

def process_documents(file_path: str, session_name: Optional[str] = None) -> None:
    """
    Procesa documentos Markdown para la ingestión en el sistema RAG.
    
    Args:
        file_path: Ruta al directorio o archivo a procesar
        session_name: Nombre personalizado para la sesión (opcional)
    """
    import os
    from pathlib import Path
    import time
    import logging
    from typing import List, Dict, Any
    
    # Importaciones de módulos del sistema
    from config import config
    from modulos.chunks import ChunkerFactory
    from modulos.embeddings.embeddings_factory import EmbeddingFactory
    from modulos.databases.FactoryDatabase import DatabaseFactory
    from modulos.session_manager.session_manager import SessionManager
    from modulos.doc_processor.markdown_processor import MarkdownProcessor
    
    logger = logging.getLogger(__name__)
    
    # Medir tiempo de inicio
    start_time = time.time()
    
    # Obtener configuraciones
    chunks_config = config.get_chunks_config()
    embeddings_config = config.get_embedding_config()
    database_config = config.get_database_config()
    
    chunking_method = chunks_config.get("method", "character")
    embedding_model = embeddings_config.get("model", "modernbert")
    db_type = database_config.get("type", "sqlite")
    
    logger.info(f"{C_HIGHLIGHT}Iniciando ingestión con chunking: {C_VALUE}{chunking_method}{C_HIGHLIGHT}, embeddings: {C_VALUE}{embedding_model}{C_HIGHLIGHT}, db: {C_VALUE}{db_type}")
    
    # Inicializar componentes
    embedding_manager = EmbeddingFactory.get_embedding_manager(embedding_model)
    # Obtener dimensiones del embedding
    embedding_dim = embedding_manager.embedding_dim
    
    # Inicializar chunker y asignarle el modelo de embeddings
    chunker = ChunkerFactory.get_chunker(chunking_method, embedding_manager)
    
    # Crear un nombre de base de datos único basado en timestamp
    db_name = f"ragdb_{int(time.time())}"
    if session_name:
        # Sanitizar el nombre personalizado para usarlo como nombre de archivo
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in session_name)
        db_name = f"{safe_name}_{int(time.time())}"
    
    # Inicializar la base de datos utilizando una interfaz más completa
    db = DatabaseFactory.get_database_instance(
        db_type=db_type,
        embedding_dim=embedding_dim,
        embedding_model=embedding_model,
        chunking_method=chunking_method,
        custom_name=db_name
    )
    
    # Recuperar la ruta de la base de datos del objeto db
    db_path = db.get_db_path()
    
    # Registrar la base de datos y sus metadatos en el SessionManager
    session_manager = SessionManager()
    
    # Preparar metadatos completos de la base de datos
    db_metadata = {
        "db_type": db_type,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "chunking_method": chunking_method,
        "created_at": time.time(),
        "last_used": time.time(),
        "custom_name": session_name,
        "db_path": db_path
    }
    
    # Registrar la base de datos con todos sus metadatos
    session_manager.register_database(db_name, db_metadata)
    
    logger.info(f"Base de datos creada: {db_name}")
    
    # Inicializar el procesador de Markdown
    markdown_processor = MarkdownProcessor()
    
    # Obtener la lista de archivos a procesar
    file_path_obj = Path(file_path)
    if file_path_obj.is_dir():
        # Si es un directorio, buscar todos los archivos markdown
        md_files = list(file_path_obj.glob("**/*.md"))
        logger.info(f"Se encontraron {len(md_files)} archivos Markdown para procesar")
    else:
        # Si es un archivo, procesarlo directamente
        md_files = [file_path_obj] if file_path_obj.suffix.lower() == '.md' else []
        if not md_files:
            logger.error(f"El archivo especificado no es un archivo Markdown válido: {file_path}")
            return
    
    # Procesar cada documento
    successful_docs = 0
    failed_docs = 0
    
    for doc_path in md_files:
        logger.info(f"Procesando documento: {doc_path}")
        doc_id = process_single_document(str(doc_path), markdown_processor, chunker, db)
        
        if doc_id:
            successful_docs += 1
            logger.info(f"Documento procesado correctamente, ID: {doc_id}")
        else:
            failed_docs += 1
            logger.error(f"Error al procesar documento: {doc_path}")
    
    # Mostrar resumen de procesamiento
    elapsed_time = time.time() - start_time
    logger.info(f"Procesamiento completo en {elapsed_time:.2f} segundos")
    logger.info(f"Documentos procesados correctamente: {successful_docs}")
    
    if failed_docs:
        logger.warning(f"Documentos con errores: {failed_docs}")
    
    # Optimizar la base de datos después de insertar todos los documentos
    logger.info("Optimizando base de datos...")
    db.optimize_database()
    logger.info("Proceso de ingestión completado")

def process_single_document(file_path: str, 
                           markdown_processor: Any, 
                           chunker: Any, 
                           db: Any) -> Optional[int]:
    """
    Procesa un único documento Markdown y lo inserta en la base de datos.
    
    Args:
        file_path: Ruta al archivo Markdown
        markdown_processor: Instancia del procesador de Markdown
        chunker: Instancia del chunker
        db: Instancia de la base de datos
        
    Returns:
        ID del documento insertado o None si falla
    """
    try:
        # Procesar el documento
        metadata, content = markdown_processor.process_document(file_path)
        
        # Generar chunks
        raw_chunks = chunker.process_content(content)
        
        # Preparar chunks con embeddings
        prepared_chunks = []
        for chunk in raw_chunks:
            # Obtener embedding combinando encabezado y texto
            embedding = chunker.model.get_document_embedding(chunk['header'], chunk['text'])
            
            # Crear chunk final con embedding
            prepared_chunk = {
                'text': chunk['text'],
                'header': chunk['header'],
                'page': chunk['page'],
                'embedding': embedding,
                'embedding_dim': chunker.model.get_dimensions()
            }
            
            prepared_chunks.append(prepared_chunk)
        
        # Insertar en la base de datos
        document_id = db.insert_document(metadata, prepared_chunks)
        
        return document_id
    
    except Exception as e:
        logger.error(f"{C_ERROR}Error al procesar documento {file_path}: {e}")
        return None

def process_query(query: str, n_chunks: int = 5, model: Optional[str] = None, 
                 session_id: Optional[str] = None, db_index: Optional[int] = None) -> str:
    """
    Procesa una consulta utilizando el sistema RAG.
    
    Args:
        query: Texto de la consulta
        n_chunks: Número de chunks a recuperar
        model: Modelo de IA a utilizar (opcional)
        session_id: ID de sesión específica a utilizar (opcional)
        db_index: Índice de la base de datos a utilizar (opcional)
        
    Returns:
        Respuesta generada
    """
    # Importaciones bajo demanda
    from config import config  
    from modulos.session_manager.session_manager import SessionManager
    from modulos.embeddings.embeddings_factory import EmbeddingFactory
    from modulos.clientes.FactoryClient import ClientFactory
    
    try:
        # Usar session_manager para obtener la configuración correcta
        session_manager = SessionManager()
    
        # Obtener la base de datos y configuración
        if db_index is not None:
            # Si se especificó un índice, usamos ese índice (pasando el session_id si está disponible)
            db, session = session_manager.get_database_by_index(db_index, session_id=session_id)
        elif session_id:
            # Si se especificó un ID de sesión pero no un índice, usamos la base de datos más reciente asociada a la sesión
            session_dbs = session_manager.get_session_databases(session_id)
            if session_dbs:
                # Usar la primera base de datos asociada a la sesión (la más reciente)
                db, session = session_manager.get_database_by_index(0, session_id=session_id)
            else:
                # Si no hay bases de datos asociadas, usar la más reciente en general
                db, session = session_manager.get_database_by_index(0)
        else:
            # Si no se especificó nada, usamos la base de datos más reciente
            db, session = session_manager.get_database_by_index(0)
        
        # Inicialización del modelo específico que usa esta base de datos
        embedding_model = session.get("embedding_model", "modernbert")
        embedding_manager = EmbeddingFactory.get_embedding_manager(embedding_model)
        
        try:
            # Cargar modelo con manejo de excepciones específico
            embedding_manager.load_model()
        except Exception as e:
            logger.error(f"Error al cargar el modelo de embeddings: {e}")
            return "No se pudo cargar el modelo de embeddings. Por favor, verifica la configuración o intenta con otra base de datos."
        
        # Generar embedding de la consulta
        query_embedding = embedding_manager.get_query_embedding(query)
        
        # Buscar los chunks más relevantes con manejo de errores mejorado
        try:
            search_results = db.vector_search(query_embedding, n_results=n_chunks)
        except Exception as e:
            logger.error(f"Error en la búsqueda vectorial: {e}")
            return "Hubo un problema al buscar información relevante. Es posible que la base de datos seleccionada no sea compatible con la consulta actual."
        
        if not search_results:
            return "No se encontró información relevante para responder a esta consulta."
        
        # Usar el modelo de IA especificado o el predeterminado
        if model is None:
            ai_config = config.get_ai_client_config()
            model = ai_config.get("type", "openai")
            
        # Obtener cliente de IA con manejo de errores mejorado
        try:
            ai_client = ClientFactory.get_client(client_type=model)
        except Exception as e:
            logger.error(f"Error al crear cliente IA: {e}")
            return "No se pudo inicializar el modelo de IA. Por favor, verifica tu configuración y API keys."
        
        # Preparar contexto para la respuesta
        context_chunks = []
        for chunk in search_results:
            context_chunks.append({
                "text": chunk["text"],
                "header": chunk.get("header", ""),
                "similarity": chunk.get("similarity", 0.0),
                "page": chunk.get("page", "N/A")  # Asegurar que se incluya el número de página
            })
        
        # Generar respuesta con manejo de errores mejorado
        try:
            # Siempre mostrar el contexto usado para la respuesta
            response = ai_client.generate_response(query, context=context_chunks, show_context=True)
            
            # Si estamos usando streaming, necesitamos asegurarnos de que se muestre el resultado completo
            if hasattr(response, '__iter__') and not isinstance(response, str):
                # Consumir el generador y devolver el texto completo
                try:
                    # Intentar iterar sobre la respuesta
                    chunks = list(response)
                    
                    # Unir los chunks en una respuesta completa
                    full_response = "".join(chunks)
                    
                    if full_response:
                        return full_response
                    else:
                        # Si la respuesta está vacía después de iterar, significa que el generador no produjo nada
                        # En este caso, vamos a obtener la respuesta original de Gemini directamente
                        if hasattr(ai_client, 'last_response_text') and ai_client.last_response_text:
                            return ai_client._format_response_with_context(ai_client.last_response_text, query)
                        else:
                            # Si todo lo demás falla, usar una respuesta genérica
                            return "La respuesta se generó correctamente pero no se pudo mostrar. Por favor, intenta nuevamente."
                except Exception as e:
                    # Si hay un error al iterar, intentar usar la respuesta como string
                    if response:
                        return str(response)
                    else:
                        return "Error al procesar la respuesta. Por favor, intenta nuevamente."
            
            # Si llegamos aquí, ya tenemos la respuesta como texto
            # Nos aseguramos de que la respuesta no sea None
            if response is None:
                return "No se recibió respuesta del modelo. Esto puede ser un problema de conexión o API."
                
            # Retornamos la respuesta (ya formateada si show_context=True)
            return response
            
        except Exception as e:
            logger.error(f"Error al generar respuesta: {e}")
            return "Hubo un problema al generar la respuesta. Verifica tu configuración y conexión a internet."
        
    except Exception as e:
        # Actualizar el timestamp de último uso de la base de datos si es posible
        try:
            if 'session' in locals() and 'db_name' in session:
                db_name = session.get("id", "")
                # Usar register_database para actualizar el timestamp
                session_manager.register_database(db_name, {
                    "last_used": time.time(),
                    **session  # Mantener el resto de metadatos
                })
        except Exception as update_err:
            # No mostramos este error al usuario final
            logger.debug(f"Error al actualizar metadatos: {update_err}")
    
        # Manejo de errores generales
        try:
            error_msg = f"{e.__class__.__name__}: {str(e)}"
        except Exception:
            error_msg = "Error desconocido durante el procesamiento"
            
        logger.error(f"Error al procesar consulta: {error_msg}")
        
        # Mensaje de error más amigable
        if "database" in error_msg.lower():
            return "Error al acceder a la base de datos. Por favor, verifica que la base de datos seleccionada existe y es accesible."
        elif "embedding" in error_msg.lower() or "model" in error_msg.lower():
            return "Error con el modelo de embeddings. Por favor, verifica la configuración o intenta con otro modelo."
        else:
            return f"Se produjo un error al procesar tu consulta: {error_msg}. Por favor, intenta nuevamente o selecciona otra base de datos."

def verify_database_file(db_path: str) -> bool:
    """
    Verifica que el archivo físico de la base de datos existe.
    
    Args:
        db_path: Ruta al archivo de la base de datos
        
    Returns:
        bool: True si el archivo existe, False en caso contrario
    """
    if db_path == ":memory:":
        return True
    
    file_path = Path(db_path)
    exists = file_path.exists()
    
    if exists:
        size_kb = file_path.stat().st_size / 1024
        logger.info(f"{C_SUCCESS}✓ Base de datos verificada: {C_VALUE}{db_path} ({C_SUCCESS}{size_kb:.2f} KB{C_INFO})")
    else:
        logger.error(f"{C_ERROR}✗ Archivo de base de datos no encontrado: {C_VALUE}{db_path}")
        
    return exists
