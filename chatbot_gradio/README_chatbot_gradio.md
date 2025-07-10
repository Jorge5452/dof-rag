# Chat RAG - Sistema de Consulta DOF

Sistema de chat inteligente que permite consultar documentos del **Diario Oficial de la Federación (DOF)** mediante tecnología **RAG (Retrieval-Augmented Generation)**. Utiliza búsqueda semántica para encontrar información relevante y genera respuestas contextuales respaldadas por fuentes específicas.

## ¿Qué hace?

**Conversación inteligente con documentos oficiales**: Haz preguntas en lenguaje natural y obtén respuestas precisas extraídas directamente de documentos del DOF, con referencias a las fuentes utilizadas.

**Ejemplo de uso:**
- Pregunta: *"¿Qué decretos relacionados con infraestructura se publicaron en enero 2025?"*
- Respuesta: Información específica con enlaces a documentos exactos del DOF

## ✨ Características Principales

- 🤖 **Chat conversacional** con historial de preguntas y respuestas
- � **Búsqueda semántica** en base de datos vectorial de documentos DOF
- � **Fuentes verificables** con fragmentos exactos y enlaces a documentos
- ⚙️ **Configuración ajustable** del número de fuentes consultadas (1-10)
- 🎨 **Interfaz moderna** con paneles colapsables y diseño responsivo

## 🏗️ Arquitectura Técnica

```
┌─ Interface (Gradio) ────────────────────────┐
│ Chat UI + Paneles de Fuentes                │
├─ RAG Pipeline ──────────────────────────────┤
│ 1. Query → Embedding (ModernBERT)           │
│ 2. Vector Search (DuckDB + VSS)             │
│ 3. Context Assembly                         │
│ 4. LLM Generation (Gemini/OpenAI/Claude)    │
├─ Data Layer ────────────────────────────────┤
│ DuckDB + VSS Extension                      │
│ Embeddings: 768-dim vectors                 │
│ Documents: DOF metadata + text chunks       │
└─────────────────────────────────────────────┘
```

### Stack Tecnológico

- **Frontend**: Gradio Blocks con ChatInterface nativa
- **Vector DB**: DuckDB con extensión VSS (Vector Similarity Search)
- **Embeddings**: ModernBERT (768 dimensiones)
- **LLM**: Gemini 2.0 Flash (por defecto), OpenAI GPT-4o-mini, Claude 3.5 Sonnet, Ollama
- **Processing**: Python con arquitectura modular

## 🚀 Inicio Rápido

### 1. Prerrequisitos
- Python 3.12+
- Base de datos DOF creada (desde proyecto principal)
- API key de al menos un proveedor LLM

### 2. Instalación de Dependencias
```bash
# Instalar dependencias con uv
uv add openai
uv add python-dotenv
uv add duckdb
uv add "gradio<=5.34.2"
uv add sentence-transformers
```

**Nota importante**: Se requiere Gradio versión 5.34.2 o inferior debido a problemas de compatibilidad con versiones más recientes.

### 3. Configuración
```bash
# Desde la raíz del proyecto dof-rag
cd chatbot_gradio

# Crear .env en la raíz del proyecto
echo "GEMINI_API_KEY=tu_api_key_aqui" >> ../.env
```

### 4. Ejecutar
```bash
python app.py
# URL: http://localhost:8888
```

## ⚙️ Configuración de LLM

### Proveedores Soportados
| Proveedor | Variable | Modelo | Límite/min |
|-----------|----------|--------|------------|
| **Gemini** (default) | `GEMINI_API_KEY` | gemini-2.0-flash | 15 |
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini | 10 |
| Anthropic | `ANTHROPIC_API_KEY` | claude-3-5-sonnet | 5 |
| Ollama | (local) | llama3.1 | ∞ |

Configura en `.env` en la **raíz del proyecto**:
```env
GEMINI_API_KEY=tu_gemini_key
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-api03-...
```

## 🎯 Casos de Uso

- **Investigación legal**: Búsqueda de decretos, leyes y regulaciones específicas
- **Análisis normativo**: Consulta de cambios en políticas públicas
- **Estudios académicos**: Investigación en documentos oficiales históricos
- **Consultoría**: Verificación rápida de información gubernamental

## � Para Desarrolladores

### Estructura Modular
```
config/     → Configuración centralizada
core/       → Lógica RAG (database, embeddings, llm, pipeline)
interface/  → UI components (chat, rendering)
```

### Logs
- Consola: Nivel INFO
- Archivo: `chatbot_gradio.log`

### Extensibilidad
- Agregar nuevos proveedores LLM en `core/llm_client.py`
- Personalizar UI en `interface/chat_ui.py`
- Modificar procesamiento en `core/rag_pipeline.py`

---

**Nota**: Este sistema es solo para **consulta**. Para actualizar la base de datos con nuevos documentos DOF, utiliza las herramientas del proyecto principal `dof-rag`.