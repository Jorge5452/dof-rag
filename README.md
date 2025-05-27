# Sistema RAG (Retrieval-Augmented Generation)

Un sistema completo de RAG para análisis de documentos Markdown que combina búsqueda vectorial con generación de respuestas mediante IA.

## 📋 Índice

- [🚀 Características Principales](#-características-principales)
- [🛠️ Instalación](#️-instalación)
- [📖 Guía de Uso de Comandos](#-guía-de-uso-de-comandos)
  - [🔄 Comandos Principales](#-comandos-principales-mutuamente-exclusivos)
  - [🔧 Opciones Complementarias](#-opciones-complementarias)
  - [💡 Ejemplos Prácticos](#-ejemplos-prácticos-de-combinaciones)
  - [⚠️ Restricciones Importantes](#️-restricciones-importantes)
  - [🎯 Casos de Uso Típicos](#-casos-de-uso-típicos)
- [🌐 Interfaz Web y Chatbot](#-interfaz-web-y-chatbot)
- [🏗️ Arquitectura del Sistema](#️-arquitectura-del-sistema)
  - [📁 Estructura del Proyecto](#-estructura-del-proyecto)
  - [🔄 Flujo de Datos](#-flujo-de-datos)
  - [🔗 Interacción de Componentes](#-interacción-de-componentes)
- [🔧 Extensión del Sistema](#-extensión-del-sistema)
  - [➕ Agregar Nuevo Cliente IA](#-agregar-nuevo-cliente-ia)
  - [🗄️ Agregar Nueva Base de Datos](#️-agregar-nueva-base-de-datos)
  - [✂️ Agregar Nuevo Método de Chunking](#️-agregar-nuevo-método-de-chunking)
  - [📊 Agregar Nuevo Modelo de Embeddings](#-agregar-nuevo-modelo-de-embeddings)
- [⚙️ Configuración](#️-configuración)
  - [🎛️ Configuración Global (config.yaml)](#️-configuración-global-configyaml)
  - [🔑 Variables de Entorno](#-variables-de-entorno)
  - [📝 Configuración por Módulos](#-configuración-por-módulos)
- [📚 Documentación de Módulos](#-documentación-de-módulos)
- [🔗 Enlaces Útiles](#-enlaces-útiles)
- [🤝 Contribuciones](#-contribuciones)
- [📄 Licencia](#-licencia)

## 🚀 Características Principales

- **📄 Procesamiento de documentos**: Ingesta y procesa archivos Markdown con chunking inteligente y optimización de memoria
- **🔍 Búsqueda vectorial**: Encuentra información relevante usando embeddings semánticos con múltiples algoritmos de similitud
- **🤖 Múltiples modelos IA**: Compatible con OpenAI, Google Gemini y Ollama con configuración unificada
- **🗄️ Bases de datos vectoriales**: Soporte para SQLite y DuckDB con optimizaciones específicas y búsqueda eficiente
- **💾 Gestión de recursos**: Monitoreo automático de memoria y CPU con limpieza inteligente y concurrencia adaptativa
- **💬 Modo interactivo**: Interfaz conversacional para consultas continuas con historial y comandos especiales
- **🌐 Interfaz web**: Chatbot web integrado para acceso fácil y amigable al sistema
- **📊 Gestión de sesiones**: Sistema unificado para organizar proyectos y bases de datos con metadatos completos
- **⚡ Optimización de rendimiento**: Procesamiento paralelo, batching dinámico y liberación automática de recursos

## 🛠️ Instalación

### Requisitos del Sistema

- **Python 3.8 o superior**
- **8GB RAM mínimo** (16GB recomendado para documentos grandes)
- **2GB espacio libre** para modelos y bases de datos

### Instalación Rápida

1. **Clonar el repositorio:**
```bash
git clone <repository-url>
cd dof-rag
```

2. **Crear entorno virtual:**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows
```

3. **Instalar dependencias:**
```bash
# El proyecto usa pyproject.toml para gestión de dependencias
pip install -e .

# O instalación alternativa si hay problemas:
pip install -r requirements.txt
```

4. **Configurar variables de entorno:**
```bash
cp .env-example .env
# Editar .env con tus API keys (ver sección Variables de Entorno)
```

5. **Verificar instalación:**
```bash
python run.py --resource-status
```

## 📖 Guía de Uso de Comandos

### 🔄 Comandos Principales (Mutuamente Exclusivos)

Estos comandos definen el **modo de operación** del sistema. Solo puedes usar uno a la vez:

#### **📥 Modo Ingesta** (`--ingest`)
Procesa documentos Markdown y los almacena en la base de datos vectorial.

```bash
# Crear nueva sesión/base de datos
python run.py --ingest --files documentos/

# Agregar documentos a base de datos existente
python run.py --ingest --files nuevos_docs/ --db-index 0

# Crear sesión con nombre personalizado
python run.py --ingest --files documentos/ --session-name "mi_proyecto"

# Procesar archivo único
python run.py --ingest --files documento.md

# Procesamiento con debug activado
python run.py --ingest --files documentos/ --debug
```

**Características del modo ingesta:**
- **Procesamiento recursivo**: Busca archivos `.md` en subdirectorios
- **Optimización de memoria**: Procesamiento streaming para documentos grandes
- **Transacciones atómicas**: Rollback automático si hay errores
- **Concurrencia adaptativa**: Paralelización inteligente según recursos
- **Metadatos completos**: Extracción de títulos, encabezados y estructura

#### **🔍 Modo Consulta** (`--query`)
Realiza búsquedas y genera respuestas basadas en los documentos procesados.

```bash
# Consulta única
python run.py --query "¿Cuál es el tema principal del documento?"

# Modo interactivo (sin texto de consulta)
python run.py --query

# Consulta con parámetros específicos
python run.py --query "Mi pregunta" --chunks 10 --db-index 2

# Consulta con modelo específico
python run.py --query "Explica el proceso" --model gemini --chunks 8

# Mostrar bases disponibles antes de consultar
python run.py --query "Mi pregunta" --show-dbs
```

**Características del modo consulta:**
- **Búsqueda semántica**: Recuperación basada en similitud vectorial
- **Contexto enriquecido**: Combinación inteligente de fragmentos relevantes
- **Múltiples modelos**: Soporte para OpenAI, Gemini, Ollama
- **Respuestas contextuales**: Basadas únicamente en documentos procesados
- **Tiempo de respuesta**: Medición y optimización automática

#### **📋 Gestión y Mantenimiento**
```bash
# Listar sesiones/bases de datos disponibles
python run.py --list-sessions

# Mostrar estadísticas detalladas de bases de datos
python run.py --db-stats

# Optimizar base de datos específica
python run.py --optimize-db 0

# Optimizar todas las bases de datos
python run.py --optimize-all

# Ver estado del gestor de recursos
python run.py --resource-status
```

### 🔧 Opciones Complementarias

Estas opciones **se pueden combinar** con los comandos principales:

#### **📤 Exportar Chunks** (`--export-chunks`)
Exporta los fragmentos procesados a archivos de texto para revisión y análisis.

```bash
# Solo exportar (requiere documentos ya procesados)
python run.py --export-chunks --files documentos/

# ✨ Combinar: Procesar Y exportar en un solo comando
python run.py --ingest --export-chunks --files documentos/

# Exportar con base de datos específica
python run.py --export-chunks --files documentos/ --db-index 1
```

**Formato de exportación:**
- **Archivos .txt**: Un archivo por documento original
- **Estructura preservada**: Encabezados y metadatos incluidos
- **Numeración de chunks**: Fragmentos numerados secuencialmente
- **Información técnica**: Dimensiones de embeddings y métricas

#### **🎯 Opciones de Consulta Avanzadas**
```bash
# Especificar número de fragmentos a recuperar (1-20)
--chunks 10

# Usar modelo IA específico  
--model gemini|openai|ollama

# Usar sesión específica por ID
--session mi_sesion_20241220_142030

# Usar base de datos por índice (más rápido)
--db-index 2

# Mostrar bases disponibles antes de consultar
--show-dbs
```

#### **🐛 Depuración y Monitoreo**
```bash
# Modo debug (logs detallados)
--debug

# Combinado con procesamiento para análisis detallado
python run.py --ingest --files test/ --debug
```

### 💡 Ejemplos Prácticos de Combinaciones

#### **🏗️ Flujo Completo: Procesar + Exportar**
```bash
# Procesa documentos Y exporta chunks en un solo paso
python run.py --ingest --export-chunks --files mis_documentos/
```

#### **🔍 Consultas Avanzadas**
```bash
# Consulta con configuración específica
python run.py --query "Explica el proceso de instalación" \
  --chunks 8 \
  --db-index 1 \
  --debug

# Consulta mostrando bases disponibles primero
python run.py --query "Mi pregunta" --show-dbs

# Comparar respuestas entre modelos
python run.py --query "Mismo concepto" --model openai --db-index 0
python run.py --query "Mismo concepto" --model gemini --db-index 0
```

#### **📊 Mantenimiento Integral**
```bash
# 1. Ver estado actual del sistema
python run.py --resource-status

# 2. Ver sesiones disponibles
python run.py --list-sessions

# 3. Optimizar base específica
python run.py --optimize-db 0

# 4. Ver estadísticas actualizadas  
python run.py --db-stats
```

#### **🔄 Gestión de Proyectos**
```bash
# 1. Crear proyecto nuevo
python run.py --ingest --files proyecto_alpha/ --session-name "Proyecto Alpha"

# 2. Ver bases disponibles
python run.py --list-sessions

# 3. Agregar nuevos documentos al proyecto
python run.py --ingest --files nuevos_documentos/ --db-index 0

# 4. Consultar el proyecto actualizado
python run.py --query "Resumen del proyecto" --db-index 0
```

### ⚠️ Restricciones Importantes

1. **Un solo comando principal**: No puedes usar `--ingest` y `--query` juntos
2. **Archivos requeridos**: `--ingest` y `--export-chunks` requieren `--files`
3. **Prioridad de bases**: `--db-index` tiene prioridad sobre `--session`
4. **Límites de chunks**: Máximo 20 chunks por consulta (configuración ajustable)
5. **Formatos soportados**: Solo archivos `.md` (Markdown)

### 🎯 Casos de Uso Típicos

| Escenario | Comando |
|-----------|---------|
| **Primera vez** | `python run.py --ingest --files docs/` |
| **Consulta rápida** | `python run.py --query "mi pregunta"` |  
| **Modo interactivo** | `python run.py --query` |
| **Agregar documentos** | `python run.py --ingest --files nuevos/ --db-index 0` |
| **Revisar chunks** | `python run.py --export-chunks --files docs/` |
| **Proceso completo** | `python run.py --ingest --export-chunks --files docs/` |
| **Mantenimiento** | `python run.py --optimize-all && python run.py --db-stats` |
| **Análisis de recursos** | `python run.py --resource-status` |

## 🌐 Interfaz Web y Chatbot

### Chatbot Web Integrado

El sistema incluye una interfaz web moderna para interactuar con el RAG de manera intuitiva:

#### **Características del Chatbot**
- **Interfaz moderna**: Diseño responsive y amigable
- **Selección de bases**: Cambio dinámico entre proyectos/sesiones
- **Historial de conversación**: Mantiene contexto de la sesión
- **Streaming de respuestas**: Respuestas en tiempo real
- **Indicadores visuales**: Estado de procesamiento y errores
- **Exportación**: Descarga de conversaciones

#### **Ejecución del Chatbot**
```bash
# Iniciar servidor web (puerto 8000 por defecto)
python -m http.server 8000 --directory web/

# Acceder en navegador
# http://localhost:8000
```

#### **Configuración del Chatbot**
```javascript
// web/config.js
const config = {
    apiUrl: 'http://localhost:5000',  // URL del API RAG
    defaultChunks: 5,
    models: ['gemini', 'openai', 'ollama'],
    theme: 'auto'  // 'light', 'dark', 'auto'
};
```

#### **API Endpoints**
```bash
# Listar sesiones disponibles
GET /api/sessions

# Realizar consulta
POST /api/query
{
    "query": "Mi pregunta",
    "db_index": 0,
    "chunks": 5,
    "model": "gemini"
}

# Estado del sistema
GET /api/status
```

### Integración con Aplicaciones Externas

```python
# Ejemplo de integración Python
import requests

def consultar_rag(pregunta, db_index=0):
    response = requests.post('http://localhost:5000/api/query', json={
        'query': pregunta,
        'db_index': db_index,
        'chunks': 5
    })
    return response.json()

resultado = consultar_rag("¿Cómo instalar el sistema?")
print(resultado['response'])
```

## 🏗️ Arquitectura del Sistema

### 📁 Estructura del Proyecto

```
📁 new_rag/
├── 🔧 modulos/                    # Componentes principales del sistema
│   ├── 📚 chunks/                 # Procesamiento de fragmentos
│   │   ├── ChunkerFactory.py      # Factory para chunkers
│   │   └── implementaciones/      
│   │       ├── character_chunker.py    # Por caracteres
│   │       ├── token_chunker.py        # Por tokens (recomendado)
│   │       ├── context_chunker.py      # Por contexto semántico
│   │       └── page_chunker.py         # Por páginas
│   │
│   ├── 🤖 clientes/               # Clientes IA
│   │   ├── AbstractClient.py      # Interfaz base
│   │   ├── FactoryClient.py       # Factory pattern
│   │   └── implementaciones/      
│   │       ├── openai.py          # Cliente OpenAI
│   │       ├── gemini.py          # Cliente Google Gemini  
│   │       └── ollama.py          # Cliente Ollama (local)
│   │
│   ├── 🗄️ databases/             # Bases de datos vectoriales
│   │   ├── VectorialDatabase.py   # Interfaz base
│   │   ├── FactoryDatabase.py     # Factory pattern
│   │   └── implementaciones/
│   │       ├── sqlite.py          # SQLite con sqlite-vec
│   │       └── duckdb.py          # DuckDB (recomendado)
│   │
│   ├── 📊 embeddings/             # Modelos de embeddings
│   │   ├── embeddings_factory.py  # Factory para embeddings
│   │   └── implementaciones/
│   │       ├── modernbert.py      # ModernBERT (recomendado)
│   │       ├── e5_small.py        # E5-small multilingüe
│   │       └── cde_small.py       # CDE-small
│   │
│   ├── 📄 doc_processor/          # Procesamiento de documentos
│   │   └── markdown_processor.py  # Procesador Markdown
│   │
│   ├── 🎯 session_manager/        # Gestión de sesiones
│   │   └── session_manager.py     # Gestión unificada
│   │
│   ├── 💾 resource_management/    # Gestión de recursos
│   │   ├── resource_manager.py    # Gestor principal
│   │   └── memory_manager.py      # Gestión de memoria
│   │
│   ├── 🖼️ view_chunks/            # Exportación y visualización
│   │   └── chunk_exporter.py      # Exportador de chunks
│   │
│   └── 🛠️ utils/                 # Utilidades del sistema
│       ├── formatting.py          # Formateo de output
│       └── logging_utils.py       # Configuración de logs
│
├── 🌐 web/                        # Interfaz web del chatbot
│   ├── index.html                 # Página principal
│   ├── config.js                  # Configuración del frontend
│   ├── styles.css                 # Estilos CSS
│   └── app.js                     # Lógica JavaScript
│
├── ⚙️ config.yaml                 # Configuración principal
├── 🚀 run.py                      # Punto de entrada CLI
├── 📋 main.py                     # Lógica principal del sistema
├── 🔧 config.py                   # Gestión de configuración
├── 📦 pyproject.toml              # Configuración del proyecto
├── 📄 requirements.txt            # Dependencias (alternativo)
└── 📁 sessions/                   # Almacén de sesiones
    └── metadata/                  # Metadatos de sesiones
```

### 🔄 Flujo de Datos

#### **Ingesta de Documentos**
```
┌──────────────┐    ┌─────────────────┐    ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐
│   📄 .md     │───►│ 🔍 MarkdownProc │───►│ ✂️ Chunker  │───►│ 📊 EmbeddingMod │───►│ 🗄️ Database │
│   Files      │    │   - Metadata    │    │ - Strategy  │    │ - Vectorization │    │ - Storage   │
│              │    │   - Headers     │    │ - Overlap   │    │ - Normalization │    │ - Indexing  │
└──────────────┘    └─────────────────┘    └─────────────┘    └─────────────────┘    └─────────────┘
```

#### **Procesamiento de Consultas**
```
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐    ┌─────────────┐
│ ❓ Query    │───►│ 📊 EmbeddingMod │───►│ 🔍 VectorSearch │───►│ 🤖 IAClient │───►│ 💬 Response │
│ - Text      │    │ - Vectorization │    │ - Similarity    │    │ - Context   │    │ - Streaming │
│ - Intent    │    │ - Normalization │    │ - Top-K         │    │ - Generate  │    │ - Formatted │
└─────────────┘    └─────────────────┘    └─────────────────┘    └─────────────┘    └─────────────┘
```

### 🔗 Interacción de Componentes

```
┌─────────────────────┐     ┌────────────────────┐     ┌─────────────────────┐
│  Chunking Module    │     │ Embedding Module   │     │   Database Module   │
│  ---------------    │     │ ---------------    │     │   ---------------   │
│  - Character        │     │  - ModernBERT      │     │   - SQLite          │
│  - Token            │◄────┤  - CDE             ├────►│   - DuckDB          │
│  - Context          │     │  - E5              │     │                     │
│  - Page             │     │                    │     │                     │
└─────────────────────┘     └────────────────────┘     └─────────────────────┘
          ▲                          ▲                           ▲
          │                          │                           │
          │                          │                           │
          │                          ▼                           │
┌─────────┴──────────┐     ┌────────────────────┐     ┌─────────┴─────────┐
│Resource Management │     │   IA Clients       │     │  Session Manager  │
│------------------  │     │   ------------     │     │  --------------   │
│- ResourceManager   │◄───►│   - OpenAI         │◄───►│  - User Sessions  │
│- MemoryManager     │     │   - Gemini         │     │  - Conversations  │
│- ConcurrencyManager│     │   - Ollama         │     │  - Context Storage│
└────────────────────┘     └────────────────────┘     └───────────────────┘
          ▲                          ▲                           ▲
          │                          │                           │
          └──────────────────────────┼───────────────────────────┘
                                     │
                                     ▼
                           ┌────────────────────┐
                           │   Configuration    │
                           │   ------------     │
                           │   - config.yaml    │
                           │   - .env           │
                           └────────────────────┘
```

## 🔧 Extensión del Sistema

### ➕ Agregar Nuevo Cliente IA

#### **1. Crear Implementación**

```python
# modulos/clientes/implementaciones/mi_cliente.py
from ..AbstractClient import IAClient
from typing import List, Dict, Any, Optional

class MiClienteIA(IAClient):
    def __init__(self, api_key: str = None, **kwargs):
        """Inicializar cliente con configuración específica"""
        # Tu implementación específica
        pass
    
    def generate_response(self, prompt: str, context: List[Dict] = None, **kwargs) -> str:
        """Generar respuesta usando tu API/modelo"""
        # Implementar lógica de generación
        pass
    
    def get_embedding(self, text: str) -> List[float]:
        """Generar embeddings si el cliente lo soporta"""
        # Opcional: implementar si tu cliente maneja embeddings
        pass
```

#### **2. Registrar en Factory**

```python
# modulos/clientes/FactoryClient.py
def get_client(client_type: str, **kwargs) -> IAClient:
    # ...existing code...
    elif client_type == 'mi_cliente':
        from modulos.clientes.implementaciones.mi_cliente import MiClienteIA
        return MiClienteIA(**params)
```

#### **3. Configurar en config.yaml**

```yaml
# config.yaml
ai_client:
  mi_cliente:
    model: "mi_modelo_v1"
    api_key_env: "MI_CLIENTE_API_KEY"
    api_url: "https://mi-api.com/v1"
    timeout: 30
    # Parámetros específicos de tu cliente
    temperatura_custom: 0.8
```

#### **4. Configurar Variables de Entorno**

```bash
# .env
MI_CLIENTE_API_KEY=tu_api_key_aqui
```

### 🗄️ Agregar Nueva Base de Datos

#### **1. Implementar Interfaz**

```python
# modulos/databases/implementaciones/mi_db.py
from modulos.databases.VectorialDatabase import VectorialDatabase
from typing import List, Dict, Any

class MiBaseDatos(VectorialDatabase):
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        # Tu implementación específica
    
    def connect(self, db_path: str) -> bool:
        """Conectar a tu base de datos"""
        pass
    
    def vector_search(self, query_embedding: List[float], n_results: int = 5) -> List[Dict]:
        """Implementar búsqueda vectorial específica"""
        pass
    
    def insert_document(self, document: Dict[str, Any], chunks: List[Dict]) -> int:
        """Insertar documento con chunks"""
        pass
    
    def optimize_database(self) -> bool:
        """Optimizar tu base de datos"""
        pass
```

#### **2. Registrar en Factory**

```python
# modulos/databases/FactoryDatabase.py
def get_database_instance(db_type: str, embedding_dim: int):
    # ...existing code...
    elif db_type == "mi_db":
        from .implementaciones.mi_db import MiBaseDatos
        return MiBaseDatos(embedding_dim=embedding_dim)
```

#### **3. Configurar en config.yaml**

```yaml
# config.yaml
database:
  type: "mi_db"
  mi_db:
    connection_string: "mi://localhost:5432/rag_db"
    pool_size: 10
    timeout: 30
    similarity_threshold: 0.3
```

### ✂️ Agregar Nuevo Método de Chunking

#### **1. Implementar Chunker**

```python
# modulos/chunks/implementaciones/mi_chunker.py
from typing import Generator, Dict, Any, List
from modulos.chunks.AbstractChunker import AbstractChunker

class MiChunker(AbstractChunker):
    def __init__(self, modelo_embedding, **kwargs):
        super().__init__(modelo_embedding)
        # Configuración específica
        self.chunk_strategy = kwargs.get('strategy', 'semantic')
    
    def process_content_stream(self, content: str, doc_title: str = "") -> Generator[Dict[str, Any], None, None]:
        """Generar chunks usando tu estrategia específica"""
        # Implementar lógica de chunking
        for chunk in self._tu_logica_chunking(content):
            yield {
                'text': chunk['texto'],
                'header': chunk.get('encabezado', ''),
                'page': chunk.get('pagina', ''),
                'metadata': chunk.get('metadatos', {})
            }
    
    def _tu_logica_chunking(self, content: str) -> List[Dict]:
        """Tu algoritmo específico de chunking"""
        pass
```

#### **2. Registrar en Factory**

```python
# modulos/chunks/ChunkerFactory.py
def get_chunker(chunker_type: str, embedding_model):
    # ...existing code...
    elif chunker_type == "mi_chunker":
        from .implementaciones.mi_chunker import MiChunker
        return MiChunker(embedding_model, **config)
```

#### **3. Configurar en config.yaml**

```yaml
# config.yaml
chunks:
  method: "mi_chunker"
  mi_chunker:
    strategy: "semantic"
    max_chunk_size: 1500
    overlap_ratio: 0.1
    preserve_structure: true
```

### 📊 Agregar Nuevo Modelo de Embeddings

#### **1. Implementar Modelo**

```python
# modulos/embeddings/implementaciones/mi_embedding.py
from modulos.embeddings.AbstractEmbedding import AbstractEmbedding
from typing import List

class MiModeloEmbedding(AbstractEmbedding):
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name', 'mi-modelo-v1')
        self.dimensions = kwargs.get('dimensions', 768)
        # Inicializar tu modelo
    
    def get_dimensions(self) -> int:
        return self.dimensions
    
    def get_document_embedding(self, header: str, text: str) -> List[float]:
        """Generar embedding para documento"""
        combined_text = f"{header}\n{text}" if header else text
        return self._generate_embedding(combined_text)
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Generar embedding para consulta"""
        return self._generate_embedding(query)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Tu implementación específica"""
        pass
```

#### **2. Registrar en Factory**

```python
# modulos/embeddings/embeddings_factory.py
def get_embedding_manager(model_type: str):
    # ...existing code...
    elif model_type == "mi_embedding":
        from .implementaciones.mi_embedding import MiModeloEmbedding
        return MiModeloEmbedding(**config)
```

#### **3. Configurar en config.yaml**

```yaml
# config.yaml
embeddings:
  model: "mi_embedding"
  mi_embedding:
    model_name: "mi-empresa/mi-modelo-v1"
    dimensions: 1024
    device: "cuda"
    normalize: true
    batch_size: 16
```

## ⚙️ Configuración

### 🎛️ Configuración Global (config.yaml)

El sistema utiliza un archivo de configuración centralizado `config.yaml` que controla todos los aspectos del funcionamiento. Esta configuración sigue un patrón jerárquico que permite especificaciones generales y específicas por módulo.

#### **🏗️ Estructura de Configuración**

```yaml
# Configuración general del sistema
general:
  debug: false                    # Modo debug global
  log_level: "INFO"              # Nivel de logging
  log_dir: "logs"                # Directorio de logs
  sessions_dir: "sessions"       # Directorio de sesiones

# Configuración específica por módulo
ai_client:
  type: "gemini"                 # Cliente por defecto
  general:                       # Configuración compartida
    temperature: 0.7
    max_tokens: 2048
    system_prompt: "..."
  gemini:                        # Configuración específica
    model: "gemini-2.0-flash"
    api_key_env: "GEMINI_API_KEY"

embeddings:
  model: "modernbert"            # Modelo por defecto
  modernbert:
    model_name: "nomic-ai/modernbert-embed-base"
    device: "cpu"
    normalize: true

database:
  type: "duckdb"                # Base de datos por defecto
  duckdb:
    memory_limit: "2GB"
    threads: 4
    similarity_threshold: 0.3

chunks:
  method: "token"               # Método de chunking por defecto
  token:
    max_tokens: 2048
    token_overlap: 100
    tokenizer: "nomic-ai/modernbert-embed-base"

resource_management:
  monitoring:
    interval_sec: 120           # Intervalo de monitoreo
    aggressive_threshold_mem_pct: 85
  memory:
    auto_suspend_memory_mb: 1000
  concurrency:
    cpu_workers: "auto"         # Detección automática
    io_workers: "auto"
```

#### **🔄 Jerarquía de Configuración**

1. **Configuración general**: Valores aplicables a todos los módulos
2. **Configuración específica**: Valores específicos por implementación
3. **Variables de entorno**: Sobrescriben configuración del archivo
4. **Parámetros de línea de comandos**: Máxima prioridad

#### **🎯 Configuraciones Clave**

**Rendimiento y Recursos:**
```yaml
resource_management:
  monitoring:
    aggressive_threshold_mem_pct: 85  # Umbral memoria crítica
    warning_threshold_mem_pct: 75     # Umbral advertencia
  concurrency:
    cpu_workers: "auto"               # Workers automáticos
    max_total_workers: null           # Sin límite por defecto
  memory:
    model_release:
      inactive_timeout_sec: 300       # Liberar modelos inactivos
```

**Calidad de Respuestas:**
```yaml
ai_client:
  general:
    temperature: 0.7                  # Balance creatividad/precisión
    top_p: 0.85                       # Diversidad de respuestas
    top_k: 50                         # Vocabulario permitido
    system_prompt: "..."              # Instrucciones del sistema
processing:
  max_chunks_to_retrieve: 5           # Contexto por consulta
```

**Optimización de Memoria:**
```yaml
chunks:
  memory_optimization:
    enabled: true                     # Habilitar optimización
    batch_size: 50                    # Tamaño de lote base
    memory_check_interval: 15         # Verificación cada 15s
    force_gc: true                    # Garbage collection forzado
```

### 🔑 Variables de Entorno

Las variables de entorno tienen prioridad sobre `config.yaml` y permiten configuración sensible sin exposer secretos:

```bash
# .env
# APIs principales
OPENAI_API_KEY=sk-tu_openai_key_aqui
GEMINI_API_KEY=tu_gemini_key_aqui
OLLAMA_API_URL=http://localhost:11434

# APIs adicionales
ANTHROPIC_API_KEY=tu_anthropic_key_aqui

# Configuración de desarrollo
RAG_DEBUG_MODE=true
RAG_LOG_LEVEL=DEBUG

# Configuración de base de datos
RAG_DB_TYPE=duckdb
RAG_DB_PATH=/custom/path/to/db

# Configuración de recursos
RAG_MEMORY_LIMIT=8GB
RAG_CPU_WORKERS=4
```

#### **🔧 Acceso a Configuración desde Código**

```python
from config import config

# Obtener configuración completa
full_config = config.get_config()

# Obtener configuración específica
ai_config = config.get_ai_client_config()
db_config = config.get_database_config()
embed_config = config.get_embedding_config()

# Obtener configuración filtrada para un cliente específico
gemini_config = config.get_specific_ai_config('gemini')
```

### 📝 Configuración por Módulos

#### **🤖 Clientes IA**
- **Parámetros generales**: Compartidos entre todos los clientes
- **Configuración específica**: Por proveedor (OpenAI, Gemini, Ollama)
- **Manejo de APIs**: Variables de entorno para claves
- **Timeouts y reintentos**: Configuración de robustez

#### **📊 Embeddings**
- **Selección de modelo**: ModernBERT, E5-small, CDE-small
- **Configuración de dispositivo**: CPU/CUDA/MPS
- **Parámetros de normalización**: Para consistencia vectorial
- **Optimización de memoria**: Batch processing

#### **🗄️ Bases de Datos**
- **Tipo de BD**: SQLite vs DuckDB
- **Configuración de memoria**: Límites y optimización
- **Paralelización**: Número de threads
- **Umbrales de similitud**: Para búsqueda vectorial

#### **✂️ Chunking**
- **Método de fragmentación**: Token, carácter, contexto
- **Optimización de memoria**: Procesamiento streaming
- **Configuración de solapamiento**: Para continuidad
- **Extracción de metadatos**: Encabezados y estructura

## 📚 Documentación de Módulos

### 📖 Documentación Detallada

- **[🛠️ Guía del Desarrollador](DEVELOPER_GUIDE.md)** - Documentación técnica completa para desarrolladores. Incluye configuración del entorno de desarrollo, arquitectura del sistema, APIs internas, patrones de diseño, estrategias de testing y debugging, y guías paso a paso para contribuir al proyecto.

- **[🎯 Session Manager](modulos/session_manager/README.md)** - Documentación del gestor de sesiones que coordina el acceso a bases de datos del sistema RAG. Cubre la gestión unificada de sesiones=bases de datos, comandos para listar y gestionar sesiones, casos de uso comunes, y troubleshooting específico.

- **[💾 Resource Management](modulos/resource_management/README.md)** - Documentación del sistema centralizado para gestión inteligente de recursos (memoria, CPU, concurrencia). Explica el monitoreo automático, gestión de modelos de embedding, configuración de umbrales, y optimización de rendimiento.

### 🔧 APIs Internas

Cada módulo implementa el **patrón Factory** para creación de instancias:

- **🤖 Clientes IA**: `modulos/clientes/FactoryClient.py` - Abstracciones unificadas para OpenAI, Gemini, Ollama con manejo consistente de errores y configuración
- **📊 Embeddings**: `modulos/embeddings/embeddings_factory.py` - Gestión de modelos ModernBERT, E5, CDE-small con optimización de memoria y dispositivo
- **🗄️ Bases de Datos**: `modulos/databases/FactoryDatabase.py` - Implementaciones SQLite y DuckDB con búsqueda vectorial optimizada y transacciones ACID
- **✂️ Chunking**: `modulos/chunks/ChunkerFactory.py` - Procesamiento por tokens, caracteres, contexto con streaming y preservación de metadatos
- **📄 Procesamiento**: `modulos/doc_processor/markdown_processor.py` - Análisis de documentos Markdown con extracción de estructura y metadatos

## 🔗 Enlaces Útiles

- **[⚙️ Configuración Principal](config.yaml)** - Archivo de configuración centralizado con todas las opciones del sistema, parámetros de modelos, umbrales de recursos, y configuraciones específicas por módulo

- **[🔑 Variables de Entorno](.env-example)** - Plantilla de configuración para APIs y variables sensibles. Incluye ejemplos para OpenAI, Gemini, Ollama y configuraciones de desarrollo

- **[📁 Módulos del Sistema](modulos/)** - Directorio principal con toda la implementación modular. Cada subdirectorio contiene componentes específicos con documentación propia y ejemplos de uso

- **[🌐 Interfaz Web](web/)** - Chatbot web integrado con interfaz moderna, selección dinámica de bases de datos, historial de conversaciones y streaming de respuestas

## 🤝 Contribuciones

Las contribuciones son bienvenidas y valoradas. El proyecto sigue estándares de código abierto:

### 📋 Proceso de Contribución

1. **Fork** el repositorio en tu cuenta
2. **Crear rama** de feature: `git checkout -b feature/nueva-funcionalidad`
3. **Desarrollar** usando los comandos de debugging: `python run.py --debug`
4. **Testear** extensivamente: `python run.py --ingest --files test_docs/ --debug`
5. **Documentar** cambios en READMEs correspondientes
6. **Commit** con mensajes descriptivos: `git commit -m "feat: nueva funcionalidad X"`
7. **Push** a tu fork: `git push origin feature/nueva-funcionalidad`
8. **Pull Request** con descripción detallada y pruebas

### 🎯 Áreas de Contribución

- **🔌 Nuevos clientes IA**: Anthropic, Cohere, clientes locales
- **🗄️ Bases de datos**: PostgreSQL+pgvector, Qdrant, Weaviate
- **📊 Modelos de embedding**: Nuevos modelos de HuggingFace
- **✂️ Métodos de chunking**: Chunking semántico, por entidades
- **🌐 Interfaz web**: Mejoras UX, nuevas funcionalidades
- **📚 Documentación**: Tutoriales, ejemplos, casos de uso

### 📏 Estándares de Código

- **Type hints** obligatorios para funciones públicas
- **Docstrings** estilo Google para documentación
- **Logging** apropiado con niveles consistentes
- **Tests** para nuevas funcionalidades
- **Configuración** centralizada en `config.yaml`
- **Manejo de errores** robusto con logging detallado

Para desarrollo detallado, consulta la **[🛠️ Guía del Desarrollador](DEVELOPER_GUIDE.md)** que incluye:
- Configuración completa del entorno de desarrollo
- Arquitectura interna y patrones de diseño
- Comandos específicos de testing y debugging
- Guías paso a paso para agregar nuevos componentes
- Estándares de código y mejores prácticas

## 📄 Licencia

Este proyecto está bajo la **licencia MIT**. Ver archivo `LICENSE` para detalles completos.

### Resumen de la Licencia
- ✅ **Uso comercial** permitido
- ✅ **Modificación** permitida  
- ✅ **Distribución** permitida
- ✅ **Uso privado** permitido
- ❗ **Sin garantía** - el software se proporciona "como está"
- 📋 **Atribución requerida** - incluir aviso de copyright