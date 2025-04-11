# Sistema RAG Modular

Sistema de Retrieval Augmented Generation (RAG) altamente modular y extensible, diseñado para procesar documentos, dividirlos en chunks, generar embeddings y consultar modelos de IA con el contexto relevante.

## Características

- **Arquitectura modular**: Basada en interfaces y factories que permiten intercambiar componentes fácilmente.
- **Múltiples estrategias de chunking**: Por caracteres, tokens o contexto semántico.
- **Soporte para diferentes modelos de embeddings**: ModernBERT, CDE-small, E5-small, etc.
- **Almacenamiento en base de datos vectorial**: Utilizando SQLite con extensiones vectoriales, DuckDB, entre otros.
- **Múltiples clientes de IA**: OpenAI, Gemini, Ollama, entre otros.
- **Configuración centralizada**: Mediante archivo YAML y variables de entorno.
- **Interfaz de usuario mejorada**: Colorización con colorama, modo interactivo y comandos de ayuda.
- **Medición de rendimiento**: Temporizadores que muestran el tiempo de respuesta en consultas.

## Estructura del Proyecto

```
project/
├── run.py              # Punto de entrada con modos de ingestión y consulta
├── main.py             # Orquesta la lógica principal de ejecución
├── config.py           # Carga configuraciones desde YAML (implementado como Singleton)
├── config.yaml         # Archivo de configuración centralizada
├── pyproject.toml      # Definición de dependencias y configuración del proyecto
├── .env                # Variables de entorno (API keys, URLs, etc.)
├── run_tests.py        # Orquestador de pruebas unitarias
└── modulos/
    ├── chunks/
    │   ├── __init__.py
    │   ├── ChunkAbstract.py        # Clase abstracta para chunkers
    │   ├── ChunkerFactory.py       # Factory para instanciar diferentes chunkers
    │   └── implementaciones/       # Métodos concretos de chunking
    │       ├── character_chunker.py
    │       ├── context_chunker.py
    │       └── token_chunker.py
    ├── databases/
    │   ├── VectorialDatabase.py    # Clase abstracta para bases de datos vectoriales
    │   ├── FactoryDatabase.py      # Factory para instanciar diferentes bases de datos
    │   ├── db/                     # Archivos físicos de bases de datos
    │   └── implementaciones/       # Conectores concretos a bases de datos
    │       ├── sqlite.py
    │       └── duckdb.py
    ├── embeddings/
    │   ├── __init__.py
    │   ├── embeddings_factory.py   # Factory para modelos de embeddings
    │   ├── embeddings_manager.py   # Gestor para modelos de embeddings
    │   └── implementaciones/       # Implementaciones específicas para embeddings
    ├── clientes/
    │   └── implementaciones/       # Implementaciones de clientes de IA
    ├── doc_processor/
    │   ├── __init__.py
    │   └── markdown_processor.py   # Procesa archivos Markdown de forma recursiva
    ├── session_manager/
    │   ├── __init__.py
    │   └── session_manager.py      # Gestión de sesiones para el sistema RAG
    └── view_chunks/                # Módulo para visualizar o exportar chunks en txt
└── test/
    ├── __init__.py
    ├── test_chunkers.py           # Pruebas para estrategias de chunking
    ├── databases/                 # Pruebas para bases de datos vectoriales
    │   ├── __init__.py
    │   ├── benchmark_vector_search.py
    │   ├── test_chunk_insertion.py
    │   ├── test_chunk_operations.py
    │   ├── test_duckdb_database.py
    │   ├── test_factory_database.py
    │   ├── test_sqlite_database.py
    │   └── test_vectorial_database.py
    ├── resultados_db_tests/       # Resultados de pruebas de bases de datos
    └── resultados_pruebas/        # Almacena resultados de evaluaciones
```

## Arquitectura Modular

El sistema está diseñado con un enfoque modular basado en patrones de diseño como **Factory**, **Clases Abstractas** y **Singleton** para la configuración centralizada:

- **Chunkers**: Diferentes estrategias para dividir documentos:
  - **Character Chunker**: División por número de caracteres con solapamientos.
  - **Token Chunker**: Tokenización usando el modelo de embeddings.
  - **Context Chunker**: División respetando estructura semántica (párrafos, encabezados).

- **Embeddings**: Soporte para múltiples modelos:
  - **ModernBERT**: Modelo de embedding general.
  - **CDE**: Contextual Dense Embeddings con procesamiento en dos etapas.
  - **E5**: Modelo optimizado para consultas y recuperación.

- **Bases de Datos**: Almacenamiento vectorial optimizado:
  - **SQLite**: Con extensiones vectoriales.
  - **DuckDB**: Para análisis y consultas avanzadas.

- **Clientes IA**: Interfaces para diferentes proveedores:
  - **OpenAI**: Integración con GPT y API de embeddings.
  - **Gemini**: Soporte para modelos de Google.
  - **Ollama**: Integración con modelos locales.

## Pipeline RAG

El sistema implementa el siguiente pipeline:

1. **Configuración del Sistema**: Carga centralizada desde `config.yaml`.
2. **Ingestión de Documentos**: Procesamiento de archivos Markdown, extrayendo contenido y metadatos.
3. **Procesamiento de Embeddings**: Generación de vectores usando el modelo configurado.
4. **Chunking y Segmentación**: División del contenido en fragmentos relevantes.
5. **Almacenamiento en DB Vectorial**: Persistencia optimizada de documentos, chunks y embeddings.
6. **Recuperación de Información**: Búsqueda por similitud vectorial.
7. **Generación de Respuestas**: Interacción con LLM utilizando el contexto recuperado.
8. **Pruebas y Evaluación**: Validación de precisión y rendimiento.

## Instalación

1. Clonar el repositorio:
```bash
git clone <repositorio>
cd <directorio>
```

2. Crear y activar entorno virtual:
```bash
# Con venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Con conda
conda create -n rag python=3.12
conda activate rag
```

3. Instalar dependencias usando pip con pyproject.toml:
```bash
pip install .
```

4. Configurar variables de entorno:
Crear un archivo `.env` en la raíz del proyecto con las siguientes variables (según sea necesario):
```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
OLLAMA_API_URL=http://localhost:11434/api
```

## Configuración

El sistema se configura a través del archivo `config.yaml`. Este archivo contiene secciones para cada módulo, permitiendo personalizar el comportamiento del sistema sin modificar el código.

Principales secciones:
- `general`: Configuración general y de logging
- `chunks`: Método de chunking y parámetros
- `embeddings`: Modelo de embedding a utilizar
- `database`: Configuración de la base de datos
- `ai_client`: Cliente de IA y configuración
- `processing`: Parámetros de procesamiento

## Uso

### Modo Línea de Comandos

**Procesar un documento:**
```bash
python run.py process archivo.txt
```

**Procesar múltiples documentos con títulos:**
```bash
python run.py process doc1.txt doc2.txt --titles "Documento 1" "Documento 2"
```

**Hacer una consulta única:**
```bash
python run.py query "¿Cuál es la idea principal del documento?"
```

**Iniciar modo interactivo:**
```bash
python run.py interactive
```

**Ingerir documentos desde un directorio:**
```bash
python run.py --ingest --file <directorio>
```

### Como Biblioteca

También puedes utilizar el sistema como una biblioteca en tu propio código:

```python
from main import RAGSystem

# Inicializar el sistema
rag = RAGSystem()

# Procesar un documento
doc_id = rag.process_document("ruta/al/documento.txt", title="Mi Documento")

# Realizar una consulta
respuesta = rag.query("¿Cuáles son los puntos clave?")
print(respuesta["answer"])

# Cerrar recursos
rag.close()
```

## Extensión del Sistema

El sistema está diseñado para ser fácilmente extensible. A continuación se muestra cómo agregar nuevos componentes a cada módulo:

### Añadir un Nuevo Método de Chunking

1. Crea una nueva clase en `modulos/chunks/implementaciones/` que herede de la clase base abstracta del chunker
```python
from modulos.chunks.interfaces.base_chunker import BaseChunker

class MiNuevoChunker(BaseChunker):
    def process(self, text: str) -> List[Chunk]:
        # Implementación personalizada
        return chunks
```

2. Registra tu implementación en el factory correspondiente de la carpeta de interfaces

3. Añade la configuración específica en `config.yaml`:
```yaml
chunks:
  method: "mi_nuevo_chunker"
  parameters:
    # Parámetros específicos
```

### Añadir un Nuevo Modelo de Embeddings

1. Crea una nueva clase en `modulos/embeddings/implementaciones/` que implemente la interfaz correspondiente
```python
from modulos.embeddings.interfaces.base_embedder import BaseEmbedder

class MiNuevoEmbedder(BaseEmbedder):
    def get_embedding(self, text: str) -> List[float]:
        # Implementación de tu modelo
        return vector
```

2. Registra tu modelo en el sistema de factories de embeddings

3. Configura tu modelo en `config.yaml`:
```yaml
embeddings:
  model: "mi_nuevo_embedder"
  parameters:
    # Configuración específica
```

### Añadir un Nuevo Cliente de IA

1. Crea una nueva clase en `modulos/clientes/implementaciones/` implementando la interfaz base
```python
from modulos.clientes.interfaces.base_client import AIClient

class MiNuevoClienteIA(AIClient):
    def generate_response(self, prompt: str, context: List[Chunk]) -> str:
        # Implementación de la generación de respuestas
        return respuesta
```

2. Registra el cliente en el sistema de factories

3. Configura el cliente en `config.yaml`:
```yaml
ai_client:
  provider: "mi_nuevo_cliente"
  parameters:
    # Parámetros específicos
```

### Añadir una Nueva Base de Datos Vectorial

1. Crea una implementación que se ajuste a la interfaz de base de datos vectorial
```python
from modulos.databases.interfaces.base_database import VectorDatabase

class MiNuevaDB(VectorDatabase):
    # Implementar métodos requeridos
    def insert_chunk(self, chunk: Chunk) -> int:
        # Implementación
        return id
```

2. Registra tu nueva base de datos en el sistema

3. Configura en `config.yaml`:
```yaml
database:
  type: "mi_nueva_db"
  parameters:
    # Configuración específica
```

## Pruebas y Evaluación

El sistema incluye un framework de pruebas para validar cada componente:

- **Pruebas unitarias** para chunkers, embeddings y bases de datos
- **Evaluación de precisión** en la recuperación de información
- **Registro de resultados** en la carpeta de logs configurada

Ejecutar pruebas:
```bash
python run_tests.py
```

Ejecutar pruebas específicas:
```bash
python -m unittest test.databases.test_sqlite_database
```

## Consideraciones Técnicas

- **Tipado estricto**: El proyecto utiliza anotaciones de tipos para garantizar la consistencia
- **Diseño para experimentación**: La configuración centralizada facilita probar diferentes combinaciones de componentes
- **Logging detallado**: Registro de operaciones para depuración y análisis de rendimiento
- **Gestión de errores**: Mecanismos de reintentos y manejo de excepciones en componentes críticos

## Licencia

[MIT](LICENSE)