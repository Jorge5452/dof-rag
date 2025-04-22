# Sistema RAG Modular

Sistema de Retrieval Augmented Generation (RAG) altamente modular y extensible, diseñado para procesar documentos, dividirlos en chunks, generar embeddings y consultar modelos de IA con el contexto relevante.

## Índice

1. [Características](#características)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Arquitectura Modular](#arquitectura-modular)
4. [Pipeline RAG](#pipeline-rag)
5. [Módulos Principales](#módulos-principales)
   - [Módulo RAG](#módulo-rag)
   - [Módulo View Chunks](#módulo-view-chunks)
6. [Optimizaciones de Memoria](#optimizaciones-de-memoria)
7. [Instalación](#instalación)
8. [Configuración](#configuración)
9. [Uso](#uso)
   - [Modo Línea de Comandos](#modo-línea-de-comandos)
   - [Servidor Web y API REST](#servidor-web-y-api-rest)
   - [Como Biblioteca](#como-biblioteca)
10. [Pruebas y Evaluación](#pruebas-y-evaluación)
    - [Ejecución de Pruebas](#ejecución-de-pruebas)
    - [Análisis de Resultados](#análisis-de-resultados)
    - [Pruebas de Bases de Datos](#pruebas-de-bases-de-datos)
    - [Benchmarking](#benchmarking)
11. [Extensión del Sistema](#extensión-del-sistema)
    - [Añadir un Nuevo Método de Chunking](#añadir-un-nuevo-método-de-chunking)
    - [Añadir un Nuevo Modelo de Embeddings](#añadir-un-nuevo-modelo-de-embeddings)
    - [Añadir un Nuevo Cliente de IA](#añadir-un-nuevo-cliente-de-ia)
    - [Añadir una Nueva Base de Datos Vectorial](#añadir-una-nueva-base-de-datos-vectorial)
12. [Consideraciones Técnicas](#consideraciones-técnicas)
13. [Licencia](#licencia)

## Características

- **Arquitectura modular**: Basada en interfaces y factories que permiten intercambiar componentes fácilmente.
- **Múltiples estrategias de chunking**: Por caracteres, tokens, contexto semántico o páginas.
- **Soporte para diferentes modelos de embeddings**: ModernBERT, CDE-small, E5-small, etc.
- **Almacenamiento en base de datos vectorial**: Utilizando SQLite con extensiones vectoriales, DuckDB, entre otros.
- **Múltiples clientes de IA**: OpenAI, Gemini, Ollama, entre otros.
- **Configuración centralizada**: Mediante archivo YAML y variables de entorno.
- **Interfaz de usuario mejorada**: Colorización con colorama, modo interactivo y comandos de ayuda.
- **Medición de rendimiento**: Temporizadores que muestran el tiempo de respuesta en consultas.
- **Optimización de memoria**: Procesamiento por streaming para documentos de gran tamaño.
- **Servidor API REST**: Interfaz web para interactuar con el sistema RAG a través de un chatbot.
- **Gestión de sesiones**: Gestión centralizada de sesiones y persistencia de contexto entre interacciones.
- **Visualizaciones**: Exportación de chunks y generación de visualizaciones t-SNE para análisis de embeddings.

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
├── chatbot_server.py   # Servidor web para la API REST del chatbot
└── modulos/
    ├── chunks/
    │   ├── __init__.py
    │   ├── ChunkAbstract.py        # Clase abstracta para chunkers
    │   ├── ChunkerFactory.py       # Factory para instanciar diferentes chunkers
    │   └── implementaciones/       # Métodos concretos de chunking
    │       ├── character_chunker.py
    │       ├── context_chunker.py
    │       ├── token_chunker.py
    │       └── page_chunker.py     # Chunker basado en páginas
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
    │   ├── FactoryClient.py        # Factory para instanciar clientes de IA
    │   └── implementaciones/       # Implementaciones de clientes de IA
    ├── doc_processor/
    │   ├── __init__.py
    │   └── markdown_processor.py   # Procesa archivos Markdown de forma recursiva
    ├── session_manager/
    │   ├── __init__.py
    │   └── session_manager.py      # Gestión de sesiones para el sistema RAG
    ├── view_chunks/                # Módulo para visualizar o exportar chunks en txt
    │   ├── __init__.py
    │   ├── chunk_exporter.py       # Exportador de chunks a archivos de texto
    │   └── tsne_visualizer.py      # Visualizador de embeddings mediante t-SNE
    ├── utils/                      # Utilidades generales del sistema
    │   ├── __init__.py
    │   └── logging_utils.py        # Configuración y utilidades de logging
    └── rag/
        ├── api.py                  # Implementación de la API REST
        ├── app.py                  # Aplicación RAG para el servidor
        ├── chatbot.py              # Lógica del chatbot
        └── static/                 # Archivos estáticos para la interfaz web
└── test/
    ├── __init__.py
    ├── run_tests.py               # Script unificado para ejecutar pruebas
    ├── analizar_resultados.py     # Analiza los resultados de las pruebas
    ├── chunkers/                  # Pruebas para estrategias de chunking
    ├── clients/                   # Pruebas para clientes de IA
    ├── databases/                 # Pruebas para bases de datos vectoriales
    │   ├── __init__.py
    │   ├── benchmark_vector_search.py
    │   ├── test_chunk_insertion.py
    │   ├── test_chunk_operations.py
    │   ├── test_duckdb_database.py
    │   ├── test_factory_database.py
    │   ├── test_sqlite_database.py
    │   └── test_vectorial_database.py
    ├── embeddings/                # Pruebas para modelos de embeddings
    ├── doc_processor/            # Pruebas para procesador de documentos
    ├── rag/                      # Pruebas para el módulo RAG
    ├── session_manager/          # Pruebas para el gestor de sesiones
    ├── view_chunks/              # Pruebas para visualización de chunks
    ├── integration/              # Pruebas de integración
    ├── utils/                    # Utilidades compartidas para pruebas
    └── results/                  # Almacena resultados de pruebas
```

## Arquitectura Modular

El sistema está diseñado con un enfoque modular basado en patrones de diseño como **Factory**, **Clases Abstractas** y **Singleton** para la configuración centralizada:

- **Chunkers**: Diferentes estrategias para dividir documentos:
  - **Character Chunker**: División por número de caracteres con solapamientos.
  - **Token Chunker**: Tokenización usando el modelo de embeddings.
  - **Context Chunker**: División respetando estructura semántica (párrafos, encabezados).
  - **Page Chunker**: División basada en marcadores de página con extracción de encabezados.

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

## Módulos Principales

### Módulo RAG

El módulo RAG implementa un sistema de Retrieval Augmented Generation con una arquitectura modular diseñada para flexibilidad y extensibilidad. Está completamente integrado con el SessionManager para mejorar la gestión de sesiones, administración de recursos y persistencia de contexto.

#### Componentes Principales

- **app.py**: Clase principal que integra bases de datos, embeddings y clientes de IA.
- **api.py**: Implementación de API REST para acceder a la funcionalidad RAG mediante HTTP.
- **chatbot.py**: Implementación especializada para aplicaciones tipo chatbot.
- **server_example.py**: Servidor web de ejemplo con soporte para streaming.

#### Integración con SessionManager

El módulo RAG y SessionManager están estrechamente integrados para proporcionar:

1. **Gestión Centralizada de Sesiones**: Todas las sesiones se gestionan a través del singleton SessionManager.
2. **Registro de Bases de Datos**: Las bases de datos se registran y asocian con sesiones.
3. **Almacenamiento de Contexto**: Los contextos de consulta se almacenan y pueden recuperarse por ID de mensaje.
4. **Optimización de Recursos**: Uso mejorado de memoria y limpieza automática de sesiones inactivas.
5. **Almacenamiento de Configuración**: Las configuraciones específicas de sesión persisten.

#### Endpoints API

El servidor API proporciona estos endpoints principales:

- `GET /api/databases`: Lista las bases de datos disponibles.
- `POST /api/sessions`: Crea una nueva sesión.
- `POST /api/query`: Procesa una consulta y devuelve una respuesta.
- `GET /api/context/<session_id>/latest`: Obtiene el contexto de la última consulta.
- `DELETE /api/sessions/<session_id>`: Elimina una sesión.
- `GET /api/diagnostics`: Obtiene información de diagnóstico del sistema.

### Módulo View Chunks

Este módulo permite exportar los chunks procesados de documentos Markdown a archivos de texto plano (.txt) y generar visualizaciones para facilitar su análisis.

#### Características

- Exporta chunks directamente desde la base de datos a archivos de texto.
- Genera un archivo txt por cada documento Markdown en la misma ubicación.
- Muestra metadatos completos del documento y cada chunk.
- Procesa directorios recursivamente para exportar todos los Markdown encontrados.
- Optimizado para manejar documentos grandes con uso eficiente de memoria.
- Visualizaciones t-SNE 2D y 3D de los embeddings de chunks.

#### Visualizaciones t-SNE

Además de los archivos de texto, el módulo genera visualizaciones t-SNE de los embeddings de chunks:

- **Visualización 2D**: Genera un archivo PNG con una representación bidimensional de los embeddings.
- **Visualización 3D**: Genera un archivo PNG con una representación tridimensional de los embeddings.

Estas visualizaciones son útiles para:
- Identificar clusters semánticos en los documentos.
- Detectar outliers o chunks atípicos.
- Analizar la efectividad del método de chunking.
- Visualizar la distribución de los temas dentro del documento.

Los archivos se guardan en la misma ubicación que el documento Markdown original con los nombres:
- `<nombre_documento>_tsne_2d.png`
- `<nombre_documento>_tsne_3d.png`

## Optimizaciones de Memoria

El sistema incluye optimizaciones específicas para el procesamiento de documentos de gran tamaño:

- **Procesamiento por streaming**: Procesa e inserta chunks individualmente en lugar de cargarlos todos en memoria.
- **Gestión de transacciones**: Optimiza operaciones de base de datos para grandes volúmenes de datos.
- **Garbage collection periódico**: Libera memoria durante el procesamiento por lotes.
- **Monitoreo de recursos**: Opciones para seguimiento del uso de memoria durante el procesamiento.

Esta arquitectura permite manejar documentos muy grandes sin limitaciones por la memoria disponible, manteniendo la retrocompatibilidad con implementaciones anteriores.

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
- `sessions`: Configuración de gestión de sesiones

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

**Exportar chunks a archivos de texto y generar visualizaciones:**
```bash
python run.py --export-chunks --files <ruta>
```

**Exportar chunks con opciones:**
```bash
# Usar una base de datos específica por índice
python run.py --export-chunks --files <ruta> --db-index 2

# Usar una sesión específica
python run.py --export-chunks --files <ruta> --session <session_id>

# Modo de depuración con logs detallados
python run.py --export-chunks --files <ruta> --debug
```

### Servidor Web y API REST

**Iniciar el servidor de chatbot:**
```bash
python chatbot_server.py
```

**Opciones adicionales:**
```bash
python chatbot_server.py --port 8000 --host 127.0.0.1 --debug
```

**Desactivar streaming de respuestas:**
```bash
python chatbot_server.py --no-streaming
```

La interfaz web estará disponible en `http://localhost:5000/` (o el puerto especificado).

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

**Uso del módulo RAG:**

```python
from modulos.rag.chatbot import RagChatbot

# Crear un chatbot con una base de datos específica
chatbot = RagChatbot(database_name="my_database", streaming=True)

# Procesar una consulta
result = chatbot.process_query("¿Qué es RAG?")

# Si el streaming está habilitado
if result.get("streaming"):
    response_generator = result.get("response")
    for chunk in response_generator:
        print(chunk, end="", flush=True)
else:
    print(result.get("response"))

# Obtener el contexto del último mensaje
message_id = result.get("message_id")
context = chatbot.extract_context_from_response(result.get("response"))

# Cerrar el chatbot
chatbot.close()
```

## Pruebas y Evaluación

El sistema incluye un framework de pruebas completo para validar cada componente:

### Ejecución de Pruebas

**Ejecutar todas las pruebas:**
```bash
python test/run_tests.py
```

**Ejecutar pruebas específicas:**
```bash
python test/run_tests.py --type chunkers --chunkers character,token,context
```

**Ejecutar pruebas con parámetros específicos:**
```bash
python test/run_tests.py --dir pruebas --chunkers character,token,context --results-dir resultados
```

### Análisis de Resultados

El sistema genera informes detallados de las pruebas en los directorios de resultados.
Para analizar los resultados:

```bash
python test/analizar_resultados.py --dir <directorio_resultados> --out <directorio_analisis>
```

### Pruebas de Bases de Datos

**Ejecutar todas las pruebas de bases de datos:**
```bash
python test/run_tests.py --type databases
```

**Ejecutar pruebas específicas de bases de datos:**
```bash
python test/run_tests.py --type databases --db-type sqlite
```

**Ejecutar pruebas individuales:**
```bash
python -m unittest test.databases.test_sqlite_database.SQLiteDatabaseTest.test_vector_search
```

**Crear bases de datos de prueba:**
```bash
python test/databases/create_test_data.py --type sqlite --path ./test_db.sqlite --docs 100 --chunks 20
```

**Comparar rendimiento de búsqueda vectorial:**
```bash
python test/databases/benchmark_vector_search.py --sqlite ./test_sqlite.db --duckdb ./test_duckdb.db --queries 50 --plot comparison.png
```

### Benchmarking

Se incluyen herramientas para medir el rendimiento:
- **Búsqueda vectorial**: Comparación de rendimiento entre diferentes bases de datos.
- **Velocidad de chunking**: Análisis de rendimiento para diferentes métodos de chunking.
- **Uso de memoria**: Seguimiento de consumo de recursos para optimización.

## Extensión del Sistema

El sistema está diseñado para ser fácilmente extensible. A continuación se muestra cómo agregar nuevos componentes a cada módulo:

### Añadir un Nuevo Método de Chunking

1. Crea una nueva clase en `modulos/chunks/implementaciones/` que herede de la clase base abstracta del chunker
```python
from modulos.chunks.ChunkAbstract import ChunkAbstract

class MiNuevoChunker(ChunkAbstract):
    def process(self, text: str) -> List[Chunk]:
        # Implementación personalizada
        return chunks
```

2. Registra tu implementación en el factory correspondiente

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
from modulos.embeddings.embeddings_manager import EmbeddingManager

class MiNuevoEmbedder(EmbeddingManager):
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
from modulos.clientes.IAClient import IAClient

class MiNuevoClienteIA(IAClient):
    def generate_response(self, prompt: str, context: List[Dict]) -> str:
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
from modulos.databases.VectorialDatabase import VectorialDatabase

class MiNuevaDB(VectorialDatabase):
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

## Consideraciones Técnicas

- **Tipado estricto**: El proyecto utiliza anotaciones de tipos para garantizar la consistencia
- **Diseño para experimentación**: La configuración centralizada facilita probar diferentes combinaciones de componentes
- **Logging detallado**: Registro de operaciones para depuración y análisis de rendimiento
- **Gestión de errores**: Mecanismos de reintentos y manejo de excepciones en componentes críticos
- **Sesiones y contexto**: Gestión centralizada de sesiones y mantenimiento del contexto de conversación
- **Interfaz dual**: CLI y web con streaming para diferentes casos de uso

## Licencia

[MIT](LICENSE)