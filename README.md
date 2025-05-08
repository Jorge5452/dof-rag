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
7. [Gestión de Recursos Centralizada](#gestión-de-recursos-centralizada)
8. [Instalación](#instalación)
9. [Configuración](#configuración)
10. [Uso](#uso)
   - [Modo Línea de Comandos](#modo-línea-de-comandos)
   - [Servidor Web y API REST](#servidor-web-y-api-rest)
   - [Como Biblioteca](#como-biblioteca)
11. [Pruebas y Evaluación](#pruebas-y-evaluación)
    - [Ejecución de Pruebas](#ejecución-de-pruebas)
    - [Análisis de Resultados](#análisis-de-resultados)
    - [Pruebas de Bases de Datos](#pruebas-de-bases-de-datos)
    - [Benchmarking](#benchmarking)
12. [Extensión del Sistema](#extensión-del-sistema)
    - [Añadir un Nuevo Método de Chunking](#añadir-un-nuevo-método-de-chunking)
    - [Añadir un Nuevo Modelo de Embeddings](#añadir-un-nuevo-modelo-de-embeddings)
    - [Añadir un Nuevo Cliente de IA](#añadir-un-nuevo-cliente-de-ia)
    - [Añadir una Nueva Base de Datos Vectorial](#añadir-una-nueva-base-de-datos-vectorial)
13. [Consideraciones Técnicas](#consideraciones-técnicas)
14. [Licencia](#licencia)

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
- **Gestión de Recursos Centralizada**: Sistema dedicado para monitorear y optimizar el uso de memoria y concurrencia.

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
    ├── resource_management/        # NUEVO: Gestión centralizada de recursos
    │   ├── __init__.py
    │   ├── resource_manager.py     # Orquestador principal (Singleton)
    │   ├── memory_manager.py       # Gestión específica de memoria
    │   └── concurrency_manager.py  # Gestión de concurrencia (pools)
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

El sistema incluye varias estrategias para optimizar el uso de la memoria, especialmente al manejar grandes volúmenes de datos o documentos extensos:

- **Procesamiento por Streaming**: Durante la ingestión de documentos, los chunks se procesan y se insertan en la base de datos de forma individual o en pequeños lotes, evitando cargar todo el documento o todos sus chunks en memoria simultáneamente. Esto es gestionado por `doc_processor` y `databases`.
- **Gestión de Transacciones**: Las operaciones de inserción masiva en la base de datos (durante la ingestión por streaming) se agrupan en transacciones para mejorar el rendimiento de I/O.
- **Optimización Dinámica de Batch Size**: El `MemoryManager` ajusta el tamaño de los lotes (`batch_size`) utilizados en operaciones como la generación de embeddings o el procesamiento de chunks, basándose en el uso actual de memoria del sistema para prevenir picos de consumo.
- **Limpieza Coordinada de Recursos**: El `ResourceManager` monitoriza el uso de recursos y coordina la limpieza proactiva a través del `MemoryManager` y otros componentes (`SessionManager`, `EmbeddingFactory`) para liberar memoria (e.g., ejecutando GC, eliminando sesiones inactivas, descargando modelos no utilizados).

Estas optimizaciones, ahora coordinadas por el [Sistema de Gestión de Recursos Centralizada](#gestión-de-recursos-centralizada), permiten manejar documentos grandes y cargas concurrentes de manera más estable y eficiente.

## Gestión de Recursos Centralizada

Para mejorar la estabilidad y eficiencia bajo diferentes cargas de trabajo, el sistema implementa un módulo dedicado a la gestión centralizada de recursos (`modulos/resource_management/`).

### Componentes

- **`ResourceManager`**: Actúa como el orquestador principal (Singleton). Monitoriza el uso de CPU y memoria, carga la configuración específica (`resource_management` en `config.yaml`), y coordina las acciones de los otros gestores.
- **`MemoryManager`**: Se enfoca en la optimización de memoria. Ejecuta garbage collection, libera modelos de embedding inactivos, y ajusta dinámicamente los tamaños de lote (`batch_size`) según la memoria disponible.
- **`ConcurrencyManager`**: Gestiona la ejecución concurrente de tareas. Mantiene pools de hilos (`ThreadPoolExecutor`) y procesos (`ProcessPoolExecutor`), ajustando el número de workers según la configuración y los cores del sistema.

### Beneficios

- **Monitoreo Activo**: Permite observar el comportamiento del sistema bajo carga.
- **Limpieza Proactiva**: Libera recursos automáticamente cuando se alcanzan umbrales definidos, previniendo errores por falta de memoria (OOM).
- **Concurrencia Adaptativa**: Optimiza el paralelismo para tareas I/O-bound y CPU-bound.
- **Configurabilidad**: Permite ajustar umbrales, intervalos de monitoreo y número de workers a través de `config.yaml`.

### Visualización

Puedes consultar el estado actual del gestor de recursos usando el comando:

```bash
python run.py --resource-status
```

Esto mostrará métricas de memoria, CPU, sesiones activas, modelos cargados y la configuración de gestión de recursos activa.

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

La configuración principal se gestiona a través del archivo `config.yaml`. Este archivo permite ajustar parámetros para:

- Configuración general (logs, directorios).
- Métodos de chunking (tamaño, solapamiento, método por defecto).
- Modelos de embeddings (nombre del modelo, dispositivo, normalización).
- Bases de datos vectoriales (tipo, ruta, umbrales).
- Clientes de IA (tipo, modelo, parámetros de generación, API keys vía `.env`).
- Parámetros de procesamiento (chunks a recuperar).
- **Gestión de Recursos** (intervalos de monitoreo, umbrales de memoria/CPU, configuración de concurrencia).

Las claves de API y otras credenciales sensibles deben configurarse en un archivo `.env` en la raíz del proyecto, siguiendo el ejemplo de `.env-example`.

```dotenv
# Ejemplo .env
OPENAI_API_KEY="tu_clave_openai"
GEMINI_API_KEY="tu_clave_gemini"
# OLLAMA_API_URL="http://localhost:11434/api" # Si usas una URL diferente
```

## Uso

### Modo Línea de Comandos

El script `run.py` es el punto de entrada principal.

**Ingestión de Documentos:**

```bash
# Ingestar un directorio recursivamente
python run.py --ingest --files /ruta/a/tus/documentos

# Ingestar un solo archivo
python run.py --ingest --files /ruta/a/documento.md

# Ingestar y asignar un nombre a la sesión/base de datos
python run.py --ingest --files /ruta/ --session-name "Mi Corpus V1"

# Ingestar y exportar chunks a archivos .txt
python run.py --ingest --files /ruta/ --export-chunks 
```

**Consulta:**

```bash
# Consulta simple (usará la base de datos más reciente)
python run.py --query "¿Cuál es el propósito principal del documento X?"

# Consulta especificando el número de chunks a recuperar
python run.py --query "Explica el concepto Y" --chunks 10

# Consulta usando una base de datos específica por índice (0 es la más reciente)
python run.py --query "Resume la sección Z" --db-index 1

# Iniciar modo interactivo (pregunta por la base de datos)
python run.py --query

# Iniciar modo interactivo usando una base de datos específica
python run.py --query --db-index 0
```

**Gestión de Bases de Datos y Sesiones:**

```bash
# Listar bases de datos disponibles (ordenadas por más recientes)
python run.py --list-dbs

# Listar sesiones activas
python run.py --list-sessions

# Optimizar una base de datos específica (ej. la más reciente)
python run.py --optimize-db 0

# Optimizar todas las bases de datos
python run.py --optimize-all

# Mostrar estadísticas de las bases de datos
python run.py --db-stats

# Mostrar estado del gestor de recursos
python run.py --resource-status
```

**Exportar Chunks:**

```bash
# Exportar chunks de documentos previamente ingestados (usa la última DB por defecto)
python run.py --export-chunks --files /ruta/a/documentos/originales

# Exportar chunks usando una base de datos específica
python run.py --export-chunks --files /ruta/a/documentos/originales --db-index 1
```

### Servidor Web y API REST

El script `chatbot_server.py` inicia un servidor Flask con una interfaz web y una API REST.

```bash
python chatbot_server.py [--host 0.0.0.0] [--port 5000] [--debug]
```

Accede a la interfaz web en `http://localhost:5000/`.

**Endpoints API Principales:**

- `GET /api/databases`: Lista bases de datos.
- `POST /api/sessions`: Crea una nueva sesión.
- `POST /api/query`: Procesa una consulta y devuelve una respuesta.
- `GET /api/context/<session_id>/latest`: Obtiene el contexto de la última consulta.
- `DELETE /api/sessions/<session_id>`: Elimina una sesión.
- `GET /api/diagnostics`: Obtiene información de diagnóstico.
- `GET /api/message-context/<message_id>`: Obtiene el contexto de un mensaje.
- `POST /api/cleanup`: Desencadena limpieza manual de recursos.

### Como Biblioteca

Los módulos están diseñados para ser importables. Puedes usar `main.py` o las clases individuales (`RagApp`, `EmbeddingManager`, etc.) en tus propios scripts.

```python
from main import process_query

response = process_query("Mi pregunta", n_chunks=5, db_index=0)
print(response)
```

## Pruebas y Evaluación

El proyecto incluye un sistema de pruebas unificado en `test/run_tests.py` y análisis de resultados.

### Ejecución de Pruebas

Desde la raíz del proyecto:

```bash
# Ejecutar todas las pruebas
python test/run_tests.py

# Ejecutar pruebas de un tipo específico (e.g., chunkers)
python test/run_tests.py --test-type chunkers

# Ejecutar pruebas de múltiples tipos
python test/run_tests.py --test-type databases clients

# Listar tipos de prueba disponibles
python test/run_tests.py --list-types

# Ver ayuda
python test/run_tests.py --help
```

### Análisis de Resultados

Los resultados detallados se guardan en `test/results/`. El script `test/analizar_resultados.py` puede usarse para generar resúmenes o visualizaciones.

```bash
python test/analizar_resultados.py --input-dir test/results/ --output-file analisis.txt
```

### Pruebas de Bases de Datos

Se incluyen pruebas específicas para:
- Inserción y recuperación de chunks.
- Búsqueda vectorial (precisión y rendimiento).
- Operaciones específicas de SQLite y DuckDB.

### Benchmarking

El directorio `performance_test/benchmarks/` contiene scripts para medir el rendimiento:

- `benchmark_ingestion.py`: Mide el tiempo de ingesta para un corpus estándar.
- `benchmark_queries.py`: Mide la latencia y QPS para un conjunto de consultas estándar.

Ejecuta estos scripts para evaluar el impacto de cambios en la configuración o el código.

## Extensión del Sistema

La arquitectura modular facilita la adición de nuevos componentes.

### Añadir un Nuevo Método de Chunking

1. Crea un nuevo archivo en `modulos/chunks/implementaciones/` (e.g., `mi_chunker.py`).
2. Registra tu implementación en el factory correspondiente
3. Añade la configuración específica en `config.yaml`:
```yaml
chunks:
  method: "mi_chunker"
  parameters:
    # Parámetros específicos
```
4. (Opcional) Añade pruebas unitarias en `test/databases/`.

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

- **Tipado Estricto**: Se utilizan anotaciones de tipo (`typing`) para mejorar la legibilidad y robustez.
- **Documentación**: Se espera que las clases y métodos complejos incluyan docstrings.
- **Gestión de Dependencias**: Se utiliza `pyproject.toml` (compatible con `uv`, `pip`, `poetry`).
- **Licencia**: Revisa el archivo `LICENSE` para detalles.

## Licencia

[MIT License](LICENSE)