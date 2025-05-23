# Sistema RAG Modular

Sistema modular de Retrieval Augmented Generation (RAG) con arquitectura flexible basada en clases abstractas, factories y gestión optimizada de recursos.

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

## Características Principales

- **Arquitectura Modular:** Basada en clases abstractas y factories para máxima flexibilidad y extensibilidad.
- **Múltiples Métodos de Chunking:** División por caracteres, tokens, contexto o páginas.
- **Embeddings Configurables:** Soporte para ModernBERT, CDE, E5 y otros modelos.
- **Bases de Datos Vectoriales:** Implementaciones para SQLite y DuckDB con optimizaciones vectoriales.
- **Múltiples Clientes IA:** OpenAI, Gemini, Ollama y más.
- **Gestión de Sesiones:** Sistema completo para mantener estado de usuarios.
- **Exportación de Chunks:** Visualización y exportación de chunks generados.
- **API Web y Streaming:** Interfaz web completa con streaming de respuestas.
- **Sistema de Gestión de Recursos Centralizado:** Optimización avanzada de memoria y concurrencia con `ResourceManager`, `MemoryManager` y `ConcurrencyManager`.

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

## Arquitectura del Sistema

La arquitectura modular se basa en interfaces claras (clases abstractas) con múltiples implementaciones gestionadas por factories:

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

## Sistema de Gestión de Recursos Centralizado

El sistema implementa una gestión avanzada y centralizada de recursos mediante tres componentes principales:

- **ResourceManager:** Gestiona y monitoriza los recursos del sistema, sirviendo como punto central de control para memoria, CPU y concurrencia.
- **MemoryManager:** Implementa estrategias proactivas de optimización de memoria como limpieza inteligente de caché, garbage collection adaptativo y ajuste dinámico del batch size.
- **ConcurrencyManager:** Optimiza la concurrencia con pools de workers adaptativo según el tipo de tarea y recursos disponibles.

Esta arquitectura proporciona:

- Adaptabilidad a diferentes entornos y cargas de trabajo
- Optimización automática del uso de recursos
- Monitoreo continuo del rendimiento
- Estrategias de degradación elegante bajo presión de recursos

## Uso Básico

### Instalación

```bash
pip install -r requirements.txt
```

### Ingestión de Documentos

```bash
python run.py --ingest --files docs/
```

### Ingestión y Exportación de Chunks

```bash
python run.py --ingest --export-chunks --files docs/
```

### Consulta

```bash
python run.py --query "¿Cómo funciona el sistema RAG?"
```

### Modo Interactivo

```bash
python run.py --query
```

### Servidor Web

```bash
python chatbot_server.py
```

## Configuración

Todo el sistema se configura desde `config.yaml`, donde se pueden ajustar:

- Método de chunking y sus parámetros
- Modelo de embeddings a utilizar
- Tipo de base de datos y configuración
- Cliente de IA y parámetros de generación
- Configuración del gestor de recursos y optimizaciones de memoria

## Contribuciones

Contribuciones son bienvenidas. Por favor revisa el archivo DEVELOPER_GUIDE.md para más información.

## Licencia

MIT License