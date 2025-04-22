# Pruebas del Sistema RAG

Este directorio contiene pruebas unitarias, de integración y rendimiento para el sistema RAG, con una interfaz unificada para facilitar la ejecución y el análisis de resultados.

## Estructura del Sistema de Pruebas

```
test/
├── utils/                     # Utilidades comunes para pruebas
│   ├── environment.py         # Preparación del entorno
│   ├── reporting.py           # Generación de informes
│   ├── discovery.py           # Descubrimiento de pruebas
│   └── constants.py           # Constantes compartidas
├── chunkers/                  # Pruebas de chunkers
│   ├── test_chunkers.py       # Pruebas para todos los chunkers
│   └── test_page_chunker.py   # Prueba específica para PageChunker
├── databases/                 # Pruebas de bases de datos
│   ├── test_sqlite_database.py
│   ├── test_duckdb_database.py
│   └── ...
├── clients/                   # Pruebas de clientes IA
│   ├── test_client_configs.py # Prueba de configuración de clientes
│   └── test_gemini.py         # Prueba específica para GeminiClient
├── embeddings/                # Pruebas de embeddings
│   ├── test_embedding_manager.py
│   ├── test_embedding_factory.py
│   └── ...
├── doc_processor/             # Pruebas del procesador de documentos
├── rag/                       # Pruebas del pipeline RAG
├── session_manager/           # Pruebas del gestor de sesiones
├── view_chunks/               # Pruebas del visualizador de chunks
├── results/                   # Resultados de pruebas estandarizados
│   ├── database_tests/        # Resultados de pruebas de databases
│   ├── chunker_tests/         # Resultados de pruebas de chunkers
│   ├── client_tests/          # Resultados de pruebas de clientes
│   └── analysis/              # Análisis de resultados
├── run_tests.py               # Script principal unificado
├── run_all_database_tests.py  # Script específico para pruebas de base de datos
└── analizar_resultados.py     # Script para analizar resultados
```

## Cómo Ejecutar Pruebas

El sistema de pruebas está diseñado para ser fácil de usar con una interfaz unificada a través del script `run_tests.py`. Este script permite ejecutar diferentes tipos de pruebas, filtrar resultados y generar informes estandarizados.

### Comandos Básicos

#### 1. Ejecutar Todas las Pruebas

```bash
python -m test.run_tests --type all
```

#### 2. Ejecutar Pruebas por Categoría

```bash
python -m test.run_tests --type databases     # Pruebas de bases de datos
python -m test.run_tests --type chunkers      # Pruebas de chunkers
python -m test.run_tests --type clients       # Pruebas de clientes IA
python -m test.run_tests --type embeddings    # Pruebas de embeddings
```

#### 3. Ejecutar Pruebas Específicas

##### Pruebas de Bases de Datos

```bash
# Probar una base de datos específica
python -m test.run_tests --type databases --db-type sqlite

# Probar todas las bases de datos con un directorio de resultados personalizado
python -m test.run_tests --type databases --results-dir mi_directorio/resultados

# Alternativa: usar el script específico para bases de datos
python -m test.run_all_database_tests
```

##### Pruebas de Chunkers

```bash
# Probar un tipo específico de chunker
python -m test.run_tests --type chunkers --chunkers character,token

# Probar chunkers con un archivo específico
python -m test.run_tests --type chunkers --file ruta/a/documento.md --chunkers context
```

##### Pruebas de Clientes

```bash
# Probar un cliente específico
python -m test.run_tests --type clients --client gemini

# Probar la configuración de clientes
python -m test.run_tests --type clients --client config
```

##### Pruebas de Embeddings

```bash
# Probar componentes específicos de embeddings
python -m test.run_tests --type embeddings --embedding-component manager
```

#### 4. Listar Pruebas Disponibles

```bash
# Listar todas las pruebas disponibles
python -m test.run_tests --type list

# Listar pruebas para una categoría específica
python -m test.run_tests --type list --type databases
```

### Personalización de Resultados

Por defecto, los resultados se guardan en el directorio `test/results/` en subdirectorios organizados por tipo de prueba. Puedes personalizar esto con los siguientes parámetros:

```bash
# Cambiar el directorio de resultados
python -m test.run_tests --type chunkers --results-dir mi_directorio/resultados

# Cambiar el directorio de análisis
python -m test.run_tests --type chunkers --analysis-dir mi_directorio/analisis
```

## Análisis de Resultados

El sistema genera automáticamente informes en formato TXT y JSON con estadísticas detalladas. Para las pruebas de chunkers, también se generan visualizaciones con el script `analizar_resultados.py`.

```bash
# Analizar resultados manualmente (para chunkers)
python -m test.analizar_resultados --dir test/results/chunker_tests --out test/results/analysis
```

## Estructura de Informes

Los informes generados tienen un formato estandarizado que incluye:

- Estadísticas generales (pruebas ejecutadas, éxitos, fallos, errores)
- Tiempo de ejecución
- Detalles de fallos y errores
- Metadatos relevantes

Los informes se guardan en formato TXT para lectura humana y JSON para procesamiento programático.

## Integración con CI/CD

El sistema está diseñado para integrarse fácilmente con sistemas de CI/CD. Puedes ejecutar pruebas específicas y establecer el código de salida adecuado para la integración con pipelines de CI:

```bash
# Ejemplo para CI/CD
python -m test.run_tests --type databases --db-type sqlite --results-dir $CI_RESULTS_DIR
```

Si hay fallos en las pruebas, el script devolverá un código de salida distinto de cero, lo que puede usarse para indicar un fallo en el pipeline. 