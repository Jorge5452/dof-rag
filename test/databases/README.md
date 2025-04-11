# Tests de Bases de Datos Vectoriales

Este directorio contiene pruebas y utilidades para trabajar con las implementaciones de bases de datos vectoriales del proyecto.

## Estructura de archivos

- **test_vectorial_database.py**: Contiene la clase base para todas las pruebas de bases de datos vectoriales.
- **test_sqlite_database.py**: Pruebas específicas para la implementación SQLite.
- **test_duckdb_database.py**: Pruebas específicas para la implementación DuckDB.
- **test_factory_database.py**: Pruebas para el patrón Factory que crea instancias de bases de datos.
- **test_chunk_insertion.py**: Pruebas para inserción de chunks de documento.
- **test_chunk_operations.py**: Pruebas para operaciones avanzadas con chunks.
- **utils.py**: Utilidades compartidas para las pruebas.
- **create_test_data.py**: Script para generar bases de datos de prueba pobladas con datos sintéticos.
- **benchmark_vector_search.py**: Herramienta para evaluar el rendimiento de búsquedas vectoriales.

## Cómo ejecutar las pruebas

### Ejecutar todas las pruebas

```bash
python test/run_database_tests.py
```

### Ejecutar pruebas específicas

```bash
python -m unittest test.databases.test_sqlite_database
python -m unittest test.databases.test_duckdb_database
```

### Ejecutar una prueba individual

```bash
python -m unittest test.databases.test_sqlite_database.SQLiteDatabaseTest.test_vector_search
```

## Utilidades

### Crear bases de datos de prueba

Para generar una base de datos SQLite de prueba con 100 documentos y 20 chunks cada uno:

```bash
python test/databases/create_test_data.py --type sqlite --path ./test_db.sqlite --docs 100 --chunks 20
```

### Comparar rendimiento

Para comparar el rendimiento de búsqueda vectorial entre SQLite y DuckDB:

```bash
python test/databases/benchmark_vector_search.py --sqlite ./test_sqlite.db --duckdb ./test_duckdb.db --queries 50 --plot comparison.png
```

## Dimensiones de los Embeddings

Para simplificar las pruebas, utilizamos embeddings de dimensión pequeña (10) por defecto. En un entorno real, las dimensiones típicas serían mucho mayores (384, 768, 1536, etc.).

## Depuración

Los resultados de las pruebas se almacenan en `test/resultados_db_tests/`.
