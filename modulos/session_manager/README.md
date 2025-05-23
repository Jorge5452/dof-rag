# Sistema de Sesiones Unificadas en RAG

Este módulo implementa un sistema de gestión de sesiones y bases de datos unificado para el sistema RAG. A diferencia de la implementación anterior que mantenía sesiones y bases de datos como conceptos separados, el nuevo sistema los integra en una única entidad para simplificar la administración y mejorar la coherencia.

## Características Principales

- **Un archivo por sesión/base de datos**: Cada sesión se almacena en un archivo JSON individual en la carpeta `sessions/`, facilitando la gestión y evitando conflictos.
- **Metadatos completos**: Cada archivo de sesión contiene todos los metadatos necesarios (ruta de base de datos, método de chunking, modelo de embedding, etc.), eliminando la duplicación y posibles inconsistencias.
- **Soporte para migración**: Sistema automático para migrar sesiones del formato antiguo al nuevo formato cuando se inicia por primera vez.
- **Gestión de recursos mejorada**: Integración con `ResourceManager` para gestionar eficientemente la limpieza y optimización de recursos.
- **Tracking de archivos procesados**: Mantiene una lista de todos los archivos procesados y sus metadatos, facilitando la gestión del contenido.
- **Estadísticas integradas**: Cada sesión almacena estadísticas sobre el número de documentos y chunks, facilitando la monitorización.

## Estructura del Archivo de Sesión

Cada archivo de sesión (almacenado como JSON) incluye:

```json
{
  "id": "ID_de_la_sesion",
  "name": "Nombre_descriptivo",
  "created_at": 1747947065.0649545,
  "last_modified": 1747947065.0679524,
  "db_type": "sqlite",
  "db_path": "ruta/a/la/base_de_datos.db",
  "embedding_model": "modernbert",
  "embedding_dim": 768,
  "chunking_method": "character",
  "files": [
    "/ruta/absoluta/documento1.md",
    "/ruta/absoluta/documento2.md"
  ],
  "stats": {
    "total_documents": 5,
    "total_chunks": 100,
    "creation_date": "2025-05-22T14:51:05.064954"
  }
}
```

## Uso Principal

### Comandos de la CLI

- Listar sesiones disponibles:
  ```
  python run.py --list-sessions
  ```

- Crear una nueva base de datos (sesión) e ingestar documentos:
  ```
  python run.py --ingest --files documentos/ --session-name mi_sesion
  ```

- Añadir documentos a una base de datos existente por índice:
  ```
  python run.py --ingest --files nuevos_documentos/ --db-index 0
  ```

- Consultar usando una base de datos específica:
  ```
  python run.py --query "¿Qué es RAG?" --db-index 1
  ```

### Uso en Código

```python
from modulos.session_manager.session_manager import SessionManager

# Obtener instancia (singleton)
session_manager = SessionManager()

# Listar todas las sesiones disponibles
sessions = session_manager.list_sessions()

# Crear una nueva sesión
session_id = session_manager.create_session(
    session_id="my_session",
    metadata={
        "db_type": "sqlite",
        "db_path": "path/to/db.sqlite",
        "embedding_model": "modernbert",
        "embedding_dim": 768,
        "chunking_method": "character"
    }
)

# Añadir archivos a una sesión existente
session_manager.update_session_file_list(
    session_id=session_id,
    new_files=["documento1.md", "documento2.md"]
)

# Obtener base de datos por índice (0 = más reciente)
db, session = session_manager.get_database_by_index(0)

# Crear una sesión unificada completa
unified_id = session_manager.create_unified_session(
    database_metadata={
        "name": "mi_sesion_unificada",
        "db_type": "sqlite",
        "db_path": "path/to/db.sqlite",
        "embedding_model": "modernbert",
        "embedding_dim": 768,
        "chunking_method": "character"
    },
    files_list=["documento1.md", "documento2.md"]
)
```

## Coherencia de Configuración

Una característica clave del sistema de sesiones unificadas es mantener la coherencia de configuración. Cuando se utiliza una base de datos existente para añadir documentos o realizar consultas:

1. El sistema utiliza automáticamente el mismo modelo de embeddings configurado para esa sesión.
2. Utiliza el mismo método de chunking para garantizar coherencia en la ingesta.
3. Mantiene los mismos parámetros de configuración con los que se creó la base de datos.

Esto evita problemas de incompatibilidad y garantiza consistencia en los resultados.

## Gestión de Recursos

El sistema de sesiones está integrado con `ResourceManager` para:

1. Limpiar sesiones inactivas cuando los recursos son limitados.
2. Realizar limpieza periódica para liberar memoria y espacio en disco.
3. Optimizar el uso de recursos mediante limpieza proactiva.

## Migración desde el Formato Antiguo

Si existen sesiones en el formato antiguo (archivos `sessions.json` y `db_metadata.json`), el sistema las migra automáticamente al nuevo formato en el primer inicio. Los archivos antiguos se conservan en una carpeta de backup para referencia. 