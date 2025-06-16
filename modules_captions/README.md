# Módulo de Extracción de Descripciones de Imágenes (modules_captions)

Sistema modular de extracción de descripciones de imágenes con almacenamiento SQLite, manejo centralizado de errores y soporte para múltiples proveedores de IA.

## Características Principales

- **Almacenamiento en Base de Datos SQLite**: Operaciones transaccionales con interfaz CRUD
- **Manejo Centralizado de Errores**: Logging estructurado con reportes de errores en JSON
- **Arquitectura Modular**: Separación de responsabilidades en módulos especializados
- **Cliente Universal de IA**: Cliente único compatible con OpenAI para múltiples proveedores
- **Logging Avanzado**: Logging multinivel con salida colorizada
- **Procesamiento por Lotes**: Tamaños de lote configurables con limitación de velocidad

## Arquitectura

```
modules_captions/
├── __init__.py              # Module exports and version
├── extract_captions.py      # Main extraction script
├── config.json              # Provider configurations
├── debug_config.py          # Debug utilities
├── captions.db              # SQLite database (auto-generated)
├── clients/                 # AI client implementations
│   ├── __init__.py          # Client factory
│   └── openai.py            # Universal OpenAI-compatible client
├── db/                      # Database management
│   ├── __init__.py
│   ├── captions.db          # SQLite database (auto-generated)
│   └── manager.py           # SQLite CRUD operations
├── logs/                    # System logs
│   ├── caption_extractor_YYYYMMDD.log
│   └── errors_YYYYMMDD.json
└── utils/                   # Core utilities
    ├── __init__.py
    ├── error_handler.py     # Centralized error management
    └── file_processor.py    # File and image processing
```

## Instalación

### Dependencias

```bash
pip install openai pillow python-dotenv colorama tqdm
```

### Variables de Entorno

```bash
export OPENAI_API_KEY="your_api_key_here"     # OpenAI official
export GOOGLE_API_KEY="your_api_key_here"     # Google Gemini
export ANTHROPIC_API_KEY="your_api_key_here"  # Anthropic Claude
```

## Proveedores Soportados

Todos los proveedores utilizan el cliente universal compatible con OpenAI:

- **OpenAI**: gpt-4o, gpt-4-vision-preview, gpt-4o-mini
- **Google Gemini**: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash-exp
- **Anthropic Claude**: claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus
- **Ollama**: Modelos locales con API compatible con OpenAI
- **Azure OpenAI**: Modelos OpenAI desplegados en Azure
- **Endpoints Personalizados**: Cualquier API compatible con OpenAI

## Uso

### Uso Básico

```bash
# Proveedor por defecto (Gemini)
python extract_captions.py --root-dir ./images

# Especificar proveedor
python extract_captions.py --root-dir ./images --gemini
python extract_captions.py --root-dir ./images --openai
python extract_captions.py --root-dir ./images --claude

# Configuración personalizada
python extract_captions.py

# Forzar reprocesamiento
python extract_captions.py --root-dir ./images --force-reprocess
```

### Configuración Avanzada

```bash
# Procesamiento por lotes con limitación de velocidad
python extract_captions.py --root-dir ./images --batch-size 20 --cooldown-seconds 10

# Logging de depuración
python extract_captions.py --root-dir ./images --log-level DEBUG

# Estado del sistema
python extract_captions.py --status
```

### Archivo de Configuración

```json
{
  "root_dir": "./images",
  "db_path": "captions.db",
  "provider": "gemini",
  "log_dir": "logs",
  "log_level": 20,
  "providers": {
    "openai": {
      "client_config": {
        "model": "gpt-4o",
        "max_tokens": 256,
        "temperature": 0.6
      },
      "env_var": "OPENAI_API_KEY"
    },
    "gemini": {
      "client_config": {
        "model": "gemini-1.5-pro",
        "max_tokens": 256,
        "temperature": 0.6,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
      },
      "env_var": "GOOGLE_API_KEY"
    },
    "claude": {
      "client_config": {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 256,
        "temperature": 0.6,
        "base_url": "https://api.anthropic.com/v1/"
      },
      "env_var": "ANTHROPIC_API_KEY"
    }
  }
}
```

## Uso Programático

### Ejemplo Básico

```python
from modules_captions import DatabaseManager, create_client, FileProcessor

# Inicializar componentes
db_manager = DatabaseManager("captions.db")
client = create_client("openai", api_key="your_api_key")

# Procesar imágenes
processor = FileProcessor(
    root_directory="./images",
    db_manager=db_manager,
    ai_client=client,
    batch_size=10
)

results = processor.process_images()
print(processor.get_processing_summary())
```

### Manejo de Errores

```python
from modules_captions.utils import ErrorHandler

error_handler = ErrorHandler(log_dir="logs")

try:
    # Tu código de procesamiento
    pass
except Exception as e:
    error_handler.handle_api_error(e, image_path, model_info)
    
print(error_handler.get_error_report())
```

### Operaciones de Base de Datos

```python
from modules_captions.db import DatabaseManager

db = DatabaseManager("captions.db")

# Insertar descripción
db.insert_description(
    document_name="document_001",
    page_number=1,
    image_filename="image_001.png",
    description="Descripción de imagen"
)

# Verificar existencia
if db.description_exists("document_001", 1, "image_001.png"):
    description = db.get_description("document_001", 1, "image_001.png")
    print(description)

# Obtener estadísticas
stats = db.get_statistics()
print(f"Total de descripciones: {stats['total_descriptions']}")
```

## Esquema de Base de Datos

```sql
CREATE TABLE image_descriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_name TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    image_filename TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_name, page_number, image_filename)
);
```

## Logging y Monitoreo

### Archivos de Log

- `logs/caption_extractor_YYYYMMDD.log`: Log principal del sistema
- `logs/errors_YYYYMMDD.json`: Reportes detallados de errores

### Niveles de Log

- **DEBUG**: Información detallada de depuración
- **INFO**: Información general de progreso
- **WARNING**: Advertencias no bloqueantes
- **ERROR**: Errores específicos de operación
- **CRITICAL**: Errores que detienen el sistema

## Comparación con extract_captions_1

| Característica | extract_captions_1 | modules_captions |
|---------|-------------------|------------------|
| Almacenamiento | Archivos TXT | Base de datos SQLite |
| Manejo de errores | Básico | Centralizado con reportes detallados |
| Arquitectura | Monolítica | Modular (4 módulos principales) |
| Logging | Básico | Avanzado con salida colorizada |
| Recuperación | Manual | Automática con lógica de reintentos |
| Estadísticas | Limitadas | Métricas comprehensivas |
| Configuración | Hardcodeada | Flexible (CLI + JSON + env) |
| Proveedores | Múltiples clientes | Cliente universal OpenAI |
| Interfaz | Scripts separados | CLI unificado |

## Migración desde extract_captions_1

```python
from modules_captions.db import DatabaseManager
import os

def migrate_txt_to_db(txt_dir, db_path):
    """Migrar archivos TXT existentes a base de datos SQLite."""
    db = DatabaseManager(db_path)
    
    for txt_file in os.listdir(txt_dir):
        if txt_file.endswith('.txt'):
            image_name = txt_file.replace('.txt', '')
            
            with open(os.path.join(txt_dir, txt_file), 'r', encoding='utf-8') as f:
                description = f.read().strip()
            
            db.insert_description(
                document_name="migrated",
                page_number=0,
                image_filename=image_name,
                description=description
            )

# Uso: migrate_txt_to_db("/path/to/txt/files", "captions.db")
```

## Solución de Problemas

### Problemas Comunes

1. **Error de Clave API**: Verificar que la variable de entorno esté configurada correctamente
2. **Permisos de Base de Datos**: Verificar permisos de escritura en el directorio destino
3. **Problemas de Memoria**: Reducir `batch_size` para imágenes grandes
4. **Limitación de Velocidad**: Aumentar `cooldown_seconds`

### Depuración

```bash
# Logging detallado
python extract_captions.py --root-dir ./images --log-level DEBUG

# Estado del sistema
python extract_captions.py --status
```

## Contribuir

1. Seguir la estructura modular existente
2. Agregar pruebas para nueva funcionalidad
3. Actualizar documentación
4. Mantener compatibilidad con `AIClientInterface`

## Licencia

Misma licencia que el proyecto principal DOF-RAG.