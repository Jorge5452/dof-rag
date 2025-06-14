# Módulos de Extracción de Descripciones (modules_captions)

Este módulo proporciona una versión mejorada del sistema de extracción de descripciones de imágenes con almacenamiento en base de datos SQLite, mejor manejo de errores y capacidades de procesamiento mejoradas.

## Características Principales

### 🔄 Mejoras sobre `extract_captions_1`

- **Almacenamiento en Base de Datos**: Reemplaza el sistema basado en archivos TXT con SQLite
- **Manejo de Errores Mejorado**: Sistema centralizado de logging y recuperación de errores
- **Arquitectura Modular**: Separación clara de responsabilidades en módulos especializados
- **Procesamiento Transaccional**: Operaciones de base de datos con soporte para transacciones
- **Checkpoints Mejorados**: Sistema de puntos de control más robusto para recuperación
- **Cliente Universal**: Un solo cliente OpenAI que soporta múltiples proveedores

### 🏗️ Arquitectura

```
modules_captions/
├── __init__.py              # Módulo principal con imports y versión
├── extract_captions.py      # Script principal de extracción (618 líneas)
├── config.json              # Configuración unificada con todos los proveedores
├── debug_config.py          # Configuración para debugging
├── captions.db              # Base de datos SQLite (generada automáticamente)
├── clients/                 # Clientes de IA
│   ├── __init__.py         # Factory de clientes y funciones de utilidad
│   └── openai.py          # Cliente OpenAI universal para todos los proveedores
├── db/                     # Gestión de base de datos
│   ├── __init__.py
│   └── manager.py         # Gestor SQLite con operaciones CRUD
├── logs/                   # Archivos de registro del sistema
│   ├── caption_extractor_YYYYMMDD.log  # Logs principales
│   └── errors_YYYYMMDD.json            # Errores detallados
├── checkpoints/            # Puntos de control del procesamiento
│   └── processing_checkpoint.json      # Estado del procesamiento
└── utils/                  # Utilidades del sistema
    ├── __init__.py
    ├── error_handler.py   # Manejo centralizado de errores
    └── file_processor.py  # Procesador de archivos e imágenes
```

## Instalación y Configuración

### Dependencias

```bash
# Instalar dependencias requeridas
pip install openai pillow python-dotenv colorama tqdm

# Dependencias opcionales para funcionalidades adicionales
pip install sqlite3  # Incluido en Python estándar
```

### Variables de Entorno

```bash
# Configurar API key según el proveedor
export OPENAI_API_KEY="tu_api_key_aqui"        # Para OpenAI oficial
export GOOGLE_API_KEY="tu_api_key_aqui"        # Para Gemini
export ANTHROPIC_API_KEY="tu_api_key_aqui"     # Para Claude
```

## 🔑 Configuración de API Keys

### OpenAI y Proveedores Compatibles
```bash
# Opción 1: Variable de entorno
export OPENAI_API_KEY="tu_clave_api_aqui"

# Opción 2: En el archivo de configuración
{
  "api_key": "tu_clave_api_aqui",
  "provider": "openai"
}
```

### Proveedores Soportados

Todos los proveedores utilizan el cliente OpenAI universal con diferentes configuraciones:

- **OpenAI Official**: GPT-4o, GPT-4 Vision, GPT-4o-mini
- **Google Gemini**: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash-exp
- **Anthropic Claude**: claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus
- **Ollama Local**: Modelos locales con API compatible (llava, moondream, etc.)
- **Azure OpenAI**: Modelos OpenAI desplegados en Azure
- **Endpoints Personalizados**: Cualquier API compatible con OpenAI

## Uso

### Uso Básico

```bash
# Usar Gemini (por defecto)
python extract_captions.py --root-dir ./imagenes

# Especificar proveedor explícitamente
python extract_captions.py --root-dir ./imagenes --gemini

# Con configuración personalizada
python extract_captions.py --config mi_config.json

# Con diferentes proveedores (todos usan cliente OpenAI)
python extract_captions.py --root-dir ./imagenes --openai
python extract_captions.py --root-dir ./imagenes --claude
python extract_captions.py --root-dir ./imagenes --ollama

# Forzar reprocesamiento
python extract_captions.py --root-dir /ruta/a/imagenes --force-reprocess
```

### Configuración Avanzada

```bash
# Configurar tamaño de lote y tiempo de espera
python extract_captions.py --root-dir /ruta/a/imagenes --batch-size 20 --cooldown-seconds 10

# Cambiar nivel de logging
python extract_captions.py --root-dir /ruta/a/imagenes --log-level DEBUG

# Ver estado del sistema
python extract_captions.py --status
```

### Archivo de Configuración

Crea un archivo `config.json`:

```json
{
  "root_dir": "/ruta/a/imagenes",
  "db_path": "captions.db",
  "provider": "openai",
  "api_key": null,
  "log_dir": "logs",
  "checkpoint_dir": "checkpoints",
  "log_level": 20,
  "prompt": "Resume brevemente la imagen en español (máximo 3-4 oraciones por categoría)...",
  "providers": {
    "openai": {
      "client_config": {
        "model": "gpt-4o",
        "max_tokens": 256,
        "temperature": 0.6,
        "top_p": 0.6,
        "base_url": null
      },
      "env_var": "OPENAI_API_KEY",
      "rate_limits": {
        "requests_per_minute": 500,
        "tokens_per_minute": 100000
      }
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
from modules_captions import DatabaseManager, create_client, FileProcessor, ErrorHandler

# Configurar componentes
db_manager = DatabaseManager("captions.db")
client = create_client("openai", api_key="tu_api_key")
error_handler = ErrorHandler()

# Procesar archivos
processor = FileProcessor(
    root_directory="/ruta/a/imagenes",
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
    # Tu código aquí
    pass
except Exception as e:
    error_handler.handle_api_error(e, image_path, model_info)
    
# Obtener reporte de errores
print(error_handler.get_error_report())
```

### Operaciones de Base de Datos

```python
from modules_captions.db import DatabaseManager

db = DatabaseManager("captions.db")

# Insertar descripción
db.insert_description(
    document_name="documento_001",
    page_number=1,
    image_filename="imagen_001.png",
    description="Descripción de la imagen"
)

# Verificar si existe
if db.description_exists("documento_001", 1, "imagen_001.png"):
    description = db.get_description("documento_001", 1, "imagen_001.png")
    print(description)

# Obtener estadísticas
stats = db.get_statistics()
print(f"Total descripciones: {stats['total_descriptions']}")
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
- `logs/errors_YYYYMMDD.json`: Errores detallados en formato JSON
- `checkpoints/processing_checkpoint.json`: Punto de control del procesamiento

### Niveles de Log

- **DEBUG**: Información detallada para debugging
- **INFO**: Información general del progreso
- **WARNING**: Advertencias que no detienen el procesamiento
- **ERROR**: Errores que afectan operaciones específicas
- **CRITICAL**: Errores que pueden detener el sistema

## Comparación con `extract_captions_1`

| Característica | extract_captions_1 | modules_captions |
|---|---|---|
| Almacenamiento | Archivos TXT | Base de datos SQLite |
| Manejo de errores | Básico | Centralizado y detallado |
| Checkpoints | Archivos JSON simples | Sistema robusto con transacciones |
| Arquitectura | Monolítica | Modular (4 módulos principales) |
| Logging | Básico | Avanzado con colorama y múltiples niveles |
| Recuperación | Manual | Automática con reintentos |
| Estadísticas | Limitadas | Completas con métricas detalladas |
| Configuración | Hardcoded | Flexible (CLI + JSON + .env) |
| Proveedores | Múltiples clientes | Cliente universal OpenAI |
| Interfaz | Scripts separados | CLI unificado con flags |

## Migración desde `extract_captions_1`

### Script de Migración

```python
# Migrar datos existentes de TXT a SQLite
from modules_captions.db import DatabaseManager
import os
import re

def migrate_txt_to_db(txt_dir, db_path):
    db = DatabaseManager(db_path)
    
    for txt_file in os.listdir(txt_dir):
        if txt_file.endswith('.txt'):
            # Extraer información del nombre del archivo
            image_name = txt_file.replace('.txt', '')
            
            # Leer descripción
            with open(os.path.join(txt_dir, txt_file), 'r', encoding='utf-8') as f:
                description = f.read().strip()
            
            # Insertar en base de datos
            db.insert_description(
                document_name="migrated",
                page_number=0,
                image_filename=image_name,
                description=description
            )

# migrate_txt_to_db("/ruta/a/archivos/txt", "captions.db")
```

## Solución de Problemas

### Problemas Comunes

1. **Error de API Key**: Verificar que `OPENAI_API_KEY` esté configurada
2. **Permisos de Base de Datos**: Verificar permisos de escritura en el directorio
3. **Memoria Insuficiente**: Reducir `batch_size` para imágenes grandes
4. **Rate Limiting**: Aumentar `cooldown_seconds`

### Debugging

```bash
# Ejecutar con logging detallado
python extract_captions.py --root-dir /ruta --log-level DEBUG

# Verificar estado del sistema
python extract_captions.py --status
```

## Contribución

Para contribuir al proyecto:

1. Seguir la estructura modular existente
2. Añadir tests para nuevas funcionalidades
3. Documentar cambios en este README
4. Mantener compatibilidad con la interfaz `AIClientInterface`

## Licencia

Este proyecto está bajo la misma licencia que el proyecto DOF-RAG principal.