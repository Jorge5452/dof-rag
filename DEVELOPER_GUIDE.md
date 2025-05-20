# Guía del Desarrollador: Sistema RAG Modular

## Introducción

Esta guía proporciona información detallada para desarrolladores que deseen extender, modificar o contribuir al Sistema RAG Modular. Se explican los patrones de diseño utilizados, la estructura del código, las interfaces principales y cómo implementar nuevas funcionalidades.

## Patrones de Diseño Utilizados

El sistema utiliza varios patrones de diseño para proporcionar flexibilidad y extensibilidad:

- **Clases Abstractas**: Definen interfaces comunes para componentes sustituibles.
- **Factories**: Crean instancias concretas basadas en la configuración.
- **Singleton**: Utilizado para gestores centrales como `Config`, `SessionManager` y `ResourceManager`.
- **Strategy**: Los diferentes chunkers, modelos de embeddings, etc. son estrategias intercambiables.
- **Dependency Injection**: La configuración y las dependencias se inyectan en los componentes.

## Estructura del Proyecto

La estructura modular está organizada en directorios específicos:

```
project/
├── run.py              # Punto de entrada con modos de ingestión y consulta.
├── main.py             # Orquesta la lógica principal de ejecución.
├── config.py           # Configuración central, carga del config.yaml.
├── config.yaml         # Archivo de configuración.
├── chatbot_server.py   # Servidor web para la API REST.
├── pyproject.toml      # Definición de dependencias del proyecto.
├── run_tests.py        # Orquestador de pruebas unitarias.
└── modulos/
    ├── databases/      # Gestión de bases de datos vectoriales
    ├── chunks/         # Estrategias de chunking
    ├── embeddings/     # Gestión y generación de embeddings
    ├── clientes/       # Clientes para proveedores de IA
    ├── resource_management/ # Gestión centralizada de recursos
    ├── session_manager/ # Gestión de sesiones
    ├── rag/            # Sistema RAG y API
    ├── view_chunks/    # Visualización de chunks
    └── utils/          # Utilidades comunes
```

## Interfaces y Clases Principales

### Clases Abstractas

Las principales clases abstractas que definen las interfaces del sistema:

- `ChunkAbstract`: Define la interfaz para estrategias de chunking.
- `VectorialDatabase`: Define operaciones para bases de datos vectoriales.
- `IAClient`: Define métodos para interactuar con modelos de lenguaje.
- `EmbeddingManager`: Gestiona la generación de vectores de embedding.

### Factories

Factories principales para instanciar componentes:

- `ChunkerFactory`: Crea instancias de chunkers según configuración.
- `DatabaseFactory`: Crea instancias de bases de datos vectoriales.
- `ClientFactory`: Crea instancias de clientes de IA.
- `EmbeddingFactory`: Gestiona y crea instancias de gestores de embeddings.

### Gestores Principales

Componentes principales que orquestan operaciones:

- `Config`: Singleton para gestionar configuración centralizada.
- `SessionManager`: Gestión de sesiones de usuario y persistencia.
- `ResourceManager`: Gestión centralizada de recursos del sistema.
- `RagApp`: Clase principal para operaciones RAG.

## Sistema de Gestión de Recursos Centralizado

El sistema de gestión de recursos (`modulos/resource_management/`) proporciona una infraestructura centralizada para optimizar y controlar el uso de recursos del sistema. Está compuesto por tres componentes principales que trabajan juntos:

### ResourceManager

Actúa como el punto central de coordinación y monitoreo:

```python
# Obtener la instancia del ResourceManager (singleton)
from modulos.resource_management.resource_manager import ResourceManager
resource_manager = ResourceManager()

# Verificar el estado actual de los recursos
metrics = resource_manager.metrics
print(f"Memoria usada: {metrics['system_memory_percent']}%")

# Solicitar limpieza de recursos cuando sea necesario
resource_manager.request_cleanup(aggressive=False, reason="manual_request")
```

### MemoryManager

Gestiona la optimización de memoria y proporciona métodos útiles:

```python
# El MemoryManager se obtiene a través del ResourceManager
memory_manager = resource_manager.memory_manager

# Verificar uso de memoria
memory_status = memory_manager.check_memory_usage()
print(f"Memoria disponible: {memory_status['available_mb']} MB")

# Optimizar tamaño de lote dinámicamente
base_batch_size = 50
optimized_batch_size = memory_manager.optimize_batch_size(
    base_batch_size=base_batch_size,
    min_batch_size=10,
    max_batch_size=200
)
```

### ConcurrencyManager

Gestiona la ejecución paralela de tareas con pools optimizados:

```python
# El ConcurrencyManager se obtiene a través del ResourceManager
concurrency_manager = resource_manager.concurrency_manager

# Ejecutar tarea en pool de hilos (para operaciones I/O bound)
def io_task(file_path):
    # leer archivo, acceder a API, etc.
    return result

results = concurrency_manager.map_tasks(
    io_task, 
    file_paths_list,
    task_type="io"  # Especificar el tipo de tarea ayuda a optimizar
)

# O con procesos para tareas CPU-intensivas
def cpu_task(data):
    # procesamiento intensivo...
    return processed_data

results = concurrency_manager.map_tasks(
    cpu_task,
    data_chunks,
    task_type="cpu",
    prefer_process=True  # Forzar uso de ProcessPool
)
```

### Integración con Componentes Existentes

Los componentes existentes del sistema RAG están integrados con el sistema de gestión de recursos:

1. **SessionManager**: Se comunica con ResourceManager para limpiar sesiones cuando los recursos están bajo presión.
2. **EmbeddingFactory**: Registra modelos activos y responde a solicitudes de liberación del MemoryManager.
3. **main.py**: Utiliza optimize_batch_size para ajustar dinámicamente el procesamiento de chunks durante la ingesta.
4. **chatbot_server.py**: Proporciona endpoints para supervisar y gestionar recursos.

### Configuración del Sistema de Recursos

El comportamiento del sistema de gestión de recursos se configura en la sección `resource_management` de `config.yaml`:

```yaml
resource_management:
  # Nivel de verbosidad en los logs (minimal, normal, detailed)
  log_verbosity: "normal"
  
  # Configuración de monitoreo
  monitoring:
    interval_sec: 120  # Intervalo de monitoreo en segundos
    aggressive_threshold_mem_pct: 85  # Umbral para limpieza agresiva
    warning_threshold_mem_pct: 75  # Umbral para advertencias
```

### Mejores Prácticas para Desarrolladores

Al extender el sistema, considera estas prácticas para una gestión óptima de recursos:

1. **Accede a ResourceManager como Singleton**: Siempre obtén la instancia existente.
2. **Evita Gestión Manual de Recursos**: Para GC, liberación de modelos o gestión de pools, usa los métodos del ResourceManager.
3. **Registra Uso de Recursos**: Si tu componente consume recursos significativos, considera registrarlo con ResourceManager.
4. **Optimiza Tamaños de Lote**: Usa MemoryManager.optimize_batch_size() para procesos intensivos.
5. **Utiliza ConcurrencyManager**: Para tareas paralelas, utiliza map_tasks o run_in_executor en lugar de crear threads/procesos directamente.

## Implementación de Nuevos Componentes

### Crear un Nuevo Método de Chunking

1. Crea una nueva clase que herede de `ChunkAbstract`:

```python
from modulos.chunks.ChunkAbstract import ChunkAbstract
from typing import List, Dict, Optional

class MyCustomChunker(ChunkAbstract):
    def __init__(self, config_dict=None):
        super().__init__(config_dict)
        # Inicializar parámetros específicos
        
    def process(self, text: str, **kwargs) -> List[Dict]:
        """
        Implementar la lógica de chunking personalizada
        """
        chunks = []
        # Tu lógica de chunking aquí
        return chunks
```

2. Registra tu chunker en `ChunkerFactory.py`:

```python
def get_chunker(name=None):
    if name == "my_custom":
        from modulos.chunks.implementaciones.my_custom_chunker import MyCustomChunker
        return MyCustomChunker(config.get_chunker_method_config("my_custom"))
    # ... resto del código existente ...
```

3. Añade configuración en `config.yaml`:

```yaml
chunks:
  my_custom:
    param1: value1
    param2: value2
```

### Crear un Nuevo Cliente de IA

1. Crea una nueva clase que implemente la interfaz `IAClient`:

```python
from modulos.clientes.IAClient import IAClient
from typing import List, Dict, Optional

class MyCustomAIClient(IAClient):
    def __init__(self, config=None):
        super().__init__(config)
        # Inicializar cliente específico
        
    def generate_response(self, prompt: str, context: List[Dict], streaming=False) -> str:
        """
        Implementar generación de respuesta
        """
        # Tu lógica específica
        return response
```

2. Registra tu cliente en `ClientFactory.py`.
3. Añade configuración en `config.yaml`.

## Convenciones de Código

Para mantener un código consistente y de alta calidad:

- **Tipado**: Usar anotaciones de tipo en todos los métodos públicos.
- **Docstrings**: Documentar todas las clases y métodos siguiendo formato [Google style](https://google.github.io/styleguide/pyguide.html).
- **Manejo de Errores**: Utilizar excepciones específicas con mensajes descriptivos.
- **Logging**: Usar el sistema de logging en lugar de print().
- **Tests**: Escribir pruebas unitarias para nuevas funcionalidades.

## Pruebas

El proyecto utiliza un sistema de pruebas unificado:

- Pruebas unitarias organizadas en `test/` por componente.
- Ejecución con `python test/run_tests.py`.
- Resultados guardados en `test/results/`.

Para crear nuevas pruebas:

1. Añade un archivo de prueba en el directorio correspondiente.
2. Registra la prueba en `run_tests.py`.

## Integración con ResourceManager

Si estás creando un nuevo componente que necesita integrarse con el sistema de gestión de recursos:

1. Obtén la instancia del ResourceManager:

```python
from modulos.resource_management.resource_manager import ResourceManager

class MyComponent:
    def __init__(self):
        self.resource_manager = ResourceManager()
        
    def process_large_data(self, data):
        # Verificar estado de recursos antes de operación intensiva
        metrics = self.resource_manager.metrics
        
        if metrics["system_memory_percent"] > 90:
            # Implementar estrategia de degradación elegante
            return self._process_with_reduced_batch(data)
            
        # Uso del ConcurrencyManager para procesamiento paralelo
        return self.resource_manager.concurrency_manager.map_tasks(
            self._process_chunk, 
            self._split_data(data),
            task_type="cpu"
        )
```

2. Registra limpiadores de recursos si es necesario:

```python
def cleanup_resources(self, aggressive=False):
    """Método que puede ser llamado por ResourceManager para liberar recursos."""
    # Liberar caché, cerrar conexiones, etc.
    pass
    
# Registrar el limpiador con ResourceManager
resource_manager.register_cleanup_handler(self.cleanup_resources)
```

3. Notifica sobre cambios significativos en el uso de recursos:

```python
def allocate_large_resource(self):
    # Notificar sobre cambio significativo en uso de recursos
    self.resource_manager.notify_significant_allocation("large_model_loaded", size_mb=1500)
```

## Contribuciones

Para contribuir:

1. Crea un fork del repositorio.
2. Crea una rama para tu funcionalidad.
3. Implementa tus cambios siguiendo las convenciones.
4. Asegúrate de que las pruebas pasen.
5. Envía un pull request con una descripción clara.

## Solución de Problemas

- **Errores de Memoria**: Verifica la sección `resource_management` en config.yaml, puede necesitar ajustes.
- **Problemas de Concurrencia**: Revisa si estás creando pools/threads manualmente en lugar de usar ConcurrencyManager.
- **Liberación de Recursos**: Si encuentras fugas de memoria, verifica que tu componente se integre correctamente con ResourceManager. 