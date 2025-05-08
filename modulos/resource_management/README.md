# Sistema de Gestión de Recursos Centralizado

Este módulo (`modulos/resource_management/`) es responsable de la gestión centralizada y optimizada de los recursos del sistema RAG, incluyendo la memoria y la concurrencia.

## Componentes Principales

El sistema de gestión de recursos consta de los siguientes componentes principales:

### 1. `resource_manager.py` (`ResourceManager`)

-   **Rol Principal:** Es el orquestador central y el punto de entrada Singleton para la gestión de recursos.
-   **Funcionalidades:**
    -   Monitoriza continuamente el uso de CPU y memoria del sistema y del proceso RAG.
    -   Carga la configuración específica para la gestión de recursos desde `config.yaml` (sección `resource_management`).
    -   Coordina las operaciones de limpieza de memoria delegando a `MemoryManager`.
    -   Gestiona y proporciona acceso a pools de concurrencia (hilos y procesos) a través de `ConcurrencyManager`.
    -   Recolecta métricas de componentes clave del sistema RAG como `SessionManager` (sesiones activas) y `EmbeddingFactory` (modelos activos).
    -   Inicia acciones de limpieza (normal o agresiva) basadas en umbrales de uso de recursos predefinidos.
    -   Proporciona información estática y dinámica sobre el estado de los recursos del sistema.

### 2. `memory_manager.py` (`MemoryManager`)

-   **Rol Principal:** Implementa las estrategias y operaciones específicas para la gestión y optimización de la memoria.
-   **Funcionalidades:**
    -   Ejecuta la recolección de basura (Garbage Collection) de Python de forma controlada.
    -   Libera recursos asociados a modelos de embedding que ya no están en uso activo (interactuando con `EmbeddingFactory`).
    -   Proporciona una función para optimizar dinámicamente el tamaño de los lotes (`batch_size`) para operaciones intensivas en memoria, basándose en el uso actual de recursos del sistema.
    -   Incluye placeholders para futuras estrategias como la limpieza de cachés específicas de Python o la comprobación de fragmentación de memoria.
-   **Interacción:** Es instanciado y gestionado por `ResourceManager`, que le delega las tareas de limpieza cuando es necesario.

### 3. `concurrency_manager.py` (`ConcurrencyManager`)

-   **Rol Principal:** Gestiona la ejecución concurrente de tareas, optimizando el uso de hilos y procesos.
-   **Funcionalidades:**
    -   Inicializa y mantiene pools de `ThreadPoolExecutor` (para tareas I/O bound) y `ProcessPoolExecutor` (para tareas CPU bound).
    -   Calcula dinámicamente el número de workers para cada pool basándose en los cores de CPU disponibles y la configuración (`default_cpu_workers`, `default_io_workers`, `max_total_workers` desde `config.yaml` via `ResourceManager`).
    -   Proporciona métodos para obtener instancias de estos ejecutores y para enviar tareas a ellos (`run_in_thread_pool`, `run_in_process_pool`, `map_tasks_in_thread_pool`, `map_tasks_in_process_pool`).
    -   Maneja el cierre controlado (`shutdown`) de los pools de ejecutores.
-   **Interacción:** Es instanciado y gestionado por `ResourceManager`. Otros módulos del sistema RAG (como `main.py` para la ingesta de documentos) pueden solicitar ejecutores a `ConcurrencyManager` (generalmente a través de `ResourceManager`) para paralelizar operaciones.

## Interacción General

1.  El `ResourceManager` actúa como el cerebro, monitorizando el sistema.
2.  Cuando se superan ciertos umbrales de uso de recursos (e.g., memoria alta), `ResourceManager` solicita una limpieza.
3.  Esta solicitud se delega al `MemoryManager`, que ejecuta sus estrategias de limpieza (GC, liberación de modelos, etc.).
4.  Paralelamente, otros módulos pueden solicitar al `ResourceManager` acceso al `ConcurrencyManager` para ejecutar tareas de forma eficiente.
5.  La configuración de todos estos componentes (intervalos de monitoreo, umbrales, número de workers) se define centralizadamente en la sección `resource_management` del archivo `config.yaml`.

La documentación detallada de la API de cada clase y sus métodos se encuentra en los docstrings dentro de los respectivos archivos `.py`. 