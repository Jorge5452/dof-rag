# Guía del Desarrollador: Sistema de Gestión de Recursos Centralizado

Esta guía describe cómo interactuar con el sistema de gestión de recursos (`ResourceManager`, `MemoryManager`, `ConcurrencyManager`) y cómo desarrollar nuevos módulos que se integren correctamente con él.

## 1. Uso del `ResourceManager`

El `ResourceManager` es el punto central para interactuar con el sistema de gestión de recursos. Es un Singleton, por lo que siempre obtendrás la misma instancia.

### 1.1 Obtener la Instancia

```python
from modulos.resource_management.resource_manager import ResourceManager

# Obtener la instancia única
resource_manager = ResourceManager() 
```
Asegúrate de que `ResourceManager` ya haya sido inicializado en algún punto de la aplicación (normalmente al inicio, p.ej., en `main.py` o `run.py`) para que la configuración y los sub-gestores estén listos.

### 1.2 Acceder a Métricas Actuales

El `ResourceManager` mantiene un diccionario actualizado con métricas del sistema, del proceso y de componentes específicos del RAG.

```python
# Acceder al diccionario completo de métricas
current_metrics = resource_manager.metrics

# Ejemplos de acceso a métricas específicas:
memory_usage_percent = current_metrics.get("system_memory_percent", 0.0)
active_sessions = current_metrics.get("active_sessions_rag", 0)
last_update_timestamp = current_metrics.get("last_metrics_update_ts", 0.0)

print(f"Uso de memoria actual: {memory_usage_percent}%")
```
Las métricas se actualizan periódicamente si el hilo de monitoreo está habilitado (`monitoring_enabled: true` en `config.yaml`), o puedes forzar una actualización (aunque no es lo común) llamando a `resource_manager.update_metrics()`.

### 1.3 Solicitar Limpieza Manual

Aunque la limpieza automática basada en umbrales es la norma, puedes solicitar una limpieza manual si es necesario (p.ej., después de una operación muy intensiva).

```python
# Solicitar limpieza normal
resource_manager.request_cleanup(aggressive=False, reason="manual_request_after_large_op")

# Solicitar limpieza agresiva (usar con precaución)
resource_manager.request_cleanup(aggressive=True, reason="critical_manual_request") 
```
Esto activará las rutinas de limpieza en `MemoryManager` y notificará a `SessionManager`.

### 1.4 Usar `ConcurrencyManager` para Tareas

Para ejecutar tareas de forma concurrente, obtén el `ConcurrencyManager` a través del `ResourceManager` y solicita el ejecutor adecuado (`ThreadPoolExecutor` para I/O, `ProcessPoolExecutor` para CPU).

**Ejemplo: Ejecutar una tarea I/O bound (e.g., llamada API):**
```python
def my_io_task(arg1, arg2):
    # ... lógica I/O ...
    return result

if resource_manager.concurrency_manager:
    future = resource_manager.concurrency_manager.run_in_thread_pool(my_io_task, "valor1", arg2="valor2")
    if future:
        try:
            result = future.result(timeout=60) # Esperar resultado con timeout
            print(f"Resultado tarea I/O: {result}")
        except Exception as e:
            print(f"Error en tarea I/O: {e}")
else:
    print("ConcurrencyManager no disponible.")
```

**Ejemplo: Mapear una tarea CPU bound sobre un iterable:**
```python
def my_cpu_task(item):
    # ... lógica CPU intensiva ...
    return processed_item

items_to_process = [1, 2, 3, 4, 5]

if resource_manager.concurrency_manager:
    results_iterator = resource_manager.concurrency_manager.map_tasks_in_process_pool(my_cpu_task, items_to_process)
    if results_iterator:
        processed_results = list(results_iterator) # Recoger resultados
        print(f"Resultados procesados: {processed_results}")
else:
    print("ConcurrencyManager no disponible.")
```
**Importante:** Para `ProcessPoolExecutor`, asegúrate de que la función y sus argumentos sean "picklables".

### 1.5 Consultar `MemoryManager` (Ej: Optimizar Batch Size)

Puedes acceder a funcionalidades del `MemoryManager` a través de la instancia en `ResourceManager`.

```python
base_batch = 64
if resource_manager.memory_manager:
    optimized_batch = resource_manager.memory_manager.optimize_batch_size(
        base_batch_size=base_batch, 
        min_batch_size=16 
    )
    print(f"Tamaño de lote optimizado: {optimized_batch}")
else:
    optimized_batch = base_batch
    print("MemoryManager no disponible, usando tamaño de lote base.")
```

## 2. Integración de Nuevos Módulos

Si desarrollas un nuevo módulo que maneja recursos significativos (memoria, conexiones, etc.), considera lo siguiente para integrarlo con el `ResourceManager`:

-   **Informar Estado:** Si tu módulo mantiene un estado relevante para las métricas globales (e.g., número de conexiones activas, tamaño de caché interna), modifica `ResourceManager.update_metrics()` para que llame a un método de tu módulo y recolecte esa información, añadiéndola al diccionario `resource_manager.metrics`.
-   **Participar en Limpieza Coordinada:** Si tu módulo necesita liberar recursos durante una limpieza (normal o agresiva), modifica `ResourceManager.request_cleanup()` para que, además de llamar a `MemoryManager` y `SessionManager`, llame a un método `cleanup(aggressive=...)` en tu módulo.
-   **Utilizar Concurrencia Centralizada:** En lugar de crear tus propios hilos o procesos, utiliza los pools gestionados por `ConcurrencyManager` (accedido vía `ResourceManager`) para beneficiarte de la gestión adaptativa de workers.
-   **Configuración Centralizada:** Si tu módulo necesita parámetros relacionados con la gestión de recursos (e.g., tamaño máximo de caché), considera añadirlos a la sección `resource_management` en `config.yaml` y léelos a través de la instancia `Config` en `ResourceManager` o directamente.

## 3. Configuración y Ajuste

La configuración del sistema de gestión de recursos se encuentra en la sección `resource_management` del archivo `config.yaml`.

```yaml
resource_management:
  monitoring_interval: 30         # Intervalo (s) para comprobar recursos. Más bajo = más reactivo, más overhead.
  aggressive_threshold_memory: 85 # % memoria sistema para limpieza agresiva.
  warning_threshold_memory: 75  # % memoria sistema para limpieza normal/advertencia.
  warning_threshold_cpu: 80     # % CPU sistema para advertencia (actualmente solo log).
  monitoring_enabled: true      # Habilita/deshabilita el monitoreo automático.
  
  concurrency:
    default_cpu_workers: "auto"   # Workers para CPU bound: "auto" (cores), o número fijo.
    default_io_workers: "auto"    # Workers para I/O bound: "auto" (heurística), o número fijo.
    max_total_workers: null     # Límite global opcional para workers (null = sin límite).
```

**Consejos de Ajuste:**

-   **`monitoring_interval`:** Valores más bajos (e.g., 15-30s) aumentan la reactividad pero también el overhead. Valores más altos (e.g., 60-120s) reducen el overhead pero reaccionan más lento a picos.
-   **Umbrales de Memoria:** Ajusta `warning_threshold_memory` y `aggressive_threshold_memory` según la estabilidad observada bajo carga. Si ocurren OOMs, baja los umbrales. Si la limpieza es demasiado frecuente y elimina cachés útiles, súbelos con precaución.
-   **`concurrency`:** "auto" suele ser un buen punto de partida. Si observas subutilización o sobrecarga en tareas específicas, puedes probar con valores fijos para `default_cpu_workers` o `default_io_workers`. `max_total_workers` es útil en entornos con recursos muy limitados.

Consulta los informes generados en `performance_test/results/` (si ejecutaste las pruebas de carga y benchmarks) para obtener datos que ayuden a guiar estos ajustes. 