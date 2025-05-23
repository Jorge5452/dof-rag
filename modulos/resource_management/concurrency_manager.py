# Standard library imports
import os
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import inspect
import functools
import types
import re
import sys
import pickle
import weakref
from collections import defaultdict
import threading
import platform
import psutil

# Type checking imports
from typing import Optional, Union, Callable, Iterable, Iterator, TYPE_CHECKING, Any, Dict, Tuple, List, Literal, Set, TypeVar, cast

# Type hints for avoiding circular imports
if TYPE_CHECKING:
    from .resource_manager import ResourceManager

# Definición de estados de pool
PoolState = Literal["active", "hibernated", "shutdown"]

# Tipo genérico para resultados
T = TypeVar('T')
R = TypeVar('R')

# Caché global de verificaciones de serialización
_PICKLEABILITY_CACHE: Dict[str, bool] = {}
_PICKLEABILITY_CACHE_MAX_SIZE = 500  # Limitar tamaño para evitar crecimiento descontrolado
_PICKLEABILITY_CACHE_HITS = 0
_PICKLEABILITY_CACHE_MISSES = 0
_LAST_CACHE_CLEANUP = time.time()

# Lista de tipos nativos que son siempre serializables
_ALWAYS_SERIALIZABLE_TYPES = {
    int, float, bool, str, bytes, 
    tuple, list, dict, set, frozenset,
    type(None)
}

# Patrones para detectar tareas según el nombre
_CPU_TASK_PATTERNS = [
    r'process', r'calc', r'comput', r'transform', r'convert', 
    r'generat', r'hash', r'encod', r'encrypt', r'decrypt'
]
_IO_TASK_PATTERNS = [
    r'read', r'write', r'load', r'save', r'fetch', r'download',
    r'upload', r'request', r'query', r'get', r'post', r'send'
]

def _cleanup_pickleability_cache():
    """
    Limpia la caché de verificación de serialización cuando supera el tamaño máximo.
    
    Esto evita que la caché crezca indefinidamente, manteniendo las entradas más recientes.
    """
    global _PICKLEABILITY_CACHE, _LAST_CACHE_CLEANUP
    
    # Solo limpiar si realmente es necesario y ha pasado suficiente tiempo
    now = time.time()
    if len(_PICKLEABILITY_CACHE) > _PICKLEABILITY_CACHE_MAX_SIZE and now - _LAST_CACHE_CLEANUP > 300:
        # Crear una nueva caché con solo la mitad de las entradas más recientes
        cache_items = list(_PICKLEABILITY_CACHE.items())
        half_size = _PICKLEABILITY_CACHE_MAX_SIZE // 2
        _PICKLEABILITY_CACHE = dict(cache_items[-half_size:])
        _LAST_CACHE_CLEANUP = now

# Clase nueva para el tracking histórico de rendimiento de los workers
class WorkerPerformanceTracker:
    """
    Rastrea el rendimiento histórico de diferentes configuraciones de workers
    para permitir análisis predictivo y toma de decisiones más inteligentes.

    Atributos:
        max_history_entries (int): Número máximo de entradas históricas a mantener por tipo
        history (Dict): Historial de rendimiento para diferentes configuraciones
        last_recalculation_time (float): Timestamp de la última recalculación
        cooling_period_sec (float): Periodo de enfriamiento tras recálculos
        stability_threshold (int): Número mínimo de muestras antes de considerar cambios
        performance_variance_threshold (float): Umbral de variación para considerar cambios
    """
    def __init__(self, max_history_entries: int = 20, cooling_period_sec: float = 300.0):
        self.max_history_entries = max_history_entries
        self.cooling_period_sec = cooling_period_sec
        self.last_recalculation_time = 0.0
        self.stability_threshold = 3  # Mínimo de muestras para considerar estable
        self.performance_variance_threshold = 0.15  # 15% de variación
        
        # Estructura para almacenar historial por tipo de pool y configuración
        self.history: Dict[str, List[Dict[str, Any]]] = {
            "thread_pool": [],
            "process_pool": [],
            "task_performance": {},  # Rendimiento por tipo de tarea
            "global_metrics": []     # Métricas globales del sistema
        }
        
        # Métricas actuales que representan la última ejecución conocida
        self.current_metrics = {
            "thread_pool": None,
            "process_pool": None,
            "system_pressure": 0.0,  # Combinación de CPU y memoria
            "last_update_time": 0.0
        }
        
        # Contador para establecer un período de calentamiento
        self.warmup_count = 0
        self.warmup_threshold = 5  # Mínimo de ejecuciones de calentamiento

    def record_pool_performance(self, pool_type: str, worker_count: int, 
                               tasks_completed: int, execution_time: float,
                               task_type: str, system_metrics: Dict[str, float]) -> None:
        """
        Registra el rendimiento de un pool de workers para una configuración específica.
        
        Args:
            pool_type (str): Tipo de pool ("thread_pool" o "process_pool")
            worker_count (int): Número de workers utilizados
            tasks_completed (int): Número de tareas completadas
            execution_time (float): Tiempo total de ejecución en segundos
            task_type (str): Tipo de tarea ejecutada
            system_metrics (Dict[str, float]): Métricas del sistema durante la ejecución
        """
        if pool_type not in self.history:
            return
            
        # Calcular métricas derivadas
        tasks_per_second = tasks_completed / max(execution_time, 0.001)
        tasks_per_worker = tasks_completed / max(worker_count, 1)
        efficiency = tasks_per_second / max(worker_count, 1)
        
        # Calcular presión del sistema (0-1.0)
        cpu_pressure = system_metrics.get("cpu_percent", 0) / 100
        memory_pressure = system_metrics.get("memory_percent", 0) / 100
        system_pressure = (cpu_pressure * 0.6) + (memory_pressure * 0.4)  # Ponderado
        
        # Crear entrada para el historial
        entry = {
            "timestamp": time.time(),
            "worker_count": worker_count,
            "tasks_completed": tasks_completed,
            "execution_time": execution_time,
            "tasks_per_second": tasks_per_second,
            "tasks_per_worker": tasks_per_worker,
            "efficiency": efficiency,
            "task_type": task_type,
            "system_pressure": system_pressure,
            "cpu_percent": system_metrics.get("cpu_percent", 0),
            "memory_percent": system_metrics.get("memory_percent", 0)
        }
        
        # Añadir al historial del tipo específico de pool
        self.history[pool_type].append(entry)
        
        # Limitar el tamaño del historial
        if len(self.history[pool_type]) > self.max_history_entries:
            self.history[pool_type] = self.history[pool_type][-self.max_history_entries:]
            
        # Registrar en historial específico de tipo de tarea
        if task_type not in self.history["task_performance"]:
            self.history["task_performance"][task_type] = []
            
        task_entry = entry.copy()
        self.history["task_performance"][task_type].append(task_entry)
        
        # Limitar el historial por tipo de tarea
        if len(self.history["task_performance"][task_type]) > self.max_history_entries:
            self.history["task_performance"][task_type] = self.history["task_performance"][task_type][-self.max_history_entries:]
            
        # Actualizar métricas actuales
        self.current_metrics[pool_type] = entry
        self.current_metrics["system_pressure"] = system_pressure
        self.current_metrics["last_update_time"] = time.time()
        
        # Incrementar contador de calentamiento
        self.warmup_count += 1

    def record_system_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Registra métricas del sistema para análisis de tendencias.
        
        Args:
            metrics (Dict[str, float]): Métricas del sistema (CPU, memoria, etc.)
        """
        entry = {
            "timestamp": time.time(),
            "cpu_percent": metrics.get("cpu_percent", 0),
            "memory_percent": metrics.get("memory_percent", 0),
            "cpu_count": metrics.get("cpu_count", os.cpu_count() or 1)
        }
        
        self.history["global_metrics"].append(entry)
        
        if len(self.history["global_metrics"]) > self.max_history_entries:
            self.history["global_metrics"] = self.history["global_metrics"][-self.max_history_entries:]

    def predict_optimal_worker_count(self, pool_type: str, task_type: str, 
                                    system_metrics: Dict[str, float]) -> Optional[int]:
        """
        Predice el número óptimo de workers basado en el historial de rendimiento.
        
        Args:
            pool_type (str): Tipo de pool ("thread_pool" o "process_pool")
            task_type (str): Tipo de tarea a ejecutar
            system_metrics (Dict[str, float]): Métricas actuales del sistema
            
        Returns:
            Optional[int]: Número óptimo de workers o None si no hay suficientes datos
        """
        # Verificar si estamos en período de calentamiento
        if self.warmup_count < self.warmup_threshold:
            return None
            
        # Verificar suficientes datos históricos
        if pool_type not in self.history or not self.history[pool_type]:
            return None
            
        # Si tenemos historial específico de tipo de tarea, usarlo
        task_history = self.history["task_performance"].get(task_type, [])
        pool_history = self.history[pool_type]
        
        # Si tenemos suficiente historial específico de tarea, priorizarlo
        if len(task_history) >= self.stability_threshold:
            history_to_use = task_history
        else:
            history_to_use = pool_history
            
        if not history_to_use:
            return None
            
        # Calcular presión del sistema actual
        cpu_pressure = system_metrics.get("cpu_percent", 0) / 100
        memory_pressure = system_metrics.get("memory_percent", 0) / 100
        current_pressure = (cpu_pressure * 0.6) + (memory_pressure * 0.4)
        
        # Búsqueda de configuraciones similares en condiciones de sistema similares
        # 1. Filtrar por similitud de presión del sistema
        pressure_threshold = 0.2  # Tolerancia de desviación
        similar_entries = [
            entry for entry in history_to_use
            if abs(entry.get("system_pressure", 0) - current_pressure) <= pressure_threshold
        ]
        
        # Si no hay suficientes entradas similares, usar todo el historial
        if len(similar_entries) < 2:
            similar_entries = history_to_use
            
        # 2. Encontrar configuración con mejor eficiencia
        if similar_entries:
            # Ordenar por eficiencia (mayor a menor)
            sorted_entries = sorted(similar_entries, key=lambda x: x.get("efficiency", 0), reverse=True)
            
            # Tomar las mejores configuraciones (top 3)
            top_entries = sorted_entries[:3] if len(sorted_entries) >= 3 else sorted_entries
            
            # Calcular la media de workers de las mejores configuraciones
            optimal_workers = int(sum(entry.get("worker_count", 1) for entry in top_entries) / len(top_entries))
            
            # Aplicar ajuste basado en presión actual
            if current_pressure > 0.8:  # Alta presión, reducir workers
                optimal_workers = max(1, int(optimal_workers * 0.8))
            elif current_pressure < 0.3:  # Baja presión, posible incremento
                optimal_workers = int(optimal_workers * 1.2)
                
            return optimal_workers
        
        return None

    def should_recalculate(self, current_metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Determina si se debe recalcular el número de workers basado en:
        - Tiempo desde la última recalculación (periodo de enfriamiento)
        - Cambios significativos en las métricas del sistema
        - Historial de rendimiento
        
        Args:
            current_metrics (Dict[str, float]): Métricas actuales del sistema
            
        Returns:
            Tuple[bool, str]: (Recalcular?, Razón)
        """
        current_time = time.time()
        
        # 1. Verificar periodo de enfriamiento
        time_since_last_recalc = current_time - self.last_recalculation_time
        if time_since_last_recalc < self.cooling_period_sec:
            return False, f"Periodo de enfriamiento activo ({time_since_last_recalc:.1f}s < {self.cooling_period_sec}s)"
            
        # 2. Verificar cambios significativos en métricas del sistema
        if self.current_metrics["system_pressure"] > 0:
            # Calcular presión actual
            cpu_pressure = current_metrics.get("cpu_percent", 0) / 100
            memory_pressure = current_metrics.get("memory_percent", 0) / 100
            current_pressure = (cpu_pressure * 0.6) + (memory_pressure * 0.4)
            
            # Calcular cambio en presión
            pressure_delta = abs(current_pressure - self.current_metrics["system_pressure"])
            
            # Si cambio significativo, recalcular
            if pressure_delta > 0.25:  # 25% de cambio en presión total
                return True, f"Cambio significativo en presión del sistema: {pressure_delta:.2f}"
                
        # 3. Verificar si estamos en warmup (siempre recalcular)
        if self.warmup_count < self.warmup_threshold:
            return True, f"Periodo de calentamiento activo ({self.warmup_count}/{self.warmup_threshold})"
            
        # 4. Recalculación periódica (cada 10 minutos si nada más lo provoca)
        if time_since_last_recalc > 600:  # 10 minutos
            return True, "Recalculación periódica (10 minutos)"
            
        return False, "No se requiere recálculo"
        
    def record_recalculation(self) -> None:
        """Registra que se ha realizado una recalculación."""
        self.last_recalculation_time = time.time()

class ConcurrencyManager:
    """
    Gestiona la concurrencia y paralelismo en el sistema RAG.

    Se encarga de:
    - Proporcionar pools de hilos y procesos optimizados para diferentes tipos de tareas
    - Ajustar dinámicamente el número de workers según la carga del sistema
    - Ejecutar tareas en paralelo/concurrencia con manejo automático de recursos
    - Hibernar y despertar pools según sea necesario para optimizar recursos

    Atributos:
        resource_manager (ResourceManager): Instancia del gestor de recursos
        thread_pool (ThreadPoolExecutor): Pool de hilos para operaciones I/O bound
        process_pool (ProcessPoolExecutor): Pool de procesos para operaciones CPU bound
        logger (logging.Logger): Logger para esta clase
        cpu_workers (int): Número de workers para tareas intensivas en CPU
        io_workers (int): Número de workers para tareas I/O bound
        default_timeout (float): Tiempo máximo de espera para tareas (segundos)
        task_types (Dict[str, Dict]): Configuraciones para diferentes tipos de tareas
        disable_process_pool (bool): Indica si se debe desactivar el ProcessPoolExecutor
        performance_tracker (WorkerPerformanceTracker): Tracker de rendimiento histórico
        pools_status (Dict): Estado de los pools (active, hibernated, shutdown)
        recalculation_frequency_sec (float): Frecuencia de recálculo de workers
    """

    def __init__(self, resource_manager_instance: 'ResourceManager'):
        """
        Inicializa el gestor de concurrencia.
        
        Args:
            resource_manager_instance (ResourceManager): Instancia del gestor de recursos
        """
        # Inicializar atributos básicos
        self.resource_manager = resource_manager_instance
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Para seguimiento de rendimiento histórico y decisiones inteligentes
        self.performance_tracker = WorkerPerformanceTracker()
        
        # Configurar la frecuencia de recálculo desde ResourceManager
        self.recalculation_frequency_sec = getattr(
            resource_manager_instance, 
            'worker_recalculation_frequency', 
            300.0  # 5 minutos por defecto
        )
        self.performance_tracker.cooling_period_sec = self.recalculation_frequency_sec
        
        # Estado de los pools
        self.pools_status = {
            "thread_pool": {
                "status": "inactive",  # inactive, active, hibernated, shutdown
                "created_at": 0.0,
                "last_used": 0.0,
                "task_count": 0,
                "hibernated_at": 0.0,
                "workers": 0,
                "stored_state": None,  # Para almacenar estado durante hibernación
                "last_status_change": 0.0  # Para cooldown entre cambios de estado
            },
            "process_pool": {
                "status": "inactive",
                "created_at": 0.0,
                "last_used": 0.0,
                "task_count": 0,
                "hibernated_at": 0.0,
                "workers": 0,
                "stored_state": None,  # Para almacenar estado durante hibernación
                "last_status_change": 0.0  # Para cooldown entre cambios de estado
            }
        }
        
        # Umbrales para reinicialización de pools
        self.worker_change_threshold_pct = 20.0  # Porcentaje mínimo de cambio de workers para reiniciar
        self.pool_cooldown_period_sec = 60.0  # Segundos mínimos entre cambios significativos de estado
        
        # Configuración de recálculo
        self.last_worker_calculation_time = 0.0
        self.worker_calculation_interval_sec = 120.0
        self.worker_recalc_count = 0
        self.dynamic_recalc_interval = True  # Ajustar intervalo dinámicamente
        
        # Inicializar número de workers
        self.cpu_workers = self._calculate_optimal_workers("cpu")
        self.io_workers = self._calculate_optimal_workers("io")
        
        # Inicializar atributos adicionales
        self.thread_pool = None
        self.process_pool = None
        self.disable_process_pool = getattr(resource_manager_instance, 'disable_process_pool', False)
        self.max_total_workers = getattr(resource_manager_instance, 'max_total_workers', None)
        self.default_timeout = getattr(resource_manager_instance, 'default_timeout_sec', 120)
        
        # Configuración de tipos de tareas
        self.task_types = {
            "default": {"prefer_process": False},
            "cpu_intensive": {"prefer_process": True},
            "io_intensive": {"prefer_process": False}
        }
        
        # Actualizar estado inicial de pools
        self.pools_status["process_pool"]["workers"] = self.cpu_workers
        self.pools_status["thread_pool"]["workers"] = self.io_workers
        
    def hibernate_pool(self, pool_type: str) -> bool:
        """
        Pone un pool en estado de hibernación sin destruirlo completamente.
        Guarda su configuración para futura restauración pero libera recursos.
        
        Args:
            pool_type (str): Tipo de pool ("thread_pool" o "process_pool")
            
        Returns:
            bool: True si el pool fue hibernado, False en caso contrario
        """
        # Verificar si el pool existe y está activo
        if pool_type not in self.pools_status:
            return False
            
        pool_info = self.pools_status[pool_type]
        current_time = time.time()
        
        # Si no está activo, no hay nada que hibernar
        if pool_info["status"] != "active":
            return False
            
        # Verificar periodo de enfriamiento
        time_since_last_change = current_time - pool_info.get("last_status_change", 0)
        if time_since_last_change < self.pool_cooldown_period_sec:
            self.logger.debug(f"Pool {pool_type} en período de enfriamiento ({time_since_last_change:.1f}s < {self.pool_cooldown_period_sec}s)")
            return False
            
        try:
            # Hibernar el pool según su tipo
            if pool_type == "thread_pool" and self.thread_pool is not None:
                # Guardar estado para restauración futura
                pool_info["stored_state"] = {
                    "workers": self.thread_pool._max_workers,
                    "tasks_pending": sum(1 for _ in self.thread_pool._work_queue.unfinished_tasks) 
                                    if hasattr(self.thread_pool._work_queue, "unfinished_tasks") else 0
                }
                
                # Hibernar thread pool - no iniciar nuevas tareas pero permitir que terminen las actuales
                self.thread_pool._shutdown = True  # Marcar como shutdown pero sin afectar tareas en curso
                
                # Actualizar estado
                pool_info["status"] = "hibernated"
                pool_info["hibernated_at"] = current_time
                pool_info["last_status_change"] = current_time
                
                self.logger.info(f"Pool de hilos hibernado con {pool_info['stored_state']['workers']} workers")
                return True
                
            elif pool_type == "process_pool" and self.process_pool is not None:
                # Guardar estado para restauración futura
                pool_info["stored_state"] = {
                    "workers": self.process_pool._max_workers,
                    "tasks_pending": sum(1 for _ in self.process_pool._pending_work_items.values()) 
                                   if hasattr(self.process_pool, "_pending_work_items") else 0
                }
                
                # Hibernar process pool - no iniciar nuevas tareas pero permitir que terminen las actuales
                self.process_pool._shutdown = True
                
                # Actualizar estado
                pool_info["status"] = "hibernated"  
                pool_info["hibernated_at"] = current_time
                pool_info["last_status_change"] = current_time
                
                self.logger.info(f"Pool de procesos hibernado con {pool_info['stored_state']['workers']} workers")
                return True
                
        except Exception as e:
            self.logger.error(f"Error al hibernar pool {pool_type}: {e}")
        
        return False
        
    def wake_pool(self, pool_type: str) -> bool:
        """
        Restaura un pool hibernado a su estado activo.
        Crea un nuevo pool con la configuración almacenada durante la hibernación.
        
        Args:
            pool_type (str): Tipo de pool ("thread_pool" o "process_pool")
            
        Returns:
            bool: True si el pool fue restaurado, False en caso contrario
        """
        # Verificar si el pool está hibernado
        if pool_type not in self.pools_status:
            return False
            
        pool_info = self.pools_status[pool_type]
        current_time = time.time()
        
        # Solo despertar pools hibernados
        if pool_info["status"] != "hibernated":
            return False
            
        # Verificar periodo de enfriamiento
        time_since_last_change = current_time - pool_info.get("last_status_change", 0)
        if time_since_last_change < self.pool_cooldown_period_sec:
            self.logger.debug(f"Pool {pool_type} en período de enfriamiento ({time_since_last_change:.1f}s < {self.pool_cooldown_period_sec}s)")
            return False
        
        try:
            # Obtener número correcto de workers (desde estado almacenado o valores actuales)
            stored_state = pool_info.get("stored_state", {})
            workers = stored_state.get("workers", 0) if stored_state else 0
            
            # Si no hay información guardada, usar valores actuales
            if workers <= 0:
                workers = self.io_workers if pool_type == "thread_pool" else self.cpu_workers
            
            # Despertar el pool según su tipo
            if pool_type == "thread_pool":
                # Crear nuevo pool de hilos con la configuración almacenada
                self.thread_pool = ThreadPoolExecutor(max_workers=workers)
                
                # Actualizar estado
                pool_info["status"] = "active"
                pool_info["created_at"] = current_time
                pool_info["last_used"] = current_time
                pool_info["last_status_change"] = current_time
                pool_info["workers"] = workers
                
                self.logger.info(f"Pool de hilos restaurado de hibernación con {workers} workers")
                return True
                
            elif pool_type == "process_pool" and not self.disable_process_pool:
                # Crear nuevo pool de procesos con la configuración almacenada
                self.process_pool = ProcessPoolExecutor(max_workers=workers)
                
                # Actualizar estado
                pool_info["status"] = "active"
                pool_info["created_at"] = current_time
                pool_info["last_used"] = current_time
                pool_info["last_status_change"] = current_time
                pool_info["workers"] = workers
                
                self.logger.info(f"Pool de procesos restaurado de hibernación con {workers} workers")
                return True
                
        except Exception as e:
            self.logger.error(f"Error al despertar pool {pool_type}: {e}")
            # Marcar como inactivo en caso de error
            pool_info["status"] = "inactive"
        
        return False
        
    def _is_executor_shutdown(self, executor) -> bool:
        """
        Verifica si un executor está apagado o hibernado.
        Esta función soporta tanto ThreadPoolExecutor como ProcessPoolExecutor.
        
        Args:
            executor: El executor a verificar
            
        Returns:
            bool: True si está apagado o hibernado, False en caso contrario
        """
        if executor is None:
            return True
            
        # Verificar si el executor está apagado
        try:
            return getattr(executor, '_shutdown', False)
        except (AttributeError, TypeError):
            # Si no podemos acceder a _shutdown, asumir que está apagado por seguridad
            return True
            
    def get_thread_pool_executor(self) -> ThreadPoolExecutor:
        """
        Obtiene o inicializa un ThreadPoolExecutor para tareas I/O-bound.
        Implementación mejorada con soporte para hibernación y restauración.
        
        Returns:
            ThreadPoolExecutor: El pool de hilos
        """
        # Si el pool es None, crearlo
        if self.thread_pool is None:
            # Verificar si está hibernado y despertarlo
            if self.pools_status["thread_pool"]["status"] == "hibernated":
                if not self.wake_pool("thread_pool"):
                    # Si falló la restauración, crear un nuevo pool
                    self.thread_pool = ThreadPoolExecutor(max_workers=self.io_workers)
                    self.pools_status["thread_pool"]["status"] = "active"
                    self.pools_status["thread_pool"]["created_at"] = time.time()
                    self.pools_status["thread_pool"]["last_used"] = time.time()
                    self.pools_status["thread_pool"]["workers"] = self.io_workers
            else:
                # Crear un nuevo pool desde cero
                self.thread_pool = ThreadPoolExecutor(max_workers=self.io_workers)
                self.pools_status["thread_pool"]["status"] = "active"
                self.pools_status["thread_pool"]["created_at"] = time.time()
                self.pools_status["thread_pool"]["last_used"] = time.time()
                self.pools_status["thread_pool"]["workers"] = self.io_workers
        
        # Si el pool está apagado pero no hibernado, recrearlo
        elif self._is_executor_shutdown(self.thread_pool) and self.pools_status["thread_pool"]["status"] != "hibernated":
            self.thread_pool = ThreadPoolExecutor(max_workers=self.io_workers)
            self.pools_status["thread_pool"]["status"] = "active"
            self.pools_status["thread_pool"]["created_at"] = time.time()
            self.pools_status["thread_pool"]["last_used"] = time.time()
            self.pools_status["thread_pool"]["workers"] = self.io_workers
        
        # Actualizar último uso
        self.pools_status["thread_pool"]["last_used"] = time.time()
        
        return self.thread_pool
    
    def get_process_pool_executor(self) -> Optional[ProcessPoolExecutor]:
        """
        Obtiene o inicializa un ProcessPoolExecutor para tareas CPU-bound.
        Implementación mejorada con soporte para hibernación y restauración.
        
        Returns:
            Optional[ProcessPoolExecutor]: El pool de procesos, o None si está desactivado
        """
        # Si los process pools están deshabilitados, devolver None
        if self.disable_process_pool:
            return None
        
        # Si el pool es None, crearlo
        if self.process_pool is None:
            # Verificar si está hibernado y despertarlo
            if self.pools_status["process_pool"]["status"] == "hibernated":
                if not self.wake_pool("process_pool"):
                    # Si falló la restauración, crear un nuevo pool
                    self.process_pool = ProcessPoolExecutor(max_workers=self.cpu_workers)
                    self.pools_status["process_pool"]["status"] = "active"
                    self.pools_status["process_pool"]["created_at"] = time.time()
                    self.pools_status["process_pool"]["last_used"] = time.time()
                    self.pools_status["process_pool"]["workers"] = self.cpu_workers
            else:
                # Crear un nuevo pool desde cero
                self.process_pool = ProcessPoolExecutor(max_workers=self.cpu_workers)
                self.pools_status["process_pool"]["status"] = "active"
                self.pools_status["process_pool"]["created_at"] = time.time()
                self.pools_status["process_pool"]["last_used"] = time.time()
                self.pools_status["process_pool"]["workers"] = self.cpu_workers
        
        # Si el pool está apagado pero no hibernado, recrearlo
        elif self._is_executor_shutdown(self.process_pool) and self.pools_status["process_pool"]["status"] != "hibernated":
            self.process_pool = ProcessPoolExecutor(max_workers=self.cpu_workers)
            self.pools_status["process_pool"]["status"] = "active"
            self.pools_status["process_pool"]["created_at"] = time.time()
            self.pools_status["process_pool"]["last_used"] = time.time()
            self.pools_status["process_pool"]["workers"] = self.cpu_workers
        
        # Actualizar último uso
        self.pools_status["process_pool"]["last_used"] = time.time()
        
        return self.process_pool
        
    def shutdown_executors(self, wait: bool = True) -> None:
        """
        Cierra ordenadamente todos los pools de executors.
        
        Args:
            wait (bool): Si es True, espera a que terminen las tareas pendientes
        """
        # Cerrar thread pool
        if self.thread_pool is not None:
            try:
                # Si estaba hibernado, actualizar estado directamente
                if self.pools_status["thread_pool"]["status"] == "hibernated":
                    self.pools_status["thread_pool"]["status"] = "shutdown"
                else:
                    self.thread_pool.shutdown(wait=wait)
                    self.pools_status["thread_pool"]["status"] = "shutdown"
                
                # Liberar referencia
                self.thread_pool = None
                self.logger.debug("Pool de hilos cerrado correctamente")
            except Exception as e:
                self.logger.error(f"Error al cerrar thread pool: {e}")
        
        # Cerrar process pool
        if self.process_pool is not None:
            try:
                # Si estaba hibernado, actualizar estado directamente
                if self.pools_status["process_pool"]["status"] == "hibernated":
                    self.pools_status["process_pool"]["status"] = "shutdown"
                else:
                    self.process_pool.shutdown(wait=wait)
                    self.pools_status["process_pool"]["status"] = "shutdown"
                
                # Liberar referencia
                self.process_pool = None
                self.logger.debug("Pool de procesos cerrado correctamente")
            except Exception as e:
                self.logger.error(f"Error al cerrar process pool: {e}")
        
        # Forzar GC para liberar recursos
        try:
            import gc
            gc.collect()
        except ImportError:
            pass

    def _reinitialize_pools_with_new_workers(self) -> None:
        """
        Reinicializa los pools de executors con el número actualizado de workers.
        Versión mejorada que preserva pools existentes cuando es posible.
        """
        # Verificar si realmente necesitamos reinicializar los pools
        need_thread_reinit = False
        need_process_reinit = False
        
        # Para thread pool
        if self.thread_pool is not None and not self._is_executor_shutdown(self.thread_pool):
            # Obtener el número actual de workers
            current_io_workers = getattr(self.thread_pool, "_max_workers", 0)
            
            # Solo reinicializar si el cambio es significativo (> threshold%)
            if current_io_workers > 0:
                change_percent = abs(self.io_workers - current_io_workers) / current_io_workers * 100
                need_thread_reinit = change_percent > self.worker_change_threshold_pct
                
                if need_thread_reinit:
                    self.logger.info(f"Reinicializando thread pool: {current_io_workers} -> {self.io_workers} workers (cambio {change_percent:.1f}%)")
            else:
                need_thread_reinit = True
        else:
            # Si no hay pool activo, necesitamos inicializar
            need_thread_reinit = True
        
        # Para process pool
        if self.process_pool is not None and not self._is_executor_shutdown(self.process_pool):
            # Obtener el número actual de workers
            current_cpu_workers = getattr(self.process_pool, "_max_workers", 0)
            
            # Solo reinicializar si el cambio es significativo (> threshold%)
            if current_cpu_workers > 0:
                change_percent = abs(self.cpu_workers - current_cpu_workers) / current_cpu_workers * 100
                need_process_reinit = change_percent > self.worker_change_threshold_pct
                
                if need_process_reinit:
                    self.logger.info(f"Reinicializando process pool: {current_cpu_workers} -> {self.cpu_workers} workers (cambio {change_percent:.1f}%)")
            else:
                need_process_reinit = True
        else:
            # Si no hay pool activo, necesitamos inicializar (excepto si process pool está deshabilitado)
            need_process_reinit = not self.disable_process_pool
        
        # Reinicializar solo los pools que realmente lo necesitan
        if need_thread_reinit:
            try:
                # Cerrar el pool anterior si existe
                if self.thread_pool is not None:
                    self.thread_pool.shutdown(wait=False)
                    
                # Crear nuevo pool
                self.thread_pool = ThreadPoolExecutor(max_workers=self.io_workers)
                
                # Actualizar estado
                self.pools_status["thread_pool"]["status"] = "active"
                self.pools_status["thread_pool"]["created_at"] = time.time()
                self.pools_status["thread_pool"]["workers"] = self.io_workers
                self.pools_status["thread_pool"]["last_status_change"] = time.time()
            except Exception as e:
                self.logger.error(f"Error al reinicializar thread pool: {e}")
        
        if need_process_reinit and not self.disable_process_pool:
            try:
                # Cerrar el pool anterior si existe
                if self.process_pool is not None:
                    self.process_pool.shutdown(wait=False)
                    
                # Crear nuevo pool
                self.process_pool = ProcessPoolExecutor(max_workers=self.cpu_workers)
                
                # Actualizar estado
                self.pools_status["process_pool"]["status"] = "active"
                self.pools_status["process_pool"]["created_at"] = time.time()
                self.pools_status["process_pool"]["workers"] = self.cpu_workers
                self.pools_status["process_pool"]["last_status_change"] = time.time()
            except Exception as e:
                self.logger.error(f"Error al reinicializar process pool: {e}")

    def recalculate_workers_if_needed(self) -> bool:
        """
        Verifica si es necesario recalcular el número de workers y lo hace si aplica.
        
        Mejora con sistema de histéresis y tracking de rendimiento para evitar recálculos
        innecesarios y adaptar dinámicamente basado en rendimiento histórico.
        
        Returns:
            bool: True si se recalcularon los workers, False en caso contrario
        """
        # Obtener métricas actuales del sistema
        current_metrics = {
            "cpu_percent": self.resource_manager.metrics.get("cpu_percent_system", 0),
            "memory_percent": self.resource_manager.metrics.get("system_memory_percent", 0),
            "cpu_count": os.cpu_count() or 1
        }
        
        # Registrar métricas actuales en tracker de rendimiento
        self.performance_tracker.record_system_metrics(current_metrics)
        
        # Verificar si debemos recalcular según el tracker de rendimiento
        should_recalc, reason = self.performance_tracker.should_recalculate(current_metrics)
        
        # Si estamos usando intervalo dinámico, ajustarlo basado en recálculos anteriores
        if self.dynamic_recalc_interval:
            # Después de varios recálculos, aumentar el intervalo gradualmente
            if self.worker_recalc_count > 5:
                # Máximo 3 veces la configuración base
                max_interval = self.recalculation_frequency_sec * 3
                # Incremento logarítmico que crece más lentamente con el tiempo
                new_interval = min(
                    max_interval,
                    self.recalculation_frequency_sec * (1 + (math.log(self.worker_recalc_count - 4) / 2))
                )
                self.worker_calculation_interval_sec = new_interval
                
                # También actualizar el período de enfriamiento del tracker
                self.performance_tracker.cooling_period_sec = new_interval
        
        if not should_recalc:
            # Usar un nivel DEBUG para reducir verbosidad de logs
            self.logger.debug(f"Recálculo de workers omitido: {reason}")
            return False
            
        # Si llegamos aquí, procedemos con el recálculo
        # Verificar si hay predicciones disponibles desde el tracker
        predicted_cpu_workers = self.performance_tracker.predict_optimal_worker_count(
            "process_pool", "default", current_metrics
        )
        
        predicted_io_workers = self.performance_tracker.predict_optimal_worker_count(
            "thread_pool", "default", current_metrics
        )
        
        # Calcular la configuración óptima usando predicción si está disponible
        old_cpu_workers = self.cpu_workers
        old_io_workers = self.io_workers
        
        # Recalcular CPU workers
        if predicted_cpu_workers is not None:
            # Usar predicción con pequeño ajuste basado en presión actual
            new_cpu_workers = predicted_cpu_workers
            # Limitar por configuración mínima y máxima
            cpu_min = 1
            cpu_max = os.cpu_count() or 4  # Mínimo 4 si no se puede determinar
            tentative_cpu_workers = max(cpu_min, min(new_cpu_workers, cpu_max))
        else:
            # Si no hay predicción, usar el método tradicional
            tentative_cpu_workers = self._calculate_optimal_workers("cpu")
            
        # Recalcular IO workers
        if predicted_io_workers is not None:
            new_io_workers = predicted_io_workers
            # IO puede tener más workers ya que son tareas bloqueantes
            io_min = 2
            io_max = (os.cpu_count() or 4) * 2
            tentative_io_workers = max(io_min, min(new_io_workers, io_max))
        else:
            tentative_io_workers = self._calculate_optimal_workers("io")
        
        # Obtener umbral de cambio (porcentaje) desde el ResourceManager
        change_threshold_pct = getattr(self.resource_manager, 'worker_change_threshold_pct', 15)
        
        # Aplicar histéresis para evitar cambios pequeños y oscilaciones
        apply_cpu_change = False
        apply_io_change = False
        
        # Solo aplicar cambios si superan el umbral de variación
        if old_cpu_workers > 0:
            cpu_change_pct = abs(tentative_cpu_workers - old_cpu_workers) / old_cpu_workers * 100
            apply_cpu_change = cpu_change_pct >= change_threshold_pct
        else:
            apply_cpu_change = True  # Siempre aplicar si no hay configuración previa
            
        if old_io_workers > 0:
            io_change_pct = abs(tentative_io_workers - old_io_workers) / old_io_workers * 100
            apply_io_change = io_change_pct >= change_threshold_pct
        else:
            apply_io_change = True  # Siempre aplicar si no hay configuración previa
            
        # Aplicar cambios solo si superan el umbral o son los primeros
        if apply_cpu_change:
            self.cpu_workers = tentative_cpu_workers
        if apply_io_change:
            self.io_workers = tentative_io_workers
            
        # Verificar límite global de workers si está configurado
        if self.max_total_workers:
            total_workers = self.cpu_workers + self.io_workers
            if total_workers > self.max_total_workers:
                # Reducir proporcionalmente
                reduction_factor = self.max_total_workers / total_workers
                self.cpu_workers = max(1, int(self.cpu_workers * reduction_factor))
                self.io_workers = max(2, int(self.io_workers * reduction_factor))
        
        # Registrar recálculo y actualizar contadores
        self.performance_tracker.record_recalculation()
        self.last_worker_calculation_time = time.time()
        self.worker_recalc_count += 1
        
        # Verificar si realmente cambiaron los valores
        if old_cpu_workers != self.cpu_workers or old_io_workers != self.io_workers:
            changes = []
            if old_cpu_workers != self.cpu_workers:
                cpu_change_pct = abs(self.cpu_workers - old_cpu_workers) / max(1, old_cpu_workers) * 100
                changes.append(f"CPU: {old_cpu_workers} → {self.cpu_workers} ({cpu_change_pct:.1f}%)")
            if old_io_workers != self.io_workers:
                io_change_pct = abs(self.io_workers - old_io_workers) / max(1, old_io_workers) * 100
                changes.append(f"IO: {old_io_workers} → {self.io_workers} ({io_change_pct:.1f}%)")
                
            changes_str = ", ".join(changes)
            self.logger.info(f"Workers recalculados - {changes_str} (Razón: {reason}, Umbral: {change_threshold_pct}%)")
            
            # Actualizar estado de pools
            self.pools_status["process_pool"]["workers"] = self.cpu_workers
            self.pools_status["thread_pool"]["workers"] = self.io_workers
            
            return True
        else:
            self.logger.debug(f"Recálculo completado sin cambios. Razón: {reason}")
            return False

    def map_tasks(self, func: Callable[[Any], R], iterable: Iterable[Any], 
                 chunksize: Optional[int] = None, task_type: str = "default",
                 timeout: Optional[float] = None, prefer_process: bool = False) -> List[R]:
        """
        Ejecuta la función para cada elemento del iterable en paralelo, seleccionando
        automáticamente el mejor executor según el tipo de tarea.
        
        Versión optimizada que reduce la sobrecarga y mejora el rendimiento:
        1. Evita conversiones innecesarias a listas cuando es posible
        2. Realiza un tracking de rendimiento mínimo
        3. Utiliza procesamiento por lotes eficiente
        
        Args:
            func: La función a ejecutar para cada elemento
            iterable: Los elementos a procesar
            chunksize: Tamaño de los lotes para procesamiento
            task_type: Tipo de tarea para seleccionar configuración apropiada
            timeout: Tiempo máximo de espera en segundos
            prefer_process: Si se debe usar ProcessPool incluso para tareas I/O bound
            
        Returns:
            Lista con los resultados de aplicar la función a cada elemento
        """
        # Heurística para verificar el tamaño del iterable sin convertirlo a lista
        # cuando es posible para evitar sobrecarga de memoria
        try:
            # Intentar obtener la longitud directamente
            n_items = len(iterable)
            items = iterable  # Mantener el iterable original
        except (TypeError, AttributeError):
            # Si no tiene método len(), convertir a lista
            items = list(iterable)
            n_items = len(items)
        
        if n_items == 0:
            return []
            
        # Seleccionar el pool apropiado sin operaciones costosas
        start_time = time.time()
        pool_type = "process_pool" if prefer_process or self._is_cpu_bound_task(task_type) else "thread_pool"
        
        # Obtener el pool y el número actual de workers
        if pool_type == "process_pool":
            use_pool = self.get_process_pool_executor()
            workers = self.cpu_workers
        else:
            use_pool = self.get_thread_pool_executor()
            workers = self.io_workers
        
        # Obtener chunksize óptimo pero solo si no fue especificado
        if chunksize is None:
            # Cálculo simplificado para evitar overhead
            if n_items < workers * 2:
                chunksize = 1  # Para pocos elementos
            elif n_items < 1000:  
                chunksize = max(1, n_items // (workers * 2))  # Granularidad media
            else:
                chunksize = max(1, n_items // workers)  # Chunks grandes
        
        # Métricas del sistema - Minimizar acceso para reducir overhead
        system_metrics = {
            "cpu_percent": self.resource_manager.metrics.get("cpu_percent_system", 0),
            "memory_percent": self.resource_manager.metrics.get("system_memory_percent", 0)
        }
        
        # Ejecutar las tareas
        try:
            if pool_type == "process_pool" and self.disable_process_pool:
                # Si los process pools están deshabilitados, usar ejecución secuencial
                results = [func(item) for item in items]
            else:
                results = list(use_pool.map(func, items, chunksize=chunksize or 1))
            
            # Calcular tiempo de ejecución
            execution_time = time.time() - start_time
            
            # Actualizar estado de pool - solo métricas esenciales
            self.pools_status[pool_type]["last_used"] = time.time()
            self.pools_status[pool_type]["task_count"] += n_items
            
            # Registrar rendimiento pero solo si la tarea es significativa
            if n_items > 10 or execution_time > 1.0:
                self.performance_tracker.record_pool_performance(
                    pool_type=pool_type,
                    worker_count=workers,
                    tasks_completed=n_items,
                    execution_time=execution_time,
                    task_type=task_type,
                    system_metrics=system_metrics
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error en map_tasks ({pool_type}): {e}")
            # Intentar proceso secuencial como fallback
            self.logger.info("Intentando procesamiento secuencial como fallback")
            return [func(item) for item in items]
            
    def get_optimal_chunksize(self, task_type: str, iterable_length: int, pool_type: Optional[str] = None) -> int:
        """
        Calcula el tamaño de lote óptimo para una operación map basado en
        características de la tarea y tamaño del iterable.
        
        Versión optimizada para maximizar rendimiento y minimizar overhead.
        
        Args:
            task_type: El tipo de tarea ("default", "cpu_intensive", "io_intensive")
            iterable_length: La longitud del iterable a procesar
            pool_type: Tipo de pool que se usará (si se conoce)
            
        Returns:
            int: Tamaño de lote óptimo
        """
        # Si no se especificó pool_type, determinarlo basado en tipo de tarea
        if pool_type is None:
            pool_type = "process_pool" if self._is_cpu_bound_task(task_type) else "thread_pool"
        
        # Obtener número de workers según pool_type
        workers = self.cpu_workers if pool_type == "process_pool" else self.io_workers
        
        # Asegurar mínimo 1 worker
        workers = max(1, workers)
        
        # Cálculo simplificado de chunksize basado en tipo de tarea y longitud del iterable
        if iterable_length <= workers * 2:
            # Si hay pocas tareas, usar un chunksize pequeño
            return 1
        
        if task_type == "io_intensive":
            # Para IO-bound, usar chunks más pequeños para aprovechar paralelismo de IO
            base_chunksize = max(1, iterable_length // (workers * 4))
        elif task_type == "cpu_intensive":
            # Para CPU-bound, chunks más grandes reducen overhead
            base_chunksize = max(1, iterable_length // workers)
        else:  # default
            # Para tareas generales, equilibrio
            base_chunksize = max(1, iterable_length // (workers * 2))
        
        # Ajustes adicionales basados en tamaño del iterable
        if iterable_length > 10000:
            # Para iterables muy grandes, aumentar chunksize para reducir overhead
            base_chunksize = max(base_chunksize, iterable_length // 500)
        elif iterable_length < 100:
            # Para iterables pequeños, limitar chunksize
            base_chunksize = min(base_chunksize, 5)
        
        return base_chunksize

    def get_worker_counts(self) -> Dict[str, int]:
        """
        Obtiene el conteo actual de workers para los diferentes pools.
        
        Returns:
            Dict[str, int]: Diccionario con conteo de workers por tipo
        """
        return {
            "cpu_workers": self.cpu_workers,
            "io_workers": self.io_workers,
            "max_total": self.max_total_workers,
            "cpu_pool_status": self.pools_status["process_pool"]["status"],
            "io_pool_status": self.pools_status["thread_pool"]["status"],
            "recalc_interval": self.worker_calculation_interval_sec,
            "recalc_count": self.worker_recalc_count
        }

    def hibernate_pool_if_unused(self, pool_type: str, idle_seconds: float = 600) -> bool:
        """
        Pone en modo hibernación un pool sin destruirlo si no se ha usado 
        por cierto tiempo. Esto libera recursos manteniendo la estructura.
        
        Args:
            pool_type (str): "thread_pool" o "process_pool"
            idle_seconds (float): Segundos de inactividad para hibernar
            
        Returns:
            bool: True si el pool fue hibernado, False en caso contrario
        """
        # Verificar si el pool existe y está activo
        if pool_type not in self.pools_status:
            return False
            
        pool_info = self.pools_status[pool_type]
        current_time = time.time()
        
        # Si el pool no está activo, no hay nada que hacer
        if pool_info["status"] != "active":
            return False
            
        # Si no ha pasado suficiente tiempo, no hibernar
        time_since_last_use = current_time - pool_info["last_used"]
        if time_since_last_use < idle_seconds:
            return False
            
        # Hibernar el pool adecuado
        try:
            if pool_type == "thread_pool" and hasattr(self, "thread_pool"):
                # Guardar el número de workers antes de hibernar
                workers = self.thread_pool._max_workers
                self.thread_pool.shutdown(wait=False)
                self.thread_pool = None
                pool_info["workers"] = workers
                pool_info["status"] = "hibernated"
                pool_info["hibernated_at"] = current_time
                self.logger.info(f"Pool de hilos hibernado tras {time_since_last_use:.1f}s de inactividad")
                return True
                
            elif pool_type == "process_pool" and hasattr(self, "process_pool"):
                # Guardar el número de workers antes de hibernar
                workers = self.process_pool._max_workers
                self.process_pool.shutdown(wait=False)
                self.process_pool = None
                pool_info["workers"] = workers
                pool_info["status"] = "hibernated"
                pool_info["hibernated_at"] = current_time
                self.logger.info(f"Pool de procesos hibernado tras {time_since_last_use:.1f}s de inactividad")
                return True
                
        except Exception as e:
            self.logger.error(f"Error al hibernar pool {pool_type}: {e}")
            
        return False

    def restore_pool_from_hibernation(self, pool_type: str) -> bool:
        """
        Restaura un pool previamente hibernado a su estado activo.
        
        Args:
            pool_type (str): "thread_pool" o "process_pool"
            
        Returns:
            bool: True si el pool fue restaurado, False en caso contrario
        """
        if pool_type not in self.pools_status:
            return False
            
        pool_info = self.pools_status[pool_type]
        
        # Solo restaurar si está hibernado
        if pool_info["status"] != "hibernated":
            return False
            
        try:
            if pool_type == "thread_pool":
                # Restaurar con el mismo número de workers de antes
                workers = pool_info["workers"] if pool_info["workers"] > 0 else self.io_workers
                self.thread_pool = ThreadPoolExecutor(max_workers=workers)
                pool_info["status"] = "active"
                pool_info["last_used"] = time.time()
                pool_info["created_at"] = time.time()
                self.logger.info(f"Pool de hilos restaurado de hibernación con {workers} workers")
                return True
                
            elif pool_type == "process_pool" and not self.disable_process_pool:
                # Restaurar con el mismo número de workers de antes
                workers = pool_info["workers"] if pool_info["workers"] > 0 else self.cpu_workers
                self.process_pool = ProcessPoolExecutor(max_workers=workers)
                pool_info["status"] = "active"
                pool_info["last_used"] = time.time()
                pool_info["created_at"] = time.time()
                self.logger.info(f"Pool de procesos restaurado de hibernación con {workers} workers")
                return True
                
        except Exception as e:
            self.logger.error(f"Error al restaurar pool {pool_type} de hibernación: {e}")
            pool_info["status"] = "inactive"  # Marcar como inactivo en caso de error
            
        return False

    def _calculate_optimal_workers(self, worker_type: str) -> int:
        """
        Calcula el número óptimo de workers según el tipo y las condiciones del sistema.
        
        Esta función considera:
        - Configuración explícita desde ResourceManager
        - Número de cores disponibles
        - Uso actual de CPU y memoria
        - Tipo de entorno (desarrollo, producción, etc.)
        
        Args:
            worker_type (str): Tipo de workers a calcular ("cpu" o "io")
            
        Returns:
            int: Número óptimo de workers
        """
        # Obtener la configuración desde ResourceManager
        if worker_type == "cpu":
            config_value = getattr(self.resource_manager, 'default_cpu_workers', "auto")
        else:  # io
            config_value = getattr(self.resource_manager, 'default_io_workers', "auto")
            
        # Si hay una configuración explícita que no sea "auto", usarla
        if isinstance(config_value, int) and config_value > 0:
            return config_value
            
        # Obtener número de cores disponibles
        cpu_count = os.cpu_count() or 4  # Fallback a 4 si no se puede determinar
        
        # Obtener métricas actuales del sistema
        cpu_percent = self.resource_manager.metrics.get("cpu_percent_system", 0)
        memory_percent = self.resource_manager.metrics.get("system_memory_percent", 0)
        
        # Calcular factor de ajuste basado en uso de recursos
        # A mayor uso, menor cantidad de workers para no sobrecargar
        resource_factor = 1.0
        if cpu_percent > 80:
            resource_factor *= 0.7  # Reducir 30% si CPU está muy cargada
        elif cpu_percent > 60:
            resource_factor *= 0.85  # Reducir 15% si CPU está moderadamente cargada
            
        if memory_percent > 85:
            resource_factor *= 0.7  # Reducir 30% si memoria está muy cargada
        elif memory_percent > 70:
            resource_factor *= 0.85  # Reducir 15% si memoria está moderadamente cargada
            
        # Ajustar según tipo de entorno
        environment_type = getattr(self.resource_manager, 'environment_type', "development")
        
        # Entornos de desarrollo suelen tener más recursos disponibles para el proceso
        if "development" in environment_type:
            env_factor = 1.0
        # Entornos de producción suelen ser más conservadores
        elif "production" in environment_type or "server" in environment_type:
            env_factor = 0.9
        # Entornos cloud suelen tener recursos compartidos
        elif any(env in environment_type for env in ["cloud", "aws", "gcp", "azure"]):
            env_factor = 0.8
        # Contenedores suelen tener recursos muy limitados
        elif any(env in environment_type for env in ["container", "kubernetes"]):
            env_factor = 0.7
        else:
            env_factor = 0.9  # Valor por defecto
            
        # Calcular el número base de workers según el tipo
        if worker_type == "cpu":
            # Para CPU-bound, usar número de cores físicos
            # Intentar obtener cores físicos, fallback a lógicos
            physical_cores = psutil.cpu_count(logical=False) or cpu_count
            base_workers = max(1, int(physical_cores * resource_factor * env_factor))
            
            # Asegurar al menos 1 worker y no más que cores lógicos
            return max(1, min(base_workers, cpu_count))
        else:  # io
            # Para IO-bound, se pueden usar más workers ya que están bloqueados
            # Típicamente 2x o más que el número de cores
            io_multiplier = 2.0  # Multiplicador para IO vs CPU
            base_workers = max(2, int(cpu_count * io_multiplier * resource_factor * env_factor))
            
            # Asegurar al menos 2 workers para IO y no más que 4x cores
            return max(2, min(base_workers, cpu_count * 4))

    def _is_cpu_bound_task(self, task_type: str) -> bool:
        """
        Determina si una tarea es CPU-intensiva basándose en su tipo y patrones de nombre.
        
        Args:
            task_type (str): Tipo de tarea a evaluar
            
        Returns:
            bool: True si la tarea es CPU-intensiva, False si es I/O-intensiva
        """
        # Tipos explícitos
        if task_type == "cpu_intensive":
            return True
        elif task_type == "io_intensive":
            return False
        
        # Para "default" u otros tipos, usar patrones heurísticos
        task_lower = task_type.lower()
        
        # Verificar patrones de tareas CPU-intensivas
        for pattern in _CPU_TASK_PATTERNS:
            if re.search(pattern, task_lower):
                return True
        
        # Patrones adicionales para I/O
        io_patterns = [
            r'read', r'write', r'download', r'upload', r'fetch', 
            r'save', r'load', r'request', r'response', r'network'
        ]
        
        for pattern in io_patterns:
            if re.search(pattern, task_lower):
                return False
        
        # Por defecto, asumir I/O-bound (más conservador para evitar sobrecarga del sistema)
        return False

    def hibernate_pool_if_unused(self, pool_type: str, idle_seconds: float = 600) -> bool:
        """
        Pone en modo hibernación un pool sin destruirlo si no se ha usado 
        por cierto tiempo. Esto libera recursos manteniendo la estructura.
        
        Args:
            pool_type (str): "thread_pool" o "process_pool"
            idle_seconds (float): Segundos de inactividad para hibernar
            
        Returns:
            bool: True si el pool fue hibernado, False en caso contrario
        """
        # Verificar si el pool existe y está activo
        if pool_type not in self.pools_status:
            return False
            
        pool_info = self.pools_status[pool_type]
        current_time = time.time()
        
        # Si el pool no está activo, no hay nada que hacer
        if pool_info["status"] != "active":
            return False
            
        # Si no ha pasado suficiente tiempo, no hibernar
        time_since_last_use = current_time - pool_info["last_used"]
        if time_since_last_use < idle_seconds:
            return False
            
        # Hibernar el pool adecuado
        try:
            if pool_type == "thread_pool" and hasattr(self, "thread_pool"):
                # Guardar el número de workers antes de hibernar
                workers = self.thread_pool._max_workers
                self.thread_pool.shutdown(wait=False)
                self.thread_pool = None
                pool_info["workers"] = workers
                pool_info["status"] = "hibernated"
                pool_info["hibernated_at"] = current_time
                self.logger.info(f"Pool de hilos hibernado tras {time_since_last_use:.1f}s de inactividad")
                return True
                
            elif pool_type == "process_pool" and hasattr(self, "process_pool"):
                # Guardar el número de workers antes de hibernar
                workers = self.process_pool._max_workers
                self.process_pool.shutdown(wait=False)
                self.process_pool = None
                pool_info["workers"] = workers
                pool_info["status"] = "hibernated"
                pool_info["hibernated_at"] = current_time
                self.logger.info(f"Pool de procesos hibernado tras {time_since_last_use:.1f}s de inactividad")
                return True
                
        except Exception as e:
            self.logger.error(f"Error al hibernar pool {pool_type}: {e}")
            
        return False

    def restore_pool_from_hibernation(self, pool_type: str) -> bool:
        """
        Restaura un pool previamente hibernado a su estado activo.
        
        Args:
            pool_type (str): "thread_pool" o "process_pool"
            
        Returns:
            bool: True si el pool fue restaurado, False en caso contrario
        """
        if pool_type not in self.pools_status:
            return False
            
        pool_info = self.pools_status[pool_type]
        
        # Solo restaurar si está hibernado
        if pool_info["status"] != "hibernated":
            return False
            
        try:
            if pool_type == "thread_pool":
                # Restaurar con el mismo número de workers de antes
                workers = pool_info["workers"] if pool_info["workers"] > 0 else self.io_workers
                self.thread_pool = ThreadPoolExecutor(max_workers=workers)
                pool_info["status"] = "active"
                pool_info["last_used"] = time.time()
                pool_info["created_at"] = time.time()
                self.logger.info(f"Pool de hilos restaurado de hibernación con {workers} workers")
                return True
                
            elif pool_type == "process_pool" and not self.disable_process_pool:
                # Restaurar con el mismo número de workers de antes
                workers = pool_info["workers"] if pool_info["workers"] > 0 else self.cpu_workers
                self.process_pool = ProcessPoolExecutor(max_workers=workers)
                pool_info["status"] = "active"
                pool_info["last_used"] = time.time()
                pool_info["created_at"] = time.time()
                self.logger.info(f"Pool de procesos restaurado de hibernación con {workers} workers")
                return True
                
        except Exception as e:
            self.logger.error(f"Error al restaurar pool {pool_type} de hibernación: {e}")
            pool_info["status"] = "inactive"  # Marcar como inactivo en caso de error
            
        return False

    def _calculate_optimal_workers(self, worker_type: str) -> int:
        """
        Calcula el número óptimo de workers según el tipo y las condiciones del sistema.
        
        Esta función considera:
        - Configuración explícita desde ResourceManager
        - Número de cores disponibles
        - Uso actual de CPU y memoria
        - Tipo de entorno (desarrollo, producción, etc.)
        
        Args:
            worker_type (str): Tipo de workers a calcular ("cpu" o "io")
            
        Returns:
            int: Número óptimo de workers
        """
        # Obtener la configuración desde ResourceManager
        if worker_type == "cpu":
            config_value = getattr(self.resource_manager, 'default_cpu_workers', "auto")
        else:  # io
            config_value = getattr(self.resource_manager, 'default_io_workers', "auto")
            
        # Si hay una configuración explícita que no sea "auto", usarla
        if isinstance(config_value, int) and config_value > 0:
            return config_value
            
        # Obtener número de cores disponibles
        cpu_count = os.cpu_count() or 4  # Fallback a 4 si no se puede determinar
        
        # Obtener métricas actuales del sistema
        cpu_percent = self.resource_manager.metrics.get("cpu_percent_system", 0)
        memory_percent = self.resource_manager.metrics.get("system_memory_percent", 0)
        
        # Calcular factor de ajuste basado en uso de recursos
        # A mayor uso, menor cantidad de workers para no sobrecargar
        resource_factor = 1.0
        if cpu_percent > 80:
            resource_factor *= 0.7  # Reducir 30% si CPU está muy cargada
        elif cpu_percent > 60:
            resource_factor *= 0.85  # Reducir 15% si CPU está moderadamente cargada
            
        if memory_percent > 85:
            resource_factor *= 0.7  # Reducir 30% si memoria está muy cargada
        elif memory_percent > 70:
            resource_factor *= 0.85  # Reducir 15% si memoria está moderadamente cargada
            
        # Ajustar según tipo de entorno
        environment_type = getattr(self.resource_manager, 'environment_type', "development")
        
        # Entornos de desarrollo suelen tener más recursos disponibles para el proceso
        if "development" in environment_type:
            env_factor = 1.0
        # Entornos de producción suelen ser más conservadores
        elif "production" in environment_type or "server" in environment_type:
            env_factor = 0.9
        # Entornos cloud suelen tener recursos compartidos
        elif any(env in environment_type for env in ["cloud", "aws", "gcp", "azure"]):
            env_factor = 0.8
        # Contenedores suelen tener recursos muy limitados
        elif any(env in environment_type for env in ["container", "kubernetes"]):
            env_factor = 0.7
        else:
            env_factor = 0.9  # Valor por defecto
            
        # Calcular el número base de workers según el tipo
        if worker_type == "cpu":
            # Para CPU-bound, usar número de cores físicos
            # Intentar obtener cores físicos, fallback a lógicos
            physical_cores = psutil.cpu_count(logical=False) or cpu_count
            base_workers = max(1, int(physical_cores * resource_factor * env_factor))
            
            # Asegurar al menos 1 worker y no más que cores lógicos
            return max(1, min(base_workers, cpu_count))
        else:  # io
            # Para IO-bound, se pueden usar más workers ya que están bloqueados
            # Típicamente 2x o más que el número de cores
            io_multiplier = 2.0  # Multiplicador para IO vs CPU
            base_workers = max(2, int(cpu_count * io_multiplier * resource_factor * env_factor))
            
            # Asegurar al menos 2 workers para IO y no más que 4x cores
            return max(2, min(base_workers, cpu_count * 4))

    # ... resto del código ...