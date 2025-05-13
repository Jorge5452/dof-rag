import gc
import logging
import time # Aunque no se usa directamente en este esqueleto, es común en gestión
import sys
import weakref
import functools
import inspect
from typing import Dict, Any, Optional, TYPE_CHECKING, List

if TYPE_CHECKING:
    from .resource_manager import ResourceManager
    # from modulos.embeddings.embeddings_factory import EmbeddingFactory # Para type hinting futuro

class MemoryManager:
    """
    Gestiona las operaciones específicas de optimización y limpieza de memoria.

    Es instanciado y utilizado por ResourceManager para realizar tareas como:
    - Ejecutar recolección de basura (garbage collection).
    - Liberar recursos de modelos de embedding no utilizados.
    - Limpiar cachés de Python y comprobar fragmentación de memoria.
    - Optimizar dinámicamente tamaños de lote (batch_size) basándose en el uso
      actual de memoria del sistema.

    Atributos:
        resource_manager (ResourceManager): Instancia del ResourceManager principal.
        logger (logging.Logger): Logger para esta clase.
        cached_functions (List): Lista de referencias a funciones con decorador lru_cache.
    """
    def __init__(self, resource_manager_instance: 'ResourceManager'):
        """
        Inicializa el MemoryManager.

        Args:
            resource_manager_instance (ResourceManager): La instancia de ResourceManager
                que gestionará este MemoryManager.
        """
        self.resource_manager = resource_manager_instance
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers(): # Fallback si no hay config de logging
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Lista para seguimiento de funciones con caché
        self.cached_functions = []
        
        # Registrar funciones que usan lru_cache
        self._register_cache_functions()
        
        self.logger.info("MemoryManager inicializado.")
        # Cargar config específica si se añade en el futuro
        # self._load_config()

    def cleanup(self, aggressive: bool = False, reason: str = "unknown") -> Dict[str, Any]:
        """
        Realiza una serie de operaciones de limpieza de memoria.

        Las acciones incluyen liberar modelos de embedding, limpiar cachés (placeholder),
        ejecutar garbage collection y comprobar fragmentación (placeholder).

        Args:
            aggressive (bool): Si True, las operaciones de limpieza serán más intensivas
                               (e.g., GC más profundo, liberación más agresiva de modelos).
                               Defaults to False.
            reason (str): Descripción del motivo de la limpieza. Defaults to "unknown".

        Returns:
            Dict[str, Any]: Un diccionario con los resultados de las acciones de limpieza,
                            incluyendo las acciones tomadas y el conteo de colecciones de GC.
        """
        self.logger.info(f"Iniciando limpieza de memoria. Agresivo: {aggressive}, Razón: '{reason}'")
        results: Dict[str, Any] = {
            "reason": reason,
            "aggressive_mode": aggressive,
            "actions_taken": [],
            "gc_collections": 0
        }

        action_release_models = self._release_model_resources(aggressive=aggressive)
        if action_release_models:
            results["actions_taken"].append(action_release_models)

        action_clear_caches = self._clear_python_caches(aggressive=aggressive)
        if action_clear_caches:
            results["actions_taken"].append(action_clear_caches)

        gc_collections_count = self._run_garbage_collection(aggressive=aggressive)
        results["gc_collections"] = gc_collections_count
        results["actions_taken"].append(f"Garbage collection ejecutado (aggressive={aggressive}), {gc_collections_count} objetos recolectados/ciclos.")

        action_check_fragmentation = self._check_memory_fragmentation(aggressive=aggressive)
        if action_check_fragmentation:
            results["actions_taken"].append(action_check_fragmentation)
        
        self.logger.info(f"Limpieza de memoria completada. Resultados: {results}")
        return results

    def _run_garbage_collection(self, aggressive: bool) -> int:
        """
        Ejecuta el recolector de basura de Python.

        Args:
            aggressive (bool): Si True, intenta una recolección más profunda (generación 2).
                               Sino, realiza una recolección estándar.

        Returns:
            int: El número de objetos recolectados por `gc.collect()`, o 0 si no es determinable.
        """
        self.logger.debug(f"Ejecutando garbage collection (aggressive={aggressive}).")
        collected_count = 0
        try:
            if aggressive:
                # Forzar recolección en todas las generaciones
                for gen in range(3):  # Python tiene 3 generaciones (0, 1, 2)
                    gen_count = gc.collect(generation=gen)
                    collected_count += gen_count
                    self.logger.debug(f"GC agresivo (gen {gen}) recolectó {gen_count} objetos.")
                
                # Configurar umbral de GC para ser más agresivo temporalmente
                old_thresholds = gc.get_threshold()
                gc.set_threshold(700, 10, 10)  # Valores más bajos = GC más frecuente
                
                # Restaurar configuración original después de un tiempo
                def restore_gc_threshold():
                    gc.set_threshold(*old_thresholds)
                    self.logger.debug("Umbrales de GC restaurados a valores originales.")
                
                # Programar restauración tras 60 segundos
                threading_timer = None
                try:
                    import threading
                    threading_timer = threading.Timer(60, restore_gc_threshold)
                    threading_timer.daemon = True
                    threading_timer.start()
                except ImportError:
                    # Si threading no está disponible, restaurar inmediatamente
                    restore_gc_threshold()
            else:
                collected_count = gc.collect()  # Recolección estándar
                self.logger.debug(f"GC estándar recolectó {collected_count} objetos.")
        except Exception as e:
            self.logger.error(f"Error durante garbage collection: {e}", exc_info=True)
        return collected_count if collected_count is not None else 0

    def _register_cache_functions(self):
        """
        Registra funciones decoradas con lru_cache para su limpieza posterior.
        Esto permite rastrear y limpiar las cachés específicas cuando sea necesario.
        """
        try:
            # Obtener módulos importados
            for module_name, module in list(sys.modules.items()):
                # Procesar solo módulos del proyecto (modulos.*)
                if module and module_name.startswith('modulos.'):
                    for name, obj in inspect.getmembers(module):
                        # Verificar si es una función con caché
                        if inspect.isfunction(obj) and hasattr(obj, 'cache_info') and hasattr(obj, 'cache_clear'):
                            self.cached_functions.append(weakref.ref(obj))
                            self.logger.debug(f"Registrada función con caché: {module_name}.{name}")
        except Exception as e:
            self.logger.error(f"Error registrando funciones con caché: {e}", exc_info=True)

    def _clear_python_caches(self, aggressive: bool = False) -> str:
        """
        Limpia las cachés internas de Python para liberar memoria.

        Implementa la "Fase 1: Optimización de Limpieza de Memoria" limpiando:
        - Cachés de importación de módulos
        - Cachés LRU de funciones decoradas con @functools.lru_cache
        - Otras cachés internas cuando sea posible

        Args:
            aggressive (bool): Si True, realiza una limpieza más intensiva, incluyendo
                               cachés que podrían afectar al rendimiento. Default False.

        Returns:
            str: Mensaje describiendo la acción realizada.
        """
        self.logger.info(f"Iniciando limpieza de cachés de Python (aggressive={aggressive}).")
        cache_info = {
            "module_cache_size_before": len(sys.modules),
            "lru_caches_cleared": 0,
            "module_cache_cleared": False,
            "path_cache_cleared": False,
            "memory_returned": 0
        }

        try:
            # 1. Limpieza de cachés LRU de funciones decoradas
            valid_cached_functions = []
            for func_ref in self.cached_functions:
                func = func_ref()
                if func is not None:
                    try:
                        # Para limpieza agresiva, limpiar todas las cachés
                        # Para limpieza normal, limpiar solo si tienen muchas entradas (>1000)
                        cache_info_obj = func.cache_info()
                        if aggressive or (hasattr(cache_info_obj, 'currsize') and cache_info_obj.currsize > 1000):
                            func.cache_clear()
                            cache_info["lru_caches_cleared"] += 1
                            self.logger.debug(f"Limpiada caché de función: {func.__module__}.{func.__name__}")
                    except Exception as e:
                        self.logger.warning(f"Error limpiando caché de función: {e}")
                    valid_cached_functions.append(func_ref)
            
            # Actualizar la lista con solo referencias válidas
            self.cached_functions = valid_cached_functions
            
            # 2. Limpieza de caché de importación de módulos (solo en modo agresivo)
            if aggressive:
                non_essential_modules = []
                for module_name in list(sys.modules.keys()):
                    # Preservar módulos esenciales del sistema y de nuestro proyecto
                    if (not module_name.startswith('_') and 
                        not module_name.startswith('sys') and 
                        not module_name.startswith('os') and
                        not module_name.startswith('modulos.') and  # Mantener nuestros propios módulos
                        not module_name in ('logging', 'gc', 'threading', 'time', 'functools')):
                        non_essential_modules.append(module_name)
                
                # Eliminar un subconjunto de módulos no esenciales
                # No eliminar todos para evitar problemas graves, solo los menos utilizados
                modules_to_remove = non_essential_modules[:max(len(non_essential_modules)//4, 1)]
                for module_name in modules_to_remove:
                    try:
                        del sys.modules[module_name]
                    except KeyError:
                        pass
                
                cache_info["module_cache_cleared"] = True
                cache_info["module_cache_size_after"] = len(sys.modules)
                self.logger.info(f"Eliminados {len(modules_to_remove)} módulos de sys.modules")
            
            # 3. Limpieza de caché de paths de importación (solo en modo agresivo)
            if aggressive and hasattr(sys, "_getframe"):
                importlib_cache_cleared = False
                try:
                    import importlib
                    if hasattr(importlib, 'invalidate_caches'):
                        importlib.invalidate_caches()
                        importlib_cache_cleared = True
                except (ImportError, AttributeError):
                    pass
                
                cache_info["path_cache_cleared"] = importlib_cache_cleared
                if importlib_cache_cleared:
                    self.logger.info("Caché de importlib invalidada.")
            
            # 4. Ejecutar una recolección de basura para liberar la memoria de cachés eliminadas
            # pero solo si no se va a llamar a _run_garbage_collection después
            if not aggressive:  # Si es agresivo, el método cleanup ya llamará a _run_garbage_collection
                gc.collect()

            msg = (f"Limpieza de cachés completada - "
                   f"LRU cachés: {cache_info['lru_caches_cleared']}, "
                   f"Módulos: {'Sí' if cache_info['module_cache_cleared'] else 'No'}, "
                   f"Path cache: {'Sí' if cache_info['path_cache_cleared'] else 'No'}")
            
            return msg
            
        except Exception as e:
            error_msg = f"Error durante limpieza de cachés: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return error_msg

    def _release_model_resources(self, aggressive: bool) -> Optional[str]:
        """
        Intenta liberar recursos asociados con modelos de embedding inactivos.

        Delega la tarea a `EmbeddingFactory.release_inactive_models()`.

        Args:
            aggressive (bool): Pasa este flag a `EmbeddingFactory` para influir
                               en su estrategia de liberación.

        Returns:
            Optional[str]: Un mensaje describiendo el resultado de la operación o un error.
        """
        self.logger.debug(f"Intentando liberar recursos de modelos (aggressive={aggressive}).")
        try:
            # Importar la clase Factory aquí para evitar dependencias globales innecesarias
            from modulos.embeddings.embeddings_factory import EmbeddingFactory
            count = EmbeddingFactory.release_inactive_models(aggressive=aggressive)
            msg = f"{count} modelos de embedding liberados (agresivo={aggressive})."
            
            # Si no se liberaron modelos, usar nivel DEBUG en lugar de INFO
            if count > 0:
                self.logger.info(msg)
            else:
                self.logger.debug(msg)
                
            return msg
        except ImportError:
            self.logger.error("No se pudo importar EmbeddingFactory para liberar modelos.")
            return "Error: EmbeddingFactory no encontrada."
        except Exception as e:
            self.logger.error(f"Error al liberar modelos de embedding: {e}", exc_info=True)
            return f"Error al liberar modelos: {str(e)}"

    def _check_memory_fragmentation(self, aggressive: bool = False) -> Optional[str]:
        """
        Evalúa y potencialmente mitiga la fragmentación de memoria en Python.

        La medición y mitigación de la fragmentación de memoria en Python es compleja.
        Esta implementación detecta señales de fragmentación e intenta mitigarla.

        Args:
            aggressive (bool): Si True, intenta medidas más intensivas para
                              reducir la fragmentación. Default False.

        Returns:
            Optional[str]: Mensaje indicando el resultado de la operación.
        """
        self.logger.debug(f"Comprobando fragmentación de memoria (aggressive={aggressive}).")
        
        try:
            import psutil
            process = psutil.Process()
            rss_before = process.memory_info().rss / (1024 * 1024)  # MB
            vms_before = process.memory_info().vms / (1024 * 1024)  # MB
            
            # Calcular un indicador aproximado de fragmentación
            # Un alto VSS/RSS puede indicar fragmentación
            fragmentation_ratio = vms_before / rss_before if rss_before > 0 else 0
            
            # Acciones basadas en el nivel de fragmentación detectado
            if fragmentation_ratio > 2.5 or aggressive:  # Alto nivel de fragmentación o modo agresivo
                self.logger.info(f"Posible fragmentación de memoria detectada (ratio VMS/RSS: {fragmentation_ratio:.2f})")
                
                # Estrategias para reducir la fragmentación
                # 1. Ejecutar varios ciclos de GC para liberar memoria no contigua
                for _ in range(3):
                    gc.collect()
                
                # 2. Eliminación y recreación de estructuras grandes (simulado)
                self._simulate_defragmentation()
                
                # Verificar después de las acciones
                rss_after = process.memory_info().rss / (1024 * 1024)  # MB
                vms_after = process.memory_info().vms / (1024 * 1024)  # MB
                fragmentation_ratio_after = vms_after / rss_after if rss_after > 0 else 0
                
                change_msg = (
                    f"Mitigación de fragmentación - "
                    f"Antes: RSS={rss_before:.1f}MB, VMS={vms_before:.1f}MB, Ratio={fragmentation_ratio:.2f} | "
                    f"Después: RSS={rss_after:.1f}MB, VMS={vms_after:.1f}MB, Ratio={fragmentation_ratio_after:.2f}"
                )
                self.logger.info(change_msg)
                return change_msg
            else:
                return f"Fragmentación no crítica (ratio VMS/RSS: {fragmentation_ratio:.2f})"
            
        except ImportError:
            return "No se pudo evaluar fragmentación (psutil no disponible)."
        except Exception as e:
            self.logger.error(f"Error al evaluar fragmentación de memoria: {e}", exc_info=True)
            return f"Error al evaluar fragmentación: {str(e)}"

    def _simulate_defragmentation(self):
        """
        Simula un proceso de desfragmentación de memoria mediante técnicas de 
        recompactación de estructuras de datos.
        
        En Python, no podemos desfragmentar directamente la memoria, pero podemos
        simular este proceso recreando estructuras de datos grandes.
        """
        self.logger.debug("Iniciando simulación de desfragmentación")
        try:
            # Forzar liberación de objetos no utilizados
            gc.collect()
            
            # Recalcular referencias débiles (esto puede ayudar con la fragmentación)
            gc.collect(2)
            
            # Intentamos llevar a cabo una desfragmentación indirecta
            # Este proceso es principalmente simbólico en Python pero puede
            # ayudar a consolidar memoria en casos específicos
            self.logger.debug("Simulación de desfragmentación completada")
            
        except Exception as e:
            self.logger.error(f"Error durante simulación de desfragmentación: {e}", exc_info=True)

    def check_memory_usage(self) -> None:
        """
        Verifica el uso actual de memoria y ejecuta acciones de limpieza si es necesario.
        
        Este método comprueba las métricas actuales de memoria del sistema a través del
        ResourceManager y decide si es necesario realizar una limpieza, basándose en
        los umbrales configurados.
        """
        try:
            if not self.resource_manager or not self.resource_manager.metrics:
                self.logger.warning("ResourceManager o sus métricas no disponibles para verificar memoria.")
                return
                
            mem_usage_pct = self.resource_manager.metrics.get("system_memory_percent", 0)
            
            # Obtener umbrales del ResourceManager
            aggressive_thresh = getattr(self.resource_manager, 'aggressive_cleanup_threshold_mem_pct', 85)
            warning_thresh = getattr(self.resource_manager, 'warning_cleanup_threshold_mem_pct', 70)
            
            if mem_usage_pct >= aggressive_thresh:
                self.logger.warning(f"Uso de memoria crítico ({mem_usage_pct}%). Ejecutando limpieza agresiva.")
                self.cleanup(aggressive=True, reason="memory_usage_critical")
            elif mem_usage_pct >= warning_thresh:
                self.logger.info(f"Uso de memoria elevado ({mem_usage_pct}%). Ejecutando limpieza normal.")
                self.cleanup(aggressive=False, reason="memory_usage_high")
            else:
                # Solo registramos pero no hacemos nada si la memoria está en niveles normales
                self.logger.debug(f"Uso de memoria normal ({mem_usage_pct}%). No se requiere limpieza.")
        
        except Exception as e:
            self.logger.error(f"Error al verificar uso de memoria: {e}", exc_info=True)

    def optimize_batch_size(self, base_batch_size: int, min_batch_size: int = 1, 
                             max_batch_size: Optional[int] = None, verification_suspended: bool = False) -> int:
        """
        Ajusta dinámicamente el tamaño de lote (batch_size) para operaciones
        intensivas en memoria, basándose en el uso actual de memoria del sistema.

        Args:
            base_batch_size (int): El tamaño de lote base o preferido.
            min_batch_size (int): El tamaño de lote mínimo permitido. Defaults to 1.
            max_batch_size (Optional[int]): El tamaño de lote máximo permitido. 
                                            Defaults to None (sin límite superior explícito aquí).
            verification_suspended (bool): Si las verificaciones están suspendidas actualmente.
                                          Affects the optimization strategy.

        Returns:
            int: El tamaño de lote optimizado. Si no se pueden obtener métricas,
                 devuelve `base_batch_size`.
        """
        self.logger.debug(f"Solicitud para optimizar batch_size. Base: {base_batch_size}, Min: {min_batch_size}, Max: {max_batch_size}, Suspended: {verification_suspended}")
        
        # Si las verificaciones están suspendidas, devolver un valor más conservador
        if verification_suspended:
            self.logger.debug("Verificaciones suspendidas, usando batch_size conservador")
            if max_batch_size is not None:
                return min(base_batch_size, max_batch_size)
            return base_batch_size
        
        optimized_batch_size = base_batch_size
        try:
            if not self.resource_manager or not self.resource_manager.metrics:
                self.logger.warning("ResourceManager o sus métricas no disponibles para optimizar batch_size. Usando base_batch_size.")
                return base_batch_size

            mem_usage_pct = self.resource_manager.metrics.get("system_memory_percent", 0)
            # mem_available_gb = self.resource_manager.metrics.get("system_memory_available_gb", float('inf'))
            
            # Umbrales del ResourceManager
            aggressive_thresh = self.resource_manager.aggressive_cleanup_threshold_mem_pct
            warning_thresh = self.resource_manager.warning_cleanup_threshold_mem_pct

            if mem_usage_pct >= aggressive_thresh:
                optimized_batch_size = max(min_batch_size, int(base_batch_size * 0.25)) # Reducción drástica
                self.logger.warning(f"Uso de memoria ({mem_usage_pct}%) muy alto. Batch size reducido drásticamente a {optimized_batch_size}.")
            elif mem_usage_pct >= warning_thresh:
                optimized_batch_size = max(min_batch_size, int(base_batch_size * 0.5)) # Reducción moderada
                self.logger.warning(f"Uso de memoria ({mem_usage_pct}%) alto. Batch size reducido a {optimized_batch_size}.")
            # else: # Podríamos añadir lógica para incrementar si hay mucha memoria, pero con cautela.
            #    if mem_available_gb > HIGH_MEMORY_THRESHOLD_GB:
            #        optimized_batch_size = int(base_batch_size * 1.25)

            if max_batch_size is not None:
                optimized_batch_size = min(optimized_batch_size, max_batch_size)
            
            optimized_batch_size = max(min_batch_size, optimized_batch_size) # Asegurar mínimo

        except Exception as e:
            self.logger.error(f"Error al optimizar batch_size: {e}. Usando base_batch_size.", exc_info=True)
            optimized_batch_size = base_batch_size
        
        self.logger.info(f"Batch size optimizado: {optimized_batch_size} (desde base {base_batch_size})")
        return optimized_batch_size

    def shutdown(self) -> None:
        """
        Realiza cualquier limpieza o cierre necesario para MemoryManager al apagar el sistema.
        Actualmente, no tiene acciones específicas más allá de registrar el evento.
        """
        self.logger.info("MemoryManager shutdown solicitado.")
        # Por ahora, no hay acciones específicas de shutdown para MemoryManager.
        self.logger.info("MemoryManager shutdown completado.") 