import gc
import logging
import sys
import weakref
import functools
import inspect
import time
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

    Nota importante: Los modelos de embedding NO son liberados durante el procesamiento
    de documentos, ya que se utilizan continuamente. Solo se liberarán cuando se hayan
    terminado de procesar todos los documentos.

    Atributos:
        resource_manager (ResourceManager): Instancia del ResourceManager principal.
        logger (logging.Logger): Logger para esta clase.
        cached_functions (List): Lista de referencias a funciones con decorador lru_cache.
        last_gc_time (float): Timestamp del último garbage collection completo.
        last_gc_threshold_change (float): Timestamp del último cambio de umbrales de GC.
        gc_stats (Dict): Estadísticas sobre las operaciones de garbage collection.
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
        
        # Variables para gestión optimizada de GC
        self.last_gc_time = 0.0
        self.last_gc_threshold_change = 0.0
        self.gc_stats = {
            "total_collections": 0,
            "objects_collected": 0,
            "aggressive_collections": 0,
            "standard_collections": 0,
            "last_collection_objects": 0
        }
        
        # Obtener intervalos mínimos desde la configuración, si está disponible
        # con valores predeterminados si no se puede acceder a la configuración
        if self.resource_manager and hasattr(self.resource_manager, 'config'):
            try:
                resource_config = self.resource_manager.config.get_resource_management_config()
                monitoring_config = resource_config.get('monitoring', {})
                
                # Obtener intervalos configurados o usar valores predeterminados
                self.min_gc_interval = float(monitoring_config.get('min_cleanup_interval_sec', 30.0))
                self.min_aggressive_gc_interval = float(monitoring_config.get('min_aggressive_cleanup_interval_sec', 300.0))
                self.min_memory_change_pct = float(monitoring_config.get('min_memory_change_pct', 5.0))
                self.extended_cooling_period = float(monitoring_config.get('extended_cooling_period_sec', 900.0))
                
                self.logger.debug(f"Intervalos de limpieza configurados: normal={self.min_gc_interval}s, " 
                                 f"agresivo={self.min_aggressive_gc_interval}s, " 
                                 f"cambio mínimo={self.min_memory_change_pct}%, "
                                 f"enfriamiento extendido={self.extended_cooling_period}s")
            except Exception as e:
                self.logger.warning(f"Error al cargar intervalos de configuración: {e}. Usando valores predeterminados.")
                # Valores predeterminados si hay error en la configuración
                self.min_gc_interval = 30.0
                self.min_aggressive_gc_interval = 300.0
                self.min_memory_change_pct = 5.0
                self.extended_cooling_period = 900.0
        else:
            # Intervalos mínimos entre operaciones de GC (en segundos) - valores predeterminados
            self.min_gc_interval = 30.0  # Mínimo 30 segundos entre GCs normales
            self.min_aggressive_gc_interval = 300.0  # Mínimo 5 minutos entre GCs agresivos
            self.min_memory_change_pct = 5.0  # Mínimo 5% de cambio para considerar una limpieza efectiva
            self.extended_cooling_period = 900.0  # 15 minutos de enfriamiento extendido después de limpieza ineficaz
        
        # Registrar funciones que usan lru_cache
        self._register_cache_functions()
        
        self.logger.info("MemoryManager inicializado con gestión de GC optimizada.")
        # Cargar config específica si se añade en el futuro
        # self._load_config()

    def cleanup(self, aggressive: bool = False, reason: str = "manual", skip_model_cleanup: bool = False) -> bool:
        """
        Realiza limpieza de memoria y liberación de recursos.

        Versión optimizada con mejoras para reducir el overhead durante procesamiento 
        de documentos y evitar liberación innecesaria de modelos.
        
        Args:
            aggressive (bool): Si es True, realiza una limpieza más profunda.
            reason (str): Razón para la limpieza (para logging y seguimiento).
            skip_model_cleanup (bool): Si es True, evita la liberación de modelos de embeddings.
                                     Útil durante procesamiento continuo de documentos.
            
        Returns:
            bool: True si se realizó limpieza efectiva, False si se omitió.
        """
        # Verificar tiempo desde última limpieza
        current_time = time.time()
        time_since_last_gc = current_time - self.last_gc_time
        
        # Verificar si podemos omitir esta limpieza por estar en periodo de enfriamiento
        # (Excepto para razones que requieren limpieza inmediata)
        force_immediate_cleanup = reason in ["oom_prevention", "critical_memory", "processing_completed"]
        
        if not force_immediate_cleanup:
            # Criterios para omitir limpieza basados en tiempo transcurrido
            if aggressive and time_since_last_gc < self.min_aggressive_gc_interval:
                self.logger.debug(f"Omitiendo limpieza agresiva: {time_since_last_gc:.1f}s < {self.min_aggressive_gc_interval:.1f}s")
                return False
            elif not aggressive and time_since_last_gc < self.min_gc_interval:
                self.logger.debug(f"Omitiendo limpieza normal: {time_since_last_gc:.1f}s < {self.min_gc_interval:.1f}s")
                return False
        
        # Actualizar timestamp de última limpieza
        self.last_gc_time = current_time
        
        self.logger.info(f"Ejecutando limpieza (agresivo={aggressive}, razón='{reason}'{', sin_liberar_modelos' if skip_model_cleanup else ''})")
        
        # 1. Ejecutar recolección de basura de Python
        try:
            collected = self._run_garbage_collection(aggressive)
            self.gc_stats["total_collections"] += 1
            self.gc_stats["objects_collected"] += collected
            self.gc_stats["last_collection_objects"] = collected
            
            if aggressive:
                self.gc_stats["aggressive_collections"] += 1
            else:
                self.gc_stats["standard_collections"] += 1
        except Exception as e:
            self.logger.error(f"Error durante garbage collection: {e}")
            collected = 0
        
        # 2. Limpiar cachés de Python
        try:
            self._clear_python_caches()
        except Exception as e:
            self.logger.error(f"Error al limpiar cachés de Python: {e}")
            
        # 3. Liberación de recursos específicos según razón y agresividad
        resources_released = 0
        
        # 4. Liberar recursos de modelos solo si:
        # - No estamos en modo skip_model_cleanup
        # - La limpieza es agresiva o la razón es compatible con liberar modelos
        if not skip_model_cleanup and (aggressive or reason in ["processing_completed", "oom_prevention", "idle_timeout"]):
            try:
                models_released = self._release_model_resources(aggressive=aggressive)
                resources_released += models_released
                self.logger.debug(f"Modelos liberados: {models_released}")
            except Exception as e:
                self.logger.error(f"Error al liberar recursos de modelos: {e}")
        elif skip_model_cleanup:
            self.logger.debug("Omitiendo liberación de modelos por solicitud explícita (skip_model_cleanup)")
        
        # Logging más discreto con sistema de niveles (Reducir overhead)
        if aggressive:
            self.logger.info(f"Limpieza agresiva completada. Objetos recolectados: {collected}, recursos liberados: {resources_released}")
        else:
            self.logger.debug(f"Limpieza estándar completada. Objetos recolectados: {collected}")
            
        # Devolver True si la limpieza liberó objetos o recursos
        return collected > 0 or resources_released > 0

    def _run_garbage_collection(self, aggressive: bool) -> int:
        """
        Ejecuta el recolector de basura de Python con optimizaciones para
        minimizar el impacto en rendimiento y mejorar la eficiencia.

        Esta implementación mejorada se centra principalmente en la generación 0
        para mayor rendimiento, y solo ejecuta recolecciones completas cuando 
        es absolutamente necesario. Se eliminan técnicas anti-fragmentación pesadas.

        Args:
            aggressive (bool): Si True, intenta una recolección más profunda (generación 2).
                               Sino, realiza una recolección más selectiva.

        Returns:
            int: El número de objetos recolectados por `gc.collect()`, o 0 si no es determinable.
        """
        self.logger.debug(f"Ejecutando garbage collection optimizado (aggressive={aggressive}).")
        collected_count = 0
        
        try:
            # Obtener estadísticas iniciales
            initial_count = len(gc.get_objects())
            current_time = time.time()

            # Verificar si GC está habilitado
            was_enabled = gc.isenabled()
            if not was_enabled:
                gc.enable()
                self.logger.debug("GC estaba deshabilitado, se ha activado temporalmente.")
            
            if aggressive:
                # Estrategia para GC agresivo - simplificada sin anti-fragmentación
                # Solo ajustar umbrales cuando realmente sea necesario
                old_thresholds = gc.get_threshold()
                threshold_adjusted = False
                
                # Solo modificar umbrales si ha pasado suficiente tiempo
                if (current_time - self.last_gc_threshold_change) > 180.0:  # 3 minutos
                    # Valores más bajos = GC más frecuente
                    new_thresholds = (700, 10, 10)  
                    gc.set_threshold(*new_thresholds)
                    self.logger.debug(f"Umbrales GC temporalmente ajustados de {old_thresholds} a {new_thresholds}")
                    self.last_gc_threshold_change = current_time
                    threshold_adjusted = True
                    
                    # Restauración de umbrales programada aquí...
                    try:
                        import threading
                        def restore_gc_threshold():
                            try:
                                gc.set_threshold(*old_thresholds)
                                self.logger.debug(f"Umbrales GC restaurados a {old_thresholds}")
                                self.last_gc_threshold_change = time.time()
                            except Exception as e:
                                self.logger.error(f"Error restaurando umbrales GC: {e}")
                        
                        # Restaurar después de 60 segundos
                        timer = threading.Timer(60.0, restore_gc_threshold)
                        timer.daemon = True
                        timer.start()
                    except ImportError:
                        # No hacer nada si threading no está disponible
                        pass
                
                # Primero enfocarse en gen 0 (más eficiente)
                gen0_count = gc.collect(0)
                self.logger.debug(f"GC gen 0 recolectó {gen0_count} objetos")
                collected_count = gen0_count
                
                # Solo ejecutar gen 1 si el gen 0 encontró varios objetos
                if gen0_count > 50:  # Umbral reducido para mejor rendimiento
                    gen1_count = gc.collect(1)
                    self.logger.debug(f"GC gen 1 recolectó {gen1_count} objetos")
                    collected_count += gen1_count
                    
                    # Solo ejecutar gen 2 (muy costoso) si realmente hay presión severa
                    # y han pasado al menos 5 minutos desde el último GC completo
                    time_since_last_full_gc = current_time - getattr(self, "last_full_gc_time", 0)
                    mem_percent = self.resource_manager.metrics.get("system_memory_percent", 0)
                    
                    if (gen1_count > 50 and time_since_last_full_gc > 300) or mem_percent > 90:
                        gen2_count = gc.collect(2)
                        self.logger.debug(f"GC gen 2 recolectó {gen2_count} objetos")
                        collected_count += gen2_count
                        
                        # Registrar timestamp del último GC completo
                        self.last_full_gc_time = current_time
                
            else:
                # Estrategia estándar - más conservadora y enfocada en gen 0
                mem_percent = self.resource_manager.metrics.get("system_memory_percent", 0)
                
                if mem_percent <= 60:  # Presión baja
                    # Solo generación 0 (más rápida)
                    collected_count = gc.collect(0)
                    self.logger.debug(f"GC selectivo gen 0 recolectó {collected_count} objetos")
                elif mem_percent <= 75:  # Presión moderada
                    # Generaciones 0 y 1
                    gen0_count = gc.collect(0)
                    gen1_count = gc.collect(1)
                    collected_count = gen0_count + gen1_count
                    self.logger.debug(f"GC selectivo gen 0+1 recolectó {collected_count} objetos")
                else:  # Presión alta
                    # Primero probar gen 0+1, y solo si recogen muchos objetos, hacer gen 2
                    gen0_count = gc.collect(0)
                    gen1_count = gc.collect(1)
                    collected_count = gen0_count + gen1_count
                    
                    # Solo hacer gen 2 si hay muchos objetos en gen 0+1
                    if collected_count > 100:
                        gen2_count = gc.collect(2)
                        collected_count += gen2_count
                        self.logger.debug(f"GC completo recolectó total {collected_count} objetos")
                    else:
                        self.logger.debug(f"GC selectivo gen 0+1 suficiente, recolectó {collected_count} objetos")
            
            # Restaurar estado de GC si estaba deshabilitado
            if not was_enabled:
                gc.disable()
                self.logger.debug("GC restaurado a su estado anterior (deshabilitado).")
                
            # Calcular eficiencia y reportar
            final_count = len(gc.get_objects())
            objects_diff = initial_count - final_count
            
            # Corregir diferencia negativa (objetos creados durante GC)
            if objects_diff < 0:
                objects_diff = 0
                
            self.logger.info(f"GC completado - Recolectados: {collected_count}, "
                           f"Diferencia de objetos: {objects_diff}")
            
            return collected_count
            
        except Exception as e:
            self.logger.error(f"Error durante garbage collection: {e}")
            return 0  # En caso de error, reportar 0 objetos recolectados

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

    def _release_model_resources(self, aggressive: bool) -> int:
        """
        Libera recursos de modelos de embedding y realiza limpieza avanzada de memoria.
        
        Versión optimizada que solo se enfoca en liberar modelos de embedding, retornando
        el número de modelos liberados para mejor control y monitoreo.
        
        Args:
            aggressive (bool): Si es True, se usa una estrategia más agresiva de liberación.
            
        Returns:
            int: El número de modelos de embedding liberados.
        """
        # Inicializar contadores y marcadores si no existen
        if not hasattr(self, '_consecutive_failed_cleanups'):
            self._consecutive_failed_cleanups = 0
            self._last_cleanup_memory_before = 0
            self._last_cleanup_memory_after = 0
            self._cleanup_history = []
            
        # Determinar si las limpiezas anteriores han sido efectivas
        ineffective_cleanups = self._consecutive_failed_cleanups
        current_memory = 0
        
        if self.resource_manager and hasattr(self.resource_manager, 'metrics'):
            current_memory = self.resource_manager.metrics.get("system_memory_percent", 0)
            
            # Si tenemos datos de la última limpieza, evaluar efectividad
            if self._last_cleanup_memory_before > 0:
                if current_memory >= (self._last_cleanup_memory_before - self.min_memory_change_pct):
                    # La limpieza anterior no fue efectiva
                    self._consecutive_failed_cleanups += 1
                    self.logger.debug(f"Limpieza {self._consecutive_failed_cleanups} consecutiva ineficaz. "
                                     f"Memoria antes: {self._last_cleanup_memory_before:.1f}%, "
                                     f"Ahora: {current_memory:.1f}%")
                else:
                    # La limpieza anterior fue efectiva
                    self._consecutive_failed_cleanups = 0
                    self.logger.debug(f"Limpieza anterior efectiva. "
                                     f"Memoria antes: {self._last_cleanup_memory_before:.1f}%, "
                                     f"Ahora: {current_memory:.1f}%")
        
        # Guardar el valor de memoria actual para la próxima evaluación
        self._last_cleanup_memory_before = current_memory
        
        # Variables para seguimiento de acciones
        models_released = 0
        
        try:
            # Liberar modelos de embedding a través de EmbeddingFactory
            self.logger.debug(f"Intentando liberar modelos inactivos (aggressive={aggressive}, failed_cleanups={ineffective_cleanups}).")
            
            try:
                from modulos.embeddings.embeddings_factory import EmbeddingFactory
                
                # Ajustar agresividad basada en fallos consecutivos
                adjusted_aggressive = aggressive or (ineffective_cleanups >= 2)
                
                # Cuando hay muchos fallos consecutivos, liberar incluso modelos activos
                force_active = False
                if ineffective_cleanups >= 3 or (aggressive and ineffective_cleanups >= 1):
                    force_active = True
                    self.logger.warning(f"Activando liberación forzada de modelos activos debido a {ineffective_cleanups} limpiezas ineficaces")
                
                models_released = EmbeddingFactory.release_inactive_models(
                    aggressive=adjusted_aggressive,
                    force_release_active=force_active
                )
                
                if models_released > 0:
                    self.logger.info(f"Liberados {models_released} modelos de embedding")
                
            except ImportError:
                self.logger.error("No se pudo importar EmbeddingFactory para liberar modelos.")
                return 0
            except Exception as e:
                self.logger.error(f"Error al liberar modelos de embedding: {e}")
                return 0
            
            # Limpieza específica de memoria PyTorch (si está disponible)
            if aggressive or ineffective_cleanups >= 1 or models_released > 0:
                try:
                    import torch
                    
                    # Comprobar CUDA
                    if hasattr(torch, 'cuda') and torch.cuda.is_available():
                        # Limpiar cachés de CUDA
                        torch.cuda.empty_cache()
                        self.logger.debug("Caché CUDA liberada")
                    
                    # Comprobar MPS (Apple Silicon)
                    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                            self.logger.debug("Caché MPS liberada")
                    
                except ImportError:
                    self.logger.debug("PyTorch no disponible")
                except Exception as e:
                    self.logger.debug(f"Error durante limpieza de PyTorch: {e}")
        
        except Exception as e:
            self.logger.error(f"Error general durante liberación de modelos: {e}")
            return 0
        
        return models_released

    def check_memory_usage(self) -> Dict[str, Any]:
        """
        Verifica el uso actual de la memoria y realiza limpiezas si es necesario.
        
        Esta función:
        1. Verifica el estado actual de memoria y CPU
        2. Compara con umbrales configurados 
        3. Ejecuta limpieza proactiva si se exceden determinados umbrales
        4. Implementa período de enfriamiento para evitar limpiezas consecutivas ineficaces
        5. Toma en cuenta el historial de limpiezas para ajustar estrategia
        
        Returns:
            Dict[str, Any]: Informe del estado de memoria y acciones tomadas
        """
        result = {
            "memory_checked": True,
            "cleanup_performed": False,
            "memory_percent": None,
            "threshold_exceeded": False,
            "threshold_type": None,
            "actions_taken": []
        }
        
        try:
            # Verificar que tenemos acceso al ResourceManager y sus métricas
            if not self.resource_manager or not hasattr(self.resource_manager, 'metrics'):
                self.logger.warning("ResourceManager no disponible o no tiene métricas configuradas")
                result["error"] = "ResourceManager no disponible"
                return result
            
            # Obtener métricas actuales
            memory_percent = self.resource_manager.metrics.get("system_memory_percent", 0)
            result["memory_percent"] = memory_percent
            
            # Obtener umbrales configurados (o usar valores predeterminados)
            aggressive_threshold = getattr(self.resource_manager, 'aggressive_cleanup_threshold_mem_pct', 85)
            warning_threshold = getattr(self.resource_manager, 'warning_cleanup_threshold_mem_pct', 75)
            
            # Umbral de liberación forzada (desde configuración, o calcular)
            force_release_threshold = 90  # Valor predeterminado
            
            # Intentar obtener de configuración
            if hasattr(self.resource_manager, 'config') and self.resource_manager.config:
                try:
                    resource_config = self.resource_manager.config.get_resource_management_config()
                    memory_config = resource_config.get('memory', {})
                    model_release_config = memory_config.get('model_release', {})
                    force_release_threshold = model_release_config.get('force_release_memory_threshold_pct', 90)
                except:
                    # Si hay error, usar el valor predeterminado
                    pass
            
            # Tiempo transcurrido desde la última limpieza
            current_time = time.time()
            time_since_last_gc = current_time - self.last_gc_time
            
            # Evaluar eficacia de limpieza anterior
            had_recent_ineffective_cleanup = False
            previous_memory_after_cleanup = self.resource_manager.metrics.get("previous_memory_after_cleanup", 0)
            
            # Si hubo una limpieza reciente y la memoria sigue siendo alta,
            # consideramos que la limpieza anterior fue ineficaz
            if time_since_last_gc < 300 and previous_memory_after_cleanup > 0:
                memory_change = previous_memory_after_cleanup - memory_percent
                if abs(memory_change) < self.min_memory_change_pct:
                    had_recent_ineffective_cleanup = True
                    self.logger.warning(f"Limpieza reciente ineficaz detectada. Cambio de memoria: {memory_change:.1f}% < {self.min_memory_change_pct}%")
                    result["actions_taken"].append(f"Detección de limpieza ineficaz reciente (cambio: {memory_change:.1f}%)")
            
            # Períodos de enfriamiento ajustados
            cooling_period_aggressive = self.extended_cooling_period if had_recent_ineffective_cleanup else self.min_aggressive_gc_interval
            cooling_period_warning = self.extended_cooling_period if had_recent_ineffective_cleanup else self.min_gc_interval * 1.5
            
            # Determinar si necesitamos forzar liberación de modelos activos
            force_release_active = False
            
            if memory_percent >= force_release_threshold:
                self.logger.warning(f"Uso de memoria crítico ({memory_percent:.1f}% >= {force_release_threshold}%). Activando liberación forzada de modelos.")
                force_release_active = True
                result["actions_taken"].append("Liberación forzada de modelos activada por uso crítico de memoria")
            elif hasattr(self, '_consecutive_failed_cleanups') and self._consecutive_failed_cleanups >= 3:
                self.logger.warning(f"Múltiples limpiezas ineficaces ({self._consecutive_failed_cleanups}). Activando liberación forzada de modelos.")
                force_release_active = True
                result["actions_taken"].append("Liberación forzada de modelos activada por limpiezas ineficaces consecutivas")
            
            # Evaluar estado actual contra umbrales
            if memory_percent >= aggressive_threshold:
                # Situación crítica - Limpieza agresiva inmediata
                self.logger.warning(f"Uso de memoria crítico: {memory_percent:.1f}% >= {aggressive_threshold}% (umbral agresivo)")
                result["threshold_exceeded"] = True
                result["threshold_type"] = "aggressive"
                
                # Período de enfriamiento dinámico basado en eficacia anterior
                if time_since_last_gc >= cooling_period_aggressive:
                    cleanup_result = self.cleanup(aggressive=True, reason="memory_critical")
                    result["cleanup_performed"] = True
                    result["cleanup_result"] = cleanup_result
                    result["actions_taken"].append("Limpieza agresiva ejecutada")
                    
                    # Guardar información para evaluar eficacia en próxima verificación
                    self.resource_manager.metrics["previous_memory_after_cleanup"] = memory_percent
                else:
                    self.logger.info(f"Omitiendo limpieza agresiva - Última limpieza hace solo {time_since_last_gc:.1f}s (período enfriamiento: {cooling_period_aggressive:.1f}s)")
                    result["actions_taken"].append(f"Limpieza omitida (enfriamiento: {time_since_last_gc:.1f}s < {cooling_period_aggressive:.1f}s)")
                    
            elif memory_percent >= warning_threshold:
                # Situación de advertencia - Limpieza estándar
                self.logger.info(f"Uso de memoria alto: {memory_percent:.1f}% >= {warning_threshold}% (umbral advertencia)")
                result["threshold_exceeded"] = True
                result["threshold_type"] = "warning"
                
                # Liberar modelos si estamos acercándonos al umbral crítico
                if memory_percent >= (aggressive_threshold - 5) and time_since_last_gc >= self.min_gc_interval:
                    self.logger.info(f"Memoria cerca del umbral crítico ({memory_percent:.1f}% vs {aggressive_threshold}%), liberando modelos proactivamente")
                    cleanup_result = self.cleanup(aggressive=False, reason="memory_near_critical")
                    result["cleanup_performed"] = True
                    result["cleanup_result"] = cleanup_result
                    result["actions_taken"].append("Limpieza proactiva ejecutada (cerca del umbral crítico)")
                    self.resource_manager.metrics["previous_memory_after_cleanup"] = memory_percent
                # Período de enfriamiento extendido si detección de ineficacia
                elif time_since_last_gc >= cooling_period_warning:
                    # Solo ejecutar limpieza si es X% mayor que después de la limpieza anterior
                    if not had_recent_ineffective_cleanup or (memory_percent > previous_memory_after_cleanup + self.min_memory_change_pct):
                        cleanup_result = self.cleanup(aggressive=False, reason="memory_high")
                        result["cleanup_performed"] = True
                        result["cleanup_result"] = cleanup_result
                        result["actions_taken"].append("Limpieza estándar ejecutada")
                        
                        # Guardar información para evaluar eficacia en próxima verificación
                        self.resource_manager.metrics["previous_memory_after_cleanup"] = memory_percent
                    else:
                        self.logger.info(f"Omitiendo limpieza - Cambio de memoria insuficiente ({memory_percent:.1f}% vs {previous_memory_after_cleanup:.1f}% anterior, mínimo requerido: {self.min_memory_change_pct}%)")
                        result["actions_taken"].append(f"Limpieza omitida (cambio insuficiente: {memory_percent-previous_memory_after_cleanup:.1f}% < {self.min_memory_change_pct}%)")
                else:
                    self.logger.debug(f"Omitiendo limpieza estándar - En período de enfriamiento ({time_since_last_gc:.1f}s < {cooling_period_warning:.1f}s)")
                    result["actions_taken"].append(f"Limpieza omitida (enfriamiento: {time_since_last_gc:.1f}s < {cooling_period_warning:.1f}s)")
                
            else:
                # Situación normal - No se requiere acción
                self.logger.debug(f"Uso de memoria normal: {memory_percent:.1f}%")
                result["actions_taken"].append("No se requiere limpieza")
                
                # Resetear métricas de eficacia si la memoria ha bajado significativamente
                if previous_memory_after_cleanup > 0 and memory_percent < previous_memory_after_cleanup - (self.min_memory_change_pct * 2):
                    self.resource_manager.metrics["previous_memory_after_cleanup"] = 0
                    self.logger.info(f"Memoria normalizada ({memory_percent:.1f}%), métricas de eficacia reseteadas")
            
            # Si se activó liberación forzada pero no se ejecutó limpieza, forzar ahora
            if force_release_active and not result.get("cleanup_performed", False):
                # Forzar liberación directamente desde EmbeddingFactory para casos críticos
                try:
                    from modulos.embeddings.embeddings_factory import EmbeddingFactory
                    models_released = EmbeddingFactory.release_inactive_models(
                        aggressive=True, 
                        force_release_active=True
                    )
                    if models_released > 0:
                        self.logger.warning(f"Liberación forzada de emergencia: {models_released} modelos liberados")
                        result["actions_taken"].append(f"Liberación forzada de emergencia: {models_released} modelos")
                        result["emergency_release"] = True
                        result["models_released"] = models_released
                        # Forzar GC para maximizar el efecto
                        gc.collect()
                except Exception as e:
                    self.logger.error(f"Error en liberación forzada de emergencia: {e}")
            
        except Exception as e:
            self.logger.error(f"Error al verificar uso de memoria: {e}", exc_info=True)
            result["error"] = str(e)
            
        return result

    def optimize_batch_size(self, base_batch_size: int, min_batch_size: int = 1, 
                             max_batch_size: Optional[int] = None, verification_suspended: bool = False) -> int:
        """
        Ajusta dinámicamente el tamaño de lote (batch_size) para operaciones
        intensivas en memoria, basándose en múltiples factores:
        
        1. Uso actual de memoria y CPU del sistema
        2. Resultados históricos de operaciones anteriores
        3. Eficiencia del garbage collection reciente
        4. Disponibilidad de recursos en el sistema
        
        Optimización mejorada que proporciona respuesta rápida con bajo overhead.

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
        # Reducir la verbosidad de logging para minimizar overhead
        self.logger.debug(f"Optimizando batch_size. Base: {base_batch_size}")
        
        # Si las verificaciones están suspendidas, usar un enfoque adaptado a ese escenario
        if verification_suspended:
            # Durante suspensión de verificaciones podemos ser más agresivos con el batch size
            # ya que se asume que estamos en un modo de rendimiento sobre seguridad
            suspension_factor = 1.5  # Incremento del 50% durante suspensión
            
            optimized_size = int(base_batch_size * suspension_factor)
            if max_batch_size is not None:
                optimized_size = min(optimized_size, max_batch_size)
                
            # Reducir logging
            return optimized_size
        
        # Variables para ajuste inteligente
        optimized_batch_size = base_batch_size  # Valor por defecto
        scaling_factor = 1.0  # Factor neutral por defecto
        
        try:
            # Verificar disponibilidad de ResourceManager y sus métricas
            if not self.resource_manager or not hasattr(self.resource_manager, 'metrics'):
                return base_batch_size

            # 1. Factor basado en métricas de memoria - Simplificado para reducir overhead
            mem_usage_pct = self.resource_manager.metrics.get("system_memory_percent", 0)
            mem_available_gb = self.resource_manager.metrics.get("system_memory_available_gb", 0)
            
            # 2. Cálculo simplificado del factor de memoria según presión actual
            # Usar tabla de decisión en lugar de cálculos complejos
            if mem_usage_pct >= 85:  # Memoria crítica
                memory_factor = 0.25  # Reducir drásticamente
            elif mem_usage_pct >= 75: 
                memory_factor = 0.50  # Reducción moderada
            elif mem_usage_pct >= 65:
                memory_factor = 0.75  # Reducción leve
            elif mem_available_gb >= 4.0:  # Mucha memoria disponible
                memory_factor = 1.2   # Aumentar ligeramente
            else:
                memory_factor = 1.0   # Factor neutral
            
            # 3. Factor CPU (simplificado)
            cpu_factor = 1.0  # Valor neutral por defecto
            cpu_percent = self.resource_manager.metrics.get('cpu_percent_system', 0)
            if cpu_percent >= 90:
                cpu_factor = 0.5   # Reducir a la mitad si CPU muy alta
            elif cpu_percent >= 70:
                cpu_factor = 0.7   # Reducción moderada
            
            # 4. Combinar factores con ponderación simple
            scaling_factor = (memory_factor * 0.7) + (cpu_factor * 0.3)
            
            # Limitar el factor combinado a un rango razonable
            scaling_factor = max(0.2, min(1.5, scaling_factor))
            
            # Aplicar factor al batch_size base con redondeo a enteros
            optimized_batch_size = max(min_batch_size, round(base_batch_size * scaling_factor))
            
            # Aplicar límite máximo si está configurado
            if max_batch_size is not None:
                optimized_batch_size = min(optimized_batch_size, max_batch_size)

            # Sistema simple de tracking histórico (máximo 5 entries para evitar overhead)
            if not hasattr(self, '_batch_size_history'):
                self._batch_size_history = []
                
            # Guardar historial reciente de forma eficiente
            self._batch_size_history.append({
                'time': time.time(), 
                'base': base_batch_size, 
                'optimized': optimized_batch_size,
                'factor': scaling_factor
            })
            
            # Mantener historial limitado
            if len(self._batch_size_history) > 5:
                self._batch_size_history = self._batch_size_history[-5:]
            
        except Exception as e:
            self.logger.error(f"Error optimizando batch_size: {e}. Usando valor conservador.")
            # Valor conservador en caso de error (75% del base)
            scaling_factor = 0.75
            optimized_batch_size = max(min_batch_size, int(base_batch_size * scaling_factor))
            if max_batch_size is not None:
                optimized_batch_size = min(optimized_batch_size, max_batch_size)
        
        # Reducir verbosidad de logging
        self.logger.debug(f"Batch size: {base_batch_size} → {optimized_batch_size} (factor: {scaling_factor:.2f}x)")
        return optimized_batch_size

    def shutdown(self) -> None:
        """
        Realiza cualquier limpieza o cierre necesario para MemoryManager al apagar el sistema.
        Actualmente, no tiene acciones específicas más allá de registrar el evento.
        """
        self.logger.info("MemoryManager shutdown solicitado.")
        # Por ahora, no hay acciones específicas de shutdown para MemoryManager.
        self.logger.info("MemoryManager shutdown completado.") 