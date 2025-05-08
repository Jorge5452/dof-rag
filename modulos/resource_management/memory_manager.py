import gc
import logging
import time # Aunque no se usa directamente en este esqueleto, es común en gestión
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .resource_manager import ResourceManager
    # from modulos.embeddings.embeddings_factory import EmbeddingFactory # Para type hinting futuro

class MemoryManager:
    """
    Gestiona las operaciones específicas de optimización y limpieza de memoria.

    Es instanciado y utilizado por ResourceManager para realizar tareas como:
    - Ejecutar recolección de basura (garbage collection).
    - Liberar recursos de modelos de embedding no utilizados.
    - (Placeholder) Limpiar cachés de Python y comprobar fragmentación de memoria.
    - Optimizar dinámicamente tamaños de lote (batch_size) basándose en el uso
      actual de memoria del sistema.

    Atributos:
        resource_manager (ResourceManager): Instancia del ResourceManager principal.
        logger (logging.Logger): Logger para esta clase.
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

        action_clear_caches = self._clear_python_caches()
        if action_clear_caches:
            results["actions_taken"].append(action_clear_caches)

        gc_collections_count = self._run_garbage_collection(aggressive=aggressive)
        results["gc_collections"] = gc_collections_count
        results["actions_taken"].append(f"Garbage collection ejecutado (aggressive={aggressive}), {gc_collections_count} objetos recolectados/ciclos.")

        action_check_fragmentation = self._check_memory_fragmentation()
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
                collected_count = gc.collect(generation=2) # Generación más alta
                self.logger.debug(f"GC agresivo (gen 2) recolectó {collected_count} objetos.")
                # gc.collect(generation=1)
                # gc.collect(generation=0)
            else:
                collected_count = gc.collect() # Recolección estándar
                self.logger.debug(f"GC estándar recolectó {collected_count} objetos.")
            # gc.isenabled() gc.disable() gc.enable()
            # gc.set_debug(gc.DEBUG_STATS | gc.DEBUG_COLLECTABLE | gc.DEBUG_UNCOLLECTABLE)
        except Exception as e:
            self.logger.error(f"Error durante garbage collection: {e}", exc_info=True)
        return collected_count if collected_count is not None else 0

    def _clear_python_caches(self) -> Optional[str]:
        """
        Placeholder para la lógica de limpieza de cachés internas de Python.

        Actualmente, esta función es un placeholder y no realiza operaciones
        concretas de limpieza de cachés más allá de la recolección de basura estándar.

        Returns:
            Optional[str]: Un mensaje indicando la acción realizada (o su naturaleza placeholder).
        """
        self.logger.debug("Intentando limpiar cachés de Python (actualmente placeholder).")
        # Ejemplo: Si se usara lru_cache en alguna parte:
        # from functools import lru_cache
        # mi_funcion_cacheada.cache_clear() # Se necesitaría una forma de registrar/acceder a estas funciones
        # Por ahora, es un placeholder.
        # Podría forzar una pasada adicional de GC como parte de la limpieza de "caches".
        # self._run_garbage_collection(aggressive=False)
        return "Limpieza de cachés de Python (placeholder)."

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

    def _check_memory_fragmentation(self) -> Optional[str]:
        """
        Placeholder para la lógica de comprobación de fragmentación de memoria.

        La medición y mitigación de la fragmentación de memoria en Python es compleja
        y dependiente de la plataforma. Esta función es un placeholder.

        Returns:
            Optional[str]: Mensaje indicando la naturaleza placeholder de la función.
        """
        self.logger.debug("Comprobación de fragmentación de memoria (actualmente placeholder).")
        # Esta es una tarea compleja y dependiente de la plataforma.
        # No hay herramientas estándar de Python para medir/controlar la fragmentación directamente.
        # Podría involucrar logs detallados del uso de memoria antes/después de operaciones grandes.
        return "Comprobación de fragmentación de memoria (placeholder)."

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

    def optimize_batch_size(self, base_batch_size: int, min_batch_size: int = 1, max_batch_size: Optional[int] = None) -> int:
        """
        Ajusta dinámicamente el tamaño de lote (batch_size) para operaciones
        intensivas en memoria, basándose en el uso actual de memoria del sistema.

        Args:
            base_batch_size (int): El tamaño de lote base o preferido.
            min_batch_size (int): El tamaño de lote mínimo permitido. Defaults to 1.
            max_batch_size (Optional[int]): El tamaño de lote máximo permitido. 
                                            Defaults to None (sin límite superior explícito aquí).

        Returns:
            int: El tamaño de lote optimizado. Si no se pueden obtener métricas,
                 devuelve `base_batch_size`.
        """
        self.logger.debug(f"Solicitud para optimizar batch_size. Base: {base_batch_size}, Min: {min_batch_size}, Max: {max_batch_size}")
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