import os
import logging
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from typing import Optional, Union, Callable, Iterable, Iterator, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .resource_manager import ResourceManager

class ConcurrencyManager:
    """
    Gestiona la concurrencia para tareas intensivas en CPU y I/O.

    Es instanciado y utilizado por ResourceManager. Proporciona acceso a pools
    de hilos (ThreadPoolExecutor) y procesos (ProcessPoolExecutor) configurados
    dinámicamente según los recursos del sistema y la configuración proporcionada
    a través de ResourceManager.

    Permite ejecutar tareas individuales o mapear funciones sobre iterables de
    forma concurrente.

    Atributos:
        resource_manager (ResourceManager): Instancia del ResourceManager principal.
        logger (logging.Logger): Logger para esta clase.
        thread_pool_executor (Optional[ThreadPoolExecutor]): Pool para tareas I/O bound.
        process_pool_executor (Optional[ProcessPoolExecutor]): Pool para tareas CPU bound.
    """
    def __init__(self, resource_manager_instance: 'ResourceManager'):
        """
        Inicializa el ConcurrencyManager.

        Args:
            resource_manager_instance (ResourceManager): La instancia de ResourceManager
                que gestionará este ConcurrencyManager y de donde se obtendrá
                la configuración de concurrencia.
        """
        self.resource_manager = resource_manager_instance
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers(): # Fallback
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger.info("ConcurrencyManager inicializado.")

        self.thread_pool_executor: Optional[ThreadPoolExecutor] = None
        self.process_pool_executor: Optional[ProcessPoolExecutor] = None
        
        self._initialize_pools()

    def _calculate_workers(self, config_value: Union[str, int], core_count: int, worker_type: str) -> int:
        """
        Calcula el número óptimo de workers para un tipo de tarea específico.

        Args:
            config_value (Union[str, int]): El valor de configuración para el número
                de workers (puede ser un entero o "auto").
            core_count (int): El número de cores de CPU disponibles.
            worker_type (str): Tipo de worker ("CPU" o "IO") para aplicar heurísticas.

        Returns:
            int: El número calculado de workers, asegurando un mínimo de 1.
        """
        self.logger.debug(f"Calculando workers para {worker_type}. Config: '{config_value}', Cores: {core_count}")
        num_workers = 0
        if isinstance(config_value, int):
            num_workers = config_value
        elif isinstance(config_value, str) and config_value.lower() == "auto":
            if worker_type == "CPU":
                num_workers = core_count
            elif worker_type == "IO":
                num_workers = min(32, (core_count * 2) + 4) # Ajuste común para IO bound
            else:
                num_workers = core_count # Default a core_count si el tipo no es reconocido
        else:
            self.logger.warning(f"Valor de configuración de workers '{config_value}' no reconocido para {worker_type}. Usando {core_count} workers.")
            num_workers = core_count
        
        # Asegurar un mínimo de 1 worker
        calculated_workers = max(1, num_workers)
        self.logger.debug(f"Workers calculados para {worker_type}: {calculated_workers}")
        return calculated_workers

    def _initialize_pools(self) -> None:
        """
        Inicializa los pools de ThreadPoolExecutor y ProcessPoolExecutor.

        El número de workers para cada pool se determina a partir de la configuración
        obtenida del ResourceManager (que a su vez la carga de config.yaml)
        y el número de cores del sistema. Se aplica una heurística para ajustar
        el número de workers si se excede un `max_total_workers` configurado.
        """
        self.logger.info("Inicializando pools de ejecutores...")
        core_count = os.cpu_count() or 1
        
        # Obtener configuración del ResourceManager
        # Estos atributos deben existir en ResourceManager y ser cargados desde config.yaml
        cfg_cpu_workers = getattr(self.resource_manager, 'default_cpu_workers', "auto")
        cfg_io_workers = getattr(self.resource_manager, 'default_io_workers', "auto")
        max_total_workers_config = getattr(self.resource_manager, 'max_total_workers', None)

        max_cpu_workers = self._calculate_workers(cfg_cpu_workers, core_count, "CPU")
        max_io_workers = self._calculate_workers(cfg_io_workers, core_count, "IO")

        # Lógica simple para respetar max_total_workers si está configurado
        if isinstance(max_total_workers_config, int) and max_total_workers_config > 0:
            if (max_cpu_workers + max_io_workers) > max_total_workers_config:
                self.logger.warning(
                    f"La suma de workers CPU ({max_cpu_workers}) e IO ({max_io_workers}) excede max_total_workers ({max_total_workers_config}). "
                    f"Se ajustarán proporcionalmente. Esto es una heurística simple."
                )
                total_calculated = max_cpu_workers + max_io_workers
                # Reducir proporcionalmente - se podría mejorar esta lógica
                max_cpu_workers = math.floor(max_cpu_workers * (max_total_workers_config / total_calculated))
                max_io_workers = math.floor(max_io_workers * (max_total_workers_config / total_calculated))
                max_cpu_workers = max(1, max_cpu_workers) # Asegurar al menos 1
                max_io_workers = max(1, max_io_workers)
                self.logger.info(f"Workers ajustados: CPU={max_cpu_workers}, IO={max_io_workers}")

        try:
            if self.thread_pool_executor is None or self.thread_pool_executor._shutdown:
                self.thread_pool_executor = ThreadPoolExecutor(max_workers=max_io_workers, thread_name_prefix='RM_Thread')
                self.logger.info(f"ThreadPoolExecutor inicializado con {max_io_workers} workers.")
            else:
                self.logger.info("ThreadPoolExecutor ya estaba inicializado.")
        except Exception as e:
            self.logger.error(f"Error al inicializar ThreadPoolExecutor: {e}", exc_info=True)
            self.thread_pool_executor = None

        try:
            if self.process_pool_executor is None or self.process_pool_executor._shutdown:
                self.process_pool_executor = ProcessPoolExecutor(max_workers=max_cpu_workers)
                self.logger.info(f"ProcessPoolExecutor inicializado con {max_cpu_workers} workers.")
            else:
                self.logger.info("ProcessPoolExecutor ya estaba inicializado.")
        except Exception as e:
            self.logger.error(f"Error al inicializar ProcessPoolExecutor: {e}", exc_info=True)
            self.process_pool_executor = None

    def get_thread_pool_executor(self) -> Optional[ThreadPoolExecutor]:
        """
        Obtiene la instancia del ThreadPoolExecutor para tareas I/O bound.

        Si el pool no está inicializado o fue cerrado, intenta reinicializarlo.

        Returns:
            Optional[ThreadPoolExecutor]: La instancia del ThreadPoolExecutor o None si falla.
        """
        if self.thread_pool_executor is None or self.thread_pool_executor._shutdown:
            self.logger.warning("ThreadPoolExecutor no está inicializado o fue apagado. Intentando reinicializar.")
            self._initialize_pools() # O una subrutina para recrear solo este pool
        return self.thread_pool_executor

    def get_process_pool_executor(self) -> Optional[ProcessPoolExecutor]:
        """
        Obtiene la instancia del ProcessPoolExecutor para tareas CPU bound.

        Si el pool no está inicializado o fue cerrado, intenta reinicializarlo.

        Returns:
            Optional[ProcessPoolExecutor]: La instancia del ProcessPoolExecutor o None si falla.
        """
        if self.process_pool_executor is None or self.process_pool_executor._shutdown:
            self.logger.warning("ProcessPoolExecutor no está inicializado o fue apagado. Intentando reinicializar.")
            self._initialize_pools() # O una subrutina para recrear solo este pool
        return self.process_pool_executor

    def run_in_thread_pool(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Optional[Future]:
        """
        Ejecuta una función de forma asíncrona en el ThreadPoolExecutor.

        Args:
            func (Callable[..., Any]): La función a ejecutar.
            *args (Any): Argumentos posicionales para la función.
            **kwargs (Any): Argumentos de palabra clave para la función.

        Returns:
            Optional[Future]: Un objeto Future que representa la ejecución de la función,
                              o None si el executor no está disponible.
        """
        executor = self.get_thread_pool_executor()
        if executor:
            return executor.submit(func, *args, **kwargs)
        self.logger.error("No se pudo ejecutar en ThreadPool: Executor no disponible.")
        return None

    def run_in_process_pool(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Optional[Future]:
        """
        Ejecuta una función de forma asíncrona en el ProcessPoolExecutor.

        Asegurarse de que la función y sus argumentos/keywords sean "picklables".

        Args:
            func (Callable[..., Any]): La función a ejecutar.
            *args (Any): Argumentos posicionales para la función.
            **kwargs (Any): Argumentos de palabra clave para la función.

        Returns:
            Optional[Future]: Un objeto Future que representa la ejecución de la función,
                              o None si el executor no está disponible.
        """
        executor = self.get_process_pool_executor()
        if executor:
            # Asegurarse que la función y args/kwargs son picklables
            return executor.submit(func, *args, **kwargs)
        self.logger.error("No se pudo ejecutar en ProcessPool: Executor no disponible.")
        return None

    def map_tasks_in_thread_pool(self, func: Callable[..., Any], iterables: Iterable[Any], timeout: Optional[float] = None, chunksize: int = 1) -> Optional[Iterator]:
        """
        Aplica una función a cada ítem de un iterable de forma concurrente usando ThreadPoolExecutor.

        Similar a `map()` incorporado, pero ejecutado en hilos.

        Args:
            func (Callable[..., Any]): La función a aplicar.
            iterables (Iterable[Any]): Un iterable (o varios) cuyos ítems se pasarán a `func`.
            timeout (Optional[float]): Tiempo máximo de espera para cada tarea. Defaults to None.
            chunksize (int): Tamaño de los lotes de tareas enviadas al pool. Defaults to 1.

        Returns:
            Optional[Iterator]: Un iterador que produce los resultados en el orden de los ítems
                                del iterable, o None si el executor no está disponible.
        """
        executor = self.get_thread_pool_executor()
        if executor:
            return executor.map(func, iterables, timeout=timeout, chunksize=chunksize)
        self.logger.error("No se pudo mapear tareas en ThreadPool: Executor no disponible.")
        return None

    def map_tasks_in_process_pool(self, func: Callable[..., Any], iterables: Iterable[Any], timeout: Optional[float] = None, chunksize: int = 1) -> Optional[Iterator]:
        """
        Aplica una función a cada ítem de un iterable de forma concurrente usando ProcessPoolExecutor.

        Similar a `map()` incorporado, pero ejecutado en procesos.
        Asegurarse de que `func` y los ítems en `iterables` sean "picklables".

        Args:
            func (Callable[..., Any]): La función a aplicar.
            iterables (Iterable[Any]): Un iterable cuyos ítems se pasarán a `func`.
            timeout (Optional[float]): Tiempo máximo de espera para cada tarea. Defaults to None.
            chunksize (int): Tamaño de los lotes de tareas enviadas al pool. Defaults to 1.

        Returns:
            Optional[Iterator]: Un iterador que produce los resultados en el orden de los ítems
                                del iterable, o None si el executor no está disponible.
        """
        executor = self.get_process_pool_executor()
        if executor:
            return executor.map(func, iterables, timeout=timeout, chunksize=chunksize)
        self.logger.error("No se pudo mapear tareas en ProcessPool: Executor no disponible.")
        return None

    def shutdown_executors(self, wait: bool = True) -> None:
        """
        Cierra de forma controlada los pools de ThreadPoolExecutor y ProcessPoolExecutor.

        Args:
            wait (bool): Si True, espera a que todas las tareas pendientes completen antes
                         de cerrar los pools. Si False, cierra inmediatamente. 
                         Defaults to True.
        """
        self.logger.info(f"Solicitando shutdown de ejecutores (wait={wait})...")
        if self.thread_pool_executor:
            try:
                self.logger.debug("Cerrando ThreadPoolExecutor...")
                self.thread_pool_executor.shutdown(wait=wait)
                self.logger.info("ThreadPoolExecutor cerrado.")
            except Exception as e:
                self.logger.error(f"Error al cerrar ThreadPoolExecutor: {e}", exc_info=True)
            finally:
                self.thread_pool_executor = None

        if self.process_pool_executor:
            try:
                self.logger.debug("Cerrando ProcessPoolExecutor...")
                self.process_pool_executor.shutdown(wait=wait)
                self.logger.info("ProcessPoolExecutor cerrado.")
            except Exception as e:
                self.logger.error(f"Error al cerrar ProcessPoolExecutor: {e}", exc_info=True)
            finally:
                self.process_pool_executor = None
        self.logger.info("Shutdown de ejecutores completado.") 