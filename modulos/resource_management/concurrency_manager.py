import os
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from typing import Optional, Union, Callable, Iterable, Iterator, TYPE_CHECKING, Any, Dict, Tuple

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
        adaptive_config (Dict): Configuración para la gestión adaptativa de workers.
        cpu_workers (int): Número actual de workers CPU.
        io_workers (int): Número actual de workers I/O.
        base_cpu_workers (int): Número base de workers CPU calculado inicialmente.
        base_io_workers (int): Número base de workers I/O calculado inicialmente.
        last_recalculation_time (float): Timestamp de la última recalculación de workers.
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
        
        # Atributos para gestión adaptativa de workers
        self.adaptive_config = self._load_adaptive_config()
        self.cpu_workers = 0
        self.io_workers = 0
        self.base_cpu_workers = 0
        self.base_io_workers = 0
        self.last_recalculation_time = 0
        
        self._initialize_pools()

    def _load_adaptive_config(self) -> Dict[str, Any]:
        """
        Carga la configuración para la gestión adaptativa de workers.
        
        Returns:
            Dict[str, Any]: Diccionario con la configuración adaptativa.
        """
        # Valores por defecto si no se encuentra configuración
        default_config = {
            "enabled": True,
            "recalculation_interval": 300,  # 5 minutos
            "high_cpu_threshold": 85,
            "high_memory_threshold": 80,
            "reduction_factor": 0.7,
            "increase_factor": 1.2,
            "low_cpu_threshold": 30,
            "low_memory_threshold": 50
        }
        
        # Intentar obtener configuración del ResourceManager
        try:
            if hasattr(self.resource_manager, 'config') and self.resource_manager.config:
                config = self.resource_manager.config
                if hasattr(config, 'get_resource_management_config'):
                    resource_config = config.get_resource_management_config() or {}
                    concurrency_config = resource_config.get("concurrency", {})
                    adaptive_config = concurrency_config.get("adaptive_workers", {})
                    
                    # Actualizar los valores predeterminados con los de la configuración
                    for key, default_value in default_config.items():
                        if key not in adaptive_config:
                            adaptive_config[key] = default_value
                    
                    self.logger.info(f"Configuración adaptativa cargada: {adaptive_config}")
                    return adaptive_config
        except Exception as e:
            self.logger.warning(f"Error al cargar la configuración adaptativa: {e}")
        
        self.logger.warning("Usando configuración adaptativa predeterminada")
        return default_config

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

        # Calcular el número base de workers
        self.base_cpu_workers = self._calculate_workers(cfg_cpu_workers, core_count, "CPU")
        self.base_io_workers = self._calculate_workers(cfg_io_workers, core_count, "IO")
        
        # Aplicar ajuste inicial adaptativo si está habilitado
        if self.adaptive_config["enabled"]:
            # Establecer los valores iniciales (usamos los valores base para comenzar)
            self.cpu_workers = self.base_cpu_workers
            self.io_workers = self.base_io_workers
            self.last_recalculation_time = time.time()
            
            # Registrar la configuración inicial
            self.logger.info(f"Workers iniciales: CPU={self.cpu_workers}, IO={self.io_workers}")
        else:
            # Si la gestión adaptativa está desactivada, usar valores base
            self.cpu_workers = self.base_cpu_workers
            self.io_workers = self.base_io_workers
        
        # Lógica para respetar max_total_workers
        if isinstance(max_total_workers_config, int) and max_total_workers_config > 0:
            if (self.cpu_workers + self.io_workers) > max_total_workers_config:
                self.logger.warning(
                    f"La suma de workers CPU ({self.cpu_workers}) e IO ({self.io_workers}) excede max_total_workers ({max_total_workers_config}). "
                    f"Se ajustarán proporcionalmente."
                )
                total_calculated = self.cpu_workers + self.io_workers
                # Reducir proporcionalmente
                self.cpu_workers = max(1, math.floor(self.cpu_workers * (max_total_workers_config / total_calculated)))
                self.io_workers = max(1, math.floor(self.io_workers * (max_total_workers_config / total_calculated)))
                self.logger.info(f"Workers ajustados por límite total: CPU={self.cpu_workers}, IO={self.io_workers}")

        try:
            if self._is_executor_shutdown(self.thread_pool_executor):
                self.thread_pool_executor = ThreadPoolExecutor(max_workers=self.io_workers, thread_name_prefix='RM_Thread')
                self.logger.info(f"ThreadPoolExecutor inicializado con {self.io_workers} workers.")
            else:
                self.logger.info("ThreadPoolExecutor ya estaba inicializado.")
        except Exception as e:
            self.logger.error(f"Error al inicializar ThreadPoolExecutor: {e}", exc_info=True)
            self.thread_pool_executor = None

        try:
            if self._is_executor_shutdown(self.process_pool_executor):
                self.process_pool_executor = ProcessPoolExecutor(max_workers=self.cpu_workers)
                self.logger.info(f"ProcessPoolExecutor inicializado con {self.cpu_workers} workers.")
            else:
                self.logger.info("ProcessPoolExecutor ya estaba inicializado.")
        except Exception as e:
            self.logger.error(f"Error al inicializar ProcessPoolExecutor: {e}", exc_info=True)
            self.process_pool_executor = None

    def recalculate_workers_if_needed(self) -> bool:
        """
        Recalcula el número de workers según las métricas del sistema si es necesario.
        
        Evalúa las condiciones actuales del sistema (CPU, memoria) y ajusta
        el número de workers si se superan umbrales configurados, respetando
        un intervalo mínimo entre recálculos.
        
        Returns:
            bool: True si se realizó la recalculación, False si no fue necesaria.
        """
        # Si la gestión adaptativa no está habilitada, salir
        if not self.adaptive_config["enabled"]:
            return False
        
        # Verificar si ha pasado suficiente tiempo desde la última recalculación
        current_time = time.time()
        elapsed_since_last = current_time - self.last_recalculation_time
        
        if elapsed_since_last < self.adaptive_config["recalculation_interval"]:
            return False
        
        # Obtener las métricas actuales del sistema
        metrics = self.resource_manager.metrics
        cpu_percent = metrics.get("cpu_percent_system", 0)
        memory_percent = metrics.get("system_memory_percent", 0)
        
        # Determinar si necesitamos ajustar
        need_adjustment = self._check_if_adjustment_needed(cpu_percent, memory_percent)
        
        if need_adjustment:
            self.logger.info(f"Recalculando workers (CPU: {cpu_percent:.1f}%, MEM: {memory_percent:.1f}%)")
            new_cpu, new_io = self._adjust_workers_based_on_metrics(cpu_percent, memory_percent)
            
            # Registrar y aplicar los cambios
            self.last_recalculation_time = current_time
            
            # Si realmente hay cambios, re-inicializar los pools
            if new_cpu != self.cpu_workers or new_io != self.io_workers:
                old_cpu, old_io = self.cpu_workers, self.io_workers
                self.cpu_workers, self.io_workers = new_cpu, new_io
                self.logger.info(f"Ajuste de workers: CPU {old_cpu}->{new_cpu}, IO {old_io}->{new_io}")
                
                # Reinicializar los pools con los nuevos valores
                self._reinitialize_pools_with_new_workers()
                return True
        
        # No se hicieron cambios
        return False
    
    def _check_if_adjustment_needed(self, cpu_percent: float, memory_percent: float) -> bool:
        """
        Determina si es necesario un ajuste de workers basado en métricas.
        
        Args:
            cpu_percent (float): Porcentaje de uso de CPU del sistema.
            memory_percent (float): Porcentaje de uso de memoria del sistema.
            
        Returns:
            bool: True si se necesita ajuste, False en caso contrario.
        """
        # Verificar si alguno de los umbrales se ha superado
        high_cpu = cpu_percent >= self.adaptive_config["high_cpu_threshold"]
        high_memory = memory_percent >= self.adaptive_config["high_memory_threshold"]
        
        low_cpu = cpu_percent <= self.adaptive_config["low_cpu_threshold"]
        low_memory = memory_percent <= self.adaptive_config["low_memory_threshold"]
        
        # Necesitamos ajuste si estamos en condición de alta carga o baja carga
        return (high_cpu or high_memory) or (low_cpu and low_memory)
    
    def _adjust_workers_based_on_metrics(self, cpu_percent: float, memory_percent: float) -> Tuple[int, int]:
        """
        Calcula nuevos valores de workers basados en las métricas del sistema.
        
        Ajusta dinámicamente el número de workers considerando la carga del sistema
        y respetando los límites mínimos configurados.
        
        Args:
            cpu_percent (float): Porcentaje de uso de CPU del sistema.
            memory_percent (float): Porcentaje de uso de memoria del sistema.
            
        Returns:
            Tuple[int, int]: Nuevo número de workers (CPU, IO).
        """
        # Por defecto, mantener los valores actuales
        new_cpu_workers = self.cpu_workers
        new_io_workers = self.io_workers
        
        # Obtener umbrales de configuración adaptativa
        high_cpu = cpu_percent >= self.adaptive_config["high_cpu_threshold"]
        high_memory = memory_percent >= self.adaptive_config["high_memory_threshold"]
        low_cpu = cpu_percent <= self.adaptive_config["low_cpu_threshold"]
        low_memory = memory_percent <= self.adaptive_config["low_memory_threshold"]
        
        # Obtener los valores mínimos configurados
        min_cpu_workers = self.adaptive_config.get("min_cpu_workers", 1)
        min_io_workers = self.adaptive_config.get("min_io_workers", 2)
        
        # Verificar condición de alta carga (reducir workers)
        if high_cpu or high_memory:
            # Aplicar factor de reducción
            reduction = self.adaptive_config["reduction_factor"]
            
            # Reducir workers respetando los mínimos configurados
            new_cpu_workers = max(min_cpu_workers, math.floor(self.cpu_workers * reduction))
            new_io_workers = max(min_io_workers, math.floor(self.io_workers * reduction))
            
            self.logger.info(f"Alta carga detectada (CPU:{cpu_percent:.1f}%, MEM:{memory_percent:.1f}%). "
                            f"Reduciendo workers CPU:{self.cpu_workers}->{new_cpu_workers}, "
                            f"IO:{self.io_workers}->{new_io_workers}, factor:{reduction}")
        
        # Verificar condición de baja carga (aumentar workers si ambos están bajos)
        elif low_cpu and low_memory:
            # Solo aumentar si estamos por debajo del valor base
            if self.cpu_workers < self.base_cpu_workers or self.io_workers < self.base_io_workers:
                # Aplicar factor de aumento
                increase = self.adaptive_config["increase_factor"]
                
                # Aumentar workers sin exceder los valores base
                new_cpu_workers = min(self.base_cpu_workers, math.ceil(self.cpu_workers * increase))
                new_io_workers = min(self.base_io_workers, math.ceil(self.io_workers * increase))
                
                self.logger.info(f"Baja carga detectada (CPU:{cpu_percent:.1f}%, MEM:{memory_percent:.1f}%). "
                                f"Aumentando workers CPU:{self.cpu_workers}->{new_cpu_workers}, "
                                f"IO:{self.io_workers}->{new_io_workers}, factor:{increase}")
        
        # Verificar límites máximos de workers
        max_total_workers = getattr(self.resource_manager, 'max_total_workers', None)
        if isinstance(max_total_workers, int) and max_total_workers > 0:
            if (new_cpu_workers + new_io_workers) > max_total_workers:
                # Calcular la proporción ideal CPU:IO
                total_new = new_cpu_workers + new_io_workers
                cpu_ratio = new_cpu_workers / total_new
                io_ratio = new_io_workers / total_new
                
                # Distribuir el máximo respetando la proporción y los mínimos
                new_cpu_workers = max(min_cpu_workers, math.floor(max_total_workers * cpu_ratio))
                new_io_workers = max(min_io_workers, math.floor(max_total_workers * io_ratio))
                
                # Asegurar que no excedemos el límite total
                while (new_cpu_workers + new_io_workers) > max_total_workers:
                    if new_cpu_workers > min_cpu_workers:
                        new_cpu_workers -= 1
                    elif new_io_workers > min_io_workers:
                        new_io_workers -= 1
                    else:
                        # No podemos reducir más respetando los mínimos, ajustar el límite
                        self.logger.warning(f"Los valores mínimos requeridos (CPU:{min_cpu_workers}, IO:{min_io_workers}) "
                                          f"exceden el límite total ({max_total_workers}). Ajustando límite.")
                        break
                
                self.logger.info(f"Ajuste por límite máximo ({max_total_workers}): "
                               f"CPU workers={new_cpu_workers}, IO workers={new_io_workers}")
        
        return new_cpu_workers, new_io_workers

    def _is_executor_shutdown(self, executor) -> bool:
        """
        Verifica de forma segura si un executor está cerrado.
        
        Args:
            executor: El executor a verificar (ThreadPoolExecutor o ProcessPoolExecutor)
            
        Returns:
            bool: True si el executor está cerrado o no es válido, False en caso contrario
        """
        if executor is None:
            return True
        # Verificar atributo _shutdown directamente si existe
        if hasattr(executor, '_shutdown'):
            return executor._shutdown
        # En versiones más recientes puede ser shutdown 
        if hasattr(executor, 'shutdown'):
            # Esto no es perfecto, pero es mejor que acceder a un atributo privado
            # que puede cambiar entre versiones
            try:
                # Intentar verificar si ya fue cerrado de alguna forma
                return executor._thread is None if hasattr(executor, '_thread') else False
            except:
                # Si no podemos verificar, asumimos que sigue activo
                return False
        # Si no podemos determinar el estado, asumimos que necesita ser reinicializado
        return True
        
    def _reinitialize_pools_with_new_workers(self) -> None:
        """
        Reinicializa los pools de ejecutores con los nuevos valores de workers.
        
        Cierra los pools existentes de manera controlada y crea nuevos con
        el número actualizado de workers.
        """
        # Shutdown de los pools actuales si existen
        self.shutdown_executors(wait=True)
        
        try:
            # Crear nuevo ThreadPoolExecutor
            self.thread_pool_executor = ThreadPoolExecutor(
                max_workers=self.io_workers, 
                thread_name_prefix='RM_Thread'
            )
            self.logger.info(f"ThreadPoolExecutor reinicializado con {self.io_workers} workers.")
        except Exception as e:
            self.logger.error(f"Error al reinicializar ThreadPoolExecutor: {e}", exc_info=True)
            self.thread_pool_executor = None

        try:
            # Crear nuevo ProcessPoolExecutor
            self.process_pool_executor = ProcessPoolExecutor(
                max_workers=self.cpu_workers
            )
            self.logger.info(f"ProcessPoolExecutor reinicializado con {self.cpu_workers} workers.")
        except Exception as e:
            self.logger.error(f"Error al reinicializar ProcessPoolExecutor: {e}", exc_info=True)
            self.process_pool_executor = None

    def get_thread_pool_executor(self) -> Optional[ThreadPoolExecutor]:
        """
        Obtiene la instancia del ThreadPoolExecutor para tareas I/O bound.

        Si el pool no está inicializado o fue cerrado, intenta reinicializarlo.
        
        También realiza una verificación de recalculación de workers si es necesario.

        Returns:
            Optional[ThreadPoolExecutor]: La instancia del ThreadPoolExecutor o None si falla.
        """
        # Verificar si es momento de recalcular workers
        self.recalculate_workers_if_needed()
        
        # Verificar si el pool necesita ser inicializado
        if self._is_executor_shutdown(self.thread_pool_executor):
            self.logger.warning("ThreadPoolExecutor no está inicializado o fue apagado. Intentando reinicializar.")
            self._initialize_pools() # O una subrutina para recrear solo este pool
        return self.thread_pool_executor

    def get_process_pool_executor(self) -> Optional[ProcessPoolExecutor]:
        """
        Obtiene la instancia del ProcessPoolExecutor para tareas CPU bound.

        Si el pool no está inicializado o fue cerrado, intenta reinicializarlo.
        
        También realiza una verificación de recalculación de workers si es necesario.

        Returns:
            Optional[ProcessPoolExecutor]: La instancia del ProcessPoolExecutor o None si falla.
        """
        # Verificar si es momento de recalcular workers
        self.recalculate_workers_if_needed()
        
        # Verificar si el pool necesita ser inicializado
        if self._is_executor_shutdown(self.process_pool_executor):
            self.logger.warning("ProcessPoolExecutor no está inicializado o fue apagado. Intentando reinicializar.")
            self._initialize_pools() # O una subrutina para recrear solo este pool
        return self.process_pool_executor

    def get_worker_counts(self) -> Dict[str, int]:
        """
        Obtiene información sobre el número actual de workers en uso.
        
        Returns:
            Dict[str, int]: Diccionario con información de workers.
        """
        return {
            "cpu_workers": self.cpu_workers,
            "io_workers": self.io_workers,
            "base_cpu_workers": self.base_cpu_workers,
            "base_io_workers": self.base_io_workers
        }

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
        Si ocurre un error de serialización (como con AuthenticationString), automáticamente
        cae al ThreadPoolExecutor.

        Args:
            func (Callable[..., Any]): La función a ejecutar.
            *args (Any): Argumentos posicionales para la función.
            **kwargs (Any): Argumentos de palabra clave para la función.

        Returns:
            Optional[Future]: Un objeto Future que representa la ejecución de la función,
                              o None si el executor no está disponible.
        """
        # Verificar si el proceso de serialización está deshabilitado a nivel de config
        disable_process_pool = getattr(self.resource_manager, 'disable_process_pool', False)
        
        # Si está deshabilitado, usar directamente thread pool
        if disable_process_pool:
            self.logger.info(f"ProcessPool desactivado por configuración, usando ThreadPool para {func.__name__}")
            return self.run_in_thread_pool(func, *args, **kwargs)
        
        # Intentar usar ProcessPoolExecutor
        executor = self.get_process_pool_executor()
        if executor:
            try:
                # Intentar verificar si la función es pickleable
                import pickle
                try:
                    # Verificar si la función y args son serializables
                    pickle.dumps(func)
                    if args:
                        pickle.dumps(args)
                    if kwargs:
                        pickle.dumps(kwargs)
                except Exception as e:
                    # Si hay error de serialización, loguear y usar alternativa
                    self.logger.error(f"Error de serialización para ProcessPoolExecutor: {e}")
                    self.logger.warning(f"Ejecutando función {func.__name__} en ThreadPoolExecutor como alternativa")
                    
                    # Caer con gracia a ThreadPool como alternativa
                    return self.run_in_thread_pool(func, *args, **kwargs)
                
                try:
                    # Si la verificación de pickle pasa, intentar usar el ProcessPoolExecutor
                    return executor.submit(func, *args, **kwargs)
                except TypeError as e:
                    # Capturar específicamente el error de AuthenticationString
                    if "Pickling an AuthenticationString object is disallowed" in str(e):
                        self.logger.warning(f"Detectado AuthenticationString no serializable. Usando ThreadPool para {func.__name__}")
                        # Marcar para evitar intentar usar ProcessPool en el futuro
                        setattr(self.resource_manager, 'disable_process_pool', True)
                        return self.run_in_thread_pool(func, *args, **kwargs)
                    else:
                        raise
                except Exception as e:
                    self.logger.error(f"Error al ejecutar {func.__name__} en ProcessPoolExecutor: {e}", exc_info=True)
                    self.logger.warning(f"Fallback: intentando ejecutar en ThreadPoolExecutor")
                    return self.run_in_thread_pool(func, *args, **kwargs)
                
            except Exception as e:
                self.logger.error(f"Error general al intentar ejecutar {func.__name__}: {e}", exc_info=True)
                # Intentar con ThreadPoolExecutor como último recurso
                self.logger.warning(f"Último recurso: intentando ejecutar en ThreadPoolExecutor")
                return self.run_in_thread_pool(func, *args, **kwargs)
        
        self.logger.error("No se pudo ejecutar en ProcessPool: Executor no disponible. Intentando ThreadPool.")
        return self.run_in_thread_pool(func, *args, **kwargs)

    def map_tasks_in_thread_pool(self, func: Callable[..., Any], iterables: Iterable[Any], 
                                timeout: Optional[float] = None, 
                                chunksize: Optional[int] = None,
                                task_type: str = "default") -> Optional[Iterator]:
        """
        Aplica una función a cada ítem de un iterable de forma concurrente usando ThreadPoolExecutor.

        Similar a `map()` incorporado, pero ejecutado en hilos, con tamaño de chunk optimizado.

        Args:
            func (Callable[..., Any]): La función a aplicar.
            iterables (Iterable[Any]): Un iterable (o varios) cuyos ítems se pasarán a `func`.
            timeout (Optional[float]): Tiempo máximo de espera para cada tarea. Defaults to None.
            chunksize (Optional[int]): Tamaño de los lotes de tareas. Si es None, se calcula automáticamente.
            task_type (str): Tipo de tarea para calcular chunksize óptimo si no se especifica.

        Returns:
            Optional[Iterator]: Un iterador que produce los resultados en el orden de los ítems
                                del iterable, o None si el executor no está disponible.
        """
        executor = self.get_thread_pool_executor()
        if executor:
            # Calcular el tamaño de chunk óptimo si no se proporcionó uno
            if chunksize is None:
                # Intentar determinar la longitud del iterable sin consumirlo
                try:
                    # Esto funciona para listas, tuplas, conjuntos, etc.
                    iterable_length = len(iterables)
                except (TypeError, AttributeError):
                    # Si no podemos obtener la longitud, usar None
                    iterable_length = None
                
                # Calcular el chunksize óptimo
                chunksize = self.get_optimal_chunksize(
                    task_type=task_type if task_type == "io_operations" else "default",
                    iterable_length=iterable_length
                )
                self.logger.debug(f"Chunksize calculado para ThreadPool ({task_type}): {chunksize}")
            
            # Si después de todo, el chunksize sigue siendo None, usar 1 como valor seguro
            if chunksize is None:
                chunksize = 1
            
            return executor.map(func, iterables, timeout=timeout, chunksize=chunksize)
        
        self.logger.error("No se pudo mapear tareas en ThreadPool: Executor no disponible.")
        return None

    def map_tasks_in_process_pool(self, func: Callable[..., Any], iterables: Iterable[Any], 
                                 timeout: Optional[float] = None, 
                                 chunksize: Optional[int] = None,
                                 task_type: str = "default") -> Optional[Iterator]:
        """
        Aplica una función a cada ítem de un iterable de forma concurrente usando ProcessPoolExecutor.

        Similar a `map()` incorporado, pero ejecutado en procesos, con tamaño de chunk optimizado.
        Si hay problemas de serialización, cae automáticamente a ThreadPoolExecutor.
        
        Args:
            func (Callable[..., Any]): La función a aplicar.
            iterables (Iterable[Any]): Un iterable cuyos ítems se pasarán a `func`.
            timeout (Optional[float]): Tiempo máximo de espera para cada tarea. Defaults to None.
            chunksize (Optional[int]): Tamaño de los lotes de tareas. Si es None, se calcula automáticamente.
            task_type (str): Tipo de tarea para calcular chunksize óptimo si no se especifica.

        Returns:
            Optional[Iterator]: Un iterador que produce los resultados en el orden de los ítems
                                del iterable, o None si el executor no está disponible.
        """
        # Verificar si el proceso de serialización está deshabilitado
        disable_process_pool = getattr(self.resource_manager, 'disable_process_pool', False)
        
        # Si está deshabilitado, usar directamente thread pool
        if disable_process_pool:
            self.logger.info(f"ProcessPool desactivado por configuración, usando ThreadPool para map_tasks")
            return self.map_tasks_in_thread_pool(func, iterables, timeout, chunksize, task_type)
        
        # Calcular el tamaño de chunk óptimo si no se proporcionó uno
        if chunksize is None:
            # Intentar determinar la longitud del iterable sin consumirlo
            try:
                # Esto funciona para listas, tuplas, conjuntos, etc.
                iterable_length = len(iterables)
            except (TypeError, AttributeError):
                # Si no podemos obtener la longitud, usar None
                iterable_length = None
            
            # Calcular el chunksize óptimo
            chunksize = self.get_optimal_chunksize(
                task_type=task_type,
                iterable_length=iterable_length
            )
            self.logger.debug(f"Chunksize calculado para ProcessPool ({task_type}): {chunksize}")
        
        # Si después de todo, el chunksize sigue siendo None, usar 1 como valor seguro
        if chunksize is None:
            chunksize = 1
        
        executor = self.get_process_pool_executor()
        if executor:
            try:
                # Intentar verificar si la función es pickleable
                import pickle
                pickle.dumps(func)
                # No podemos verificar todos los items en iterables sin consumirlo
                
                # Intentar usar ProcessPoolExecutor
                try:
                    return executor.map(func, iterables, timeout=timeout, chunksize=chunksize)
                except TypeError as e:
                    # Capturar específicamente el error de AuthenticationString
                    if "Pickling an AuthenticationString object is disallowed" in str(e):
                        self.logger.warning("Detectado AuthenticationString no serializable. Usando ThreadPool para map_tasks")
                        # Marcar para evitar intentar usar ProcessPool en el futuro
                        setattr(self.resource_manager, 'disable_process_pool', True)
                        return self.map_tasks_in_thread_pool(func, iterables, timeout, chunksize, task_type)
                    else:
                        raise
                except Exception as e:
                    self.logger.error(f"Error en map_tasks_in_process_pool: {e}", exc_info=True)
                    self.logger.warning("Fallback: intentando map_tasks en ThreadPoolExecutor")
                    return self.map_tasks_in_thread_pool(func, iterables, timeout, chunksize, task_type)
                
            except Exception as e:
                self.logger.error(f"Error de serialización en map_tasks_in_process_pool: {e}")
                self.logger.warning("Fallback: usando ThreadPoolExecutor para map_tasks")
                return self.map_tasks_in_thread_pool(func, iterables, timeout, chunksize, task_type)
        
        self.logger.error("No se pudo mapear tareas en ProcessPool: Executor no disponible. Intentando ThreadPool.")
        return self.map_tasks_in_thread_pool(func, iterables, timeout, chunksize, task_type)

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

    def get_optimal_chunksize(self, task_type: str = "default", iterable_length: Optional[int] = None) -> int:
        """
        Calcula el tamaño de chunk óptimo para procesamiento por lotes.
        
        Ajusta el tamaño de chunk según el tipo de tarea y la longitud del iterable
        para balancear la sobrecarga de comunicación con el paralelismo efectivo.
        
        Args:
            task_type (str): Tipo de tarea ("default", "io_operations", "embeddings")
            iterable_length (Optional[int]): Longitud del iterable a procesar, si se conoce
            
        Returns:
            int: Tamaño de chunk óptimo
        """
        # Intentar obtener configuración desde config
        try:
            if hasattr(self.resource_manager, 'config') and self.resource_manager.config:
                config = self.resource_manager.config
                if hasattr(config, 'get_resource_management_config'):
                    resource_config = config.get_resource_management_config() or {}
                    concurrency_config = resource_config.get("concurrency", {})
                    chunk_sizes = concurrency_config.get("chunk_sizes", {})
                    
                    # Obtener el valor específico para este tipo de tarea
                    if task_type in chunk_sizes:
                        configured_size = chunk_sizes.get(task_type)
                        if isinstance(configured_size, int) and configured_size > 0:
                            # Si tenemos un tamaño en la configuración, usarlo como base
                            base_size = configured_size
                            self.logger.debug(f"Usando chunk_size configurado para {task_type}: {base_size}")
                            
                            # Si conocemos la longitud del iterable, ajustar dinámicamente
                            if iterable_length is not None:
                                worker_count = max(self.cpu_workers, self.io_workers)
                                if worker_count == 0:  # Prevenir división por cero
                                    worker_count = os.cpu_count() or 4
                                
                                # Calcular un tamaño que distribuya el trabajo eficientemente
                                # entre los workers disponibles
                                dynamic_size = max(1, iterable_length // (worker_count * 2))
                                
                                # Elegir entre el tamaño base y el calculado dinámicamente
                                # privilegiando un valor que mantenga los workers ocupados
                                return max(base_size, min(dynamic_size, 32))
                            
                            return base_size
        except Exception as e:
            self.logger.warning(f"Error al calcular optimal_chunksize: {e}")
        
        # Valores por defecto si no hay configuración o hay error
        if task_type == "embeddings":
            return 16  # Batch size mayor para embeddings (proceso costoso)
        elif task_type == "io_operations":
            return 8   # Para operaciones I/O (que pueden tener latencia)
        else:
            # Para tareas CPU-bound regulares
            # Si conocemos la longitud, calcular dinámicamente
            if iterable_length is not None:
                worker_count = self.cpu_workers if self.cpu_workers > 0 else (os.cpu_count() or 4)
                return max(1, min(iterable_length // (worker_count * 2), 8))
            return 4   # Valor por defecto 

    def map_tasks(self, func: Callable[..., Any], iterables: Iterable[Any], 
                 timeout: Optional[float] = None, 
                 chunksize: Optional[int] = None,
                 task_type: str = "default",
                 prefer_process: Optional[bool] = None) -> Optional[Iterator]:
        """
        Aplica una función a cada ítem de un iterable de forma concurrente, 
        seleccionando automáticamente entre thread y process pools según la tarea.
        
        Esta función centralizada elige inteligentemente el executor más adecuado
        basándose en el tipo de tarea, características del sistema, y configuración,
        maximizando así el uso de recursos disponibles.
        
        Args:
            func (Callable[..., Any]): La función a aplicar.
            iterables (Iterable[Any]): Un iterable cuyos ítems se pasarán a `func`.
            timeout (Optional[float]): Tiempo máximo de espera para cada tarea. Defaults to None.
            chunksize (Optional[int]): Tamaño de los lotes de tareas. Si es None, se calcula automáticamente.
            task_type (str): Tipo de tarea ("default", "io_operations", "embeddings").
            prefer_process (Optional[bool]): Preferencia explícita por process pool (True) o thread pool (False).
                                           Si es None, se decide automáticamente.
                                           
        Returns:
            Optional[Iterator]: Un iterador que produce los resultados en el orden de los ítems
                              del iterable, o None si ningún executor está disponible.
        """
        # Verificar si process pool está deshabilitado globalmente
        disable_process_pool = getattr(self.resource_manager, 'disable_process_pool', False)
        
        # Determinar el tipo de executor a usar
        use_process_pool = False
        if not disable_process_pool:
            if prefer_process is not None:
                # Usar preferencia explícita si se proporciona
                use_process_pool = prefer_process
            else:
                # Decisión automática basada en tipo de tarea
                if task_type in ["io_operations", "network"]:
                    # Tareas I/O-bound se benefician más de ThreadPool
                    use_process_pool = False
                elif task_type in ["cpu_intensive", "embeddings"]:
                    # Tareas CPU-intensive se benefician más de ProcessPool
                    use_process_pool = True
                else:
                    # Para tareas por defecto, usar ProcessPool si hay muchos items
                    try:
                        # Intentar determinar longitud del iterable sin consumirlo
                        iterable_length = len(iterables)
                        # Usar ProcessPool para lotes grandes de trabajo
                        use_process_pool = iterable_length > 10
                    except (TypeError, AttributeError):
                        # Si no podemos determinar la longitud, default a ThreadPool
                        use_process_pool = False
        
        # Usar el pool seleccionado
        if use_process_pool:
            self.logger.debug(f"Usando ProcessPool para map_tasks ({task_type})")
            return self.map_tasks_in_process_pool(func, iterables, timeout, chunksize, task_type)
        else:
            self.logger.debug(f"Usando ThreadPool para map_tasks ({task_type})")
            return self.map_tasks_in_thread_pool(func, iterables, timeout, chunksize, task_type) 