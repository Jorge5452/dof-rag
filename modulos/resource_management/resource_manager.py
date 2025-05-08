import threading
import time
import logging
import os # Añadido para os.getpid()
import platform
import psutil
from typing import Optional, Dict, Any, TYPE_CHECKING # Añadido TYPE_CHECKING
# import torch # Opcional: Añadir solo si se implementará la recolección de métricas de GPU y torch es la vía.

if TYPE_CHECKING: # Evitar importación circular en runtime, permitir en type checking
    from .memory_manager import MemoryManager
    from .concurrency_manager import ConcurrencyManager
    from modulos.session_manager.session_manager import SessionManager # Nueva importación para type hint

# Asumimos que la clase Config es accesible.
# Si está en la raíz del proyecto y 'modulos' está en el sys.path, esto debería funcionar.
# Si no, ajustar la importación según la estructura del proyecto.
try:
    from config import Config
except ImportError:
    # Fallback o manejo de error si Config no se puede importar directamente.
    # Esto es crucial para la independencia del módulo si se prueba aisladamente.
    # Para la integración en el proyecto RAG, la importación directa debería funcionar.
    Config = None 

class ResourceManager:
    """
    Gestor centralizado de recursos del sistema RAG.

    Implementa el patrón Singleton para asegurar una única instancia. Se encarga de:
    - Monitorizar el uso de recursos del sistema y del proceso (CPU, memoria).
    - Coordinar la limpieza de recursos a través de MemoryManager.
    - Gestionar la concurrencia a través de ConcurrencyManager.
    - Cargar y proporcionar configuración específica para la gestión de recursos.
    - Interactuar con otros componentes del sistema RAG (como SessionManager y
      EmbeddingFactory) para obtener métricas y coordinar acciones.

    Atributos:
        config (Optional[Config]): Instancia de la configuración global.
        memory_manager (Optional[MemoryManager]): Instancia del gestor de memoria.
        concurrency_manager (Optional[ConcurrencyManager]): Instancia del gestor de concurrencia.
        session_manager_instance (Optional[SessionManager]): Instancia del gestor de sesiones.
        metrics (Dict[str, Any]): Diccionario con las métricas de recursos recolectadas.
        monitoring_interval_sec (int): Intervalo en segundos para el monitoreo.
        aggressive_cleanup_threshold_mem_pct (float): Umbral de memoria para limpieza agresiva.
        warning_cleanup_threshold_mem_pct (float): Umbral de memoria para advertencias/limpieza normal.
        warning_threshold_cpu_pct (float): Umbral de CPU para advertencias.
        monitoring_enabled (bool): Indica si el hilo de monitoreo está habilitado.
        default_cpu_workers (Union[str, int]): Configuración para workers de CPU.
        default_io_workers (Union[str, int]): Configuración para workers de I/O.
        max_total_workers (Optional[int]): Límite máximo de workers en total.
    """
    _instance = None
    _lock = threading.RLock() # RLock para permitir llamadas recursivas al lock desde el mismo hilo

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_instance: Optional[Config] = None):
        """
        Inicializa el ResourceManager.

        Como es un Singleton, la inicialización real solo ocurre una vez.
        Carga la configuración, inicializa los sub-gestores (MemoryManager,
        ConcurrencyManager) y puede iniciar el hilo de monitoreo de recursos.

        Args:
            config_instance (Optional[Config]): Una instancia opcional de Config.
                Si no se provee, intentará obtener la instancia global de Config.
        """
        # Rápida comprobación sin lock para rendimiento
        if hasattr(self, '_initialized') and self._initialized:
            return

        with self._lock: # Asegurar thread-safety durante la inicialización
            # Segunda comprobación con lock adquirido
            if hasattr(self, '_initialized') and self._initialized:
                return # Otro hilo podría haber terminado la inicialización mientras se esperaba el lock
            
            # INICIO DE LA INICIALIZACIÓN REAL
            # Configurar logger primero para usarlo en el resto de la inicialización
            self.logger = logging.getLogger(self.__class__.__name__)
            if not self.logger.hasHandlers():
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                # Sin mensaje de log aquí, lo hacemos más abajo cuando ya tengamos la configuración de verbosidad

            # Cargar config
            if config_instance:
                self.config = config_instance
            elif Config: # Si la clase Config se importó correctamente
                self.config = Config() # Obtener la instancia Singleton de Config
            else:
                self.logger.warning("Clase Config no disponible. ResourceManager operará sin configuración externa.")
                self.config = None # O una clase Config Dummy/Mock para pruebas

            # Cargar configuración para determinar la verbosidad de logs
            self._load_configuration()

            # Ya tenemos la configuración de verbosidad, ahora podemos decidir qué logs mostrar
            is_detailed = self.log_verbosity == "detailed"
            is_minimal = self.log_verbosity == "minimal"
            
            if is_detailed:
                self.logger.debug("Inicializando ResourceManager como Singleton.")

            # Inicializar atributos
            self._session_manager_instance = None # Usamos propiedad para inicialización perezosa
            
            # Inicializar managers y métricas
            self.memory_manager: Optional["MemoryManager"] = None
            self.concurrency_manager: Optional["ConcurrencyManager"] = None
            
            # Estructura inicial para métricas
            self.metrics: Dict[str, Any] = {
                "system_memory_total_gb": 0.0,
                "system_memory_available_gb": 0.0,
                "system_memory_used_gb": 0.0,
                "system_memory_percent": 0.0,
                "process_memory_rss_mb": 0.0,
                "process_memory_vms_mb": 0.0,
                "process_memory_percent": 0.0,
                "cpu_percent_system": 0.0,
                "cpu_percent_process": 0.0,
                "active_sessions_rag": 0,
                "active_embedding_models": 0,
                "last_metrics_update_ts": 0.0,
                "monitoring_thread_active": False
            }
            
            # Configuración para monitoreo
            self._stop_monitoring_event = threading.Event()
            self._monitor_thread = None
            
            # Inicializar MemoryManager
            try:
                from .memory_manager import MemoryManager as MM_Class
                self.memory_manager = MM_Class(resource_manager_instance=self)
                if not is_minimal:
                    self.logger.info("MemoryManager instanciado y vinculado a ResourceManager.")
            except ImportError:
                self.logger.error("Error al importar MemoryManager. Asegúrate que 'memory_manager.py' existe en el mismo directorio.", exc_info=True)
            except Exception as e:
                self.logger.error(f"Error al instanciar MemoryManager: {e}", exc_info=True)
            
            # Inicializar ConcurrencyManager
            try:
                from .concurrency_manager import ConcurrencyManager as CM_Class
                self.concurrency_manager = CM_Class(resource_manager_instance=self)
                if not is_minimal:
                    self.logger.info("ConcurrencyManager instanciado y vinculado a ResourceManager.")
            except ImportError:
                self.logger.error("Error al importar ConcurrencyManager. Asegúrate que 'concurrency_manager.py' existe.", exc_info=True)
            except Exception as e:
                self.logger.error(f"Error al instanciar ConcurrencyManager: {e}", exc_info=True)
            
            # Iniciar hilo de monitoreo si está habilitado
            if self.monitoring_enabled:
                self._start_monitoring_thread()
                
            # Inicializar métricas
            try:
            # Actualizar conteo de modelos de embedding activos
                from modulos.embeddings.embeddings_factory import EmbeddingFactory
                self.metrics["active_embedding_models"] = EmbeddingFactory.get_active_model_count()
            except (ImportError, Exception) as e:
                if isinstance(e, ImportError):
                    if is_detailed:
                        self.logger.warning("No se pudo importar EmbeddingFactory para obtener conteo de modelos.")
                else:
                    self.logger.error(f"Error al obtener get_active_model_count de EmbeddingFactory: {e}")

            self.metrics["last_metrics_update_ts"] = time.time()
            self._initialized = True
            
            # Log final de inicialización
            if not is_minimal:
                self.logger.info("ResourceManager inicializado y configurado.")

    # Propiedad para inicialización perezosa de SessionManager (evitar dependencia circular)
    @property
    def session_manager_instance(self):
        """Obtiene la instancia de SessionManager con inicialización perezosa."""
        if self._session_manager_instance is None:
            try:
                from modulos.session_manager.session_manager import SessionManager as SM_Class
                self._session_manager_instance = SM_Class()
                is_minimal = getattr(self, 'log_verbosity', 'normal') == "minimal"
                if not is_minimal and hasattr(self, 'logger'):
                    self.logger.debug("SessionManager recuperado para ResourceManager.")
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Error al acceder a SessionManager: {e}", exc_info=True)
        return self._session_manager_instance

    def _load_configuration(self):
        """
        Carga la configuración específica del ResourceManager desde la instancia
        global de Config.

        Establece atributos como intervalos de monitoreo, umbrales de memoria/CPU
        y configuraciones de concurrencia.
        """
        cfg: Dict[str, Any] = {}
        if self.config and hasattr(self.config, 'get_resource_management_config'):
            cfg = self.config.get_resource_management_config()
            self.logger.debug(f"Configuración de Resource Management cargada.")
        else:
            self.logger.warning("get_resource_management_config no disponible o Config no inicializada. Usando valores por defecto para ResourceManager.")

        # Parámetros del plan (Fase 1 y preparación para Fase 2)
        self.monitoring_interval_sec = cfg.get("monitoring_interval", 60)
        self.aggressive_cleanup_threshold_mem_pct = cfg.get("aggressive_threshold_memory", 85)
        self.warning_cleanup_threshold_mem_pct = cfg.get("warning_threshold_memory", 70)
        self.warning_threshold_cpu_pct = cfg.get("warning_threshold_cpu", 80)
        self.monitoring_enabled = cfg.get("monitoring_enabled", True)
        # Añadir configuración para verbosidad de logs
        self.log_verbosity = cfg.get("log_verbosity", "normal")  # puede ser "minimal", "normal", "detailed"

        # Configuración de concurrencia (para ConcurrencyManager - Fase 4)
        concurrency_cfg = cfg.get("concurrency", {})
        self.default_cpu_workers = concurrency_cfg.get("default_cpu_workers", "auto")
        self.default_io_workers = concurrency_cfg.get("default_io_workers", "auto")
        self.max_total_workers = concurrency_cfg.get("max_total_workers", None)

    def _start_monitoring_thread(self):
        """Inicia el hilo de monitoreo de recursos si está habilitado y no activo."""
        # Solo iniciar el hilo si está habilitado y no existe
        if not self.monitoring_enabled:
            return
            
        # Verificar si ya existe un hilo de monitoreo activo
        if hasattr(self, '_monitor_thread') and self._monitor_thread and self._monitor_thread.is_alive():
            is_detailed = self.log_verbosity == "detailed"
            if is_detailed:
                self.logger.debug("El hilo de monitoreo de recursos ya está activo.")
            return
            
        # Iniciar el hilo de monitoreo
        self._stop_monitoring_event.clear() # Asegurar que el evento de parada está limpio
        self._monitor_thread = ResourceMonitorThread(self, self._stop_monitoring_event)
        self._monitor_thread.start()
        self.metrics["monitoring_thread_active"] = True
        
        is_minimal = self.log_verbosity == "minimal"
        if not is_minimal:
            self.logger.info("Hilo de monitoreo de recursos iniciado.")

    # --- Resto de métodos sin cambios excepto optimizaciones de verbosidad ---
    
    def get_system_static_info(self) -> Dict[str, Any]:
        """
        Obtiene información estática del sistema operativo y hardware.

        Esta información puede ser útil para la configuración adaptativa o el diagnóstico.
        Utiliza `platform` y `psutil` para recolectar datos como tipo de OS,
        versión, arquitectura, versión de Python, número de cores de CPU y RAM total.

        Returns:
            Dict[str, Any]: Un diccionario con la información estática del sistema.
                            En caso de error, devuelve un diccionario con una clave "error".
        """
        self.logger.debug("Recolectando información estática del sistema.")
        try:
            info = {
                "os_platform": platform.system(),
                "os_release": platform.release(),
                "os_version": platform.version(),
                "architecture": platform.machine(),
                "python_version": platform.python_version(),
                "cpu_cores_physical": psutil.cpu_count(logical=False),
                "cpu_cores_logical": psutil.cpu_count(logical=True),
                "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                # "gpu_info": self._get_gpu_static_info() # Ejemplo si se implementa
            }
            # Actualizar una única vez el total de RAM en las métricas, si es relevante
            if self.metrics["system_memory_total_gb"] == 0.0:
                    self.metrics["system_memory_total_gb"] = info["total_ram_gb"]
            return info
        except Exception as e:
            self.logger.error(f"Error al obtener información estática del sistema: {e}", exc_info=True)
            return {"error": str(e)}

    def update_metrics(self) -> None:
        """
        Actualiza las métricas de recursos dinámicas del sistema y del proceso actual.

        Recolecta información sobre el uso de memoria (total, disponible, usada, porcentaje),
        uso de CPU (sistema y proceso), y métricas específicas de componentes del RAG
        como el número de sesiones activas y modelos de embedding cargados.
        Los resultados se almacenan en el atributo `self.metrics`.
        Esta función es llamada periódicamente por el hilo de monitoreo.
        """
        is_minimal = self.log_verbosity == "minimal"
        is_detailed = self.log_verbosity == "detailed"
        
        if is_detailed:
            self.logger.debug("Actualizando métricas de recursos...")
        try:
            # Métricas de memoria del sistema
            sys_mem = psutil.virtual_memory()
            self.metrics["system_memory_total_gb"] = round(sys_mem.total / (1024**3), 2)
            self.metrics["system_memory_available_gb"] = round(sys_mem.available / (1024**3), 2)
            self.metrics["system_memory_used_gb"] = round(sys_mem.used / (1024**3), 2)
            self.metrics["system_memory_percent"] = sys_mem.percent

            # Métricas de CPU del sistema
            self.metrics["cpu_percent_system"] = psutil.cpu_percent(interval=None)

            # Métricas del proceso actual
            process = psutil.Process(os.getpid())
            proc_mem_info = process.memory_info()
            self.metrics["process_memory_rss_mb"] = round(proc_mem_info.rss / (1024**2), 2)
            self.metrics["process_memory_vms_mb"] = round(proc_mem_info.vms / (1024**2), 2)
            self.metrics["process_memory_percent"] = round(process.memory_percent(), 2)
            self.metrics["cpu_percent_process"] = round(process.cpu_percent(interval=None), 2)

            # Actualizar métricas de sesiones activas (usando inicialización perezosa)
            if self.session_manager_instance:
                try:
                    self.metrics["active_sessions_rag"] = self.session_manager_instance.get_active_sessions_count()
                except Exception as e:
                    if is_detailed:
                        self.logger.error(f"Error al obtener active_sessions_count: {e}")
            
            # Actualizar conteo de modelos de embedding activos
            try:
                from modulos.embeddings.embeddings_factory import EmbeddingFactory
                self.metrics["active_embedding_models"] = EmbeddingFactory.get_active_model_count()
            except (ImportError, Exception) as e:
                if is_detailed and not isinstance(e, ImportError):
                    self.logger.error(f"Error al obtener get_active_model_count: {e}")

            self.metrics["last_metrics_update_ts"] = time.time()
            
            # Reducir drasticamente la verbosidad del log de métricas
            if is_detailed:
                mem_pct = self.metrics["system_memory_percent"]
                cpu_pct = self.metrics["cpu_percent_system"]
                sess = self.metrics["active_sessions_rag"]
                models = self.metrics["active_embedding_models"]
                self.logger.debug(f"Métricas: Memoria={mem_pct:.1f}%, CPU={cpu_pct:.1f}%, Sesiones={sess}, Modelos={models}")

        except Exception as e:
            self.logger.error(f"Error al actualizar métricas: {e}", exc_info=True)

    def request_cleanup(self, aggressive: bool = False, reason: str = "threshold_exceeded") -> None:
        """
        Solicita una operación de limpieza de recursos.

        Esta función delega la tarea de limpieza al MemoryManager y también
        puede coordinar la limpieza con otros componentes como SessionManager
        y EmbeddingFactory (a través de MemoryManager).

        Args:
            aggressive (bool): Si es True, se realizará una limpieza más intensiva.
                               Defaults to False.
            reason (str): Una cadena que describe el motivo de la solicitud de limpieza.
                          Defaults to "threshold_exceeded".
        """
        is_minimal = self.log_verbosity == "minimal"
        is_detailed = self.log_verbosity == "detailed"

        if is_detailed:
            self.logger.info(f"Solicitud de limpieza recibida. Agresivo: {aggressive}, Razón: {reason}")
        
        # Limpiar memoria a través de MemoryManager
        if self.memory_manager:
            try:
                cleanup_results = self.memory_manager.cleanup(aggressive=aggressive, reason=reason)
                if is_detailed:
                    self.logger.info(f"MemoryManager.cleanup ejecutado. Resultados: {cleanup_results}")
            except Exception as e:
                self.logger.error(f"Error durante MemoryManager.cleanup: {e}", exc_info=True)
        elif not is_minimal:
            self.logger.warning("MemoryManager no está disponible. No se puede ejecutar la limpieza de memoria.")

        # Coordinación con SessionManager para limpiar sesiones (inicialización perezosa)
        if self.session_manager_instance:
            if is_detailed:
                self.logger.info(f"Solicitando limpieza de sesiones (agresivo={aggressive}, razón='{reason}').")
            try:
                cleanup_results = self.session_manager_instance.clean_expired_sessions(
                    aggressive=aggressive, 
                    cleanup_reason=reason
                )
                if is_detailed:
                    self.logger.info(f"SessionManager.clean_expired_sessions completado.")
                    
                # Actualizar métrica de sesiones activas después de la limpieza
                self.metrics["active_sessions_rag"] = self.session_manager_instance.get_active_sessions_count()
            except AttributeError as e:
                # Manejar específicamente el error de atributo que podría ocurrir si falta un método
                if "_save_sessions" in str(e):
                    # Si el error es específicamente sobre _save_sessions, intentar un enfoque alternativo
                    self.logger.warning("SessionManager no tiene el método _save_sessions. Implementando solución alternativa...")
                    try:
                        # Intentar guardar las sesiones directamente usando _save_to_file
                        if hasattr(self.session_manager_instance, '_save_to_file') and hasattr(self.session_manager_instance, 'sessions_file'):
                            self.session_manager_instance._save_to_file(
                                self.session_manager_instance.sessions_file, 
                                self.session_manager_instance.sessions
                            )
                            if not is_minimal:
                                self.logger.info("Sesiones guardadas correctamente con método alternativo.")
                    except Exception as e2:
                        self.logger.error(f"Error al intentar guardar sesiones con método alternativo: {e2}")
                else:
                    # Para otros errores de atributo
                    self.logger.error(f"Error durante SessionManager.clean_expired_sessions (AttributeError): {e}")
            except Exception as e:
                self.logger.error(f"Error durante SessionManager.clean_expired_sessions: {e}", exc_info=True)
        elif is_detailed:
            self.logger.debug("SessionManager no está disponible para limpieza.")

    def shutdown(self) -> None:
        """
        Realiza una parada controlada del ResourceManager y sus componentes.

        Esto incluye detener el hilo de monitoreo y apagar los pools de
        ejecutores en ConcurrencyManager.
        """
        is_minimal = hasattr(self, 'log_verbosity') and self.log_verbosity == "minimal"
        
        if not is_minimal:
            self.logger.info("Iniciando shutdown de ResourceManager...")

        # Detener el hilo de monitoreo
        if hasattr(self, '_monitor_thread') and self._monitor_thread and self._monitor_thread.is_alive():
            if not is_minimal:
                self.logger.info("Deteniendo hilo de monitoreo de recursos...")
            self._stop_monitoring_event.set()
            try:
                self._monitor_thread.join(timeout=self.monitoring_interval_sec + 5)
                if self._monitor_thread.is_alive():
                    self.logger.warning("El hilo de monitoreo no terminó a tiempo.")
                elif not is_minimal:
                    self.logger.info("Hilo de monitoreo de recursos detenido.")
            except Exception as e:
                self.logger.error(f"Error al detener el hilo de monitoreo: {e}", exc_info=True)
            self.metrics["monitoring_thread_active"] = False

        # Shutdown de ConcurrencyManager
        if self.concurrency_manager:
            if not is_minimal:
                self.logger.info("Solicitando shutdown a ConcurrencyManager...")
            try:
                self.concurrency_manager.shutdown_executors(wait=True)
                if not is_minimal:
                    self.logger.info("ConcurrencyManager shutdown completado.")
            except Exception as e:
                self.logger.error(f"Error durante ConcurrencyManager.shutdown_executors: {e}", exc_info=True)

        # Shutdown de MemoryManager
        if self.memory_manager:
            if not is_minimal:
                self.logger.info("Solicitando shutdown a MemoryManager...")
            try:
                self.memory_manager.shutdown()
                if not is_minimal:
                    self.logger.info("MemoryManager shutdown completado.")
            except Exception as e:
                self.logger.error(f"Error durante MemoryManager.shutdown: {e}", exc_info=True)

        if not is_minimal:
         self.logger.info("ResourceManager shutdown completado.")


class ResourceMonitorThread(threading.Thread):
    """
    Hilo demonio para monitorear periódicamente los recursos del sistema.

    Este hilo llama al método `update_metrics` del `ResourceManager` y,
    basándose en umbrales configurados, puede solicitar operaciones de limpieza
    a través de `request_cleanup`.

    Atributos:
        resource_manager (ResourceManager): La instancia del ResourceManager a monitorear.
        stop_event (threading.Event): Evento para señalar la detención del hilo.
        interval_sec (int): Intervalo en segundos entre cada ciclo de monitoreo.
    """
    def __init__(self, resource_manager_instance: 'ResourceManager', stop_event: threading.Event):
        super().__init__(name="ResourceMonitorThread", daemon=True)
        self.resource_manager = resource_manager_instance
        self.stop_event = stop_event
        self.interval_sec = self.resource_manager.monitoring_interval_sec
        self.logger = logging.getLogger(self.__class__.__name__) # Logger para el hilo
        if not self.logger.hasHandlers(): # Fallback
            logging.basicConfig(level=logging.INFO)

    def run(self):
        """
        Lógica principal del hilo de monitoreo.

        En un bucle, actualiza métricas y verifica umbrales para solicitar limpieza
        hasta que el `stop_event` es activado.
        """
        is_minimal = self.resource_manager.log_verbosity == "minimal"
        is_detailed = self.resource_manager.log_verbosity == "detailed"
        
        if is_detailed:
            self.logger.debug(f"ResourceMonitorThread iniciado. Intervalo: {self.interval_sec}s.")
        
        # Variables para controlar la frecuencia de logs repetitivos
        last_mem_warning_time = 0
        last_mem_aggressive_time = 0
        last_cpu_warning_time = 0
        warning_cooldown_sec = 300  # 5 minutos entre warnings similares
        
        try:
            while not self.stop_event.is_set():
                # Actualizar métricas
                self.resource_manager.update_metrics()
                
                # Lógica de umbrales y solicitud de limpieza
                current_time = time.time()
                current_mem_pct = self.resource_manager.metrics.get("system_memory_percent", 0)
                current_cpu_pct = self.resource_manager.metrics.get("cpu_percent_system", 0)

                aggressive_thresh_mem = self.resource_manager.aggressive_cleanup_threshold_mem_pct
                warning_thresh_mem = self.resource_manager.warning_cleanup_threshold_mem_pct
                warning_thresh_cpu = self.resource_manager.warning_threshold_cpu_pct

                # Verificar umbrales de memoria con control de frecuencia de logs
                if current_mem_pct >= aggressive_thresh_mem:
                    should_log = is_detailed or (current_time - last_mem_aggressive_time > warning_cooldown_sec)
                    if should_log:
                        self.logger.warning(f"Uso de memoria ({current_mem_pct:.1f}%) superó umbral agresivo ({aggressive_thresh_mem}%). Solicitando limpieza agresiva.")
                        last_mem_aggressive_time = current_time
                    self.resource_manager.request_cleanup(aggressive=True, reason="memory_aggressive_threshold")
                elif current_mem_pct >= warning_thresh_mem:
                    should_log = is_detailed or (current_time - last_mem_warning_time > warning_cooldown_sec)
                    if should_log and not is_minimal:
                        self.logger.info(f"Uso de memoria ({current_mem_pct:.1f}%) superó umbral de advertencia. Solicitando limpieza.")
                        last_mem_warning_time = current_time
                    self.resource_manager.request_cleanup(aggressive=False, reason="memory_warning_threshold")
                
                # Verificar umbrales de CPU
                if current_cpu_pct >= warning_thresh_cpu:
                    should_log = is_detailed or (current_time - last_cpu_warning_time > warning_cooldown_sec)
                    if should_log:
                        self.logger.warning(f"Uso de CPU ({current_cpu_pct:.1f}%) superó umbral de advertencia ({warning_thresh_cpu}%).")
                        last_cpu_warning_time = current_time

                # Esperar hasta el próximo ciclo o hasta que se señale la parada
                self.stop_event.wait(timeout=self.interval_sec)
        except Exception as e:
            self.logger.error(f"Error inesperado en ResourceMonitorThread: {e}", exc_info=True)
        finally:
            if is_detailed:
                self.logger.debug("ResourceMonitorThread terminando.")

# Para pruebas rápidas si se ejecuta este archivo directamente (opcional)
if __name__ == '__main__':
    print("Ejecutando ResourceManager directamente para prueba básica...")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    rm = ResourceManager() # Usará la config global si está disponible
    print(f"ResourceManager instanciado: {rm}")
    print(f"Métricas iniciales: {rm.metrics}")
    static_info = rm.get_system_static_info()
    print(f"Información estática del sistema: {static_info}")

    try:
        if rm.metrics.get("monitoring_thread_active"):
            print("Hilo de monitoreo activo. Esperando algunas actualizaciones...")
            time.sleep(12)
            print(f"Métricas después de un tiempo: {rm.metrics}")
        else:
            print("Monitoreo no está activo, actualizando métricas manualmente para prueba.")
            rm.update_metrics()
            print(f"Métricas actualizadas manualmente: {rm.metrics}")
    except KeyboardInterrupt:
        print("Interrupción por teclado.")
    finally:
        print("Deteniendo ResourceManager...")
        rm.shutdown()
        print("ResourceManager detenido.") 