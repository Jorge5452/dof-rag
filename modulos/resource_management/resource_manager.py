# Standard library imports
import os
import threading
import time
import logging
import platform

# Third party imports
import psutil

# Type checking imports
from typing import Optional, Dict, Any, TYPE_CHECKING

# Type hints for avoiding circular imports
if TYPE_CHECKING:
    from .memory_manager import MemoryManager
    from .concurrency_manager import ConcurrencyManager
    from modulos.session_manager.session_manager import SessionManager

# Configuración opcional
try:
    from config import Config
except ImportError:
    # Fallback si Config no se puede importar directamente
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
        disable_process_pool (bool): Indica si el proceso debe deshabilitarse.
        worker_recalculation_frequency (float): Frecuencia de recálculo de workers en segundos.
        dynamic_recalc_interval (bool): Si el intervalo de recálculo debe ajustarse dinámicamente.
        environment_type (str): Tipo de entorno detectado ("development", "production", "cloud", etc.)
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

            # Inicializar log_verbosity con valor por defecto antes de cargar la configuración
            self.log_verbosity = "normal"  # Valor por defecto

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
            
            # Sistema de suspensión inteligente de verificaciones
            self.verification_suspended = False
            self.verification_resume_time = None
            self.verification_suspend_reason = None
            self.suspended_monitoring_interval_sec = 60  # intervalo extendido cuando está suspendido
            
            # Detección de entorno (nuevo)
            self.environment_type = self._detect_environment()
            
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
                "monitoring_thread_active": False,
                "verification_status": "active",  # puede ser "active", "suspended"
                "environment_type": self.environment_type  # Nueva métrica
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
                # Actualizar conteo de modelos de embedding activos con mejor manejo de errores
                from modulos.embeddings.embeddings_factory import EmbeddingFactory
                if hasattr(EmbeddingFactory, 'get_active_model_count'):
                    self.metrics["active_embedding_models"] = EmbeddingFactory.get_active_model_count()
                else:
                    # Alternativa si el método no existe
                    self.metrics["active_embedding_models"] = len(getattr(EmbeddingFactory, '_instances', {}))
                    self.logger.warning("El método get_active_model_count no está disponible en EmbeddingFactory")
            except ImportError:
                self.logger.warning("No se pudo importar EmbeddingFactory para obtener conteo de modelos")
                self.metrics["active_embedding_models"] = 0
            except Exception as e:
                self.logger.error(f"Error al obtener conteo de modelos activos: {e}")
                self.metrics["active_embedding_models"] = 0

            self.metrics["last_metrics_update_ts"] = time.time()
            self._initialized = True
            
            # Log final de inicialización
            if not is_minimal:
                self.logger.info(f"ResourceManager inicializado y configurado (entorno: {self.environment_type}).")

    def _detect_environment(self) -> str:
        """
        Detecta el tipo de entorno en el que se está ejecutando.
        
        Utiliza diversas señales como variables de entorno, características del sistema,
        y configuraciones para determinar si estamos en un entorno de desarrollo, 
        producción, cloud, contenedor, etc.
        
        Returns:
            str: Tipo de entorno detectado ("development", "production", "cloud", "container", etc.)
        """
        # Primero verificar si hay una configuración explícita
        if self.config:
            try:
                general_config = self.config.get_general_config()
                if general_config and "environment" in general_config:
                    return general_config["environment"]
            except Exception:
                pass
                
        # Detectar variables de entorno comunes
        env_vars = os.environ
        
        # Variables comunes para entornos de producción
        production_signals = ["PRODUCTION", "PROD"]
        for signal in production_signals:
            if signal in env_vars:
                return "production"
                
        # Entorno de desarrollo
        development_signals = ["DEVELOPMENT", "DEV"]
        for signal in development_signals:
            if signal in env_vars:
                return "development"
        
        # Detectar entornos cloud
        if "KUBERNETES_SERVICE_HOST" in env_vars:
            return "kubernetes"
            
        if any(key.startswith(("AWS_", "EC2_")) for key in env_vars):
            return "aws"
            
        if "GOOGLE_CLOUD_PROJECT" in env_vars:
            return "gcp"
            
        if "AZURE_" in " ".join(env_vars):
            return "azure"
        
        # Detectar contenedor
        container_flags = [
            os.path.exists("/.dockerenv"),
            os.path.exists("/run/.containerenv"),
        ]
        if any(container_flags):
            return "container"
        
        # Detectar Windows vs Linux
        system = platform.system().lower()
        if "darwin" in system:
            return "macos_development"
        if "win" in system:
            return "windows_development"
            
        # Verificar recursos del sistema para determinar si es probable que sea producción
        try:
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_cores = psutil.cpu_count(logical=False) or 1
            
            # Heurística básica: sistemas con muchos recursos probablemente son producción
            if memory_gb >= 16 and cpu_cores >= 8:
                return "server"
        except Exception:
            pass
            
        # Si no podemos determinar, asumimos desarrollo como caso más seguro
        return "development"

    # Propiedad para inicialización perezosa de SessionManager (evitar dependencia circular)
    @property
    def session_manager_instance(self):
        """Obtiene la instancia de SessionManager con inicialización perezosa y prevención de ciclos."""
        if self._session_manager_instance is None:
            # Detectar ciclo de inicialización
            if getattr(self, '_initializing_session_manager', False):
                self.logger.warning("Ciclo de inicialización detectado entre ResourceManager y SessionManager")
                return None
            
            self._initializing_session_manager = True
            try:
                from modulos.session_manager.session_manager import SessionManager as SM_Class
                self._session_manager_instance = SM_Class()
                is_minimal = getattr(self, 'log_verbosity', 'normal') == "minimal"
                if not is_minimal and hasattr(self, 'logger'):
                    self.logger.debug("SessionManager recuperado para ResourceManager.")
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Error al acceder a SessionManager: {e}", exc_info=True)
            finally:
                self._initializing_session_manager = False
        return self._session_manager_instance

    def _load_configuration(self):
        """
        Carga la configuración para ResourceManager desde config.yaml,
        aplicando también valores predeterminados donde sea necesario.
        """
        # Usar el valor ya inicializado de log_verbosity
        is_minimal = self.log_verbosity == "minimal"
        
        try:
            if not self.config:
                if not is_minimal:
                    self.logger.info("Instancia de Config no proporcionada, intentando obtener la global...")
                from config import config as global_config
                self.config = global_config
        
            if self.config:
                resource_config = self.config.get_resource_management_config()
                
                # Verificar que la configuración no está vacía
                if not resource_config:
                    raise ValueError("Configuración de ResourceManager está vacía. Verifica config.yaml.")
                    
                if not is_minimal:
                    self.logger.info(f"Configuración cargada correctamente: {len(resource_config)} parámetros.")
                    
                # Configurar nivel de verbosidad (actualizando el valor existente)
                self.log_verbosity = resource_config.get("log_verbosity", self.log_verbosity)
                    
                # Cargar parámetros de monitoreo (con estructura anidada)
                monitoring_config = resource_config.get("monitoring", {})
                self.monitoring_interval_sec = monitoring_config.get("interval_sec", 120)
                self.suspended_monitoring_interval_sec = monitoring_config.get("suspended_interval_sec", 300)
                self.monitoring_enabled = self.monitoring_interval_sec > 0
                
                # Umbrales de monitoreo
                self.aggressive_cleanup_threshold_mem_pct = monitoring_config.get("aggressive_threshold_mem_pct", 90)
                self.warning_cleanup_threshold_mem_pct = monitoring_config.get("warning_threshold_mem_pct", 80)
                self.warning_threshold_cpu_pct = monitoring_config.get("warning_threshold_cpu_pct", 90)
                
                # Cargar parámetros para memory_manager (con estructura anidada)
                memory_config = resource_config.get("memory", {})
                self.auto_suspend_memory_mb = memory_config.get("auto_suspend_memory_mb", 1000)
                self.max_suspend_duration_sec = memory_config.get("max_suspend_duration_sec", 1800)
                self.document_size_threshold_kb = memory_config.get("document_size_threshold_kb", 5000)
                self.min_chunks_for_suspend = memory_config.get("min_chunks_for_suspend", 200)
                
                # Parámetros para concurrency_manager
                concurrency_config = resource_config.get("concurrency", {})
                self.default_cpu_workers = concurrency_config.get("cpu_workers", "auto")
                self.default_io_workers = concurrency_config.get("io_workers", "auto")
                self.max_total_workers = concurrency_config.get("max_total_workers", None)
                self.disable_process_pool = concurrency_config.get("disable_process_pool", False)
                self.default_timeout_sec = concurrency_config.get("default_timeout_sec", 120)
                
                # Nuevos parámetros para recálculo de workers (con estructura anidada)
                worker_config = resource_config.get("worker_management", {})
                self.worker_recalculation_frequency = worker_config.get("worker_recalculation_frequency", 300.0)
                self.dynamic_recalc_interval = worker_config.get("dynamic_recalc_interval", True)
                self.max_recalc_interval = worker_config.get("max_recalc_interval", 1800.0)
                self.worker_cooldown_after_change = worker_config.get("worker_cooldown_after_change", 120.0)
                self.worker_change_threshold_pct = worker_config.get("worker_change_threshold_pct", 15)
                
                # Parámetros para el entorno (con estructura anidada)
                env_config = resource_config.get("environment", {})
                # Si se define un tipo de entorno explícito, usarlo
                explicit_env = env_config.get("environment_type", "")
                if explicit_env:
                    self.environment_type = explicit_env
                # Si no, usamos el auto-detectado en __init__
                
                # Verificar si debemos evitar ajustes automáticos por entorno
                self.override_environment_adjustments = env_config.get("override_environment_adjustments", False)
                
                # Ajustar parámetros según entorno detectado (a menos que esté desactivado)
                if not self.override_environment_adjustments:
                    self._adjust_parameters_for_environment()
                
                # Registro detallado si verbosidad adecuada
                if self.log_verbosity == "detailed":
                    self.logger.info(f"Configuración ResourceManager - Monitoreo: {self.monitoring_interval_sec}s "
                                     f"(suspendido: {self.suspended_monitoring_interval_sec}s), "
                                     f"Umbrales: agresivo={self.aggressive_cleanup_threshold_mem_pct}%, "
                                     f"advertencia={self.warning_cleanup_threshold_mem_pct}%, "
                                     f"cpu={self.warning_threshold_cpu_pct}%")
                    self.logger.info(f"Configuración de suspensión - memoria={self.auto_suspend_memory_mb}MB, "
                                     f"duración={self.max_suspend_duration_sec}s, "
                                     f"tamaño_doc={self.document_size_threshold_kb}KB, "
                                     f"min_chunks={self.min_chunks_for_suspend}")
                    self.logger.info(f"Configuración concurrencia - CPU workers={self.default_cpu_workers}, "
                                     f"I/O workers={self.default_io_workers}, "
                                     f"disable_process_pool={self.disable_process_pool}, "
                                     f"recalc_frequency={self.worker_recalculation_frequency}s")
                    self.logger.info(f"Configuración recálculo workers - Frecuencia={self.worker_recalculation_frequency}s, "
                                     f"Dinámico={self.dynamic_recalc_interval}, "
                                     f"Máx interval={self.max_recalc_interval}s, "
                                     f"Umbral cambio={self.worker_change_threshold_pct}%")
            else:
                raise ValueError("No se pudo obtener config. Verifica que config.py esté disponible.")
        
        except Exception as e:
            self.logger.error(f"Error al cargar configuración de ResourceManager: {e}", exc_info=True)
            # Valores por defecto seguros en caso de error
            self.monitoring_interval_sec = 120
            self.suspended_monitoring_interval_sec = 300
            self.monitoring_enabled = True
            self.aggressive_cleanup_threshold_mem_pct = 90
            self.warning_cleanup_threshold_mem_pct = 80
            self.warning_threshold_cpu_pct = 90
            self.default_cpu_workers = "auto"
            self.default_io_workers = "auto"
            self.disable_process_pool = False
            self.worker_recalculation_frequency = 300.0
            self.dynamic_recalc_interval = True
            self.max_recalc_interval = 1800.0
            self.worker_cooldown_after_change = 120.0
            self.worker_change_threshold_pct = 15
            self.override_environment_adjustments = False

    def _adjust_parameters_for_environment(self):
        """
        Ajusta los parámetros de configuración basándose en el entorno detectado.
        
        Diferentes entornos (desarrollo, producción, cloud, etc.) pueden beneficiarse
        de configuraciones optimizadas para sus características específicas.
        """
        # Si el entorno no se ha detectado, hacerlo ahora
        if not hasattr(self, 'environment_type'):
            self.environment_type = self._detect_environment()
        
        # Variables para decidir si se aplican ajustes
        apply_adjustments = True
        
        # No aplicar ajustes si hay configuración explícita que lo impida
        if self.config:
            try:
                resource_config = self.config.get_resource_management_config()
                if resource_config and "override_environment_adjustments" in resource_config:
                    apply_adjustments = not resource_config["override_environment_adjustments"]
            except Exception:
                pass
        
        if not apply_adjustments:
            return
            
        # Aplicar ajustes según el tipo de entorno
        if self.environment_type == "development":
            # Desarrollo: más frecuente, menos agresivo, enfoque en feedback rápido
            self.monitoring_interval_sec = min(self.monitoring_interval_sec, 90)  # Máximo 90 segundos
            self.worker_recalculation_frequency = min(self.worker_recalculation_frequency, 180.0)  # Máximo 3 minutos
            self.warning_cleanup_threshold_mem_pct = 85  # Más tolerante con memoria
            
        elif self.environment_type in ("production", "server"):
            # Producción: menos frecuente, más agresivo con limpieza
            self.worker_recalculation_frequency = max(self.worker_recalculation_frequency, 300.0)  # Mínimo 5 minutos
            self.warning_cleanup_threshold_mem_pct = min(self.warning_cleanup_threshold_mem_pct, 75)  # Más agresivo con memoria
            self.dynamic_recalc_interval = True  # Siempre habilitar ajuste dinámico
            
        elif self.environment_type in ("aws", "gcp", "azure", "cloud"):
            # Entornos cloud: optimizar para reducir costos y mejorar elasticidad
            self.worker_recalculation_frequency = max(self.worker_recalculation_frequency, 360.0)  # Mínimo 6 minutos
            self.warning_cleanup_threshold_mem_pct = 70  # Muy agresivo con memoria
            self.aggressive_cleanup_threshold_mem_pct = 85  # Muy agresivo
            
        elif self.environment_type in ("container", "kubernetes"):
            # Contenedor: recursos limitados, ser muy eficiente
            self.warning_cleanup_threshold_mem_pct = 65  # Extremadamente agresivo
            self.aggressive_cleanup_threshold_mem_pct = 80
            self.worker_recalculation_frequency = 240.0  # 4 minutos
            
        elif "windows" in self.environment_type:
            # Windows suele tener gestión de memoria diferente
            self.warning_cleanup_threshold_mem_pct = min(self.warning_cleanup_threshold_mem_pct, 75)
            
        # Log de ajustes aplicados según verbosidad
        if self.log_verbosity == "detailed":
            self.logger.info(f"Parámetros ajustados para entorno {self.environment_type}")

    def _start_monitoring_thread(self):
        """Inicia el hilo de monitoreo de recursos si está habilitado y no activo."""
        # Solo iniciar el hilo si está habilitado y no existe
        if not self.monitoring_enabled:
            return
            
        # Verificar si ya existe un hilo de monitoreo activo
        if hasattr(self, '_monitor_thread') and self._monitor_thread and self._monitor_thread.is_alive():
            # Acceso seguro a log_verbosity
            is_detailed = getattr(self, 'log_verbosity', 'normal') == "detailed"
            if is_detailed:
                self.logger.debug("El hilo de monitoreo de recursos ya está activo.")
            return
            
        # Iniciar el hilo de monitoreo
        self._stop_monitoring_event.clear() # Asegurar que el evento de parada está limpio
        self._monitor_thread = ResourceMonitorThread(
            self, 
            self._stop_monitoring_event,
            verification_suspended=self.verification_suspended,
            verification_resume_time=self.verification_resume_time
        )
        self._monitor_thread.start()
        self.metrics["monitoring_thread_active"] = True
        
        # Acceso seguro a log_verbosity
        is_minimal = getattr(self, 'log_verbosity', 'normal') == "minimal"
        if not is_minimal:
            status_msg = "suspendido" if self.verification_suspended else "activo"
            self.logger.info(f"Hilo de monitoreo de recursos iniciado (estado: {status_msg}).")

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
        
        Si las verificaciones están suspendidas, solo actualiza métricas críticas
        para reducir el overhead del sistema.
        
        Esta función es llamada periódicamente por el hilo de monitoreo.
        """
        # Acceso seguro a log_verbosity
        is_minimal = getattr(self, 'log_verbosity', 'normal') == "minimal"
        is_detailed = getattr(self, 'log_verbosity', 'normal') == "detailed"
        
        # Verificar si las verificaciones están suspendidas
        verification_suspended = self.is_verification_suspended()
        
        # En modo suspendido, reducir la frecuencia (permitir una actualización cada 60s)
        if verification_suspended:
            last_update_time = self.metrics.get("last_metrics_update_ts", 0)
            time_since_update = time.time() - last_update_time
            
            # Si no ha pasado suficiente tiempo desde la última actualización en modo suspendido
            if time_since_update < self.suspended_monitoring_interval_sec:
                if is_detailed:
                    self.logger.debug(f"Actualizaciones suspendidas: solo {time_since_update:.1f}s desde la última actualización")
                return
            elif is_detailed:
                self.logger.debug("Actualización de métricas críticas en modo suspendido")
        elif is_detailed:
            self.logger.debug("Actualizando todas las métricas de recursos...")
            
        try:
            # Métricas de memoria del sistema (siempre actualizamos, son críticas)
            sys_mem = psutil.virtual_memory()
            self.metrics["system_memory_total_gb"] = round(sys_mem.total / (1024**3), 2)
            self.metrics["system_memory_available_gb"] = round(sys_mem.available / (1024**3), 2)
            self.metrics["system_memory_used_gb"] = round(sys_mem.used / (1024**3), 2)
            self.metrics["system_memory_percent"] = sys_mem.percent

            # Métricas de CPU del sistema (siempre actualizamos, son críticas)
            self.metrics["cpu_percent_system"] = psutil.cpu_percent(interval=None)
            
            # Si verificaciones suspendidas, saltamos métricas no críticas
            if not verification_suspended:
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
                    if hasattr(EmbeddingFactory, 'get_active_model_count'):
                        self.metrics["active_embedding_models"] = EmbeddingFactory.get_active_model_count()
                    else:
                        # Alternativa si el método no existe
                        self.metrics["active_embedding_models"] = len(getattr(EmbeddingFactory, '_instances', {}))
                        self.logger.warning("El método get_active_model_count no está disponible en EmbeddingFactory")
                except ImportError:
                    self.logger.warning("No se pudo importar EmbeddingFactory para obtener conteo de modelos")
                    self.metrics["active_embedding_models"] = 0
                except Exception as e:
                    self.logger.error(f"Error al obtener conteo de modelos activos: {e}")
                    self.metrics["active_embedding_models"] = 0
                    
                # Verificar estado de los pools de workers periódicamente
                # Hacerlo cada 5 actualizaciones de métricas (para no sobrecargar)
                last_pools_check = self.metrics.get("last_pools_check_ts", 0)
                time_since_pools_check = time.time() - last_pools_check
                
                # Verificar si ha pasado suficiente tiempo desde la última verificación
                if time_since_pools_check > 300:  # 5 minutos
                    if is_detailed:
                        self.logger.debug("Verificando estado de pools de workers...")
                    
                    # Llamar a check_worker_pools_status para verificar y optimizar pools
                    pools_status = self.check_worker_pools_status()
                    self.metrics["last_pools_check_ts"] = time.time()
                    
                    # Si se realizó alguna acción, registrarla
                    if pools_status.get("recalculation_performed", False):
                        if is_detailed:
                            self.logger.debug("Recálculo de workers realizado durante verificación de pools")
                    
                    # Verificar si se realizaron acciones en los pools
                    for pool_type in ["thread_pool", "process_pool"]:
                        if "action" in pools_status.get(pool_type, {}):
                            action = pools_status[pool_type]["action"]
                            reason = pools_status[pool_type].get("reason", "")
                            if is_detailed:
                                self.logger.debug(f"Pool {pool_type}: {action} - {reason}")

            self.metrics["last_metrics_update_ts"] = time.time()
            
            # Reducir drasticamente la verbosidad del log de métricas
            if is_detailed:
                mem_pct = self.metrics["system_memory_percent"]
                cpu_pct = self.metrics["cpu_percent_system"]
                sess = self.metrics.get("active_sessions_rag", "N/A")
                models = self.metrics.get("active_embedding_models", "N/A")
                if verification_suspended:
                    self.logger.debug(f"Métricas críticas actualizadas en modo suspendido: Memoria={mem_pct:.1f}%, CPU={cpu_pct:.1f}%")
                else:
                    self.logger.debug(f"Métricas: Memoria={mem_pct:.1f}%, CPU={cpu_pct:.1f}%, Sesiones={sess}, Modelos={models}")

        except Exception as e:
            self.logger.error(f"Error al actualizar métricas: {e}", exc_info=True)

    def request_cleanup(self, aggressive: bool = False, reason: str = "manual", respect_cooldown: bool = True) -> bool:
        """
        Solicita una limpieza de recursos del sistema.
        
        Versión optimizada con procesamiento inteligente que evita liberar modelos de embeddings
        durante el procesamiento continuo de documentos, centrándose en tareas de limpieza
        eficientes con bajo overhead.

        Args:
            aggressive (bool): Si es True, realiza una limpieza más profunda, incluso 
                              liberando recursos que podrían ser costosos de reinicializar.
            reason (str): La razón para solicitar la limpieza, útil para diagnóstico y decisiones.
            respect_cooldown (bool): Si es True, respeta los periodos mínimos entre limpiezas.
                                   False fuerza la limpieza independientemente del tiempo transcurrido.

        Returns:
            bool: True si se realizó la limpieza, False si se omitió (ej. por cooldown).
        """
        if not self.memory_manager:
            self.logger.warning("No se puede solicitar limpieza: MemoryManager no inicializado")
            return False

        # Verificar periodos de cooldown si se solicita
        if respect_cooldown:
            current_time = time.time()
            time_since_last_cleanup = current_time - self.metrics.get("last_cleanup_time", 0)
            
            # Definir intervalos mínimos según nivel de agresividad
            min_interval = self.memory_manager.min_aggressive_gc_interval if aggressive else self.memory_manager.min_gc_interval
            
            if time_since_last_cleanup < min_interval:
                self.logger.debug(f"Omitiendo limpieza ({reason}): en período de enfriamiento ({time_since_last_cleanup:.1f}s < {min_interval:.1f}s)")
                return False
        
        # Determinar si estamos en procesamiento de documentos
        in_document_processing = self.metrics.get("operation_in_progress") == "document_processing"
        
        # Lógica específica para procesos de ingestión de documentos
        if in_document_processing:
            self.logger.debug(f"Limpieza durante procesamiento de documentos ({reason})")
            # Durante procesamiento, evitar la liberación de modelos de embeddings
            # pero mantener la limpieza de memoria
            result = self.memory_manager.cleanup(
                aggressive=aggressive, 
                reason=reason, 
                skip_model_cleanup=True  # No liberar modelos durante ingestión continua
            )
        else:
            # Limpieza normal fuera de procesamiento de documentos
            result = self.memory_manager.cleanup(
                aggressive=aggressive,
                reason=reason
            )
            
        # Actualizar métricas de limpieza
        if result:
            self.metrics["last_cleanup_time"] = time.time()
            self.metrics["cleanup_count"] = self.metrics.get("cleanup_count", 0) + 1
            
            if aggressive:
                self.metrics["aggressive_cleanup_count"] = self.metrics.get("aggressive_cleanup_count", 0) + 1
                
            # Actualizar métricas del sistema después de la limpieza
            self.update_metrics()
        
        return result

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

    def suspend_verifications(self, duration_seconds: int = 300, reason: str = "manual") -> bool:
        """
        Suspende temporalmente las verificaciones y limpieza de recursos.
        
        Durante el periodo de suspensión, el monitoreo de recursos se reduce 
        y solo se actualizan métricas críticas. No se realizan limpiezas automáticas.
        Esta función es útil para tareas pequeñas donde el overhead de verificación
        constante no es necesario.
        
        Args:
            duration_seconds (int): Duración en segundos de la suspensión.
                                   Después de este tiempo, las verificaciones se reanudan
                                   automáticamente. Defaults to 300 (5 minutos).
            reason (str): Razón para la suspensión. Defaults to "manual".
            
        Returns:
            bool: True si la suspensión se activó correctamente, False en caso contrario.
        """
        try:
            with self._lock:
                # Si ya está suspendido, actualizar razón y tiempo de reanudación
                if self.verification_suspended:
                    # Extender el tiempo de suspensión desde ahora
                    self.verification_resume_time = time.time() + duration_seconds
                    self.verification_suspend_reason = reason
                    
                    # Acceso seguro a log_verbosity
                    is_minimal = getattr(self, 'log_verbosity', 'normal') == "minimal"
                    if not is_minimal:
                        self.logger.info(f"Suspensión de verificaciones extendida por {duration_seconds}s. Razón: {reason}")
                    
                    # Actualizar estado en métricas
                    self.metrics["verification_status"] = "suspended"
                    return True
                
                # Activar suspensión
                self.verification_suspended = True
                self.verification_resume_time = time.time() + duration_seconds
                self.verification_suspend_reason = reason
                
                # Actualizar estado en métricas
                self.metrics["verification_status"] = "suspended"
                
                # Acceso seguro a log_verbosity
                is_minimal = getattr(self, 'log_verbosity', 'normal') == "minimal"
                if not is_minimal:
                    self.logger.info(f"Verificaciones suspendidas por {duration_seconds}s. Razón: {reason}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error al suspender verificaciones: {e}", exc_info=True)
            return False
            
    def resume_verifications(self, reason: str = "manual") -> bool:
        """
        Reanuda las verificaciones y limpieza de recursos si estaban suspendidas.
        
        Args:
            reason (str): Razón para la reanudación. Defaults to "manual".
            
        Returns:
            bool: True si la reanudación fue exitosa o si las verificaciones no
                 estaban suspendidas, False en caso de error.
        """
        try:
            with self._lock:
                # Si no estaba suspendido, no hacer nada
                if not self.verification_suspended:
                    return True
                
                # Reanudar verificaciones
                self.verification_suspended = False
                self.verification_resume_time = None
                suspend_reason = self.verification_suspend_reason
                self.verification_suspend_reason = None
                
                # Actualizar estado en métricas
                self.metrics["verification_status"] = "active"
                
                # Acceso seguro a log_verbosity
                is_minimal = getattr(self, 'log_verbosity', 'normal') == "minimal"
                if not is_minimal:
                    self.logger.info(f"Verificaciones reanudadas. Razón: {reason}, estaban suspendidas por: {suspend_reason}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error al reanudar verificaciones: {e}", exc_info=True)
            return False

    def is_verification_suspended(self) -> bool:
        """
        Verifica si las comprobaciones están actualmente suspendidas.
        
        También verifica si ha pasado el tiempo de suspensión y actualiza
        el estado si es necesario.
        
        Returns:
            bool: True si las verificaciones están suspendidas, False en caso contrario
        """
        # Si no existe el atributo, no está suspendido
        if not hasattr(self, '_verification_suspended') or not self._verification_suspended:
            return False
            
        # Comprobar si el tiempo de suspensión ha expirado
        now = time.time()
        if now >= self._verification_suspended_until:
            self._verification_suspended = False
            self.logger.debug("Suspensión de verificaciones finalizada por expiración de tiempo")
            return False
            
        # Aún suspendido
        return True

    def should_suspend_verifications(self, document_size_kb: Optional[int] = None, chunk_count: Optional[int] = None) -> bool:
        """
        Determina si se deben suspender las verificaciones basándose en los criterios establecidos.
        
        Las verificaciones se suspenden si:
        1. El documento es muy pequeño (<150KB) Y no tiene muchos chunks (menos de 20)
        2. O si el sistema tiene baja carga (CPU <20% y memoria <60%)
        
        Se ha mejorado la lógica para evitar falsos positivos con documentos grandes
        y para ser más relajado con las verificaciones.
        
        Args:
            document_size_kb: Tamaño del documento en KB
            chunk_count: Número de chunks que se están procesando
            
        Returns:
            bool: True si las verificaciones deben suspenderse, False en caso contrario
        """
        # Acceso seguro a log_verbosity
        is_minimal = getattr(self, 'log_verbosity', 'normal') == "minimal"
        is_detailed = getattr(self, 'log_verbosity', 'normal') == "detailed"
        
        # Si ya está suspendido, mantener suspendido
        if self.is_verification_suspended():
            return True
            
        # Regla 1: Documento muy pequeño (<150KB) Y pocos chunks (<20)
        if document_size_kb is not None and document_size_kb < 150:
            if chunk_count is None or chunk_count < 20:
                if not is_minimal:
                    self.logger.info(f"Documento pequeño detectado ({document_size_kb:.1f}KB). Considerando suspensión de verificaciones.")
                return True
                
        # Regla 2: Baja carga del sistema
        current_cpu = self.metrics.get("cpu_percent", 0)
        current_mem = self.metrics.get("memory_percent", 0)
        
        if current_cpu < 20 and current_mem < 60:
            if is_detailed:
                self.logger.debug(f"Carga del sistema baja (CPU: {current_cpu:.1f}%, Memoria: {current_mem:.1f}%). Suspendiendo verificaciones.")
            return True
            
        # Si nada de lo anterior aplica, no suspender verificaciones
        return False

    def auto_suspend_if_needed(self, document_size_kb: float = 0, chunk_count: int = 0, 
                           duration_seconds: int = 300) -> bool:
        """
        Decide automáticamente si suspender verificaciones basado en las
        características del documento y estado del sistema.
        
        Esta función analiza varios factores:
        1. Tamaño del documento en KB
        2. Número de chunks (si ya se conoce)
        3. Estado actual de los recursos del sistema
        4. Carga de trabajo actual
        
        Args:
            document_size_kb (float): Tamaño del documento en kilobytes
            chunk_count (int): Número de chunks (si ya se generaron)
            duration_seconds (int): Duración de la suspensión si se activa
            
        Returns:
            bool: True si se suspendieron verificaciones, False en caso contrario
        """
        # Verificar si la suspensión está configurada como activada
        verification_suspension_enabled = getattr(self, 'verification_suspension_enabled', True)
        if not verification_suspension_enabled:
            self.logger.debug("Suspensión de verificaciones desactivada en configuración")
            return False
            
        # Extraer métricas actuales del sistema
        mem_percent = self.metrics.get("system_memory_percent", 0)
        cpu_percent = self.metrics.get("cpu_percent_system", 0)
        
        # Criterios para suspender:
        suspend = False
        reason = ""
        
        # 1. Documentos extremadamente pequeños - no suspender (procesamiento muy rápido)
        if document_size_kb < 10:  # Menos de 10KB
            self.logger.debug(f"Documento muy pequeño ({document_size_kb:.1f} KB). No se suspenden verificaciones.")
            return False
            
        # 2. Documentos muy grandes - suspender para optimizar rendimiento
        if document_size_kb > 1000:  # Más de 1MB
            suspend = True
            reason = f"documento grande ({document_size_kb:.1f} KB)"
            
        # 3. Si conocemos los chunks y son muchos - suspender
        elif chunk_count > 50:
            suspend = True
            reason = f"alto número de chunks ({chunk_count})"
            
        # 4. Recursos del sistema bajo presión - suspender
        elif mem_percent > 80 or cpu_percent > 80:
            suspend = True
            reason = f"alta presión en sistema (Memoria: {mem_percent:.1f}%, CPU: {cpu_percent:.1f}%)"
            
        # 5. Documentos medianos (100KB-1MB) - evaluar caso por caso
        elif document_size_kb > 100:
            # Estimar chunks basado en tamaño (aprox. 1 chunk por cada 2-4KB)
            estimated_chunks = document_size_kb / 3
            if estimated_chunks > 30:  # Si estimamos más de 30 chunks
                suspend = True
                reason = f"documento de tamaño medio con chunks estimados: ~{int(estimated_chunks)}"
                
        # Aplicar la suspensión si se determinó necesaria
        if suspend:
            return self.suspend_verification(duration_seconds, f"Auto-suspensión: {reason}")
        else:
            self.logger.debug(f"No se requiere suspensión para documento de {document_size_kb:.1f} KB")
            return False
    
    def suspend_verification(self, duration_seconds: int = 300, reason: str = "manual") -> bool:
        """
        Suspende temporalmente ciertas verificaciones que pueden afectar el rendimiento.
        
        Durante la suspensión, algunas operaciones intensivas como verificaciones
        de pickleable, ciertas validaciones y comprobaciones pueden desactivarse
        para mejorar el rendimiento durante procesamientos intensivos.
        
        Args:
            duration_seconds (int): Duración en segundos de la suspensión
            reason (str): Razón de la suspensión (para logging)
            
        Returns:
            bool: True si se aplicó la suspensión, False si ya estaba suspendido
        """
        if hasattr(self, '_verification_suspended') and self._verification_suspended:
            # Ya está suspendido, actualizar tiempo si es necesario
            now = time.time()
            time_remaining = self._verification_suspended_until - now
            
            if duration_seconds > time_remaining:
                # Extender la suspensión si la nueva duración es mayor
                self._verification_suspended_until = now + duration_seconds
                self.logger.info(f"Suspensión extendida por {duration_seconds}s. Razón: {reason}")
                return True
            else:
                # Mantener suspensión existente
                self.logger.debug(f"Suspensión ya activa por {time_remaining:.1f}s más. No se extiende.")
                return False
        
        # Activar suspensión
        self._verification_suspended = True
        self._verification_suspended_until = time.time() + duration_seconds
        self._verification_suspended_reason = reason
        
        self.logger.info(f"Verificaciones suspendidas por {duration_seconds}s. Razón: {reason}")
        
        # Programar reactivación automática
        try:
            import threading
            
            def restore_verification():
                try:
                    now = time.time()
                    if hasattr(self, '_verification_suspended_until') and now >= self._verification_suspended_until:
                        self._verification_suspended = False
                        self.logger.info("Suspensión de verificaciones finalizada automáticamente")
                except Exception as e:
                    self.logger.error(f"Error restaurando verificaciones: {e}")
            
            # Crear y ejecutar temporizador
            timer = threading.Timer(duration_seconds, restore_verification)
            timer.daemon = True
            timer.start()
        except ImportError:
            self.logger.warning("No se pudo programar restauración automática de verificaciones (threading no disponible)")
        
        return True
    
    def resume_verification(self) -> bool:
        """
        Reactiva las verificaciones suspendidas antes del tiempo programado.
        
        Returns:
            bool: True si había verificaciones suspendidas que fueron reactivadas,
                 False si no había suspensión activa
        """
        if hasattr(self, '_verification_suspended') and self._verification_suspended:
            self._verification_suspended = False
            self.logger.info("Verificaciones reactivadas manualmente")
            return True
        return False

    def check_worker_pools_status(self) -> Dict[str, Any]:
        """
        Verifica el estado actual de los pools de workers y realiza optimizaciones
        según sea necesario. Esta función se puede llamar periódicamente para
        asegurar que los pools están en un estado óptimo.
        
        Acciones que realiza:
        1. Verifica si hay pools hibernados que deberían ser restaurados
        2. Verifica si hay pools activos que deberían ser hibernados
        3. Recalcula el número óptimo de workers si es necesario
        
        Returns:
            Dict[str, Any]: Diccionario con información sobre el estado de los pools
        """
        if not self.concurrency_manager:
            self.logger.warning("ConcurrencyManager no disponible para verificar estado de pools")
            return {"error": "ConcurrencyManager no disponible"}
        
        result = {
            "thread_pool": {},
            "process_pool": {},
            "recalculation_performed": False,
            "timestamp": time.time()
        }
        
        try:
            # 1. Verificar si es hora de recalcular workers
            recalculation_performed = self.concurrency_manager.recalculate_workers_if_needed()
            result["recalculation_performed"] = recalculation_performed
            
            # 2. Obtener información actual de los pools
            pool_status = self.concurrency_manager.pools_status
            worker_counts = self.concurrency_manager.get_worker_counts()
            
            # 3. Verificar si hay pools hibernados que deberían restaurarse
            for pool_type in ["thread_pool", "process_pool"]:
                pool_info = pool_status.get(pool_type, {})
                pool_state = pool_info.get("status", "inactive")
                
                result[pool_type]["status"] = pool_state
                result[pool_type]["workers"] = pool_info.get("workers", 0)
                result[pool_type]["last_used"] = pool_info.get("last_used", 0)
                
                # Si el pool está hibernado y ha pasado suficiente tiempo desde la última hibernación
                if pool_state == "hibernated":
                    hibernated_at = pool_info.get("hibernated_at", 0)
                    time_since_hibernation = time.time() - hibernated_at
                    
                    # Si ha estado hibernado más de 10 minutos, considerar restaurarlo
                    # para tenerlo listo para la próxima operación
                    if time_since_hibernation > 600:  # 10 minutos
                        restored = self.concurrency_manager.restore_pool_from_hibernation(pool_type)
                        result[pool_type]["action"] = "restored" if restored else "restore_failed"
                        result[pool_type]["reason"] = f"Hibernado por {time_since_hibernation:.1f}s"
                
                # Si el pool está activo, verificar si debería hibernarse
                elif pool_state == "active":
                    last_used = pool_info.get("last_used", 0)
                    time_since_last_use = time.time() - last_used
                    
                    # Si no se ha usado en 30 minutos, considerar hibernarlo
                    if time_since_last_use > 1800:  # 30 minutos
                        hibernated = self.concurrency_manager.hibernate_pool_if_unused(pool_type, idle_seconds=1800)
                        result[pool_type]["action"] = "hibernated" if hibernated else "hibernate_failed"
                        result[pool_type]["reason"] = f"Inactivo por {time_since_last_use:.1f}s"
            
            # 4. Actualizar métricas
            self.metrics["worker_pools_status"] = {
                "thread_pool": pool_status.get("thread_pool", {}).get("status", "unknown"),
                "process_pool": pool_status.get("process_pool", {}).get("status", "unknown"),
                "cpu_workers": worker_counts.get("cpu_workers", 0),
                "io_workers": worker_counts.get("io_workers", 0),
                "last_check": time.time()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error al verificar estado de pools: {e}", exc_info=True)
            return {"error": str(e)}

class ResourceMonitorThread(threading.Thread):
    """
    Hilo demonio para monitorear periódicamente los recursos del sistema.

    Este hilo llama al método `update_metrics` del `ResourceManager` y,
    basándose en umbrales configurados, puede solicitar operaciones de limpieza
    a través de `request_cleanup`. También verifica periódicamente si es necesario
    recalcular el número de workers en ConcurrencyManager.

    Atributos:
        resource_manager (ResourceManager): La instancia del ResourceManager a monitorear.
        stop_event (threading.Event): Evento para señalar la detención del hilo.
        interval_sec (int): Intervalo en segundos entre cada ciclo de monitoreo.
        verification_suspended (bool): Indica si las verificaciones están suspendidas.
        verification_resume_time (float): Tiempo en que deben reanudarse las verificaciones.
    """
    def __init__(self, resource_manager_instance: 'ResourceManager', stop_event: threading.Event, 
                verification_suspended: bool = False, verification_resume_time: Optional[float] = None):
        super().__init__(name="ResourceMonitorThread", daemon=True)
        self.resource_manager = resource_manager_instance
        self.stop_event = stop_event
        self.interval_sec = self.resource_manager.monitoring_interval_sec
        self.logger = logging.getLogger(self.__class__.__name__) # Logger para el hilo
        if not self.logger.hasHandlers(): # Fallback
            logging.basicConfig(level=logging.INFO)
        self.verification_suspended = verification_suspended
        self.verification_resume_time = verification_resume_time

    def run(self):
        """
        Lógica principal del hilo de monitoreo.

        En un bucle, actualiza métricas y verifica umbrales para solicitar limpieza
        hasta que el `stop_event` es activado. Respeta la suspensión de verificaciones.
        """
        # Acceder de forma segura a log_verbosity con getattr para proporcionar un valor por defecto
        is_minimal = getattr(self.resource_manager, 'log_verbosity', 'normal') == "minimal"
        is_detailed = getattr(self.resource_manager, 'log_verbosity', 'normal') == "detailed"
        
        if is_detailed:
            status = "suspendido" if self.verification_suspended else "activo"
            self.logger.debug(f"ResourceMonitorThread iniciado. Intervalo: {self.interval_sec}s. Estado: {status}")
        
        # Inicializar timestamps para control de frecuencia de logs y limpiezas
        last_mem_aggressive_time = 0
        last_mem_warning_time = 0
        last_cpu_warning_time = 0
        last_cleanup_time = 0
        
        # Cooldowns para prevenir limpiezas demasiado frecuentes (en segundos)
        warning_cooldown_sec = 60  # Control de frecuencia de mensajes de log (sin cambios)
        cleanup_cooldown_sec = 300  # Aumentado de 180 a 300 segundos
        
        try:
            while not self.stop_event.is_set():
                # Verificar si el tiempo de suspensión ha expirado (comprobar en el ResourceManager)
                verification_suspended = self.resource_manager.is_verification_suspended()
                
                # Sincronizar el estado local con el de ResourceManager
                if verification_suspended != self.verification_suspended:
                    self.verification_suspended = verification_suspended
                    if is_detailed:
                        status = "suspendido" if verification_suspended else "activo"
                        self.logger.debug(f"Estado de verificaciones actualizado: {status}")
                
                # Si es hora de verificar los umbrales (no está suspendido)
                # Actualizar métricas
                self.resource_manager.update_metrics()
                
                # Obtener métricas actuales
                current_mem_pct = self.resource_manager.metrics.get("system_memory_percent", 0)
                current_cpu_pct = self.resource_manager.metrics.get("cpu_percent_system", 0)
                current_time = time.time()
                
                # Verificar si ha pasado suficiente tiempo desde la última limpieza
                time_since_last_cleanup = current_time - last_cleanup_time
                if time_since_last_cleanup < cleanup_cooldown_sec:
                    skip_reason = f"Esperando cooldown ({time_since_last_cleanup:.0f}/{cleanup_cooldown_sec}s)"
                    if is_detailed:
                        self.logger.debug(f"Saltando verificación de umbrales: {skip_reason}")
                    continue

                # Si las verificaciones están suspendidas, no hacer limpieza ni recálculo de workers
                if not verification_suspended:
                    # Lógica de umbrales y solicitud de limpieza
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
                        last_cleanup_time = current_time
                    elif current_mem_pct >= warning_thresh_mem:
                        should_log = is_detailed or (current_time - last_mem_warning_time > warning_cooldown_sec)
                        if should_log:
                            self.logger.warning(f"Uso de memoria ({current_mem_pct:.1f}%) superó umbral de advertencia ({warning_thresh_mem}%). Solicitando limpieza.")
                            last_mem_warning_time = current_time
                        self.resource_manager.request_cleanup(aggressive=False, reason="memory_warning_threshold")
                        last_cleanup_time = current_time
                    
                    # Verificar umbrales de CPU
                    if current_cpu_pct >= warning_thresh_cpu:
                        should_log = is_detailed or (current_time - last_cpu_warning_time > warning_cooldown_sec)
                        if should_log:
                            self.logger.warning(f"Uso de CPU ({current_cpu_pct:.1f}%) superó umbral de advertencia ({warning_thresh_cpu}%).")
                            last_cpu_warning_time = current_time
                    
                    # Verificar si es necesario recalcular workers en ConcurrencyManager
                    if self.resource_manager.concurrency_manager:
                        try:
                            was_recalculated = self.resource_manager.concurrency_manager.recalculate_workers_if_needed()
                            if was_recalculated and is_detailed:
                                workers_info = self.resource_manager.concurrency_manager.get_worker_counts()
                                self.logger.info(f"Workers recalculados: CPU={workers_info['cpu_workers']}, IO={workers_info['io_workers']}")
                        except Exception as e:
                            if not is_minimal:
                                self.logger.error(f"Error al recalcular workers: {e}")
                else:
                    if is_detailed:
                        self.logger.debug("Verificaciones suspendidas: no se realizan verificaciones de umbrales ni limpieza.")

                # Determinar intervalo para el próximo ciclo
                # Si estamos en modo suspendido, usar intervalo extendido
                wait_interval = self.resource_manager.suspended_monitoring_interval_sec if verification_suspended else self.interval_sec
                
                # Esperar hasta el próximo ciclo o hasta que se señale la parada
                self.stop_event.wait(timeout=wait_interval)
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