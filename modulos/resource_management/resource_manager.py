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

# Optional configuration
try:
    from config import Config
except ImportError:
    # Fallback if Config cannot be imported directly
    Config = None 

class ResourceManager:
    """
    Centralized resource manager for the RAG system.

    Implements the Singleton pattern to ensure a single instance. Responsible for:
    - Monitoring system and process resource usage (CPU, memory).
    - Coordinating resource cleanup through MemoryManager.
    - Managing concurrency through ConcurrencyManager.
    - Loading and providing resource management specific configuration.
    - Interacting with other RAG system components (like SessionManager and
      EmbeddingFactory) to obtain metrics and coordinate actions.

    Attributes:
        config (Optional[Config]): Instance of global configuration.
        memory_manager (Optional[MemoryManager]): Instance of memory manager.
        concurrency_manager (Optional[ConcurrencyManager]): Instance of concurrency manager.
        session_manager_instance (Optional[SessionManager]): Instance of session manager.
        metrics (Dict[str, Any]): Dictionary with collected resource metrics.
        monitoring_interval_sec (int): Monitoring interval in seconds.
        aggressive_cleanup_threshold_mem_pct (float): Memory threshold for aggressive cleanup.
        warning_cleanup_threshold_mem_pct (float): Memory threshold for warnings/normal cleanup.
        warning_threshold_cpu_pct (float): CPU threshold for warnings.
        monitoring_enabled (bool): Indicates if the monitoring thread is enabled.
        default_cpu_workers (Union[str, int]): Configuration for CPU workers.
        default_io_workers (Union[str, int]): Configuration for I/O workers.
        max_total_workers (Optional[int]): Maximum total workers limit.
        disable_process_pool (bool): Indicates if the process pool should be disabled.
        worker_recalculation_frequency (float): Worker recalculation frequency in seconds.
        dynamic_recalc_interval (bool): If recalculation interval should be dynamically adjusted.
        environment_type (str): Detected environment type ("development", "production", "cloud", etc.)
    """
    _instance = None
    _lock = threading.RLock()  # RLock to allow recursive lock calls from the same thread

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_instance: Optional[Config] = None):
        """
        Initializes the ResourceManager.

        As a Singleton, real initialization only happens once.
        Loads configuration, initializes sub-managers (MemoryManager,
        ConcurrencyManager) and may start the resource monitoring thread.

        Args:
            config_instance (Optional[Config]): An optional Config instance.
                If not provided, it will try to get the global Config instance.
        """
        # Quick check without lock for performance
        if hasattr(self, '_initialized') and self._initialized:
            return

        with self._lock:  # Ensure thread-safety during initialization
            # Second check with acquired lock
            if hasattr(self, '_initialized') and self._initialized:
                return  # Another thread might have finished initialization while waiting for the lock
            
            # START OF ACTUAL INITIALIZATION
            # Set up logger first for use in the rest of initialization
            self.logger = logging.getLogger(self.__class__.__name__)
            if not self.logger.hasHandlers():
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                # No log message here, we'll do it below when we have verbosity configuration

            # Initialize log_verbosity with default value before loading configuration
            self.log_verbosity = "normal"  # Default value

            # Load config
            if config_instance:
                self.config = config_instance
            elif Config: # If the Config class was imported correctly
                self.config = Config() # Get the Singleton instance of Config
            else:
                self.logger.warning("Config class not available. ResourceManager will operate without external configuration.")
                self.config = None # Or a dummy/mock Config class for testing

            # Load configuration to determine log verbosity
            self._load_configuration()

            # Now we have verbosity configuration, we can decide what logs to show
            is_detailed = self.log_verbosity == "detailed"
            is_minimal = self.log_verbosity == "minimal"
            
            if is_detailed:
                self.logger.debug("Initializing ResourceManager as Singleton.")

            # Initialize attributes
            self._session_manager_instance = None # We use property for lazy initialization
            
            # Initialize managers and metrics
            self.memory_manager: Optional["MemoryManager"] = None
            self.concurrency_manager: Optional["ConcurrencyManager"] = None
            
            # Smart suspension of verification checks
            self.verification_suspended = False
            self.verification_resume_time = None
            self.verification_suspend_reason = None
            self.suspended_monitoring_interval_sec = 60  # Extended interval when suspended
            
            # Environment detection (new)
            self.environment_type = self._detect_environment()
            
            # Initial structure for metrics
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
                "verification_status": "active",  # Can be "active", "suspended"
                "environment_type": self.environment_type  # New metric
            }
            
            # Monitoring configuration
            self._stop_monitoring_event = threading.Event()
            self._monitor_thread = None
            
            # Initialize MemoryManager
            try:
                from .memory_manager import MemoryManager as MM_Class
                self.memory_manager = MM_Class(resource_manager_instance=self)
                if not is_minimal:
                    self.logger.info("MemoryManager instantiated and linked to ResourceManager.")
            except ImportError:
                self.logger.error("Error importing MemoryManager. Make sure 'memory_manager.py' exists in the same directory.", exc_info=True)
            except Exception as e:
                self.logger.error(f"Error instantiating MemoryManager: {e}", exc_info=True)
            
            # Initialize ConcurrencyManager
            try:
                from .concurrency_manager import ConcurrencyManager as CM_Class
                self.concurrency_manager = CM_Class(resource_manager_instance=self)
                if not is_minimal:
                    self.logger.info("ConcurrencyManager instantiated and linked to ResourceManager.")
            except ImportError:
                self.logger.error("Error importing ConcurrencyManager. Make sure 'concurrency_manager.py' exists.", exc_info=True)
            except Exception as e:
                self.logger.error(f"Error instantiating ConcurrencyManager: {e}", exc_info=True)
            
            # Start monitoring thread if enabled
            if self.monitoring_enabled:
                self._start_monitoring_thread()
            
            # Initialize metrics
            try:
                # Update active embedding model count with better error handling
                from modulos.embeddings.embeddings_factory import EmbeddingFactory
                if hasattr(EmbeddingFactory, 'get_active_model_count'):
                    self.metrics["active_embedding_models"] = EmbeddingFactory.get_active_model_count()
                else:
                    # Alternative if method doesn't exist
                    self.metrics["active_embedding_models"] = len(getattr(EmbeddingFactory, '_instances', {}))
                    self.logger.warning("get_active_model_count method not available in EmbeddingFactory")
            except ImportError:
                self.logger.warning("Could not import EmbeddingFactory to get embedding model count")
                self.metrics["active_embedding_models"] = 0
            except Exception as e:
                self.logger.error(f"Error getting active model count: {e}")
                self.metrics["active_embedding_models"] = 0

            self.metrics["last_metrics_update_ts"] = time.time()
            self._initialized = True
            
            # Final log of initialization
            if not is_minimal:
                self.logger.info(f"ResourceManager initialized and configured (environment: {self.environment_type}).")

    def _detect_environment(self) -> str:
        """
        Detects the type of environment in which it is running.
        
        Uses various signals like environment variables, system characteristics,
        and configurations to determine if we're in a development, 
        production, cloud, container, etc. environment.
        
        Returns:
            str: Detected environment type ("development", "production", "cloud", "container", etc.)
        """
        # First check if there's explicit configuration
        if self.config:
            try:
                general_config = self.config.get_general_config()
                if general_config and "environment" in general_config:
                    return general_config["environment"]
            except Exception:
                pass
                
        # Detect common environment variables
        env_vars = os.environ
        
        # Common signals for production environments
        production_signals = ["PRODUCTION", "PROD"]
        for signal in production_signals:
            if signal in env_vars:
                return "production"
                
        # Development environment
        development_signals = ["DEVELOPMENT", "DEV"]
        for signal in development_signals:
            if signal in env_vars:
                return "development"
        
        # Detect cloud environments
        if "KUBERNETES_SERVICE_HOST" in env_vars:
            return "kubernetes"
            
        if any(key.startswith(("AWS_", "EC2_")) for key in env_vars):
            return "aws"
            
        if "GOOGLE_CLOUD_PROJECT" in env_vars:
            return "gcp"
            
        if "AZURE_" in " ".join(env_vars):
            return "azure"
        
        # Detect container
        container_flags = [
            os.path.exists("/.dockerenv"),
            os.path.exists("/run/.containerenv"),
        ]
        if any(container_flags):
            return "container"
        
        # Detect Windows vs Linux
        system = platform.system().lower()
        if "darwin" in system:
            return "macos_development"
        if "win" in system:
            return "windows_development"
            
        # Verify system resources to determine if it's probably production
        try:
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_cores = psutil.cpu_count(logical=False) or 1
            
            # Basic heuristic: systems with many resources probably are production
            if memory_gb >= 16 and cpu_cores >= 8:
                return "server"
        except Exception:
            pass
            
        # If we can't determine, we assume development as the safest case
        return "development"

    # Property for lazy initialization of SessionManager (avoid circular dependency)
    @property
    def session_manager_instance(self):
        """Gets the SessionManager instance with lazy initialization and cycle prevention."""
        if self._session_manager_instance is None:
            # Detect cycle initialization
            if getattr(self, '_initializing_session_manager', False):
                self.logger.warning("Cycle initialization detected between ResourceManager and SessionManager")
                return None
            
            self._initializing_session_manager = True
            try:
                from modulos.session_manager.session_manager import SessionManager as SM_Class
                self._session_manager_instance = SM_Class()
                is_minimal = getattr(self, 'log_verbosity', 'normal') == "minimal"
                if not is_minimal and hasattr(self, 'logger'):
                    self.logger.debug("SessionManager retrieved for ResourceManager.")
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Error accessing SessionManager: {e}", exc_info=True)
            finally:
                self._initializing_session_manager = False
        return self._session_manager_instance

    def _load_configuration(self):
        """
        Loads ResourceManager configuration from config.yaml,
        applying also default values where necessary.
        """
        # Use the already initialized log_verbosity
        is_minimal = self.log_verbosity == "minimal"
        
        try:
            if not self.config:
                if not is_minimal:
                    self.logger.info("Config instance not provided, trying to get global...")
                from config import config as global_config
                self.config = global_config
        
            if self.config:
                resource_config = self.config.get_resource_management_config()
                
                # Verify that configuration is not empty
                if not resource_config:
                    raise ValueError("ResourceManager configuration is empty. Verify config.yaml.")
                    
                if not is_minimal:
                    self.logger.info(f"Configuration loaded successfully: {len(resource_config)} parameters.")
                    
                # Configure verbosity level (updating existing value)
                self.log_verbosity = resource_config.get("log_verbosity", self.log_verbosity)
                    
                # Load monitoring parameters (with nested structure)
                monitoring_config = resource_config.get("monitoring", {})
                self.monitoring_interval_sec = monitoring_config.get("interval_sec", 120)
                self.suspended_monitoring_interval_sec = monitoring_config.get("suspended_interval_sec", 300)
                self.monitoring_enabled = self.monitoring_interval_sec > 0
                
                # Monitoring thresholds
                self.aggressive_cleanup_threshold_mem_pct = monitoring_config.get("aggressive_threshold_mem_pct", 90)
                self.warning_cleanup_threshold_mem_pct = monitoring_config.get("warning_threshold_mem_pct", 80)
                self.warning_threshold_cpu_pct = monitoring_config.get("warning_threshold_cpu_pct", 90)
                
                # Load memory_manager parameters (with nested structure)
                memory_config = resource_config.get("memory", {})
                self.auto_suspend_memory_mb = memory_config.get("auto_suspend_memory_mb", 1000)
                self.max_suspend_duration_sec = memory_config.get("max_suspend_duration_sec", 1800)
                self.document_size_threshold_kb = memory_config.get("document_size_threshold_kb", 5000)
                self.min_chunks_for_suspend = memory_config.get("min_chunks_for_suspend", 200)
                
                # Parameters for concurrency_manager
                concurrency_config = resource_config.get("concurrency", {})
                self.default_cpu_workers = concurrency_config.get("cpu_workers", "auto")
                self.default_io_workers = concurrency_config.get("io_workers", "auto")
                self.max_total_workers = concurrency_config.get("max_total_workers", None)
                self.disable_process_pool = concurrency_config.get("disable_process_pool", False)
                self.default_timeout_sec = concurrency_config.get("default_timeout_sec", 120)
                
                # New parameters for worker recalculation (with nested structure)
                worker_config = resource_config.get("worker_management", {})
                self.worker_recalculation_frequency = worker_config.get("worker_recalculation_frequency", 300.0)
                self.dynamic_recalc_interval = worker_config.get("dynamic_recalc_interval", True)
                self.max_recalc_interval = worker_config.get("max_recalc_interval", 1800.0)
                self.worker_cooldown_after_change = worker_config.get("worker_cooldown_after_change", 120.0)
                self.worker_change_threshold_pct = worker_config.get("worker_change_threshold_pct", 15)
                
                # Parameters for the environment (with nested structure)
                env_config = resource_config.get("environment", {})
                # If an explicit environment type is defined, use it
                explicit_env = env_config.get("environment_type", "")
                if explicit_env:
                    self.environment_type = explicit_env
                # If not, we use the auto-detected in __init__
                
                # Verify if we should avoid automatic adjustments by environment
                self.override_environment_adjustments = env_config.get("override_environment_adjustments", False)
                
                # Adjust parameters according to detected environment (unless disabled)
                if not self.override_environment_adjustments:
                    self._adjust_parameters_for_environment()
                
                # Detailed log if adequate verbosity
                if self.log_verbosity == "detailed":
                    self.logger.info(f"ResourceManager Configuration - Monitoring: {self.monitoring_interval_sec}s "
                                     f"(suspended: {self.suspended_monitoring_interval_sec}s), "
                                     f"Thresholds: aggressive={self.aggressive_cleanup_threshold_mem_pct}%, "
                                     f"warning={self.warning_cleanup_threshold_mem_pct}%, "
                                     f"cpu={self.warning_threshold_cpu_pct}%")
                    self.logger.info(f"Suspension Configuration - Memory={self.auto_suspend_memory_mb}MB, "
                                     f"duration={self.max_suspend_duration_sec}s, "
                                     f"doc_size={self.document_size_threshold_kb}KB, "
                                     f"min_chunks={self.min_chunks_for_suspend}")
                    self.logger.info(f"Concurrency Configuration - CPU workers={self.default_cpu_workers}, "
                                     f"I/O workers={self.default_io_workers}, "
                                     f"disable_process_pool={self.disable_process_pool}, "
                                     f"recalc_frequency={self.worker_recalculation_frequency}s")
                    self.logger.info(f"Worker Recalculation Configuration - Frequency={self.worker_recalculation_frequency}s, "
                                     f"Dynamic={self.dynamic_recalc_interval}, "
                                     f"Max interval={self.max_recalc_interval}s, "
                                     f"Change threshold={self.worker_change_threshold_pct}%")
            else:
                raise ValueError("Could not get config. Verify that config.py is available.")
        
        except Exception as e:
            self.logger.error(f"Error loading ResourceManager configuration: {e}", exc_info=True)
            # Safe default values in case of error
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
        Adjusts configuration parameters based on detected environment.
        
        Different environments (development, production, cloud, etc.) can benefit
        from optimized configurations for their specific characteristics.
        """
        # If environment hasn't been detected, do it now
        if not hasattr(self, 'environment_type'):
            self.environment_type = self._detect_environment()
        
        # Variables to decide if adjustments should be applied
        apply_adjustments = True
        
        # Don't apply adjustments if there's explicit configuration that prevents it
        if self.config:
            try:
                resource_config = self.config.get_resource_management_config()
                if resource_config and "override_environment_adjustments" in resource_config:
                    apply_adjustments = not resource_config["override_environment_adjustments"]
            except Exception:
                pass
        
        if not apply_adjustments:
            return
            
        # Apply adjustments according to environment type
        if self.environment_type == "development":
            # Development: more frequent, less aggressive, focus on quick feedback
            self.monitoring_interval_sec = min(self.monitoring_interval_sec, 90)  # Maximum 90 seconds
            self.worker_recalculation_frequency = min(self.worker_recalculation_frequency, 180.0)  # Maximum 3 minutes
            self.warning_cleanup_threshold_mem_pct = 85  # More tolerant with memory
            
        elif self.environment_type in ("production", "server"):
            # Production: less frequent, more aggressive with cleanup
            self.worker_recalculation_frequency = max(self.worker_recalculation_frequency, 300.0)  # Minimum 5 minutes
            self.warning_cleanup_threshold_mem_pct = min(self.warning_cleanup_threshold_mem_pct, 75)  # More aggressive with memory
            self.dynamic_recalc_interval = True  # Always enable dynamic adjustment
            
        elif self.environment_type in ("aws", "gcp", "azure", "cloud"):
            # Cloud environments: optimize for reducing costs and improving elasticity
            self.worker_recalculation_frequency = max(self.worker_recalculation_frequency, 360.0)  # Minimum 6 minutes
            self.warning_cleanup_threshold_mem_pct = 70  # Very aggressive with memory
            self.aggressive_cleanup_threshold_mem_pct = 85  # Very aggressive
            
        elif self.environment_type in ("container", "kubernetes"):
            # Container: limited resources, be very efficient
            self.warning_cleanup_threshold_mem_pct = 65  # Extremely aggressive
            self.aggressive_cleanup_threshold_mem_pct = 80
            self.worker_recalculation_frequency = 240.0  # 4 minutes
            
        elif "windows" in self.environment_type:
            # Windows usually has different memory management
            self.warning_cleanup_threshold_mem_pct = min(self.warning_cleanup_threshold_mem_pct, 75)
            
        # Log of applied adjustments according to verbosity
        if self.log_verbosity == "detailed":
            self.logger.info(f"Parameters adjusted for environment {self.environment_type}")

    def _start_monitoring_thread(self):
        """Starts the resource monitoring thread if enabled and not active."""
        # Only start the thread if enabled and doesn't exist
        if not self.monitoring_enabled:
            return
            
        # Verify if a monitoring thread already exists and is active
        if hasattr(self, '_monitor_thread') and self._monitor_thread and self._monitor_thread.is_alive():
            # Safe access to log_verbosity
            is_detailed = getattr(self, 'log_verbosity', 'normal') == "detailed"
            if is_detailed:
                self.logger.debug("Resource monitoring thread already active.")
            return
            
        # Start monitoring thread
        self._stop_monitoring_event.clear() # Ensure stop event is clean
        self._monitor_thread = ResourceMonitorThread(
            self, 
            self._stop_monitoring_event,
            verification_suspended=self.verification_suspended,
            verification_resume_time=self.verification_resume_time
        )
        self._monitor_thread.start()
        self.metrics["monitoring_thread_active"] = True
        
        # Safe access to log_verbosity
        is_minimal = getattr(self, 'log_verbosity', 'normal') == "minimal"
        if not is_minimal:
            status_msg = "suspended" if self.verification_suspended else "active"
            self.logger.info(f"Resource monitoring thread started (status: {status_msg}).")

    # --- Rest of methods without changes except for verbosity optimizations ---
    
    def get_system_static_info(self) -> Dict[str, Any]:
        """
        Gets static system and hardware information.

        This information can be useful for adaptive configuration or diagnosis.
        Uses `platform` and `psutil` to collect data like OS type,
        version, architecture, Python version, number of CPU cores and total RAM.

        Returns:
            Dict[str, Any]: A dictionary with system static information.
                            In case of error, returns a dictionary with an "error" key.
        """
        self.logger.debug("Collecting system static information.")
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
                # "gpu_info": self._get_gpu_static_info() # Example if implemented
            }
            # Update total RAM in metrics only once if relevant
            if self.metrics["system_memory_total_gb"] == 0.0:
                    self.metrics["system_memory_total_gb"] = info["total_ram_gb"]
            return info
        except Exception as e:
            self.logger.error(f"Error getting system static information: {e}", exc_info=True)
            return {"error": str(e)}

    def update_metrics(self) -> None:
        """
        Updates dynamic system and process resource metrics.

        Collects information about memory usage (total, available, used, percentage),
        CPU usage (system and process), and specific metrics of RAG components
        like the number of active sessions and loaded embedding models.
        Results are stored in the `self.metrics` attribute.
        
        If checks are suspended, only updates critical metrics
        to reduce system overhead.
        
        This function is called periodically by the monitoring thread.
        """
        # Safe access to log_verbosity
        is_minimal = getattr(self, 'log_verbosity', 'normal') == "minimal"
        is_detailed = getattr(self, 'log_verbosity', 'normal') == "detailed"
        
        # Verify if checks are suspended
        verification_suspended = self.is_verification_suspended()
        
        # In suspended mode, reduce frequency (allow an update every 60s)
        if verification_suspended:
            last_update_time = self.metrics.get("last_metrics_update_ts", 0)
            time_since_update = time.time() - last_update_time
            
            # If not enough time has passed since last update in suspended mode
            if time_since_update < self.suspended_monitoring_interval_sec:
                if is_detailed:
                    self.logger.debug(f"Suspended updates: only {time_since_update:.1f}s since last update")
                return
            elif is_detailed:
                self.logger.debug("Critical metrics update in suspended mode")
        elif is_detailed:
            self.logger.debug("Updating all resource metrics...")
            
        try:
            # System memory metrics (always update, they're critical)
            sys_mem = psutil.virtual_memory()
            self.metrics["system_memory_total_gb"] = round(sys_mem.total / (1024**3), 2)
            self.metrics["system_memory_available_gb"] = round(sys_mem.available / (1024**3), 2)
            self.metrics["system_memory_used_gb"] = round(sys_mem.used / (1024**3), 2)
            self.metrics["system_memory_percent"] = sys_mem.percent

            # System CPU metrics (always update, they're critical)
            self.metrics["cpu_percent_system"] = psutil.cpu_percent(interval=None)
            
            # If checks suspended, skip non-critical metrics
            if not verification_suspended:
                # Current process metrics
                process = psutil.Process(os.getpid())
                proc_mem_info = process.memory_info()
                self.metrics["process_memory_rss_mb"] = round(proc_mem_info.rss / (1024**2), 2)
                self.metrics["process_memory_vms_mb"] = round(proc_mem_info.vms / (1024**2), 2)
                self.metrics["process_memory_percent"] = round(process.memory_percent(), 2)
                self.metrics["cpu_percent_process"] = round(process.cpu_percent(interval=None), 2)

                # Update active session metrics (using lazy initialization)
                if self.session_manager_instance:
                    try:
                        self.metrics["active_sessions_rag"] = self.session_manager_instance.get_active_sessions_count()
                    except Exception as e:
                        if is_detailed:
                            self.logger.error(f"Error getting active_sessions_count: {e}")
                
                # Update active embedding model count
                try:
                    from modulos.embeddings.embeddings_factory import EmbeddingFactory
                    if hasattr(EmbeddingFactory, 'get_active_model_count'):
                        self.metrics["active_embedding_models"] = EmbeddingFactory.get_active_model_count()
                    else:
                        # Alternative if method doesn't exist
                        self.metrics["active_embedding_models"] = len(getattr(EmbeddingFactory, '_instances', {}))
                        self.logger.warning("get_active_model_count method not available in EmbeddingFactory")
                except ImportError:
                    self.logger.warning("Could not import EmbeddingFactory to get embedding model count")
                    self.metrics["active_embedding_models"] = 0
                except Exception as e:
                    self.logger.error(f"Error getting active model count: {e}")
                    self.metrics["active_embedding_models"] = 0
                    
                # Verify worker pools status periodically
                # Do it every 5 metric updates (to not overload)
                last_pools_check = self.metrics.get("last_pools_check_ts", 0)
                time_since_pools_check = time.time() - last_pools_check
                
                # Verify if enough time has passed since last check
                if time_since_pools_check > 300:  # 5 minutes
                    if is_detailed:
                        self.logger.debug("Verifying worker pools status...")
                    
                    # Call check_worker_pools_status to verify and optimize pools
                    pools_status = self.check_worker_pools_status()
                    self.metrics["last_pools_check_ts"] = time.time()
                    
                    # If any action was performed, log it
                    if pools_status.get("recalculation_performed", False):
                        if is_detailed:
                            self.logger.debug("Worker recalculation performed during pools verification")
                    
                    # Verify if actions were performed in pools
                    for pool_type in ["thread_pool", "process_pool"]:
                        if "action" in pools_status.get(pool_type, {}):
                            action = pools_status[pool_type]["action"]
                            reason = pools_status[pool_type].get("reason", "")
                            if is_detailed:
                                self.logger.debug(f"Pool {pool_type}: {action} - {reason}")

            self.metrics["last_metrics_update_ts"] = time.time()
            
            # Reduce verbosity of metrics log drastically
            if is_detailed:
                mem_pct = self.metrics["system_memory_percent"]
                cpu_pct = self.metrics["cpu_percent_system"]
                sess = self.metrics.get("active_sessions_rag", "N/A")
                models = self.metrics.get("active_embedding_models", "N/A")
                if verification_suspended:
                    self.logger.debug(f"Critical metrics updated in suspended mode: Memory={mem_pct:.1f}%, CPU={cpu_pct:.1f}%")
                else:
                    self.logger.debug(f"Metrics: Memory={mem_pct:.1f}%, CPU={cpu_pct:.1f}%, Sessions={sess}, Models={models}")

        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}", exc_info=True)

    def request_cleanup(self, aggressive: bool = False, reason: str = "manual", respect_cooldown: bool = True) -> bool:
        """
        Requests a system resource cleanup.
        
        Optimized processing version that avoids releasing embedding models
        during continuous document processing, focusing on efficient cleanup
        tasks with low overhead.

        Args:
            aggressive (bool): If True, performs a deeper cleanup, even 
                              releasing resources that could be costly to reinitialize.
            reason (str): Reason for requesting cleanup, useful for diagnosis and decision-making.
            respect_cooldown (bool): If True, respects minimum periods between cleans.
                                   False forces cleanup regardless of elapsed time.

        Returns:
            bool: True if cleanup was performed, False if omitted (e.g. by cooldown).
        """
        if not self.memory_manager:
            self.logger.warning("Cannot request cleanup: MemoryManager not initialized")
            return False

        # Verify cooldown periods if requested
        if respect_cooldown:
            current_time = time.time()
            time_since_last_cleanup = current_time - self.metrics.get("last_cleanup_time", 0)
            
            # Define minimum intervals according to level of aggressiveness
            min_interval = self.memory_manager.min_aggressive_gc_interval if aggressive else self.memory_manager.min_gc_interval
            
            if time_since_last_cleanup < min_interval:
                self.logger.debug(f"Omitting cleanup ({reason}): in cooling period ({time_since_last_cleanup:.1f}s < {min_interval:.1f}s)")
                return False
        
        # Determine if we're in document processing
        in_document_processing = self.metrics.get("operation_in_progress") == "document_processing"
        
        # Specific logic for document ingestion processes
        if in_document_processing:
            self.logger.debug(f"Cleanup during document processing ({reason})")
            # During processing, avoid releasing embedding models
            # but keep memory cleanup
            result = self.memory_manager.cleanup(
                aggressive=aggressive, 
                reason=reason, 
                skip_model_cleanup=True  # No releasing models during continuous ingestion
            )
        else:
            # Normal cleanup outside document processing
            result = self.memory_manager.cleanup(
                aggressive=aggressive,
                reason=reason
            )
            
        # Update cleanup metrics
        if result:
            self.metrics["last_cleanup_time"] = time.time()
            self.metrics["cleanup_count"] = self.metrics.get("cleanup_count", 0) + 1
            
            if aggressive:
                self.metrics["aggressive_cleanup_count"] = self.metrics.get("aggressive_cleanup_count", 0) + 1
                
            # Update system metrics after cleanup
            self.update_metrics()
        
        return result

    def shutdown(self) -> None:
        """
        Performs a controlled shutdown of the ResourceManager and its components.

        This includes stopping the monitoring thread and shutting down executor pools
        in ConcurrencyManager.
        """
        is_minimal = hasattr(self, 'log_verbosity') and self.log_verbosity == "minimal"
        
        if not is_minimal:
            self.logger.info("Starting ResourceManager shutdown...")

        # Stop monitoring thread
        if hasattr(self, '_monitor_thread') and self._monitor_thread and self._monitor_thread.is_alive():
            if not is_minimal:
                self.logger.info("Stopping resource monitoring thread...")
            self._stop_monitoring_event.set()
            try:
                self._monitor_thread.join(timeout=self.monitoring_interval_sec + 5)
                if self._monitor_thread.is_alive():
                    self.logger.warning("Resource monitoring thread did not finish in time.")
                elif not is_minimal:
                    self.logger.info("Resource monitoring thread stopped.")
            except Exception as e:
                self.logger.error(f"Error stopping monitoring thread: {e}", exc_info=True)
            self.metrics["monitoring_thread_active"] = False

        # Shutdown ConcurrencyManager
        if self.concurrency_manager:
            if not is_minimal:
                self.logger.info("Requesting shutdown to ConcurrencyManager...")
            try:
                self.concurrency_manager.shutdown_executors(wait=True)
                if not is_minimal:
                    self.logger.info("ConcurrencyManager shutdown completed.")
            except Exception as e:
                self.logger.error(f"Error during ConcurrencyManager.shutdown_executors: {e}", exc_info=True)

        # Shutdown MemoryManager
        if self.memory_manager:
            if not is_minimal:
                self.logger.info("Requesting shutdown to MemoryManager...")
            try:
                self.memory_manager.shutdown()
                if not is_minimal:
                    self.logger.info("MemoryManager shutdown completed.")
            except Exception as e:
                self.logger.error(f"Error during MemoryManager.shutdown: {e}", exc_info=True)

        if not is_minimal:
            self.logger.info("ResourceManager shutdown completed.")

    def suspend_verifications(self, duration_seconds: int = 300, reason: str = "manual") -> bool:
        """
        Temporarily suspends checks and resource cleanup.
        
        During the suspension period, resource monitoring is reduced 
        and only critical metrics are updated. No automatic cleanups are performed.
        This function is useful for small tasks where constant verification
        overhead is not necessary.
        
        Args:
            duration_seconds (int): Duration in seconds of the suspension.
                                   After this time, checks are automatically resumed
            reason (str): Reason for suspension. Defaults to "manual".
            
        Returns:
            bool: True if suspension was activated correctly, False in case of error.
        """
        try:
            with self._lock:
                # If already suspended, update reason and resume time
                if self.verification_suspended:
                    # Extend suspension time from now
                    self.verification_resume_time = time.time() + duration_seconds
                    self.verification_suspend_reason = reason
                    
                    # Safe access to log_verbosity
                    is_minimal = getattr(self, 'log_verbosity', 'normal') == "minimal"
                    if not is_minimal:
                        self.logger.info(f"Verification suspension extended by {duration_seconds}s. Reason: {reason}")
                    
                    # Update state in metrics
                    self.metrics["verification_status"] = "suspended"
                    return True
                
                # Activate suspension
                self.verification_suspended = True
                self.verification_resume_time = time.time() + duration_seconds
                self.verification_suspend_reason = reason
                
                # Update state in metrics
                self.metrics["verification_status"] = "suspended"
                
                # Safe access to log_verbosity
                is_minimal = getattr(self, 'log_verbosity', 'normal') == "minimal"
                if not is_minimal:
                    self.logger.info(f"Verification checks suspended for {duration_seconds}s. Reason: {reason}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error suspending checks: {e}", exc_info=True)
            return False
            
    def resume_verifications(self, reason: str = "manual") -> bool:
        """
        Resumes checks and resource cleanup if they were suspended.
        
        Args:
            reason (str): Reason for resuming. Defaults to "manual".
            
        Returns:
            bool: True if resuming was successful or if checks were not
                 suspended, False in case of error.
        """
        try:
            with self._lock:
                # If not suspended, do nothing
                if not self.verification_suspended:
                    return True
                
                # Resume checks
                self.verification_suspended = False
                self.verification_resume_time = None
                suspend_reason = self.verification_suspend_reason
                self.verification_suspend_reason = None
                
                # Update state in metrics
                self.metrics["verification_status"] = "active"
                
                # Safe access to log_verbosity
                is_minimal = getattr(self, 'log_verbosity', 'normal') == "minimal"
                if not is_minimal:
                    self.logger.info(f"Verification checks resumed. Reason: {reason}, were suspended by: {suspend_reason}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error resuming checks: {e}", exc_info=True)
            return False

    def is_verification_suspended(self) -> bool:
        """
        Checks if checks are currently suspended.
        
        Also checks if suspension time has expired and updates
        state if necessary.
        
        Returns:
            bool: True if checks are suspended, False otherwise
        """
        # If attribute doesn't exist, it's not suspended
        if not hasattr(self, '_verification_suspended') or not self._verification_suspended:
            return False
            
        # Check if suspension time has expired
        now = time.time()
        if now >= self._verification_suspended_until:
            self._verification_suspended = False
            self.logger.debug("Verification suspension finalization by time expiration")
            return False
            
        # Still suspended
        return True

    def should_suspend_verifications(self, document_size_kb: Optional[int] = None, chunk_count: Optional[int] = None) -> bool:
        """
        Determines if checks should be suspended based on established criteria.
        
        Checks are suspended if:
        1. The document is very small (<150KB) AND has few chunks (less than 20)
        2. Or if the system has low load (CPU <20% and memory <60%)
        
        Improved logic to avoid false positives with large documents
        and to be more relaxed with checks.
        
        Args:
            document_size_kb: Document size in KB
            chunk_count: Number of chunks being processed
            
        Returns:
            bool: True if checks should be suspended, False otherwise
        """
        # Safe access to log_verbosity
        is_minimal = getattr(self, 'log_verbosity', 'normal') == "minimal"
        is_detailed = getattr(self, 'log_verbosity', 'normal') == "detailed"
        
        # If already suspended, keep suspended
        if self.is_verification_suspended():
            return True
            
        # Rule 1: Very small document (<150KB) AND few chunks (<20)
        if document_size_kb is not None and document_size_kb < 150:
            if chunk_count is None or chunk_count < 20:
                if not is_minimal:
                    self.logger.info(f"Small document detected ({document_size_kb:.1f}KB). Considering verification suspension.")
                return True
                
        # Rule 2: Low system load
        current_cpu = self.metrics.get("cpu_percent", 0)
        current_mem = self.metrics.get("memory_percent", 0)
        
        if current_cpu < 20 and current_mem < 60:
            if is_detailed:
                self.logger.debug(f"Low system load (CPU: {current_cpu:.1f}%, Memory: {current_mem:.1f}%). Suspending checks.")
            return True
            
        # If none of the above applies, don't suspend checks
        return False

    def auto_suspend_if_needed(self, document_size_kb: float = 0, chunk_count: int = 0, 
                           duration_seconds: int = 300) -> bool:
        """
        Automatically decides whether to suspend checks based on
        document characteristics and system state.
        
        This function analyzes several factors:
        1. Document size in KB
        2. Number of chunks (if already known)
        3. Current state of system resources
        4. Current workload
        
        Args:
            document_size_kb (float): Document size in kilobytes
            chunk_count (int): Number of chunks (if already generated)
            duration_seconds (int): Duration of suspension if activated
            
        Returns:
            bool: True if checks were suspended, False otherwise
        """
        # Verify if suspension is configured as activated
        verification_suspension_enabled = getattr(self, 'verification_suspension_enabled', True)
        if not verification_suspension_enabled:
            self.logger.debug("Verification suspension disabled in configuration")
            return False
            
        # Extract current system metrics
        mem_percent = self.metrics.get("system_memory_percent", 0)
        cpu_percent = self.metrics.get("cpu_percent_system", 0)
        
        # Criteria for suspending:
        suspend = False
        reason = ""
        
        # 1. Extremely small documents - don't suspend (very fast processing)
        if document_size_kb < 10:  # Less than 10KB
            self.logger.debug(f"Very small document ({document_size_kb:.1f} KB). Checks not suspended.")
            return False
            
        # 2. Very large documents - suspend for performance optimization
        if document_size_kb > 1000:  # More than 1MB
            suspend = True
            reason = f"large document ({document_size_kb:.1f} KB)"
            
        # 3. If we know chunks and they're many - suspend
        elif chunk_count > 50:
            suspend = True
            reason = f"high chunk count ({chunk_count})"
            
        # 4. System resources under pressure - suspend
        elif mem_percent > 80 or cpu_percent > 80:
            suspend = True
            reason = f"system pressure (Memory: {mem_percent:.1f}%, CPU: {cpu_percent:.1f}%)"
            
        # 5. Medium documents (100KB-1MB) - evaluate case by case
        elif document_size_kb > 100:
            # Estimate chunks based on size (approx. 1 chunk per every 2-4KB)
            estimated_chunks = document_size_kb / 3
            if estimated_chunks > 30:  # If we estimate more than 30 chunks
                suspend = True
                reason = f"medium size document with estimated chunks: ~{int(estimated_chunks)}"
                
        # Apply suspension if determined necessary
        if suspend:
            return self.suspend_verification(duration_seconds, f"Auto-suspension: {reason}")
        else:
            self.logger.debug(f"No suspension required for document of {document_size_kb:.1f} KB")
            return False
    
    def suspend_verification(self, duration_seconds: int = 300, reason: str = "manual") -> bool:
        """
        Temporarily suspends certain checks that can affect performance.
        
        During suspension, some intensive operations like pickleable checks,
        certain validations and checks can be disabled
        to improve performance during intensive processing.
        
        Args:
            duration_seconds (int): Duration in seconds of the suspension
            reason (str): Reason for suspension (for logging)
            
        Returns:
            bool: True if suspension was applied, False if already suspended
        """
        if hasattr(self, '_verification_suspended') and self._verification_suspended:
            # Already suspended, update time if necessary
            now = time.time()
            time_remaining = self._verification_suspended_until - now
            
            if duration_seconds > time_remaining:
                # Extend suspension if new duration is longer
                self._verification_suspended_until = now + duration_seconds
                self.logger.info(f"Suspension extended by {duration_seconds}s. Reason: {reason}")
                return True
            else:
                # Keep existing suspension
                self.logger.debug(f"Suspension already active for {time_remaining:.1f}s more. Not extended.")
                return False
        
        # Activate suspension
        self._verification_suspended = True
        self._verification_suspended_until = time.time() + duration_seconds
        self._verification_suspended_reason = reason
        
        self.logger.info(f"Verification checks suspended for {duration_seconds}s. Reason: {reason}")
        
        # Schedule automatic reactivation
        try:
            import threading
            
            def restore_verification():
                try:
                    now = time.time()
                    if hasattr(self, '_verification_suspended_until') and now >= self._verification_suspended_until:
                        self._verification_suspended = False
                        self.logger.info("Verification suspension finalization by automatic expiration")
                except Exception as e:
                    self.logger.error(f"Error restoring checks: {e}")
            
            # Create and execute timer
            timer = threading.Timer(duration_seconds, restore_verification)
            timer.daemon = True
            timer.start()
        except ImportError:
            self.logger.warning("Could not schedule automatic verification restoration (threading not available)")
        
        return True
    
    def resume_verification(self) -> bool:
        """
        Reactivates suspended checks before scheduled time.
        
        Returns:
            bool: True if there were suspended checks that were reactivated,
                 False if no active suspension
        """
        if hasattr(self, '_verification_suspended') and self._verification_suspended:
            self._verification_suspended = False
            self.logger.info("Verification checks manually reactivated")
            return True
        return False

    def check_worker_pools_status(self) -> Dict[str, Any]:
        """
        Checks current state of worker pools and performs optimizations
        as necessary. This function can be called periodically to
        ensure pools are in an optimal state.
        
        Actions performed:
        1. Checks if there are hibernated pools that should be restored
        2. Checks if there are active pools that should be hibernated
        3. Recalculates optimal worker count if necessary
        
        Returns:
            Dict[str, Any]: Dictionary with information about pools state
        """
        if not self.concurrency_manager:
            self.logger.warning("ConcurrencyManager not available to verify pools state")
            return {"error": "ConcurrencyManager not available"}
        
        result = {
            "thread_pool": {},
            "process_pool": {},
            "recalculation_performed": False,
            "timestamp": time.time()
        }
        
        try:
            # 1. Check if it's time to recalculate workers
            recalculation_performed = self.concurrency_manager.recalculate_workers_if_needed()
            result["recalculation_performed"] = recalculation_performed
            
            # 2. Get current actual pools information
            pool_status = self.concurrency_manager.pools_status
            worker_counts = self.concurrency_manager.get_worker_counts()
            
            # 3. Check if there are hibernated pools that should be restored
            for pool_type in ["thread_pool", "process_pool"]:
                pool_info = pool_status.get(pool_type, {})
                pool_state = pool_info.get("status", "inactive")
                
                result[pool_type]["status"] = pool_state
                result[pool_type]["workers"] = pool_info.get("workers", 0)
                result[pool_type]["last_used"] = pool_info.get("last_used", 0)
                
                # If pool is hibernated and has passed enough time since last hibernation
                if pool_state == "hibernated":
                    hibernated_at = pool_info.get("hibernated_at", 0)
                    time_since_hibernation = time.time() - hibernated_at
                    
                    # If has been hibernated more than 10 minutes, consider restoring it
                    # to have it ready for the next operation
                    if time_since_hibernation > 600:  # 10 minutes
                        restored = self.concurrency_manager.restore_pool_from_hibernation(pool_type)
                        result[pool_type]["action"] = "restored" if restored else "restore_failed"
                        result[pool_type]["reason"] = f"Hibernated for {time_since_hibernation:.1f}s"
                
                # If pool is active, check if it should be hibernated
                elif pool_state == "active":
                    last_used = pool_info.get("last_used", 0)
                    time_since_last_use = time.time() - last_used
                    
                    # If not used in 30 minutes, consider hibernating it
                    if time_since_last_use > 1800:  # 30 minutes
                        hibernated = self.concurrency_manager.hibernate_pool_if_unused(pool_type, idle_seconds=1800)
                        result[pool_type]["action"] = "hibernated" if hibernated else "hibernate_failed"
                        result[pool_type]["reason"] = f"Inactive for {time_since_last_use:.1f}s"
            
            # 4. Update metrics
            self.metrics["worker_pools_status"] = {
                "thread_pool": pool_status.get("thread_pool", {}).get("status", "unknown"),
                "process_pool": pool_status.get("process_pool", {}).get("status", "unknown"),
                "cpu_workers": worker_counts.get("cpu_workers", 0),
                "io_workers": worker_counts.get("io_workers", 0),
                "last_check": time.time()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error verifying pools state: {e}", exc_info=True)
            return {"error": str(e)}

class ResourceMonitorThread(threading.Thread):
    """
    Daemon thread to periodically monitor system resources.

    This thread calls the `update_metrics` method of the `ResourceManager` and,
    based on configured thresholds, can request cleanup operations
    through `request_cleanup`. It also periodically checks if it's necessary
    to recalculate the number of workers in ConcurrencyManager.

    Attributes:
        resource_manager (ResourceManager): The ResourceManager instance to monitor.
        stop_event (threading.Event): Event to signal thread stop.
        interval_sec (int): Interval in seconds between each monitoring cycle.
        verification_suspended (bool): Indicates if checks are suspended.
        verification_resume_time (float): Time checks should be resumed.
    """
    def __init__(self, resource_manager_instance: 'ResourceManager', stop_event: threading.Event, 
                verification_suspended: bool = False, verification_resume_time: Optional[float] = None):
        super().__init__(name="ResourceMonitorThread", daemon=True)
        self.resource_manager = resource_manager_instance
        self.stop_event = stop_event
        self.interval_sec = self.resource_manager.monitoring_interval_sec
        self.logger = logging.getLogger(self.__class__.__name__) # Logger for the thread
        if not self.logger.hasHandlers(): # Fallback
            logging.basicConfig(level=logging.INFO)
        self.verification_suspended = verification_suspended
        self.verification_resume_time = verification_resume_time

    def run(self):
        """
        Main logic of the monitoring thread.

        In a loop, updates metrics and verifies thresholds for requesting cleanup
        until the `stop_event` is activated. Respects check suspension.
        """
        # Safe access to log_verbosity with getattr for providing a default value
        is_minimal = getattr(self.resource_manager, 'log_verbosity', 'normal') == "minimal"
        is_detailed = getattr(self.resource_manager, 'log_verbosity', 'normal') == "detailed"
        
        if is_detailed:
            status = "suspended" if self.verification_suspended else "active"
            self.logger.debug(f"ResourceMonitorThread started. Interval: {self.interval_sec}s. State: {status}")
        
        # Initialize timestamps for frequency control of logs and cleanups
        last_mem_aggressive_time = 0
        last_mem_warning_time = 0
        last_cpu_warning_time = 0
        last_cleanup_time = 0
        
        # Cooldowns to prevent too frequent cleanups (in seconds)
        warning_cooldown_sec = 60  # Control frequency of log messages (no changes)
        cleanup_cooldown_sec = 300  # Increased from 180 to 300 seconds
        
        try:
            while not self.stop_event.is_set():
                # Verify if suspension time has expired (check in ResourceManager)
                verification_suspended = self.resource_manager.is_verification_suspended()
                
                # Synchronize local state with ResourceManager
                if verification_suspended != self.verification_suspended:
                    self.verification_suspended = verification_suspended
                    if is_detailed:
                        status = "suspended" if verification_suspended else "active"
                        self.logger.debug(f"Verification state updated: {status}")
                
                # If it's time to verify thresholds (not suspended)
                # Update metrics
                self.resource_manager.update_metrics()
                
                # Get current metrics
                current_mem_pct = self.resource_manager.metrics.get("system_memory_percent", 0)
                current_cpu_pct = self.resource_manager.metrics.get("cpu_percent_system", 0)
                current_time = time.time()
                
                # Verify if enough time has passed since last cleanup
                time_since_last_cleanup = current_time - last_cleanup_time
                if time_since_last_cleanup < cleanup_cooldown_sec:
                    skip_reason = f"Waiting cooldown ({time_since_last_cleanup:.0f}/{cleanup_cooldown_sec}s)"
                    if is_detailed:
                        self.logger.debug(f"Skipping threshold check: {skip_reason}")
                    continue

                # If checks are suspended, don't do cleanup or worker recalculation
                if not verification_suspended:
                    # Threshold logic and cleanup request
                    aggressive_thresh_mem = self.resource_manager.aggressive_cleanup_threshold_mem_pct
                    warning_thresh_mem = self.resource_manager.warning_cleanup_threshold_mem_pct
                    warning_thresh_cpu = self.resource_manager.warning_threshold_cpu_pct

                    # Verify memory thresholds with log frequency control
                    if current_mem_pct >= aggressive_thresh_mem:
                        should_log = is_detailed or (current_time - last_mem_aggressive_time > warning_cooldown_sec)
                        if should_log:
                            self.logger.warning(f"Memory usage ({current_mem_pct:.1f}%) exceeded aggressive threshold ({aggressive_thresh_mem}%). Requesting aggressive cleanup.")
                            last_mem_aggressive_time = current_time
                        self.resource_manager.request_cleanup(aggressive=True, reason="memory_aggressive_threshold")
                        last_cleanup_time = current_time
                    elif current_mem_pct >= warning_thresh_mem:
                        should_log = is_detailed or (current_time - last_mem_warning_time > warning_cooldown_sec)
                        if should_log:
                            self.logger.warning(f"Memory usage ({current_mem_pct:.1f}%) exceeded warning threshold ({warning_thresh_mem}%). Requesting cleanup.")
                            last_mem_warning_time = current_time
                        self.resource_manager.request_cleanup(aggressive=False, reason="memory_warning_threshold")
                        last_cleanup_time = current_time
                    
                    # Verify CPU thresholds
                    if current_cpu_pct >= warning_thresh_cpu:
                        should_log = is_detailed or (current_time - last_cpu_warning_time > warning_cooldown_sec)
                        if should_log:
                            self.logger.warning(f"CPU usage ({current_cpu_pct:.1f}%) exceeded warning threshold ({warning_thresh_cpu}%).")
                            last_cpu_warning_time = current_time
                    
                    # Verify if it's necessary to recalculate workers in ConcurrencyManager
                    if self.resource_manager.concurrency_manager:
                        try:
                            was_recalculated = self.resource_manager.concurrency_manager.recalculate_workers_if_needed()
                            if was_recalculated and is_detailed:
                                workers_info = self.resource_manager.concurrency_manager.get_worker_counts()
                                self.logger.info(f"Workers recalculated: CPU={workers_info['cpu_workers']}, IO={workers_info['io_workers']}")
                        except Exception as e:
                            if not is_minimal:
                                self.logger.error(f"Error recalculating workers: {e}")
                else:
                    if is_detailed:
                        self.logger.debug("Checks suspended: no threshold checks or cleanup performed.")

                # Determine interval for next cycle
                # If we're in suspended mode, use extended interval
                wait_interval = self.resource_manager.suspended_monitoring_interval_sec if verification_suspended else self.interval_sec
                
                # Wait until next cycle or until stop signal
                self.stop_event.wait(timeout=wait_interval)
        except Exception as e:
            self.logger.error(f"Unexpected error in ResourceMonitorThread: {e}", exc_info=True)
        finally:
            if is_detailed:
                self.logger.debug("ResourceMonitorThread terminating.")

# For quick tests if this file is executed directly (optional)
if __name__ == '__main__':
    print("Executing ResourceManager directly for basic test...")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    rm = ResourceManager() # Will use global config if available
    print(f"ResourceManager instantiated: {rm}")
    print(f"Initial metrics: {rm.metrics}")
    static_info = rm.get_system_static_info()
    print(f"System static information: {static_info}")

    try:
        if rm.metrics.get("monitoring_thread_active"):
            print("Monitoring thread active. Waiting for some updates...")
            time.sleep(12)
            print(f"Metrics after some time: {rm.metrics}")
        else:
            print("Monitoring not active, updating metrics manually for test.")
            rm.update_metrics()
            print(f"Manually updated metrics: {rm.metrics}")
    except KeyboardInterrupt:
        print("Keyboard interruption.")
    finally:
        print("Stopping ResourceManager...")
        rm.shutdown()
        print("ResourceManager stopped.") 