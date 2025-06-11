import gc
import logging
import sys
import weakref
import inspect
import time
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .resource_manager import ResourceManager
    # from modulos.embeddings.embeddings_factory import EmbeddingFactory # For future type hinting

class MemoryManager:
    """
    Manages specific memory optimization and cleanup operations.

    It is instantiated and used by ResourceManager to perform tasks such as:
    - Execute garbage collection.
    - Release resources from unused embedding models.
    - Clean Python caches and check memory fragmentation.
    - Dynamically optimize batch sizes based on current system memory usage.

    Important note: Embedding models are NOT released during document processing,
    as they are used continuously. They will only be released when all documents
    have finished processing.

    Attributes:
        resource_manager (ResourceManager): Instance of the main ResourceManager.
        logger (logging.Logger): Logger for this class.
        cached_functions (List): List of references to functions with lru_cache decorator.
        last_gc_time (float): Timestamp of the last complete garbage collection.
        last_gc_threshold_change (float): Timestamp of the last GC threshold change.
        gc_stats (Dict): Statistics about garbage collection operations.
    """
    def __init__(self, resource_manager_instance: 'ResourceManager'):
        """
        Initializes the MemoryManager.

        Args:
            resource_manager_instance (ResourceManager): The ResourceManager instance
                that will manage this MemoryManager.
        """
        self.resource_manager = resource_manager_instance
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers(): # Fallback if there's no logging config
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # List for tracking functions with cache
        self.cached_functions = []
        
        # Variables for optimized GC management
        self.last_gc_time = 0.0
        self.last_gc_threshold_change = 0.0
        self.gc_stats = {
            "total_collections": 0,
            "objects_collected": 0,
            "aggressive_collections": 0,
            "standard_collections": 0,
            "last_collection_objects": 0
        }
        
        # Get minimum intervals from configuration, if available
        # with default values if configuration cannot be accessed
        if self.resource_manager and hasattr(self.resource_manager, 'config'):
            try:
                resource_config = self.resource_manager.config.get_resource_management_config()
                monitoring_config = resource_config.get('monitoring', {})
                
                # Get configured intervals or use default values
                self.min_gc_interval = float(monitoring_config.get('min_cleanup_interval_sec', 30.0))
                self.min_aggressive_gc_interval = float(monitoring_config.get('min_aggressive_cleanup_interval_sec', 300.0))
                self.min_memory_change_pct = float(monitoring_config.get('min_memory_change_pct', 5.0))
                self.extended_cooling_period = float(monitoring_config.get('extended_cooling_period_sec', 900.0))
                
                self.logger.debug(f"Configured cleanup intervals: normal={self.min_gc_interval}s, " 
                                 f"aggressive={self.min_aggressive_gc_interval}s, " 
                                 f"minimum change={self.min_memory_change_pct}%, "
                                 f"extended cooling={self.extended_cooling_period}s")
            except Exception as e:
                self.logger.warning(f"Error loading configuration intervals: {e}. Using default values.")
                # Default values if there's an error in configuration
                self.min_gc_interval = 30.0
                self.min_aggressive_gc_interval = 300.0
                self.min_memory_change_pct = 5.0
                self.extended_cooling_period = 900.0
        else:
            # Minimum intervals between GC operations (in seconds) - default values
            self.min_gc_interval = 30.0  # Minimum 30 seconds between normal GCs
            self.min_aggressive_gc_interval = 300.0  # Minimum 5 minutes between aggressive GCs
            self.min_memory_change_pct = 5.0  # Minimum 5% change to consider effective cleanup
            self.extended_cooling_period = 900.0  # 15 minutes extended cooling after ineffective cleanup
        
        # Register functions that use lru_cache
        self._register_cache_functions()
        
        self.logger.info("MemoryManager initialized with optimized GC management.")
        # Load specific config if added in the future
        # self._load_config()

    def cleanup(self, aggressive: bool = False, reason: str = "manual", skip_model_cleanup: bool = False) -> bool:
        """
        Performs memory cleanup and resource release.

        Optimized version with improvements to reduce overhead during document processing
        and avoid unnecessary model release.
        
        Args:
            aggressive (bool): If True, performs deeper cleanup.
            reason (str): Reason for cleanup (for logging and tracking).
            skip_model_cleanup (bool): If True, avoids releasing embedding models.
                                     Useful during continuous document processing.
            
        Returns:
            bool: True if effective cleanup was performed, False if skipped.
        """
        # Check time since last cleanup
        current_time = time.time()
        time_since_last_gc = current_time - self.last_gc_time
        
        # Check if we can skip this cleanup due to being in cooling period
        # (Except for reasons that require immediate cleanup)
        force_immediate_cleanup = reason in ["oom_prevention", "critical_memory", "processing_completed"]
        
        if not force_immediate_cleanup:
            # Criteria for skipping cleanup based on elapsed time
            if aggressive and time_since_last_gc < self.min_aggressive_gc_interval:
                self.logger.debug(f"Skipping aggressive cleanup: {time_since_last_gc:.1f}s < {self.min_aggressive_gc_interval:.1f}s")
                return False
            elif not aggressive and time_since_last_gc < self.min_gc_interval:
                self.logger.debug(f"Skipping normal cleanup: {time_since_last_gc:.1f}s < {self.min_gc_interval:.1f}s")
                return False
        
        # Update timestamp of last cleanup
        self.last_gc_time = current_time
        
        self.logger.info(f"Executing cleanup (aggressive={aggressive}, reason='{reason}'{', without_releasing_models' if skip_model_cleanup else ''})")
        
        # 1. Execute Python garbage collection
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
            self.logger.error(f"Error during garbage collection: {e}")
            collected = 0
        
        # 2. Clean Python caches
        try:
            self._clear_python_caches()
        except Exception as e:
            self.logger.error(f"Error cleaning Python caches: {e}")
            
        # 3. Release specific resources based on reason and aggressiveness
        resources_released = 0
        
        # 4. Release model resources only if:
        # - We are not in skip_model_cleanup mode
        # - The cleanup is aggressive or the reason is compatible with releasing models
        if not skip_model_cleanup and (aggressive or reason in ["processing_completed", "oom_prevention", "idle_timeout"]):
            try:
                models_released = self._release_model_resources(aggressive=aggressive)
                resources_released += models_released
                self.logger.debug(f"Models released: {models_released}")
            except Exception as e:
                self.logger.error(f"Error releasing model resources: {e}")
        elif skip_model_cleanup:
            self.logger.debug("Skipping model release by explicit request (skip_model_cleanup)")
        
        # More discrete logging with level system (Reduce overhead)
        if aggressive:
            self.logger.info(f"Aggressive cleanup completed. Objects collected: {collected}, resources released: {resources_released}")
        else:
            self.logger.debug(f"Standard cleanup completed. Objects collected: {collected}")
            
        # Return True if cleanup released objects or resources
        return collected > 0 or resources_released > 0

    def _run_garbage_collection(self, aggressive: bool) -> int:
        """
        Executes Python's garbage collector with optimizations to
        minimize performance impact and improve efficiency.

        This improved implementation focuses mainly on generation 0
        for better performance, and only executes full collections when
        absolutely necessary. Heavy anti-fragmentation techniques are eliminated.

        Args:
            aggressive (bool): If True, attempts deeper collection (generation 2).
                               Otherwise, performs more selective collection.

        Returns:
            int: The number of objects collected by `gc.collect()`, or 0 if not determinable.
        """
        self.logger.debug(f"Executing optimized garbage collection (aggressive={aggressive}).")
        collected_count = 0
        
        try:
            # Get initial statistics
            initial_count = len(gc.get_objects())
            current_time = time.time()

            # Check if GC is enabled
            was_enabled = gc.isenabled()
            if not was_enabled:
                gc.enable()
                self.logger.debug("GC was disabled, temporarily activated.")
            
            if aggressive:
                # Strategy for aggressive GC - simplified without anti-fragmentation
                # Only adjust thresholds when really necessary
                old_thresholds = gc.get_threshold()
                threshold_adjusted = False
                
                # Only modify thresholds if enough time has passed
                if (current_time - self.last_gc_threshold_change) > 180.0:  # 3 minutes
                    # Lower values = more frequent GC
                    new_thresholds = (700, 10, 10)  
                    gc.set_threshold(*new_thresholds)
                    self.logger.debug(f"GC thresholds temporarily adjusted from {old_thresholds} to {new_thresholds}")
                    self.last_gc_threshold_change = current_time
                    threshold_adjusted = True
                    
                    # Threshold restoration scheduled here...
                    try:
                        import threading
                        def restore_gc_threshold():
                            try:
                                gc.set_threshold(*old_thresholds)
                                self.logger.debug(f"GC thresholds restored to {old_thresholds}")
                                self.last_gc_threshold_change = time.time()
                            except Exception as e:
                                self.logger.error(f"Error restoring GC thresholds: {e}")
                        
                        # Restore after 60 seconds
                        timer = threading.Timer(60.0, restore_gc_threshold)
                        timer.daemon = True
                        timer.start()
                    except ImportError:
                        # Do nothing if threading is not available
                        pass
                
                # First focus on gen 0 (more efficient)
                gen0_count = gc.collect(0)
                self.logger.debug(f"GC gen 0 collected {gen0_count} objects")
                collected_count = gen0_count
                
                # Only execute gen 1 if gen 0 found several objects
                if gen0_count > 50:  # Reduced threshold for better performance
                    gen1_count = gc.collect(1)
                    self.logger.debug(f"GC gen 1 collected {gen1_count} objects")
                    collected_count += gen1_count
                    
                    # Only execute gen 2 (very expensive) if there's really severe pressure
                    # and at least 5 minutes have passed since the last full GC
                    time_since_last_full_gc = current_time - getattr(self, "last_full_gc_time", 0)
                    mem_percent = self.resource_manager.metrics.get("system_memory_percent", 0)
                    
                    if (gen1_count > 50 and time_since_last_full_gc > 300) or mem_percent > 90:
                        gen2_count = gc.collect(2)
                        self.logger.debug(f"GC gen 2 collected {gen2_count} objects")
                        collected_count += gen2_count
                        
                        # Register timestamp of last full GC
                        self.last_full_gc_time = current_time
                
            else:
                # Standard strategy - more conservative and focused on gen 0
                mem_percent = self.resource_manager.metrics.get("system_memory_percent", 0)
                
                if mem_percent <= 60:  # Low pressure
                    # Only generation 0 (faster)
                    collected_count = gc.collect(0)
                    self.logger.debug(f"Selective GC gen 0 collected {collected_count} objects")
                elif mem_percent <= 75:  # Moderate pressure
                    # Generations 0 and 1
                    gen0_count = gc.collect(0)
                    gen1_count = gc.collect(1)
                    collected_count = gen0_count + gen1_count
                    self.logger.debug(f"Selective GC gen 0+1 collected {collected_count} objects")
                else:  # High pressure
                    # First try gen 0+1, and only if they collect many objects, do gen 2
                    gen0_count = gc.collect(0)
                    gen1_count = gc.collect(1)
                    collected_count = gen0_count + gen1_count
                    
                    # Only do gen 2 if there are many objects in gen 0+1
                    if collected_count > 100:
                        gen2_count = gc.collect(2)
                        collected_count += gen2_count
                        self.logger.debug(f"Full GC collected total {collected_count} objects")
                    else:
                        self.logger.debug(f"Selective GC gen 0+1 sufficient, collected {collected_count} objects")
            
            # Restore GC state if it was disabled
            if not was_enabled:
                gc.disable()
                self.logger.debug("GC restored to its previous state (disabled).")
                
            # Calculate efficiency and report
            final_count = len(gc.get_objects())
            objects_diff = initial_count - final_count
            
            # Correct negative difference (objects created during GC)
            if objects_diff < 0:
                objects_diff = 0
                
            self.logger.info(f"GC completed - Collected: {collected_count}, "
                           f"Object difference: {objects_diff}")
            
            return collected_count
            
        except Exception as e:
            self.logger.error(f"Error during garbage collection: {e}")
            return 0  # In case of error, report 0 objects collected

    def _register_cache_functions(self):
        """
        Registers functions decorated with lru_cache for later cleanup.
        This allows tracking and cleaning specific caches when necessary.
        """
        try:
            # Get imported modules
            for module_name, module in list(sys.modules.items()):
                # Process only project modules (modulos.*)
                if module and module_name.startswith('modulos.'):
                    for name, obj in inspect.getmembers(module):
                        # Check if it's a function with cache
                        if inspect.isfunction(obj) and hasattr(obj, 'cache_info') and hasattr(obj, 'cache_clear'):
                            self.cached_functions.append(weakref.ref(obj))
                            self.logger.debug(f"Registered function with cache: {module_name}.{name}")
        except Exception as e:
            self.logger.error(f"Error registering functions with cache: {e}", exc_info=True)

    def _clear_python_caches(self, aggressive: bool = False) -> str:
        """
        Cleans Python's internal caches to free memory.

        Implements "Phase 1: Memory Cleanup Optimization" by cleaning:
        - Module import caches
        - LRU caches of functions decorated with @functools.lru_cache
        - Other internal caches when possible

        Args:
            aggressive (bool): If True, performs more intensive cleanup, including
                               caches that could affect performance. Default False.

        Returns:
            str: Message describing the action performed.
        """
        self.logger.info(f"Starting Python cache cleanup (aggressive={aggressive}).")
        cache_info = {
            "module_cache_size_before": len(sys.modules),
            "lru_caches_cleared": 0,
            "module_cache_cleared": False,
            "path_cache_cleared": False,
            "memory_returned": 0
        }

        try:
            # 1. Cleanup of LRU caches from decorated functions
            valid_cached_functions = []
            for func_ref in self.cached_functions:
                func = func_ref()
                if func is not None:
                    try:
                        # For aggressive cleanup, clear all caches
                        # For normal cleanup, clear only if they have many entries (>1000)
                        cache_info_obj = func.cache_info()
                        if aggressive or (hasattr(cache_info_obj, 'currsize') and cache_info_obj.currsize > 1000):
                            func.cache_clear()
                            cache_info["lru_caches_cleared"] += 1
                            self.logger.debug(f"Cleared function cache: {func.__module__}.{func.__name__}")
                    except Exception as e:
                        self.logger.warning(f"Error clearing function cache: {e}")
                    valid_cached_functions.append(func_ref)
            
            # Update the list with only valid references
            self.cached_functions = valid_cached_functions
            
            # 2. Module import cache cleanup (only in aggressive mode)
            if aggressive:
                non_essential_modules = []
                for module_name in list(sys.modules.keys()):
                    # Preserve essential system modules and our project modules
                    if (not module_name.startswith('_') and 
                        not module_name.startswith('sys') and 
                        not module_name.startswith('os') and
                        not module_name.startswith('modulos.') and  # Keep our own modules
                        not module_name in ('logging', 'gc', 'threading', 'time', 'functools')):
                        non_essential_modules.append(module_name)
                
                # Remove a subset of non-essential modules
                # Don't remove all to avoid serious problems, only the least used ones
                modules_to_remove = non_essential_modules[:max(len(non_essential_modules)//4, 1)]
                for module_name in modules_to_remove:
                    try:
                        del sys.modules[module_name]
                    except KeyError:
                        pass
                
                cache_info["module_cache_cleared"] = True
                cache_info["module_cache_size_after"] = len(sys.modules)
                self.logger.info(f"Removed {len(modules_to_remove)} modules from sys.modules")
            
            # 3. Import path cache cleanup (only in aggressive mode)
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
                    self.logger.info("Importlib cache invalidated.")
            
            # 4. Run garbage collection to free memory from deleted caches
            # but only if _run_garbage_collection won't be called afterwards
            if not aggressive:  # If aggressive, the cleanup method will already call _run_garbage_collection
                gc.collect()

            msg = (f"Cache cleanup completed - "
                   f"LRU caches: {cache_info['lru_caches_cleared']}, "
                   f"Modules: {'Yes' if cache_info['module_cache_cleared'] else 'No'}, "
                   f"Path cache: {'Yes' if cache_info['path_cache_cleared'] else 'No'}")
            
            return msg
            
        except Exception as e:
            error_msg = f"Error during cache cleanup: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return error_msg

    def _release_model_resources(self, aggressive: bool) -> int:
        """
        Releases embedding model resources and performs advanced memory cleanup.
        
        Optimized version that only focuses on releasing embedding models, returning
        the number of models released for better control and monitoring.
        
        Args:
            aggressive (bool): If True, uses a more aggressive release strategy.
            
        Returns:
            int: The number of embedding models released.
        """
        # Initialize counters and markers if they don't exist
        if not hasattr(self, '_consecutive_failed_cleanups'):
            self._consecutive_failed_cleanups = 0
            self._last_cleanup_memory_before = 0
            self._last_cleanup_memory_after = 0
            self._cleanup_history = []
            
        # Determine if previous cleanups have been effective
        ineffective_cleanups = self._consecutive_failed_cleanups
        current_memory = 0
        
        if self.resource_manager and hasattr(self.resource_manager, 'metrics'):
            current_memory = self.resource_manager.metrics.get("system_memory_percent", 0)
            
            # If we have data from the last cleanup, evaluate effectiveness
            if self._last_cleanup_memory_before > 0:
                if current_memory >= (self._last_cleanup_memory_before - self.min_memory_change_pct):
                    # The previous cleanup was not effective
                    self._consecutive_failed_cleanups += 1
                    self.logger.debug(f"Cleanup {self._consecutive_failed_cleanups} consecutive ineffective. "
                                     f"Memory before: {self._last_cleanup_memory_before:.1f}%, "
                                     f"Now: {current_memory:.1f}%")
                else:
                    # The previous cleanup was effective
                    self._consecutive_failed_cleanups = 0
                    self.logger.debug(f"Previous cleanup effective. "
                                     f"Memory before: {self._last_cleanup_memory_before:.1f}%, "
                                     f"Now: {current_memory:.1f}%")
        
        # Save current memory value for next evaluation
        self._last_cleanup_memory_before = current_memory
        
        # Variables for action tracking
        models_released = 0
        
        try:
            # Release embedding models through EmbeddingFactory
            self.logger.debug(f"Attempting to release inactive models (aggressive={aggressive}, failed_cleanups={ineffective_cleanups}).")
            
            try:
                from modulos.embeddings.embeddings_factory import EmbeddingFactory
                
                # Adjust aggressiveness based on consecutive failures
                adjusted_aggressive = aggressive or (ineffective_cleanups >= 2)
                
                # When there are many consecutive failures, release even active models
                force_active = False
                if ineffective_cleanups >= 3 or (aggressive and ineffective_cleanups >= 1):
                    force_active = True
                    self.logger.warning(f"Activating forced release of active models due to {ineffective_cleanups} ineffective cleanups")
                
                models_released = EmbeddingFactory.release_inactive_models(
                    aggressive=adjusted_aggressive,
                    force_release_active=force_active
                )
                
                if models_released > 0:
                    self.logger.info(f"Released {models_released} embedding models")
                
            except ImportError:
                self.logger.error("Could not import EmbeddingFactory to release models.")
                return 0
            except Exception as e:
                self.logger.error(f"Error releasing embedding models: {e}")
                return 0
            
            # PyTorch specific memory cleanup (if available)
            if aggressive or ineffective_cleanups >= 1 or models_released > 0:
                try:
                    import torch
                    
                    # Check CUDA
                    if hasattr(torch, 'cuda') and torch.cuda.is_available():
                        # Clear CUDA caches
                        torch.cuda.empty_cache()
                        self.logger.debug("CUDA cache cleared")
                    
                    # Check MPS (Apple Silicon)
                    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                            self.logger.debug("MPS cache cleared")
                    
                except ImportError:
                    self.logger.debug("PyTorch not available")
                except Exception as e:
                    self.logger.debug(f"Error during PyTorch cleanup: {e}")
        
        except Exception as e:
            self.logger.error(f"General error during model release: {e}")
            return 0
        
        return models_released

    def check_memory_usage(self) -> Dict[str, Any]:
        """
        Checks current memory usage and performs cleanups if necessary.
        
        This function:
        1. Checks current memory and CPU status
        2. Compares with configured thresholds 
        3. Executes proactive cleanup if certain thresholds are exceeded
        4. Implements cooldown period to avoid consecutive ineffective cleanups
        5. Takes cleanup history into account to adjust strategy
        
        Returns:
            Dict[str, Any]: Report of memory status and actions taken
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
            # Verify that we have access to ResourceManager and its metrics
            if not self.resource_manager or not hasattr(self.resource_manager, 'metrics'):
                self.logger.warning("ResourceManager not available or does not have configured metrics")
                result["error"] = "ResourceManager not available"
                return result
            
            # Get current metrics
            memory_percent = self.resource_manager.metrics.get("system_memory_percent", 0)
            result["memory_percent"] = memory_percent
            
            # Get configured thresholds (or use default values)
            aggressive_threshold = getattr(self.resource_manager, 'aggressive_cleanup_threshold_mem_pct', 85)
            warning_threshold = getattr(self.resource_manager, 'warning_cleanup_threshold_mem_pct', 75)
            
            # Forced release threshold (from configuration, or calculate)
            force_release_threshold = 90  # Default value
            
            # Try to get from configuration
            if hasattr(self.resource_manager, 'config') and self.resource_manager.config:
                try:
                    resource_config = self.resource_manager.config.get_resource_management_config()
                    memory_config = resource_config.get('memory', {})
                    model_release_config = memory_config.get('model_release', {})
                    force_release_threshold = model_release_config.get('force_release_memory_threshold_pct', 90)
                except:
                    # If there's an error, use the default value
                    pass
            
            # Time elapsed since last cleanup
            current_time = time.time()
            time_since_last_gc = current_time - self.last_gc_time
            
            # Evaluate effectiveness of previous cleanup
            had_recent_ineffective_cleanup = False
            previous_memory_after_cleanup = self.resource_manager.metrics.get("previous_memory_after_cleanup", 0)
            
            # If there was a recent cleanup and memory is still high,
            # we consider the previous cleanup was ineffective
            if time_since_last_gc < 300 and previous_memory_after_cleanup > 0:
                memory_change = previous_memory_after_cleanup - memory_percent
                if abs(memory_change) < self.min_memory_change_pct:
                    had_recent_ineffective_cleanup = True
                    self.logger.warning(f"Recent ineffective cleanup detected. Memory change: {memory_change:.1f}% < {self.min_memory_change_pct}%")
                    result["actions_taken"].append(f"Recent ineffective cleanup detection (change: {memory_change:.1f}%)")
            
            # Adjusted cooling periods
            cooling_period_aggressive = self.extended_cooling_period if had_recent_ineffective_cleanup else self.min_aggressive_gc_interval
            cooling_period_warning = self.extended_cooling_period if had_recent_ineffective_cleanup else self.min_gc_interval * 1.5
            
            # Determine if we need to force release of active models
            force_release_active = False
            
            if memory_percent >= force_release_threshold:
                self.logger.warning(f"Critical memory usage ({memory_percent:.1f}% >= {force_release_threshold}%). Activating forced model release.")
                force_release_active = True
                result["actions_taken"].append("Forced model release activated due to critical memory usage")
            elif hasattr(self, '_consecutive_failed_cleanups') and self._consecutive_failed_cleanups >= 3:
                self.logger.warning(f"Multiple ineffective cleanups ({self._consecutive_failed_cleanups}). Activating forced model release.")
                force_release_active = True
                result["actions_taken"].append("Forced model release activated due to consecutive ineffective cleanups")
            
            # Evaluate current state against thresholds
            if memory_percent >= aggressive_threshold:
                # Critical situation - Immediate aggressive cleanup
                self.logger.warning(f"Critical memory usage: {memory_percent:.1f}% >= {aggressive_threshold}% (aggressive threshold)")
                result["threshold_exceeded"] = True
                result["threshold_type"] = "aggressive"
                
                # Dynamic cooling period based on previous effectiveness
                if time_since_last_gc >= cooling_period_aggressive:
                    cleanup_result = self.cleanup(aggressive=True, reason="memory_critical")
                    result["cleanup_performed"] = True
                    result["cleanup_result"] = cleanup_result
                    result["actions_taken"].append("Aggressive cleanup executed")
                    
                    # Save information to evaluate effectiveness in next check
                    self.resource_manager.metrics["previous_memory_after_cleanup"] = memory_percent
                else:
                    self.logger.info(f"Skipping aggressive cleanup - Last cleanup only {time_since_last_gc:.1f}s ago (cooling period: {cooling_period_aggressive:.1f}s)")
                    result["actions_taken"].append(f"Cleanup skipped (cooling: {time_since_last_gc:.1f}s < {cooling_period_aggressive:.1f}s)")
                    
            elif memory_percent >= warning_threshold:
                # Warning situation - Standard cleanup
                self.logger.info(f"High memory usage: {memory_percent:.1f}% >= {warning_threshold}% (warning threshold)")
                result["threshold_exceeded"] = True
                result["threshold_type"] = "warning"
                
                # Release models if we're approaching the critical threshold
                if memory_percent >= (aggressive_threshold - 5) and time_since_last_gc >= self.min_gc_interval:
                    self.logger.info(f"Memory near critical threshold ({memory_percent:.1f}% vs {aggressive_threshold}%), releasing models proactively")
                    cleanup_result = self.cleanup(aggressive=False, reason="memory_near_critical")
                    result["cleanup_performed"] = True
                    result["cleanup_result"] = cleanup_result
                    result["actions_taken"].append("Proactive cleanup executed (near critical threshold)")
                    self.resource_manager.metrics["previous_memory_after_cleanup"] = memory_percent
                # Extended cooling period if ineffectiveness detected
                elif time_since_last_gc >= cooling_period_warning:
                    # Only execute cleanup if it's X% higher than after previous cleanup
                    if not had_recent_ineffective_cleanup or (memory_percent > previous_memory_after_cleanup + self.min_memory_change_pct):
                        cleanup_result = self.cleanup(aggressive=False, reason="memory_high")
                        result["cleanup_performed"] = True
                        result["cleanup_result"] = cleanup_result
                        result["actions_taken"].append("Standard cleanup executed")
                        
                        # Save information to evaluate effectiveness in next check
                        self.resource_manager.metrics["previous_memory_after_cleanup"] = memory_percent
                    else:
                        self.logger.info(f"Skipping cleanup - Insufficient memory change ({memory_percent:.1f}% vs {previous_memory_after_cleanup:.1f}% previous, minimum required: {self.min_memory_change_pct}%)")
                        result["actions_taken"].append(f"Cleanup skipped (insufficient change: {memory_percent-previous_memory_after_cleanup:.1f}% < {self.min_memory_change_pct}%)")
                else:
                    self.logger.debug(f"Skipping standard cleanup - In cooling period ({time_since_last_gc:.1f}s < {cooling_period_warning:.1f}s)")
                    result["actions_taken"].append(f"Cleanup skipped (cooling: {time_since_last_gc:.1f}s < {cooling_period_warning:.1f}s)")
                
            else:
                # Normal situation - No action required
                self.logger.debug(f"Normal memory usage: {memory_percent:.1f}%")
                result["actions_taken"].append("No cleanup required")
                
                # Reset effectiveness metrics if memory has dropped significantly
                if previous_memory_after_cleanup > 0 and memory_percent < previous_memory_after_cleanup - (self.min_memory_change_pct * 2):
                    self.resource_manager.metrics["previous_memory_after_cleanup"] = 0
                    self.logger.info(f"Memory normalized ({memory_percent:.1f}%), effectiveness metrics reset")
            
            # If forced release was activated but no cleanup was executed, force now
            if force_release_active and not result.get("cleanup_performed", False):
                # Force release directly from EmbeddingFactory for critical cases
                try:
                    from modulos.embeddings.embeddings_factory import EmbeddingFactory
                    models_released = EmbeddingFactory.release_inactive_models(
                        aggressive=True, 
                        force_release_active=True
                    )
                    if models_released > 0:
                        self.logger.warning(f"Emergency forced release: {models_released} models released")
                        result["actions_taken"].append(f"Emergency forced release: {models_released} models")
                        result["emergency_release"] = True
                        result["models_released"] = models_released
                        # Force GC to maximize effect
                        gc.collect()
                except Exception as e:
                    self.logger.error(f"Error in emergency forced release: {e}")
            
        except Exception as e:
            self.logger.error(f"Error checking memory usage: {e}", exc_info=True)
            result["error"] = str(e)
            
        return result

    def optimize_batch_size(self, base_batch_size: int, min_batch_size: int = 1, 
                             max_batch_size: Optional[int] = None, verification_suspended: bool = False) -> int:
        """
        Dynamically adjusts batch size for memory-intensive operations,
        based on multiple factors:
        
        1. Current system memory and CPU usage
        2. Historical results from previous operations
        3. Recent garbage collection efficiency
        4. System resource availability
        
        Improved optimization that provides fast response with low overhead.

        Args:
            base_batch_size (int): The base or preferred batch size.
            min_batch_size (int): The minimum allowed batch size. Defaults to 1.
            max_batch_size (Optional[int]): The maximum allowed batch size. 
                                           Defaults to None (no explicit upper limit here).
            verification_suspended (bool): If verifications are currently suspended.
                                          Affects the optimization strategy.

        Returns:
            int: The optimized batch size. If metrics cannot be obtained,
                 returns `base_batch_size`.
        """
        # Reduce logging verbosity to minimize overhead
        self.logger.debug(f"Optimizing batch_size. Base: {base_batch_size}")
        
        # If verifications are suspended, use an approach adapted to that scenario
        if verification_suspended:
            # During verification suspension we can be more aggressive with batch size
            # since it's assumed we're in a performance over safety mode
            suspension_factor = 1.5  # 50% increase during suspension
            
            optimized_size = int(base_batch_size * suspension_factor)
            if max_batch_size is not None:
                optimized_size = min(optimized_size, max_batch_size)
                
            # Reduce logging
            return optimized_size
        
        # Variables for intelligent adjustment
        optimized_batch_size = base_batch_size  # Default value
        scaling_factor = 1.0  # Default neutral factor
        
        try:
            # Check ResourceManager and its metrics availability
            if not self.resource_manager or not hasattr(self.resource_manager, 'metrics'):
                return base_batch_size

            # 1. Memory metrics-based factor - Simplified to reduce overhead
            mem_usage_pct = self.resource_manager.metrics.get("system_memory_percent", 0)
            mem_available_gb = self.resource_manager.metrics.get("system_memory_available_gb", 0)
            
            # 2. Simplified memory factor calculation based on current pressure
            # Use decision table instead of complex calculations
            if mem_usage_pct >= 85:  # Critical memory
                memory_factor = 0.25  # Drastically reduce
            elif mem_usage_pct >= 75: 
                memory_factor = 0.50  # Moderate reduction
            elif mem_usage_pct >= 65:
                memory_factor = 0.75  # Light reduction
            elif mem_available_gb >= 4.0:  # Plenty of available memory
                memory_factor = 1.2   # Slightly increase
            else:
                memory_factor = 1.0   # Neutral factor
            
            # 3. CPU factor (simplified)
            cpu_factor = 1.0  # Default neutral value
            cpu_percent = self.resource_manager.metrics.get('cpu_percent_system', 0)
            if cpu_percent >= 90:
                cpu_factor = 0.5   # Reduce by half if CPU very high
            elif cpu_percent >= 70:
                cpu_factor = 0.7   # Moderate reduction
            
            # 4. Combine factors with simple weighting
            scaling_factor = (memory_factor * 0.7) + (cpu_factor * 0.3)
            
            # Limit combined factor to reasonable range
            scaling_factor = max(0.2, min(1.5, scaling_factor))
            
            # Apply factor to base batch_size with integer rounding
            optimized_batch_size = max(min_batch_size, round(base_batch_size * scaling_factor))
            
            # Apply maximum limit if configured
            if max_batch_size is not None:
                optimized_batch_size = min(optimized_batch_size, max_batch_size)

            # Simple historical tracking system (maximum 5 entries to avoid overhead)
            if not hasattr(self, '_batch_size_history'):
                self._batch_size_history = []
                
            # Save recent history efficiently
            self._batch_size_history.append({
                'time': time.time(), 
                'base': base_batch_size, 
                'optimized': optimized_batch_size,
                'factor': scaling_factor
            })
            
            # Keep limited history
            if len(self._batch_size_history) > 5:
                self._batch_size_history = self._batch_size_history[-5:]
            
        except Exception as e:
            self.logger.error(f"Error optimizing batch_size: {e}. Using conservative value.")
            # Conservative value in case of error (75% of base)
            scaling_factor = 0.75
            optimized_batch_size = max(min_batch_size, int(base_batch_size * scaling_factor))
            if max_batch_size is not None:
                optimized_batch_size = min(optimized_batch_size, max_batch_size)
        
        # Reduce logging verbosity
        self.logger.debug(f"Batch size: {base_batch_size} → {optimized_batch_size} (factor: {scaling_factor:.2f}x)")
        return optimized_batch_size

    def shutdown(self) -> None:
        """
        Performs any necessary cleanup or shutdown for MemoryManager when shutting down the system.
        Currently, it has no specific actions beyond logging the event.
        """
        self.logger.info("MemoryManager shutdown requested.")
        # For now, there are no specific shutdown actions for MemoryManager.
        self.logger.info("MemoryManager shutdown completed.")