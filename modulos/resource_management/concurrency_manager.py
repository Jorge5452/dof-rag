import logging
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Optional, Callable, Iterable, TYPE_CHECKING, Any, Dict, Tuple, List, Literal, TypeVar

# Third-party imports
import psutil

# Type hints for avoiding circular imports
if TYPE_CHECKING:
    from .resource_manager import ResourceManager

# Pool state definitions
PoolState = Literal["active", "hibernated", "shutdown"]

# Generic types for results
T = TypeVar('T')
R = TypeVar('R')

# Global cache for serialization checks
_PICKLEABILITY_CACHE: Dict[str, bool] = {}
_PICKLEABILITY_CACHE_MAX_SIZE = 500
_PICKLEABILITY_CACHE_HITS = 0
_PICKLEABILITY_CACHE_MISSES = 0
_LAST_CACHE_CLEANUP = time.time()

# Native types that are always serializable
_ALWAYS_SERIALIZABLE_TYPES = {
    int, float, bool, str, bytes, 
    tuple, list, dict, set, frozenset,
    type(None)
}

# Patterns for task type detection based on function names
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
    Cleans the serialization verification cache when it exceeds maximum size.
    
    Prevents unlimited cache growth by keeping only the most recent entries.
    """
    global _PICKLEABILITY_CACHE, _LAST_CACHE_CLEANUP
    
    # Only clean if necessary and enough time has passed
    now = time.time()
    if len(_PICKLEABILITY_CACHE) > _PICKLEABILITY_CACHE_MAX_SIZE and now - _LAST_CACHE_CLEANUP > 300:
        # Create new cache with only the most recent half of entries
        cache_items = list(_PICKLEABILITY_CACHE.items())
        half_size = _PICKLEABILITY_CACHE_MAX_SIZE // 2
        _PICKLEABILITY_CACHE = dict(cache_items[-half_size:])
        _LAST_CACHE_CLEANUP = now

class WorkerPerformanceTracker:
    """
    Tracks historical performance of different worker configurations
    for predictive analysis and intelligent decision making.

    Attributes:
        max_history_entries (int): Maximum historical entries to maintain per type
        history (Dict): Performance history for different configurations
        last_recalculation_time (float): Timestamp of last recalculation
        cooling_period_sec (float): Cooling period after recalculations
        stability_threshold (int): Minimum samples before considering changes
        performance_variance_threshold (float): Variance threshold for considering changes
    """
    def __init__(self, max_history_entries: int = 20, cooling_period_sec: float = 300.0):
        self.max_history_entries = max_history_entries
        self.cooling_period_sec = cooling_period_sec
        self.last_recalculation_time = 0.0
        self.stability_threshold = 3  # Minimum samples to consider stable
        self.performance_variance_threshold = 0.15  # 15% variation
        
        # Structure to store history by pool type and configuration
        self.history: Dict[str, List[Dict[str, Any]]] = {
            "thread_pool": [],
            "process_pool": [],
            "task_performance": {},  # Performance by task type
            "global_metrics": []     # Global system metrics
        }
        
        # Current metrics representing the last known execution
        self.current_metrics = {
            "thread_pool": None,
            "process_pool": None,
            "system_pressure": 0.0,  # Combined CPU and memory pressure
            "last_update_time": 0.0
        }
        
        # Counter to establish a warmup period
        self.warmup_count = 0
        self.warmup_threshold = 5  # Minimum warmup executions

    def record_pool_performance(self, pool_type: str, worker_count: int, 
                               tasks_completed: int, execution_time: float,
                               task_type: str, system_metrics: Dict[str, float]) -> None:
        """
        Records worker pool performance for a specific configuration.
        
        Args:
            pool_type (str): Pool type ("thread_pool" or "process_pool")
            worker_count (int): Number of workers used
            tasks_completed (int): Number of completed tasks
            execution_time (float): Total execution time in seconds
            task_type (str): Type of task executed
            system_metrics (Dict[str, float]): System metrics during execution
        """
        if pool_type not in self.history:
            return
            
        # Calculate derived metrics
        tasks_per_second = tasks_completed / max(execution_time, 0.001)
        tasks_per_worker = tasks_completed / max(worker_count, 1)
        efficiency = tasks_per_second / max(worker_count, 1)
        
        # Calculate system pressure (0-1.0)
        cpu_pressure = system_metrics.get("cpu_percent", 0) / 100
        memory_pressure = system_metrics.get("memory_percent", 0) / 100
        system_pressure = (cpu_pressure * 0.6) + (memory_pressure * 0.4)  # Weighted
        
        # Create history entry
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
        
        # Add to pool-specific history
        self.history[pool_type].append(entry)
        
        # Limit history size
        if len(self.history[pool_type]) > self.max_history_entries:
            self.history[pool_type] = self.history[pool_type][-self.max_history_entries:]
            
        # Record in task-specific history
        if task_type not in self.history["task_performance"]:
            self.history["task_performance"][task_type] = []
            
        task_entry = entry.copy()
        self.history["task_performance"][task_type].append(task_entry)
        
        # Limit task-specific history
        if len(self.history["task_performance"][task_type]) > self.max_history_entries:
            self.history["task_performance"][task_type] = self.history["task_performance"][task_type][-self.max_history_entries:]
            
        # Update current metrics
        self.current_metrics[pool_type] = entry
        self.current_metrics["system_pressure"] = system_pressure
        self.current_metrics["last_update_time"] = time.time()
        
        # Increment warmup counter
        self.warmup_count += 1

    def record_system_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Records system metrics for trend analysis.
        
        Args:
            metrics (Dict[str, float]): System metrics (CPU, memory, etc.)
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
        Predicts optimal worker count based on performance history.
        
        Args:
            pool_type (str): Pool type ("thread_pool" or "process_pool")
            task_type (str): Task type to execute
            system_metrics (Dict[str, float]): Current system metrics
            
        Returns:
            Optional[int]: Optimal worker count or None if insufficient data
        """
        # Check if we're in warmup period
        if self.warmup_count < self.warmup_threshold:
            return None
            
        # Verify sufficient historical data
        if pool_type not in self.history or not self.history[pool_type]:
            return None
            
        # Use task-specific history if available
        task_history = self.history["task_performance"].get(task_type, [])
        pool_history = self.history[pool_type]
        
        # Prioritize task-specific history if sufficient data
        if len(task_history) >= self.stability_threshold:
            history_to_use = task_history
        else:
            history_to_use = pool_history
            
        if not history_to_use:
            return None
            
        # Calculate current system pressure
        cpu_pressure = system_metrics.get("cpu_percent", 0) / 100
        memory_pressure = system_metrics.get("memory_percent", 0) / 100
        current_pressure = (cpu_pressure * 0.6) + (memory_pressure * 0.4)
        
        # Search for similar configurations under similar system conditions
        # 1. Filter by system pressure similarity
        pressure_threshold = 0.2  # Deviation tolerance
        similar_entries = [
            entry for entry in history_to_use
            if abs(entry.get("system_pressure", 0) - current_pressure) <= pressure_threshold
        ]
        
        # Use full history if insufficient similar entries
        if len(similar_entries) < 2:
            similar_entries = history_to_use
            
        # 2. Find configuration with best efficiency
        if similar_entries:
            # Sort by efficiency (highest to lowest)
            sorted_entries = sorted(similar_entries, key=lambda x: x.get("efficiency", 0), reverse=True)
            
            # Take best configurations (top 3)
            top_entries = sorted_entries[:3] if len(sorted_entries) >= 3 else sorted_entries
            
            # Calculate average workers from best configurations
            optimal_workers = int(sum(entry.get("worker_count", 1) for entry in top_entries) / len(top_entries))
            
            # Apply adjustment based on current pressure
            if current_pressure > 0.8:  # High pressure, reduce workers
                optimal_workers = max(1, int(optimal_workers * 0.8))
            elif current_pressure < 0.3:  # Low pressure, possible increase
                optimal_workers = int(optimal_workers * 1.2)
                
            return optimal_workers
        
        return None

    def should_recalculate(self, current_metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Determines if worker count should be recalculated based on:
        - Time since last recalculation (cooling period)
        - Significant changes in system metrics
        - Performance history
        
        Args:
            current_metrics (Dict[str, float]): Current system metrics
            
        Returns:
            Tuple[bool, str]: (Should recalculate?, Reason)
        """
        current_time = time.time()
        
        # 1. Check cooling period
        time_since_last_recalc = current_time - self.last_recalculation_time
        if time_since_last_recalc < self.cooling_period_sec:
            return False, f"Cooling period active ({time_since_last_recalc:.1f}s < {self.cooling_period_sec}s)"
            
        # 2. Check significant changes in system metrics
        if self.current_metrics["system_pressure"] > 0:
            # Calculate current pressure
            cpu_pressure = current_metrics.get("cpu_percent", 0) / 100
            memory_pressure = current_metrics.get("memory_percent", 0) / 100
            current_pressure = (cpu_pressure * 0.6) + (memory_pressure * 0.4)
            
            # Calculate pressure change
            pressure_delta = abs(current_pressure - self.current_metrics["system_pressure"])
            
            # Recalculate if significant change
            if pressure_delta > 0.25:  # 25% change in total pressure
                return True, f"Significant system pressure change: {pressure_delta:.2f}"
                
        # 3. Check if in warmup (always recalculate)
        if self.warmup_count < self.warmup_threshold:
            return True, f"Warmup period active ({self.warmup_count}/{self.warmup_threshold})"
            
        # 4. Periodic recalculation (every 10 minutes if nothing else triggers)
        if time_since_last_recalc > 600:  # 10 minutes
            return True, "Periodic recalculation (10 minutes)"
            
        return False, "No recalculation required"
        
    def record_recalculation(self) -> None:
        """Records that a recalculation has been performed."""
        self.last_recalculation_time = time.time()

class ConcurrencyManager:
    """
    Manages concurrency and parallelism in the RAG system.

    Responsibilities:
    - Provide optimized thread and process pools for different task types
    - Dynamically adjust worker count based on system load
    - Execute tasks in parallel/concurrent mode with automatic resource management
    - Hibernate and wake pools as needed to optimize resources

    Attributes:
        resource_manager (ResourceManager): Resource manager instance
        thread_pool (ThreadPoolExecutor): Thread pool for I/O bound operations
        process_pool (ProcessPoolExecutor): Process pool for CPU bound operations
        logger (logging.Logger): Logger for this class
        cpu_workers (int): Number of workers for CPU intensive tasks
        io_workers (int): Number of workers for I/O bound tasks
        default_timeout (float): Maximum wait time for tasks (seconds)
        task_types (Dict[str, Dict]): Configurations for different task types
        disable_process_pool (bool): Whether to disable ProcessPoolExecutor
        performance_tracker (WorkerPerformanceTracker): Historical performance tracker
        pools_status (Dict): Pool status (active, hibernated, shutdown)
        recalculation_frequency_sec (float): Worker recalculation frequency
    """

    def __init__(self, resource_manager_instance: 'ResourceManager'):
        """
        Initializes the concurrency manager.
        
        Args:
            resource_manager_instance (ResourceManager): Resource manager instance
        """
        # Initialize basic attributes
        self.resource_manager = resource_manager_instance
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # For historical performance tracking and intelligent decisions
        self.performance_tracker = WorkerPerformanceTracker()
        
        # Configure recalculation frequency from ResourceManager
        self.recalculation_frequency_sec = getattr(
            resource_manager_instance, 
            'worker_recalculation_frequency', 
            300.0  # 5 minutes default
        )
        self.performance_tracker.cooling_period_sec = self.recalculation_frequency_sec
        
        # Pool status tracking
        self.pools_status = {
            "thread_pool": {
                "status": "inactive",  # inactive, active, hibernated, shutdown
                "created_at": 0.0,
                "last_used": 0.0,
                "task_count": 0,
                "hibernated_at": 0.0,
                "workers": 0,
                "stored_state": None,  # Store state during hibernation
                "last_status_change": 0.0  # Cooldown between status changes
            },
            "process_pool": {
                "status": "inactive",
                "created_at": 0.0,
                "last_used": 0.0,
                "task_count": 0,
                "hibernated_at": 0.0,
                "workers": 0,
                "stored_state": None,  # Store state during hibernation
                "last_status_change": 0.0  # Cooldown between status changes
            }
        }
        
        # Thresholds for pool reinitialization
        self.worker_change_threshold_pct = 20.0  # Minimum worker change percentage to restart
        self.pool_cooldown_period_sec = 60.0  # Minimum seconds between significant status changes
        
        # Recalculation configuration
        self.last_worker_calculation_time = 0.0
        self.worker_calculation_interval_sec = 120.0
        self.worker_recalc_count = 0
        self.dynamic_recalc_interval = True  # Dynamically adjust interval
        
        # Initialize worker counts
        self.cpu_workers = self._calculate_optimal_workers("cpu")
        self.io_workers = self._calculate_optimal_workers("io")
        
        # Initialize additional attributes
        self.thread_pool = None
        self.process_pool = None
        self.disable_process_pool = getattr(resource_manager_instance, 'disable_process_pool', False)
        self.max_total_workers = getattr(resource_manager_instance, 'max_total_workers', None)
        self.default_timeout = getattr(resource_manager_instance, 'default_timeout_sec', 120)
        
        # Task type configurations
        self.task_types = {
            "default": {"prefer_process": False},
            "cpu_intensive": {"prefer_process": True},
            "io_intensive": {"prefer_process": False}
        }
        
        # Update initial pool status
        self.pools_status["process_pool"]["workers"] = self.cpu_workers
        self.pools_status["thread_pool"]["workers"] = self.io_workers
        
    def hibernate_pool(self, pool_type: str) -> bool:
        """
        Puts a pool into hibernation state without completely destroying it.
        Saves its configuration for future restoration but frees resources.
        
        Args:
            pool_type (str): Pool type ("thread_pool" or "process_pool")
            
        Returns:
            bool: True if the pool was hibernated, False otherwise
        """
        # Check if the pool exists and is active
        if pool_type not in self.pools_status:
            return False
            
        pool_info = self.pools_status[pool_type]
        current_time = time.time()
        
        # If not active, there's nothing to hibernate
        if pool_info["status"] != "active":
            return False
            
        # Check cooldown period
        time_since_last_change = current_time - pool_info.get("last_status_change", 0)
        if time_since_last_change < self.pool_cooldown_period_sec:
            self.logger.debug(f"Pool {pool_type} in cooldown period ({time_since_last_change:.1f}s < {self.pool_cooldown_period_sec}s)")
            return False
            
        try:
            # Hibernate the pool according to its type
            if pool_type == "thread_pool" and self.thread_pool is not None:
                # Save state for future restoration
                pool_info["stored_state"] = {
                    "workers": self.thread_pool._max_workers,
                    "tasks_pending": sum(1 for _ in self.thread_pool._work_queue.unfinished_tasks) 
                                    if hasattr(self.thread_pool._work_queue, "unfinished_tasks") else 0
                }
                
                # Hibernate thread pool - don't start new tasks but allow current ones to finish
                self.thread_pool._shutdown = True  # Mark as shutdown but without affecting tasks in progress
                
                # Update state
                pool_info["status"] = "hibernated"
                pool_info["hibernated_at"] = current_time
                pool_info["last_status_change"] = current_time
                
                self.logger.info(f"Thread pool hibernated with {pool_info['stored_state']['workers']} workers")
                return True
                
            elif pool_type == "process_pool" and self.process_pool is not None:
                # Save state for future restoration
                pool_info["stored_state"] = {
                    "workers": self.process_pool._max_workers,
                    "tasks_pending": sum(1 for _ in self.process_pool._pending_work_items.values()) 
                                   if hasattr(self.process_pool, "_pending_work_items") else 0
                }
                
                # Hibernate process pool - don't start new tasks but allow current ones to finish
                self.process_pool._shutdown = True
                
                # Update state
                pool_info["status"] = "hibernated"  
                pool_info["hibernated_at"] = current_time
                pool_info["last_status_change"] = current_time
                
                self.logger.info(f"Process pool hibernated with {pool_info['stored_state']['workers']} workers")
                return True
                
        except Exception as e:
            self.logger.error(f"Error hibernating pool {pool_type}: {e}")
        
        return False
        
    def wake_pool(self, pool_type: str) -> bool:
        """
        Restores a hibernated pool to its active state.
        Creates a new pool with the configuration stored during hibernation.
        
        Args:
            pool_type (str): Pool type ("thread_pool" or "process_pool")
            
        Returns:
            bool: True if the pool was restored, False otherwise
        """
        # Check if the pool is hibernated
        if pool_type not in self.pools_status:
            return False
            
        pool_info = self.pools_status[pool_type]
        current_time = time.time()
        
        # Only wake hibernated pools
        if pool_info["status"] != "hibernated":
            return False
            
        # Check cooldown period
        time_since_last_change = current_time - pool_info.get("last_status_change", 0)
        if time_since_last_change < self.pool_cooldown_period_sec:
            self.logger.debug(f"Pool {pool_type} in cooldown period ({time_since_last_change:.1f}s < {self.pool_cooldown_period_sec}s)")
            return False
        
        try:
            # Get correct worker count (from stored state or current values)
            stored_state = pool_info.get("stored_state", {})
            workers = stored_state.get("workers", 0) if stored_state else 0
            
            # If no stored information, use current values
            if workers <= 0:
                workers = self.io_workers if pool_type == "thread_pool" else self.cpu_workers
            
            # Wake pool according to its type
            if pool_type == "thread_pool":
                # Create new thread pool with stored configuration
                self.thread_pool = ThreadPoolExecutor(max_workers=workers)
                
                # Update status
                pool_info["status"] = "active"
                pool_info["created_at"] = current_time
                pool_info["last_used"] = current_time
                pool_info["last_status_change"] = current_time
                pool_info["workers"] = workers
                
                self.logger.info(f"Thread pool restored from hibernation with {workers} workers")
                return True
                
            elif pool_type == "process_pool" and not self.disable_process_pool:
                # Create new process pool with stored configuration
                self.process_pool = ProcessPoolExecutor(max_workers=workers)
                
                # Update status
                pool_info["status"] = "active"
                pool_info["created_at"] = current_time
                pool_info["last_used"] = current_time
                pool_info["last_status_change"] = current_time
                pool_info["workers"] = workers
                
                self.logger.info(f"Process pool restored from hibernation with {workers} workers")
                return True
                
        except Exception as e:
            self.logger.error(f"Error waking pool {pool_type}: {e}")
            # Mark as inactive on error
            pool_info["status"] = "inactive"
        
        return False
        
    def _is_executor_shutdown(self, executor) -> bool:
        """
        Checks if an executor is shutdown or hibernated.
        This function supports both ThreadPoolExecutor and ProcessPoolExecutor.
        
        Args:
            executor: The executor to check
            
        Returns:
            bool: True if shutdown or hibernated, False otherwise
        """
        if executor is None:
            return True
            
        # Check if executor is shutdown
        try:
            return getattr(executor, '_shutdown', False)
        except (AttributeError, TypeError):
            # If we can't access _shutdown, assume it's shutdown for safety
            return True
            
    def get_thread_pool_executor(self) -> ThreadPoolExecutor:
        """
        Gets or initializes a ThreadPoolExecutor for I/O-bound tasks.
        Enhanced implementation with hibernation and restoration support.
        
        Returns:
            ThreadPoolExecutor: The thread pool
        """
        # If pool is None, create it
        if self.thread_pool is None:
            # Check if hibernated and wake it up
            if self.pools_status["thread_pool"]["status"] == "hibernated":
                if not self.wake_pool("thread_pool"):
                    # If restoration failed, create a new pool
                    self.thread_pool = ThreadPoolExecutor(max_workers=self.io_workers)
                    self.pools_status["thread_pool"]["status"] = "active"
                    self.pools_status["thread_pool"]["created_at"] = time.time()
                    self.pools_status["thread_pool"]["last_used"] = time.time()
                    self.pools_status["thread_pool"]["workers"] = self.io_workers
            else:
                # Create a new pool from scratch
                self.thread_pool = ThreadPoolExecutor(max_workers=self.io_workers)
                self.pools_status["thread_pool"]["status"] = "active"
                self.pools_status["thread_pool"]["created_at"] = time.time()
                self.pools_status["thread_pool"]["last_used"] = time.time()
                self.pools_status["thread_pool"]["workers"] = self.io_workers
        
        # If pool is shutdown but not hibernated, recreate it
        elif self._is_executor_shutdown(self.thread_pool) and self.pools_status["thread_pool"]["status"] != "hibernated":
            self.thread_pool = ThreadPoolExecutor(max_workers=self.io_workers)
            self.pools_status["thread_pool"]["status"] = "active"
            self.pools_status["thread_pool"]["created_at"] = time.time()
            self.pools_status["thread_pool"]["last_used"] = time.time()
            self.pools_status["thread_pool"]["workers"] = self.io_workers
        
        # Update last used time
        self.pools_status["thread_pool"]["last_used"] = time.time()
        
        return self.thread_pool
    
    def get_process_pool_executor(self) -> Optional[ProcessPoolExecutor]:
        """
        Gets or initializes a ProcessPoolExecutor for CPU-bound tasks.
        Enhanced implementation with hibernation and restoration support.
        
        Returns:
            Optional[ProcessPoolExecutor]: The process pool, or None if disabled
        """
        # If process pools are disabled, return None
        if self.disable_process_pool:
            return None
        
        # If pool is None, create it
        if self.process_pool is None:
            # Check if hibernated and wake it up
            if self.pools_status["process_pool"]["status"] == "hibernated":
                if not self.wake_pool("process_pool"):
                    # If restoration failed, create a new pool
                    self.process_pool = ProcessPoolExecutor(max_workers=self.cpu_workers)
                    self.pools_status["process_pool"]["status"] = "active"
                    self.pools_status["process_pool"]["created_at"] = time.time()
                    self.pools_status["process_pool"]["last_used"] = time.time()
                    self.pools_status["process_pool"]["workers"] = self.cpu_workers
            else:
                # Create a new pool from scratch
                self.process_pool = ProcessPoolExecutor(max_workers=self.cpu_workers)
                self.pools_status["process_pool"]["status"] = "active"
                self.pools_status["process_pool"]["created_at"] = time.time()
                self.pools_status["process_pool"]["last_used"] = time.time()
                self.pools_status["process_pool"]["workers"] = self.cpu_workers
        
        # If pool is shutdown but not hibernated, recreate it
        elif self._is_executor_shutdown(self.process_pool) and self.pools_status["process_pool"]["status"] != "hibernated":
            self.process_pool = ProcessPoolExecutor(max_workers=self.cpu_workers)
            self.pools_status["process_pool"]["status"] = "active"
            self.pools_status["process_pool"]["created_at"] = time.time()
            self.pools_status["process_pool"]["last_used"] = time.time()
            self.pools_status["process_pool"]["workers"] = self.cpu_workers
        
        # Update last used time
        self.pools_status["process_pool"]["last_used"] = time.time()
        
        return self.process_pool
        
    def shutdown_executors(self, wait: bool = True) -> None:
        """
        Gracefully shuts down all executor pools.
        
        Args:
            wait (bool): If True, waits for pending tasks to complete
        """
        # Close thread pool
        if self.thread_pool is not None:
            try:
                # If hibernated, update status directly
                if self.pools_status["thread_pool"]["status"] == "hibernated":
                    self.pools_status["thread_pool"]["status"] = "shutdown"
                else:
                    self.thread_pool.shutdown(wait=wait)
                    self.pools_status["thread_pool"]["status"] = "shutdown"
                
                # Release reference
                self.thread_pool = None
                self.logger.debug("Thread pool closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing thread pool: {e}")
        
        # Close process pool
        if self.process_pool is not None:
            try:
                # If hibernated, update status directly
                if self.pools_status["process_pool"]["status"] == "hibernated":
                    self.pools_status["process_pool"]["status"] = "shutdown"
                else:
                    self.process_pool.shutdown(wait=wait)
                    self.pools_status["process_pool"]["status"] = "shutdown"
                
                # Release reference
                self.process_pool = None
                self.logger.debug("Process pool closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing process pool: {e}")
        
        # Force GC to free resources
        try:
            import gc
            gc.collect()
        except ImportError:
            pass

    def _reinitialize_pools_with_new_workers(self) -> None:
        """
        Reinitializes executor pools with updated worker count.
        Enhanced version that preserves existing pools when possible.
        """
        # Check if we really need to reinitialize pools
        need_thread_reinit = False
        need_process_reinit = False
        
        # For thread pool
        if self.thread_pool is not None and not self._is_executor_shutdown(self.thread_pool):
            # Get current worker count
            current_io_workers = getattr(self.thread_pool, "_max_workers", 0)
            
            # Only reinitialize if change is significant (> threshold%)
            if current_io_workers > 0:
                change_percent = abs(self.io_workers - current_io_workers) / current_io_workers * 100
                need_thread_reinit = change_percent > self.worker_change_threshold_pct
                
                if need_thread_reinit:
                    self.logger.info(f"Reinitializing thread pool: {current_io_workers} -> {self.io_workers} workers (change {change_percent:.1f}%)")
            else:
                need_thread_reinit = True
        else:
            # If no active pool, we need to initialize
            need_thread_reinit = True
        
        # For process pool
        if self.process_pool is not None and not self._is_executor_shutdown(self.process_pool):
            # Get current worker count
            current_cpu_workers = getattr(self.process_pool, "_max_workers", 0)
            
            # Only reinitialize if change is significant (> threshold%)
            if current_cpu_workers > 0:
                change_percent = abs(self.cpu_workers - current_cpu_workers) / current_cpu_workers * 100
                need_process_reinit = change_percent > self.worker_change_threshold_pct
                
                if need_process_reinit:
                    self.logger.info(f"Reinitializing process pool: {current_cpu_workers} -> {self.cpu_workers} workers (change {change_percent:.1f}%)")
            else:
                need_process_reinit = True
        else:
            # If no active pool, we need to initialize (except if process pool is disabled)
            need_process_reinit = not self.disable_process_pool
        
        # Reinitialize only pools that really need it
        if need_thread_reinit:
            try:
                # Close previous pool if exists
                if self.thread_pool is not None:
                    self.thread_pool.shutdown(wait=False)
                    
                # Create new pool
                self.thread_pool = ThreadPoolExecutor(max_workers=self.io_workers)
                
                # Update status
                self.pools_status["thread_pool"]["status"] = "active"
                self.pools_status["thread_pool"]["created_at"] = time.time()
                self.pools_status["thread_pool"]["workers"] = self.io_workers
                self.pools_status["thread_pool"]["last_status_change"] = time.time()
            except Exception as e:
                self.logger.error(f"Error reinitializing thread pool: {e}")
        
        if need_process_reinit and not self.disable_process_pool:
            try:
                # Close previous pool if exists
                if self.process_pool is not None:
                    self.process_pool.shutdown(wait=False)
                    
                # Create new pool
                self.process_pool = ProcessPoolExecutor(max_workers=self.cpu_workers)
                
                # Update status
                self.pools_status["process_pool"]["status"] = "active"
                self.pools_status["process_pool"]["created_at"] = time.time()
                self.pools_status["process_pool"]["workers"] = self.cpu_workers
                self.pools_status["process_pool"]["last_status_change"] = time.time()
            except Exception as e:
                self.logger.error(f"Error reinitializing process pool: {e}")

    def recalculate_workers_if_needed(self) -> bool:
        """
        Checks if worker count recalculation is needed and performs it if applicable.
        
        Enhanced with hysteresis system and performance tracking to avoid unnecessary
        recalculations and dynamically adapt based on historical performance.
        
        Returns:
            bool: True if workers were recalculated, False otherwise
        """
        # Get current system metrics
        current_metrics = {
            "cpu_percent": self.resource_manager.metrics.get("cpu_percent_system", 0),
            "memory_percent": self.resource_manager.metrics.get("system_memory_percent", 0),
            "cpu_count": os.cpu_count() or 1
        }
        
        # Record current metrics in performance tracker
        self.performance_tracker.record_system_metrics(current_metrics)
        
        # Check if we should recalculate according to performance tracker
        should_recalc, reason = self.performance_tracker.should_recalculate(current_metrics)
        
        # If using dynamic interval, adjust it based on previous recalculations
        if self.dynamic_recalc_interval:
            # After several recalculations, gradually increase interval
            if self.worker_recalc_count > 5:
                # Maximum 3 times the base configuration
                max_interval = self.recalculation_frequency_sec * 3
                # Logarithmic increment that grows more slowly over time
                new_interval = min(
                    max_interval,
                    self.recalculation_frequency_sec * (1 + (math.log(self.worker_recalc_count - 4) / 2))
                )
                self.worker_calculation_interval_sec = new_interval
                
                # Also update tracker cooling period
                self.performance_tracker.cooling_period_sec = new_interval
        
        if not should_recalc:
            # Use DEBUG level to reduce log verbosity
            self.logger.debug(f"Worker recalculation skipped: {reason}")
            return False
            
        # If we reach here, proceed with recalculation
        # Check if predictions are available from tracker
        predicted_cpu_workers = self.performance_tracker.predict_optimal_worker_count(
            "process_pool", "default", current_metrics
        )
        
        predicted_io_workers = self.performance_tracker.predict_optimal_worker_count(
            "thread_pool", "default", current_metrics
        )
        
        # Calculate optimal configuration using prediction if available
        old_cpu_workers = self.cpu_workers
        old_io_workers = self.io_workers
        
        # Recalculate CPU workers
        if predicted_cpu_workers is not None:
            # Use prediction with small adjustment based on current pressure
            new_cpu_workers = predicted_cpu_workers
            # Limit by minimum and maximum configuration
            cpu_min = 1
            cpu_max = os.cpu_count() or 4  # Minimum 4 if cannot be determined
            tentative_cpu_workers = max(cpu_min, min(new_cpu_workers, cpu_max))
        else:
            # If no prediction, use traditional method
            tentative_cpu_workers = self._calculate_optimal_workers("cpu")
            
        # Recalculate IO workers
        if predicted_io_workers is not None:
            new_io_workers = predicted_io_workers
            # IO can have more workers since they are blocking tasks
            io_min = 2
            io_max = (os.cpu_count() or 4) * 2
            tentative_io_workers = max(io_min, min(new_io_workers, io_max))
        else:
            tentative_io_workers = self._calculate_optimal_workers("io")
        
        # Get change threshold (percentage) from ResourceManager
        change_threshold_pct = getattr(self.resource_manager, 'worker_change_threshold_pct', 15)
        
        # Apply hysteresis to avoid small changes and oscillations
        apply_cpu_change = False
        apply_io_change = False
        
        # Only apply changes if they exceed the variation threshold
        if old_cpu_workers > 0:
            cpu_change_pct = abs(tentative_cpu_workers - old_cpu_workers) / old_cpu_workers * 100
            apply_cpu_change = cpu_change_pct >= change_threshold_pct
        else:
            apply_cpu_change = True  # Always apply if no previous configuration
            
        if old_io_workers > 0:
            io_change_pct = abs(tentative_io_workers - old_io_workers) / old_io_workers * 100
            apply_io_change = io_change_pct >= change_threshold_pct
        else:
            apply_io_change = True  # Always apply if no previous configuration
            
        # Apply changes only if they exceed threshold or are the first ones
        if apply_cpu_change:
            self.cpu_workers = tentative_cpu_workers
        if apply_io_change:
            self.io_workers = tentative_io_workers
            
        # Check global worker limit if configured
        if self.max_total_workers:
            total_workers = self.cpu_workers + self.io_workers
            if total_workers > self.max_total_workers:
                # Reduce proportionally
                reduction_factor = self.max_total_workers / total_workers
                self.cpu_workers = max(1, int(self.cpu_workers * reduction_factor))
                self.io_workers = max(2, int(self.io_workers * reduction_factor))
        
        # Record recalculation and update counters
        self.performance_tracker.record_recalculation()
        self.last_worker_calculation_time = time.time()
        self.worker_recalc_count += 1
        
        # Check if values actually changed
        if old_cpu_workers != self.cpu_workers or old_io_workers != self.io_workers:
            changes = []
            if old_cpu_workers != self.cpu_workers:
                cpu_change_pct = abs(self.cpu_workers - old_cpu_workers) / max(1, old_cpu_workers) * 100
                changes.append(f"CPU: {old_cpu_workers} → {self.cpu_workers} ({cpu_change_pct:.1f}%)")
            if old_io_workers != self.io_workers:
                io_change_pct = abs(self.io_workers - old_io_workers) / max(1, old_io_workers) * 100
                changes.append(f"IO: {old_io_workers} → {self.io_workers} ({io_change_pct:.1f}%)")
                
            changes_str = ", ".join(changes)
            self.logger.info(f"Workers recalculated - {changes_str} (Reason: {reason}, Threshold: {change_threshold_pct}%)")
            
            # Update pool status
            self.pools_status["process_pool"]["workers"] = self.cpu_workers
            self.pools_status["thread_pool"]["workers"] = self.io_workers
            
            return True
        else:
            self.logger.debug(f"Recalculation completed without changes. Reason: {reason}")
            return False

    def map_tasks(self, func: Callable[[Any], R], iterable: Iterable[Any], 
                 chunksize: Optional[int] = None, task_type: str = "default",
                 timeout: Optional[float] = None, prefer_process: bool = False) -> List[R]:
        """
        Executes the function for each element in the iterable in parallel, automatically
        selecting the best executor based on task type.
        
        Optimized version that reduces overhead and improves performance:
        1. Avoids unnecessary list conversions when possible
        2. Performs minimal performance tracking
        3. Uses efficient batch processing
        
        Args:
            func: Function to execute for each element
            iterable: Elements to process
            chunksize: Batch size for processing
            task_type: Task type to select appropriate configuration
            timeout: Maximum wait time in seconds
            prefer_process: Whether to use ProcessPool even for I/O bound tasks
            
        Returns:
            List with results of applying the function to each element
        """
        # Heuristic to check iterable size without converting to list
        # when possible to avoid memory overhead
        try:
            # Try to get length directly
            n_items = len(iterable)
            items = iterable  # Keep original iterable
        except (TypeError, AttributeError):
            # If no len() method, convert to list
            items = list(iterable)
            n_items = len(items)
        
        if n_items == 0:
            return []
            
        # Select appropriate pool without expensive operations
        start_time = time.time()
        pool_type = "process_pool" if prefer_process or self._is_cpu_bound_task(task_type) else "thread_pool"
        
        # Get pool and current worker count
        if pool_type == "process_pool":
            use_pool = self.get_process_pool_executor()
            workers = self.cpu_workers
        else:
            use_pool = self.get_thread_pool_executor()
            workers = self.io_workers
        
        # Get optimal chunksize but only if not specified
        if chunksize is None:
            # Simplified calculation to avoid overhead
            if n_items < workers * 2:
                chunksize = 1  # For few elements
            elif n_items < 1000:  
                chunksize = max(1, n_items // (workers * 2))  # Medium granularity
            else:
                chunksize = max(1, n_items // workers)  # Large chunks
        
        # System metrics - Minimize access to reduce overhead
        system_metrics = {
            "cpu_percent": self.resource_manager.metrics.get("cpu_percent_system", 0),
            "memory_percent": self.resource_manager.metrics.get("system_memory_percent", 0)
        }
        
        # Execute tasks
        try:
            if pool_type == "process_pool" and self.disable_process_pool:
                # If process pools are disabled, use sequential execution
                results = [func(item) for item in items]
            else:
                results = list(use_pool.map(func, items, chunksize=chunksize or 1))
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update pool status - only essential metrics
            self.pools_status[pool_type]["last_used"] = time.time()
            self.pools_status[pool_type]["task_count"] += n_items
            
            # Record performance but only if task is significant
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
            self.logger.error(f"Error in map_tasks ({pool_type}): {e}")
            # Try sequential processing as fallback
            self.logger.info("Attempting sequential processing as fallback")
            return [func(item) for item in items]
            
    def get_optimal_chunksize(self, task_type: str, iterable_length: int, pool_type: Optional[str] = None) -> int:
        """
        Calculates optimal batch size for a map operation based on
        task characteristics and iterable size.
        
        Optimized version to maximize performance and minimize overhead.
        
        Args:
            task_type: Task type ("default", "cpu_intensive", "io_intensive")
            iterable_length: Length of iterable to process
            pool_type: Type of pool to be used (if known)
            
        Returns:
            int: Optimal batch size
        """
        # If pool_type not specified, determine based on task type
        if pool_type is None:
            pool_type = "process_pool" if self._is_cpu_bound_task(task_type) else "thread_pool"
        
        # Get worker count based on pool_type
        workers = self.cpu_workers if pool_type == "process_pool" else self.io_workers
        
        # Ensure minimum 1 worker
        workers = max(1, workers)
        
        # Simplified chunksize calculation based on task type and iterable length
        if iterable_length <= workers * 2:
            # If few tasks, use small chunksize
            return 1
        
        if task_type == "io_intensive":
            # For IO-bound, use smaller chunks to leverage IO parallelism
            base_chunksize = max(1, iterable_length // (workers * 4))
        elif task_type == "cpu_intensive":
            # For CPU-bound, larger chunks reduce overhead
            base_chunksize = max(1, iterable_length // workers)
        else:  # default
            # For general tasks, balance
            base_chunksize = max(1, iterable_length // (workers * 2))
        
        # Additional adjustments based on iterable size
        if iterable_length > 10000:
            # For very large iterables, increase chunksize to reduce overhead
            base_chunksize = max(base_chunksize, iterable_length // 500)
        elif iterable_length < 100:
            # For small iterables, limit chunksize
            base_chunksize = min(base_chunksize, 5)
        
        return base_chunksize

    def get_worker_counts(self) -> Dict[str, int]:
        """
        Gets current worker count for different pools.
        
        Returns:
            Dict[str, int]: Dictionary with worker count by type
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
        Hibernates a pool without destroying it if unused for a certain time.
        This frees resources while maintaining structure.
        
        Args:
            pool_type (str): "thread_pool" or "process_pool"
            idle_seconds (float): Seconds of inactivity to hibernate
            
        Returns:
            bool: True if pool was hibernated, False otherwise
        """
        # Check if pool exists and is active
        if pool_type not in self.pools_status:
            return False
            
        pool_info = self.pools_status[pool_type]
        current_time = time.time()
        
        # If pool is not active, nothing to do
        if pool_info["status"] != "active":
            return False
            
        # If not enough time has passed, don't hibernate
        time_since_last_use = current_time - pool_info["last_used"]
        if time_since_last_use < idle_seconds:
            return False
            
        # Hibernate appropriate pool
        try:
            if pool_type == "thread_pool" and hasattr(self, "thread_pool"):
                # Save worker count before hibernating
                workers = self.thread_pool._max_workers
                self.thread_pool.shutdown(wait=False)
                self.thread_pool = None
                pool_info["workers"] = workers
                pool_info["status"] = "hibernated"
                pool_info["hibernated_at"] = current_time
                self.logger.info(f"Thread pool hibernated after {time_since_last_use:.1f}s of inactivity")
                return True
                
            elif pool_type == "process_pool" and hasattr(self, "process_pool"):
                # Save worker count before hibernating
                workers = self.process_pool._max_workers
                self.process_pool.shutdown(wait=False)
                self.process_pool = None
                pool_info["workers"] = workers
                pool_info["status"] = "hibernated"
                pool_info["hibernated_at"] = current_time
                self.logger.info(f"Process pool hibernated after {time_since_last_use:.1f}s of inactivity")
                return True
                
        except Exception as e:
            self.logger.error(f"Error hibernating pool {pool_type}: {e}")
            
        return False

    def restore_pool_from_hibernation(self, pool_type: str) -> bool:
        """
        Restores a previously hibernated pool to active state.
        
        Args:
            pool_type (str): "thread_pool" or "process_pool"
            
        Returns:
            bool: True if pool was restored, False otherwise
        """
        if pool_type not in self.pools_status:
            return False
            
        pool_info = self.pools_status[pool_type]
        
        # Only restore if hibernated
        if pool_info["status"] != "hibernated":
            return False
            
        try:
            if pool_type == "thread_pool":
                # Restore with same worker count as before
                workers = pool_info["workers"] if pool_info["workers"] > 0 else self.io_workers
                self.thread_pool = ThreadPoolExecutor(max_workers=workers)
                pool_info["status"] = "active"
                pool_info["last_used"] = time.time()
                pool_info["created_at"] = time.time()
                self.logger.info(f"Thread pool restored from hibernation with {workers} workers")
                return True
                
            elif pool_type == "process_pool" and not self.disable_process_pool:
                # Restore with same worker count as before
                workers = pool_info["workers"] if pool_info["workers"] > 0 else self.cpu_workers
                self.process_pool = ProcessPoolExecutor(max_workers=workers)
                pool_info["status"] = "active"
                pool_info["last_used"] = time.time()
                pool_info["created_at"] = time.time()
                self.logger.info(f"Process pool restored from hibernation with {workers} workers")
                return True
                
        except Exception as e:
            self.logger.error(f"Error restoring pool {pool_type} from hibernation: {e}")
            pool_info["status"] = "inactive"  # Mark as inactive on error
            
        return False

    def _calculate_optimal_workers(self, worker_type: str) -> int:
        """
        Calculates optimal worker count based on type and system conditions.
        
        This function considers:
        - Explicit configuration from ResourceManager
        - Available core count
        - Current CPU and memory usage
        - Environment type (development, production, etc.)
        
        Args:
            worker_type (str): Worker type to calculate ("cpu" or "io")
            
        Returns:
            int: Optimal number of workers
        """
        # Get configuration from ResourceManager
        if worker_type == "cpu":
            config_value = getattr(self.resource_manager, 'default_cpu_workers', "auto")
        else:  # io
            config_value = getattr(self.resource_manager, 'default_io_workers', "auto")
            
        # If there's an explicit configuration other than "auto", use it
        if isinstance(config_value, int) and config_value > 0:
            return config_value
            
        # Get available core count
        cpu_count = os.cpu_count() or 4  # Fallback to 4 if cannot determine
        
        # Get current system metrics
        cpu_percent = self.resource_manager.metrics.get("cpu_percent_system", 0)
        memory_percent = self.resource_manager.metrics.get("system_memory_percent", 0)
        
        # Calculate adjustment factor based on resource usage
        # Higher usage means fewer workers to avoid overload
        resource_factor = 1.0
        if cpu_percent > 80:
            resource_factor *= 0.7  # Reduce 30% if CPU is heavily loaded
        elif cpu_percent > 60:
            resource_factor *= 0.85  # Reduce 15% if CPU is moderately loaded
            
        if memory_percent > 85:
            resource_factor *= 0.7  # Reduce 30% if memory is heavily loaded
        elif memory_percent > 70:
            resource_factor *= 0.85  # Reduce 15% if memory is moderately loaded
            
        # Adjust based on environment type
        environment_type = getattr(self.resource_manager, 'environment_type', "development")
        
        # Development environments usually have more resources available for the process
        if "development" in environment_type:
            env_factor = 1.0
        # Production environments are usually more conservative
        elif "production" in environment_type or "server" in environment_type:
            env_factor = 0.9
        # Cloud environments usually have shared resources
        elif any(env in environment_type for env in ["cloud", "aws", "gcp", "azure"]):
            env_factor = 0.8
        # Containers usually have very limited resources
        elif any(env in environment_type for env in ["container", "kubernetes"]):
            env_factor = 0.7
        else:
            env_factor = 0.9  # Default value
            
        # Calculate base worker count by type
        if worker_type == "cpu":
            # For CPU-bound, use physical core count
            # Try to get physical cores, fallback to logical
            physical_cores = psutil.cpu_count(logical=False) or cpu_count
            base_workers = max(1, int(physical_cores * resource_factor * env_factor))
            
            # Ensure at least 1 worker and no more than logical cores
            return max(1, min(base_workers, cpu_count))
        else:  # io
            # For IO-bound, more workers can be used since they're blocked
            # Typically 2x or more than core count
            io_multiplier = 2.0  # Multiplier for IO vs CPU
            base_workers = max(2, int(cpu_count * io_multiplier * resource_factor * env_factor))
            
            # Ensure at least 2 workers for IO and no more than 4x cores
            return max(2, min(base_workers, cpu_count * 4))

    def _is_cpu_bound_task(self, task_type: str) -> bool:
        """
        Determines if a task is CPU-intensive based on its type and name patterns.
        
        Args:
            task_type (str): Task type to evaluate
            
        Returns:
            bool: True if task is CPU-intensive, False if I/O-intensive
        """
        # Explicit types
        if task_type == "cpu_intensive":
            return True
        elif task_type == "io_intensive":
            return False
        
        # For "default" or other types, use heuristic patterns
        task_lower = task_type.lower()
        
        # Check CPU-intensive task patterns
        for pattern in _CPU_TASK_PATTERNS:
            if re.search(pattern, task_lower):
                return True
        
        # Additional patterns for I/O
        io_patterns = [
            r'read', r'write', r'download', r'upload', r'fetch', 
            r'save', r'load', r'request', r'response', r'network'
        ]
        
        for pattern in io_patterns:
            if re.search(pattern, task_lower):
                return False
        
        # Default to I/O-bound (more conservative to avoid system overload)
        return False