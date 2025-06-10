import logging
import threading
import time
import gc
from typing import Optional, Dict, Any

from config import config
from .embeddings_manager import EmbeddingManager

# ModelReference is defined in this file, no need to import it from embeddings_manager.
# Type annotation for _instances will use the local ModelReference class.

# Configure logging
logger = logging.getLogger(__name__)

class ModelReference:
    """Class that maintains a reference to a model and tracks its usage."""
    def __init__(self, model_instance):
        self.model = model_instance
        self.reference_count = 0
        self.last_used = time.time()
        self.lock = threading.RLock()
    
    def increment(self):
        """Increments the model reference counter."""
        with self.lock:
            self.reference_count += 1
            self.last_used = time.time()
            return self.reference_count
    
    def decrement(self):
        """Decrements the model reference counter."""
        with self.lock:
            if self.reference_count > 0:
                self.reference_count -= 1
            self.last_used = time.time()
            return self.reference_count
    
    def get_ref_count(self):
        """Gets the current reference count."""
        with self.lock:
            return self.reference_count
    
    def update_last_used(self):
        """Updates the last used timestamp."""
        with self.lock:
            self.last_used = time.time()

class EmbeddingFactory:
    """
    Factory for creating and managing embedding model instances.
    Implements both Factory and Singleton patterns combined with resource management.
    """
    
    _instances: Dict[str, 'ModelReference'] = {}
    _lock = threading.RLock()
    _inactive_timeout = 600  # 10 minutes without use
    
    @staticmethod
    def get_embedding_manager(model_type: Optional[str] = None) -> EmbeddingManager:
        """
        Gets an instance of the embedding manager.
        
        Args:
            model_type: Type of model to use. If None, it's taken from configuration.
            
        Returns:
            Instance of the embedding manager.
        """
        # If no model is specified, use the default from configuration
        if model_type is None:
            embedding_config = config.get_embedding_config()
            model_type = embedding_config.get("model", "modernbert")
        
        # Unique key for this model type
        instance_key = f"embedding:{model_type}"
        
        with EmbeddingFactory._lock:
            # If an instance already exists for this model, return it
            if instance_key in EmbeddingFactory._instances:
                model_ref = EmbeddingFactory._instances[instance_key]
                model_ref.increment()
                logger.debug(f"Reusing existing embedding model: {model_type} (refs: {model_ref.get_ref_count()})")
                return model_ref.model
            
            # Create a new instance
            logger.info(f"Creating new embedding model instance: {model_type}")
            try:
                manager = EmbeddingManager(model_type=model_type)
                # Loading the model here can be intensive, consider error handling
                manager.load_model() 
            except Exception as e:
                logger.error(f"Error creating or loading EmbeddingManager for {model_type}: {e}", exc_info=True)
                # Propagate the exception or return None/handle according to error policy
                raise RuntimeError(f"Could not initialize embedding model {model_type}") from e

            model_ref = ModelReference(manager)
            model_ref.increment()  # First reference
            EmbeddingFactory._instances[instance_key] = model_ref
            
            return manager
    
    @staticmethod
    def release_embedding_manager(model: EmbeddingManager) -> int:
        """
        Releases a reference to an embedding model.
        
        Args:
            model: Model instance to release
            
        Returns:
            Number of remaining references to the model
        """
        if not model:
            return 0
            
        instance_key = None
        
        # Find the key for this model
        for key, model_ref in EmbeddingFactory._instances.items():
            if model_ref.model == model:
                instance_key = key
                break
        
        if not instance_key:
            return 0
            
        with EmbeddingFactory._lock:
            model_ref = EmbeddingFactory._instances[instance_key]
            remaining = model_ref.decrement()
            logger.debug(f"Released reference to model {instance_key} (remaining refs: {remaining})")
            return remaining
    
    @staticmethod
    def reset_instances() -> None:
        """
        Resets all embedding manager instances.
        Useful for tests or to free resources.
        """
        with EmbeddingFactory._lock:
            EmbeddingFactory._instances.clear()
            logger.info("All embedding manager instances have been reset")
    
    @staticmethod
    def get_active_models() -> Dict[str, Dict[str, Any]]:
        """
        Gets information about currently loaded models.
        
        Returns:
            Dictionary with information about active models
        """
        result = {}
        with EmbeddingFactory._lock:
            for key, model_ref in EmbeddingFactory._instances.items():
                result[key] = {
                    "reference_count": model_ref.get_ref_count(),
                    "last_used": model_ref.last_used,
                    "idle_time": time.time() - model_ref.last_used,
                    "model_type": model_ref.model.model_type if hasattr(model_ref.model, "model_type") else "unknown"
                }
        return result
    
    @staticmethod
    def release_inactive_models(aggressive: bool = False, force_release_active: bool = False) -> int:
        """
        Releases memory occupied by inactive embedding models.
        
        Args:
            aggressive (bool): If True, uses a more aggressive strategy releasing even
                              models with few references. Default False.
            force_release_active (bool): If True, can release even active models
                                        with references when necessary in critical situations.
                                        Default False.
            
        Returns:
            int: Number of models released.
        """
        logger = logging.getLogger("EmbeddingFactory")
        
        with EmbeddingFactory._lock:
            if not EmbeddingFactory._instances:
                logger.debug("No active embedding models to release.")
                return 0
                
            models_count_before = len(EmbeddingFactory._instances)
            models_to_remove = []
            
            # First pass: identify models to release
            for model_key, model_ref in list(EmbeddingFactory._instances.items()):
                try:
                    # Check if the model is a candidate for release
                    refs = model_ref.get_ref_count()
                    last_used = time.time() - model_ref.last_used
                    is_inactive = refs == 0
                    
                    # Release policy based on references and usage
                    release_condition = False
                    
                    if is_inactive:
                        # Models without references: always release
                        release_condition = True
                        reason = "no references"
                    elif aggressive and refs <= 1 and last_used > 300:  # 5 minutes
                        # Aggressive mode: release models with few references and no recent use
                        release_condition = True
                        reason = f"aggressive with {refs} refs and {last_used:.0f}s inactive"
                    elif force_release_active and last_used > 60:
                        # Force release: release even active models without very recent use
                        release_condition = True
                        reason = f"forced with {refs} refs and {last_used:.0f}s inactive"
                        
                    if release_condition:
                        models_to_remove.append((model_key, model_ref, reason))
                        
                except Exception as e:
                    logger.error(f"Error evaluating model {model_key} for release: {e}")
                    
            # Second pass: release models
            models_released = 0
            
            for model_key, model_ref, reason in models_to_remove:
                try:
                    # Get the model before removing it from _instances
                    model = model_ref.model
                    
                    # Try deep cleaning of the model
                    try:
                        # Access the internal model if possible
                        if hasattr(model, "_model") and model._model is not None:
                            # Move specific model to CPU first if it's on GPU
                            if hasattr(model._model, "to") and callable(model._model.to):
                                try:
                                    if hasattr(model._model, "device"):
                                        device_str = str(model._model.device)
                                        if "cuda" in device_str or "mps" in device_str:
                                            logger.debug(f"Moving model {model_key} from {device_str} to CPU before releasing it")
                                            model._model.to("cpu")
                                except Exception as move_err:
                                    logger.debug(f"Error moving model to CPU: {move_err}")
                            
                            # Remove circular references
                            if hasattr(model._model, "encoder") and model._model.encoder is not None:
                                model._model.encoder = None
                            
                            # Explicitly release embeddings layer
                            if hasattr(model._model, "embeddings") and model._model.embeddings is not None:
                                model._model.embeddings = None
                            
                            # Release model completely
                            model._model = None
                        
                        # Memory release specific for sentence-transformers
                        if hasattr(model, "model") and model.model is not None:
                            if hasattr(model.model, "to") and callable(model.model.to):
                                model.model.to("cpu")
                            model.model = None
                            
                        # Release tokenizer which may contain cache
                        if hasattr(model, "tokenizer") and model.tokenizer is not None:
                            model.tokenizer = None
                    except Exception as cleanup_err:
                        logger.debug(f"Error during deep cleaning of model: {cleanup_err}")
                    
                    # Remove the reference from _instances
                    del EmbeddingFactory._instances[model_key]
                    models_released += 1
                    
                    logger.info(f"Model released: {model_key} - Reason: {reason}")
                    
                except Exception as e:
                    logger.error(f"Error releasing model {model_key}: {e}")
                    
            # Force explicit GC after releasing models if there were releases
            if models_released > 0:
                gc.collect()
                
                # Try to free GPU memory if available
                try:
                    import torch
                    if hasattr(torch, 'cuda') and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug("CUDA cache freed after removing models")
                    elif hasattr(torch, 'mps') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                            logger.debug("MPS cache freed after removing models")
                except ImportError:
                    pass
                except Exception as torch_err:
                    logger.debug(f"Error freeing GPU cache: {torch_err}")
            
            models_count_after = len(EmbeddingFactory._instances)
            logger.info(f"Model release: {models_count_before} -> {models_count_after} ({models_released} released)")
            
            return models_released

    @staticmethod
    def get_active_model_count():
        """
        Returns the number of active embedding models.
        Used by ResourceManager for resource monitoring.
        
        Returns:
            int: Number of active models with references > 0
        """
        with EmbeddingFactory._lock:
            active_count = 0
            for model_key, model_ref in EmbeddingFactory._instances.items():
                if model_ref.get_ref_count() > 0:
                    active_count += 1
            return active_count

# Attach logger to the class for classmethods
EmbeddingFactory.logger = logger
