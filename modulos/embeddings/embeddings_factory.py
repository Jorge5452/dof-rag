import logging
import threading
import time
import weakref
import gc
import importlib.util # Para chequear torch
from typing import Optional, Dict, Set, List, Any, TYPE_CHECKING

from config import config
from .embeddings_manager import EmbeddingManager

# ModelReference se define en este archivo, no necesita ser importada desde embeddings_manager.
# La anotación de tipo para _instances usará la clase local ModelReference.

# Configurar logging
logger = logging.getLogger(__name__)

class ModelReference:
    """Clase que mantiene una referencia a un modelo y cuenta su uso."""
    def __init__(self, model_instance):
        self.model = model_instance
        self.reference_count = 0
        self.last_used = time.time()
        self.lock = threading.RLock()
    
    def increment(self):
        """Incrementa el contador de referencias al modelo."""
        with self.lock:
            self.reference_count += 1
            self.last_used = time.time()
            return self.reference_count
    
    def decrement(self):
        """Decrementa el contador de referencias al modelo."""
        with self.lock:
            if self.reference_count > 0:
                self.reference_count -= 1
            self.last_used = time.time()
            return self.reference_count
    
    def get_ref_count(self):
        """Obtiene el contador actual de referencias."""
        with self.lock:
            return self.reference_count
    
    def update_last_used(self):
        """Actualiza el timestamp de último uso."""
        with self.lock:
            self.last_used = time.time()

class EmbeddingFactory:
    """
    Factory para crear y gestionar instancias de modelos de embeddings.
    Implementa el patrón Factory y Singleton combinados con gestión de recursos.
    """
    
    _instances: Dict[str, 'ModelReference'] = {}
    _lock = threading.RLock()
    _inactive_timeout = 600  # 10 minutos sin uso
    
    @staticmethod
    def get_embedding_manager(model_type: Optional[str] = None) -> EmbeddingManager:
        """
        Obtiene una instancia del gestor de embeddings.
        
        Args:
            model_type: Tipo de modelo a utilizar. Si es None, se toma de la configuración.
            
        Returns:
            Instancia del gestor de embeddings.
        """
        # Si no se especifica modelo, usar el configurado por defecto
        if model_type is None:
            embedding_config = config.get_embedding_config()
            model_type = embedding_config.get("model", "modernbert")
        
        # Clave única para este tipo de modelo
        instance_key = f"embedding:{model_type}"
        
        with EmbeddingFactory._lock:
            # Si ya existe una instancia para este modelo, retornarla
            if instance_key in EmbeddingFactory._instances:
                model_ref = EmbeddingFactory._instances[instance_key]
                model_ref.increment()
                logger.debug(f"Reutilizando modelo de embeddings existente: {model_type} (refs: {model_ref.get_ref_count()})")
                return model_ref.model
            
            # Crear una nueva instancia
            logger.info(f"Creando nueva instancia de modelo de embeddings: {model_type}")
            try:
                manager = EmbeddingManager(model_type=model_type)
                # Cargar el modelo aquí puede ser intensivo, considerar la gestión de errores
                manager.load_model() 
            except Exception as e:
                logger.error(f"Error al crear o cargar EmbeddingManager para {model_type}: {e}", exc_info=True)
                # Propagar la excepción o devolver None/manejar según la política de errores
                raise RuntimeError(f"No se pudo inicializar el modelo de embeddings {model_type}") from e

            model_ref = ModelReference(manager)
            model_ref.increment()  # Primera referencia
            EmbeddingFactory._instances[instance_key] = model_ref
            
            return manager
    
    @staticmethod
    def release_embedding_manager(model: EmbeddingManager) -> int:
        """
        Libera una referencia a un modelo de embeddings.
        
        Args:
            model: Instancia del modelo a liberar
            
        Returns:
            Número de referencias restantes al modelo
        """
        if not model:
            return 0
            
        instance_key = None
        
        # Buscar la clave para este modelo
        for key, model_ref in EmbeddingFactory._instances.items():
            if model_ref.model == model:
                instance_key = key
                break
        
        if not instance_key:
            return 0
            
        with EmbeddingFactory._lock:
            model_ref = EmbeddingFactory._instances[instance_key]
            remaining = model_ref.decrement()
            logger.debug(f"Liberada referencia al modelo {instance_key} (refs restantes: {remaining})")
            return remaining
    
    @staticmethod
    def reset_instances() -> None:
        """
        Reinicia todas las instancias de gestores de embeddings.
        Útil para tests o para liberar recursos.
        """
        with EmbeddingFactory._lock:
            EmbeddingFactory._instances.clear()
            logger.info("Todas las instancias de gestores de embeddings han sido reiniciadas")
    
    @staticmethod
    def get_active_models() -> Dict[str, Dict[str, Any]]:
        """
        Obtiene información sobre los modelos actualmente cargados.
        
        Returns:
            Diccionario con información de modelos activos
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
        Libera la memoria ocupada por modelos de embedding inactivos.
        
        Args:
            aggressive (bool): Si es True, usa estrategia más agresiva liberando incluso
                              modelos con pocas referencias. Default False.
            force_release_active (bool): Si es True, puede liberar incluso modelos activos
                                        con referencias cuando sea necesario en situaciones críticas.
                                        Default False.
            
        Returns:
            int: Número de modelos liberados.
        """
        logger = logging.getLogger("EmbeddingFactory")
        
        with EmbeddingFactory._lock:
            if not EmbeddingFactory._instances:
                logger.debug("No hay modelos de embeddings activos para liberar.")
                return 0
                
            models_count_before = len(EmbeddingFactory._instances)
            models_to_remove = []
            
            # Primera pasada: identificar modelos a liberar
            for model_key, model_ref in list(EmbeddingFactory._instances.items()):
                try:
                    # Verificar si el modelo es candidato para liberación
                    refs = model_ref.get_ref_count()
                    last_used = time.time() - model_ref.last_used
                    is_inactive = refs == 0
                    
                    # Política de liberación basada en referencias y uso
                    release_condition = False
                    
                    if is_inactive:
                        # Modelos sin referencias: liberar siempre
                        release_condition = True
                        reason = "sin referencias"
                    elif aggressive and refs <= 1 and last_used > 300:  # 5 minutos
                        # Modo agresivo: liberar modelos con pocas referencias y sin uso reciente
                        release_condition = True
                        reason = f"agresivo con {refs} refs y {last_used:.0f}s inactivo"
                    elif force_release_active and last_used > 60:
                        # Forzar liberación: liberar incluso modelos activos sin uso muy reciente
                        release_condition = True
                        reason = f"forzado con {refs} refs y {last_used:.0f}s inactivo"
                        
                    if release_condition:
                        models_to_remove.append((model_key, model_ref, reason))
                        
                except Exception as e:
                    logger.error(f"Error evaluando modelo {model_key} para liberación: {e}")
                    
            # Segunda pasada: liberar modelos
            models_released = 0
            
            for model_key, model_ref, reason in models_to_remove:
                try:
                    # Obtener el modelo antes de eliminarlo de _instances
                    model = model_ref.model
                    
                    # Intentar limpieza profunda del modelo
                    try:
                        # Acceder al modelo interno si es posible
                        if hasattr(model, "_model") and model._model is not None:
                            # Descargar modelo específico a CPU primero si está en GPU
                            if hasattr(model._model, "to") and callable(model._model.to):
                                try:
                                    if hasattr(model._model, "device"):
                                        device_str = str(model._model.device)
                                        if "cuda" in device_str or "mps" in device_str:
                                            logger.debug(f"Moviendo modelo {model_key} de {device_str} a CPU antes de liberarlo")
                                            model._model.to("cpu")
                                except Exception as move_err:
                                    logger.debug(f"Error moviendo modelo a CPU: {move_err}")
                            
                            # Eliminar referencias circulares
                            if hasattr(model._model, "encoder") and model._model.encoder is not None:
                                model._model.encoder = None
                            
                            # Liberar capa embeddings explícitamente
                            if hasattr(model._model, "embeddings") and model._model.embeddings is not None:
                                model._model.embeddings = None
                            
                            # Liberar modelo completamente
                            model._model = None
                        
                        # Liberación de memoria específica para sentence-transformers
                        if hasattr(model, "model") and model.model is not None:
                            if hasattr(model.model, "to") and callable(model.model.to):
                                model.model.to("cpu")
                            model.model = None
                            
                        # Liberar tokenizador que puede contener caché
                        if hasattr(model, "tokenizer") and model.tokenizer is not None:
                            model.tokenizer = None
                    except Exception as cleanup_err:
                        logger.debug(f"Error durante limpieza profunda del modelo: {cleanup_err}")
                    
                    # Eliminar la referencia de _instances
                    del EmbeddingFactory._instances[model_key]
                    models_released += 1
                    
                    logger.info(f"Modelo liberado: {model_key} - Razón: {reason}")
                    
                except Exception as e:
                    logger.error(f"Error liberando modelo {model_key}: {e}")
                    
            # Forzar GC explícito después de liberar modelos si hubo liberaciones
            if models_released > 0:
                gc.collect()
                
                # Intentar liberar memoria GPU si está disponible
                try:
                    import torch
                    if hasattr(torch, 'cuda') and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug("Caché CUDA liberada después de eliminar modelos")
                    elif hasattr(torch, 'mps') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                            logger.debug("Caché MPS liberada después de eliminar modelos")
                except ImportError:
                    pass
                except Exception as torch_err:
                    logger.debug(f"Error liberando caché GPU: {torch_err}")
            
            models_count_after = len(EmbeddingFactory._instances)
            logger.info(f"Liberación de modelos: {models_count_before} -> {models_count_after} ({models_released} liberados)")
            
            return models_released

    @staticmethod
    def get_active_model_count():
        """
        Devuelve el número de modelos de embeddings activos.
        Utilizado por ResourceManager para monitoreo de recursos.
        
        Returns:
            int: Número de modelos activos con referencias > 0
        """
        with EmbeddingFactory._lock:
            active_count = 0
            for model_key, model_ref in EmbeddingFactory._instances.items():
                if model_ref.get_ref_count() > 0:
                    active_count += 1
            return active_count

# Adjuntar logger a la clase para los classmethods
EmbeddingFactory.logger = logger
