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
    
    @classmethod
    def release_inactive_models(cls, aggressive: bool = False) -> int:
        """
        Libera modelos de embedding que no han sido utilizados recientemente
        y no tienen referencias activas.

        Esta función es típicamente invocada por `MemoryManager.cleanup`,
        que a su vez es llamado por `ResourceManager.request_cleanup`.

        Args:
            aggressive (bool): Si True, el tiempo de inactividad para considerar
                               un modelo como liberable se reduce (hace la
                               liberación más probable).
                               Defaults to False.

        Returns:
            int: El número de modelos que fueron liberados.
        """
        cls.logger.debug(f"Solicitud para liberar modelos inactivos (agresivo={aggressive}).")
        released_count = 0
        keys_to_remove = []
        # Determinar timeout efectivo
        effective_timeout = cls._inactive_timeout / 2 if aggressive else cls._inactive_timeout
        if aggressive:
            cls.logger.warning(f"Modo agresivo: Timeout de inactividad reducido a {effective_timeout}s.")

        with cls._lock:
            current_time = time.time()
            for key, model_ref in cls._instances.items():
                if model_ref.get_ref_count() == 0 and (current_time - model_ref.last_used) > effective_timeout:
                    keys_to_remove.append(key)
            
            model_was_on_gpu = False # Flag para saber si llamar a empty_cache
            for key in keys_to_remove:
                try:
                    cls.logger.info(f"Liberando memoria de modelo inactivo: {key}")
                    model_instance = cls._instances[key].model
                    
                    # Intentar liberar memoria específica del modelo (depende de la implementación de EmbeddingManager)
                    if hasattr(model_instance, 'release_resources'):
                        model_instance.release_resources()
                        cls.logger.debug(f"Llamado a release_resources() para {key}.")
                    
                    # Comprobar si el modelo usaba GPU (esto es heurístico, necesita info del modelo)
                    if hasattr(model_instance, 'device') and 'cuda' in str(model_instance.device):
                         model_was_on_gpu = True

                    # Eliminar la referencia de la factory
                    del cls._instances[key]
                    released_count += 1

                except Exception as e:
                    cls.logger.error(f"Error al liberar modelo {key}: {e}", exc_info=True)

        # Limpiar caché de GPU si se liberaron modelos de GPU y torch está disponible
        if released_count > 0 and model_was_on_gpu:
            if importlib.util.find_spec("torch"):
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        cls.logger.info("torch.cuda.empty_cache() llamado tras liberar modelo GPU.")
                except ImportError:
                    cls.logger.warning("Torch importado pero no se pudo llamar a empty_cache (quizás no instalado correctamente).")
                except Exception as e:
                    cls.logger.error(f"Error al llamar a torch.cuda.empty_cache(): {e}", exc_info=True)
            else:
                 cls.logger.debug("Torch no parece estar instalado, no se limpiará caché de GPU.")

        # Usar nivel de log apropiado según si se liberaron modelos o no
        if released_count > 0:
            # Forzar GC después de eliminar referencias puede ayudar
            gc.collect()
            cls.logger.info(f"{released_count} modelos de embedding fueron liberados.")
        else:
            cls.logger.debug("No se encontraron modelos inactivos para liberar.")
            
        return released_count

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
