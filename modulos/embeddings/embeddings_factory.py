import logging
import threading
import time
import weakref
from typing import Optional, Dict, Set, List, Any

from config import config
from modulos.embeddings.embeddings_manager import EmbeddingManager

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
    
    _instances: Dict[str, ModelReference] = {}
    _lock = threading.RLock()
    _cleanup_thread = None
    _cleanup_interval = 300  # 5 minutos
    _inactive_timeout = 600  # 10 minutos sin uso
    _running = True
    
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
        
        # Iniciar el hilo de limpieza si aún no está corriendo
        EmbeddingFactory._ensure_cleanup_thread()
        
        with EmbeddingFactory._lock:
            # Si ya existe una instancia para este modelo, retornarla
            if instance_key in EmbeddingFactory._instances:
                model_ref = EmbeddingFactory._instances[instance_key]
                model_ref.increment()
                logger.debug(f"Reutilizando modelo de embeddings existente: {model_type} (refs: {model_ref.get_ref_count()})")
                return model_ref.model
            
            # Crear una nueva instancia
            logger.info(f"Creando nueva instancia de modelo de embeddings: {model_type}")
            manager = EmbeddingManager(model_type)
            
            # Almacenar la instancia con su contador de referencias
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
    def _cleanup_unused_models():
        """
        Limpia modelos que no han sido utilizados por un tiempo.
        """
        to_remove = []
        
        with EmbeddingFactory._lock:
            current_time = time.time()
            
            for key, model_ref in EmbeddingFactory._instances.items():
                # Si no hay referencias activas y ha pasado suficiente tiempo
                if model_ref.get_ref_count() == 0 and (current_time - model_ref.last_used) > EmbeddingFactory._inactive_timeout:
                    to_remove.append(key)
            
            # Eliminar modelos inactivos
            for key in to_remove:
                logger.info(f"Liberando memoria de modelo inactivo: {key}")
                # Liberar explícitamente el modelo
                model = EmbeddingFactory._instances[key].model
                
                # Intentar liberar la memoria del modelo
                try:
                    if hasattr(model, "_model") and model._model:
                        model._model = None
                    
                    if hasattr(model, "embedding_dim"):
                        model.embedding_dim = None
                except:
                    pass
                
                # Eliminar la referencia
                del EmbeddingFactory._instances[key]
        
        # Forzar recolección de basura si se eliminaron modelos
        if to_remove:
            import gc
            gc.collect()
            logger.info(f"Limpieza completada: {len(to_remove)} modelos eliminados")
    
    @staticmethod
    def _ensure_cleanup_thread():
        """
        Asegura que el hilo de limpieza está en ejecución.
        """
        if EmbeddingFactory._cleanup_thread is None or not EmbeddingFactory._cleanup_thread.is_alive():
            EmbeddingFactory._running = True
            EmbeddingFactory._cleanup_thread = threading.Thread(target=EmbeddingFactory._cleanup_worker, daemon=True)
            EmbeddingFactory._cleanup_thread.start()
            logger.info("Hilo de limpieza de modelos de embeddings iniciado")
    
    @staticmethod
    def _cleanup_worker():
        """
        Hilo de trabajo para limpieza periódica.
        """
        while EmbeddingFactory._running:
            try:
                time.sleep(EmbeddingFactory._cleanup_interval)
                EmbeddingFactory._cleanup_unused_models()
            except Exception as e:
                logger.error(f"Error en hilo de limpieza de modelos: {str(e)}")
    
    @staticmethod
    def shutdown():
        """
        Detiene el hilo de limpieza y libera todos los recursos.
        """
        EmbeddingFactory._running = False
        
        if EmbeddingFactory._cleanup_thread and EmbeddingFactory._cleanup_thread.is_alive():
            EmbeddingFactory._cleanup_thread.join(timeout=1.0)
        
        with EmbeddingFactory._lock:
            EmbeddingFactory._instances.clear()
        
        logger.info("Gestor de modelos de embeddings detenido y recursos liberados")
