import logging
from typing import Optional

from config import config
from modulos.embeddings.embeddings_manager import EmbeddingManager

# Configurar logging
logger = logging.getLogger(__name__)

class EmbeddingFactory:
    """
    Factory para crear y gestionar instancias de modelos de embeddings.
    Implementa el patrón Factory y Singleton combinados.
    """
    
    _instances = {}
    
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
        
        # Si ya existe una instancia para este modelo, retornarla
        if instance_key in EmbeddingFactory._instances:
            return EmbeddingFactory._instances[instance_key]
        
        # Crear una nueva instancia
        manager = EmbeddingManager(model_type)
        
        # Almacenar la instancia para futuras referencias
        EmbeddingFactory._instances[instance_key] = manager
        
        return manager
    
    @staticmethod
    def reset_instances() -> None:
        """
        Reinicia todas las instancias de gestores de embeddings.
        Útil para tests o para liberar recursos.
        """
        EmbeddingFactory._instances.clear()
        logger.info("Todas las instancias de gestores de embeddings han sido reiniciadas")
