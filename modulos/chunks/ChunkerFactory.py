from typing import Dict, Any, Optional
import logging

from config import config
from modulos.chunks.ChunkAbstract import ChunkAbstract
from modulos.chunks.implementaciones.character_chunker import CharacterChunker
from modulos.chunks.implementaciones.token_chunker import TokenChunker
from modulos.chunks.implementaciones.context_chunker import ContextChunker

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChunkerFactory:
    """
    Factory para crear instancias de chunkers.
    Implementa el patrón Factory para seleccionar la implementación adecuada 
    según el tipo de chunking requerido.
    """
    
    _instances = {}
    
    @staticmethod
    def get_chunker(
        chunker_type: Optional[str] = None,
        embedding_model = None
    ) -> ChunkAbstract:
        """
        Retorna una instancia del chunker adecuado según el chunker_type.
        Si ya existe una instancia para ese tipo, retorna la misma instancia (patrón Singleton).
        
        Parámetros:
            chunker_type (str, opcional): Tipo de chunker ('character', 'token', 'context').
                                         Si es None, se toma del archivo de configuración.
            embedding_model: Modelo de embeddings inicializado. Si se proporciona, se asigna al chunker.
        
        Retorna:
            ChunkAbstract: Instancia de una clase que implementa ChunkAbstract.
            
        Raises:
            ValueError: Si el tipo de chunker no está soportado.
        """
        # Si no se proporciona tipo, obtenerlo de la configuración
        chunks_config = config.get_chunks_config()
        if chunker_type is None:
            chunker_type = chunks_config.get("method", "context")
        
        # Clave única para este tipo de chunker
        instance_key = f"chunker:{chunker_type}"
        
        # Si ya existe una instancia para este tipo, retornarla
        if instance_key in ChunkerFactory._instances:
            # Si se proporciona un modelo de embeddings, actualizarlo
            if embedding_model is not None:
                ChunkerFactory._instances[instance_key].set_embedding_model(embedding_model)
            return ChunkerFactory._instances[instance_key]
        
        # Crear una nueva instancia según el tipo
        if chunker_type.lower() == 'character':
            chunker = CharacterChunker(embedding_model)
        elif chunker_type.lower() == 'token':
            chunker = TokenChunker(embedding_model)
        elif chunker_type.lower() == 'context':
            chunker = ContextChunker(embedding_model)
        else:
            raise ValueError(f"Tipo de chunker no soportado: {chunker_type}")
        
        # Almacenar la instancia para futuras referencias
        ChunkerFactory._instances[instance_key] = chunker
        
        return chunker
    
    @staticmethod
    def reset_instances() -> None:
        """
        Reinicia todas las instancias de chunkers.
        Útil para tests o para liberar recursos.
        """
        ChunkerFactory._instances.clear()
        logger.info("Todas las instancias de chunkers han sido reiniciadas")
