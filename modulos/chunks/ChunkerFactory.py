from typing import Dict, Any, Optional
import logging

from config import config
from modulos.chunks.ChunkAbstract import ChunkAbstract
from modulos.chunks.implementaciones.character_chunker import CharacterChunker
from modulos.chunks.implementaciones.token_chunker import TokenChunker
from modulos.chunks.implementaciones.context_chunker import ContextChunker
from modulos.chunks.implementaciones.page_chunker import PageChunker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChunkerFactory:
    """
    Factory for creating chunker instances.
    Implements the Factory pattern to select the appropriate implementation
    based on the required chunking type.
    """
    
    _instances = {}
    
    @staticmethod
    def get_chunker(
        chunker_type: Optional[str] = None,
        embedding_model = None
    ) -> ChunkAbstract:
        """
        Returns an instance of the appropriate chunker based on the chunker_type.
        If an instance already exists for that type, it returns the same instance (Singleton pattern).
        
        Parameters:
            chunker_type (str, optional): Chunker type ('character', 'token', 'context', 'page').
                                         If None, it's taken from the configuration file.
            embedding_model: Initialized embedding model. If provided, it's assigned to the chunker.
        
        Returns:
            ChunkAbstract: Instance of a class that implements ChunkAbstract.
            
        Raises:
            ValueError: If the chunker type is not supported.
        """
        # If no type is provided, get it from the configuration
        chunks_config = config.get_chunks_config()
        if chunker_type is None:
            chunker_type = chunks_config.get("method", "context")
        
        # Unique key for this type of chunker
        instance_key = f"chunker:{chunker_type}"
        
        # If an instance already exists for this type, return it
        if instance_key in ChunkerFactory._instances:
            # If an embedding model is provided, update it
            if embedding_model is not None:
                ChunkerFactory._instances[instance_key].set_embedding_model(embedding_model)
            return ChunkerFactory._instances[instance_key]
        
        # Create a new instance based on the type
        if chunker_type.lower() == 'character':
            chunker = CharacterChunker(embedding_model)
        elif chunker_type.lower() == 'token':
            chunker = TokenChunker(embedding_model)
        elif chunker_type.lower() == 'context':
            chunker = ContextChunker(embedding_model)
        elif chunker_type.lower() == 'page':
            chunker = PageChunker(embedding_model)
        else:
            raise ValueError(f"Unsupported chunker type: {chunker_type}")
        
        # Store the instance for future references
        ChunkerFactory._instances[instance_key] = chunker
        
        return chunker
    
    @staticmethod
    def reset_instances() -> None:
        """
        Resets all chunker instances.
        Useful for tests or to free resources.
        """
        ChunkerFactory._instances.clear()
        logger.info("All chunker instances have been reset")
