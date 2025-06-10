from typing import Dict, Any, Optional
import logging

from config import config
from modulos.chunks.ChunkAbstract import ChunkAbstract
from modulos.chunks.implementaciones.character_chunker import CharacterChunker
from modulos.chunks.implementaciones.token_chunker import TokenChunker
from modulos.chunks.implementaciones.context_chunker import ContextChunker
from modulos.chunks.implementaciones.page_chunker import PageChunker

# Configure logging
logger = logging.getLogger(__name__)

class ChunkerFactory:
    """
    Factory for creating chunker instances.
    Implements the Factory pattern to select the appropriate implementation
    based on the required chunking type, and the Singleton pattern to
    manage unique chunker instances.
    """
    
    # Dictionary to store created instances (Singleton)
    _instances = {}
    
    # Mapping of chunker types to their classes
    _chunker_classes = {
        'character': CharacterChunker,
        'token': TokenChunker,
        'context': ContextChunker,
        'page': PageChunker
    }
    
    def __init__(self) -> None:
        """
        Initializes the ChunkerFactory.
        """
        pass
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Loads the chunks configuration.
        
        Returns:
            Chunks configuration
        """
        return config.get_chunks_config()
    
    def get_chunker(self, chunker_type: Optional[str] = None, embedding_model: Optional[Any] = None) -> ChunkAbstract:
        """
        Gets a chunker instance of the specified type.
        
        Args:
            chunker_type: Type of chunker ('character', 'token', 'context', 'page'). If None, the config value is used.
            embedding_model: Embedding model to use (optional)
            
        Returns:
            ChunkAbstract instance
            
        Raises:
            ValueError: If the chunker type is not valid
        """
        # Load configuration if no specific type is provided
        if chunker_type is None:
            chunks_config = self._load_config()
            chunker_type = chunks_config.get("method", "character")
        
        # Normalize the chunker type
        chunker_type = chunker_type.lower()
        
        # Verify that the implementation exists
        if chunker_type not in self._chunker_classes:
            raise ValueError(f"Invalid chunker type: {chunker_type}. Available options: {list(self._chunker_classes.keys())}")
        
        # Get the chunker class
        chunker_class = self._chunker_classes[chunker_type]
        
        # Check if an instance already exists
        instance_key = f"{chunker_type}:{id(embedding_model) if embedding_model else 'default'}"
        
        if instance_key in self._instances:
            return self._instances[instance_key]
        
        # Create new instance
        chunker_instance = chunker_class(embedding_model=embedding_model)
        
        # Store the instance for reuse
        self._instances[instance_key] = chunker_instance
        
        return chunker_instance
    
    @classmethod
    def reset_instances(cls) -> None:
        """
        Resets all chunker instances.
        Useful for testing or to free resources.
        """
        cls._instances.clear()
