"""
Embedding manager for the RAG system.

This module is responsible for loading and managing embedding models,
using sentence-transformers to automatically handle the models.
"""
import logging
from typing import List
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Manager for embedding models.
    Handles loading, management and generation of embeddings using
    various models such as ModernBERT, CDE, E5, etc.
    """
    
    def __init__(self, model_type: str = None):
        """
        Initializes the EmbeddingManager with a specific model type.
        
        Args:
            model_type: Type of model to use. If None, the configuration value is used.
        """
        # Get configuration if no specific model is provided
        if model_type is None:
            from config import config
            embedding_config = config.get_embedding_config()
            model_type = embedding_config.get("model", "modernbert")
        
        self.model_type = model_type
        self._model = None
        self.model_name = None
        self._embedding_dim = None
        
        # Initialize the model (lazy loading)
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initializes the embedding model according to the specified type.
        """
        # Mapping of model types to actual model names
        model_mapping = {
            "modernbert": "nomic-ai/modernbert-embed-base",
            "bge": "BAAI/bge-large-en-v1.5",
            "e5": "intfloat/e5-large-v2",
            "gte": "thenlper/gte-large",
            "instructor": "hkunlp/instructor-large",
            "minilm": "sentence-transformers/all-MiniLM-L6-v2",
            "mpnet": "sentence-transformers/all-mpnet-base-v2"
        }
        
        # Get the actual model name
        self.model_name = model_mapping.get(self.model_type.lower(), self.model_type)
    
    def load_model(self):
        """
        Loads the embedding model if not already loaded.
        """
        if self._model is None:
            try:
                # Import SentenceTransformer here to avoid early import
                from sentence_transformers import SentenceTransformer
                
                # Load the model
                self._model = SentenceTransformer(self.model_name)
                
                # Get model dimensions
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                raise
    
    @property
    def model(self):
        """
        Access to the model with automatic loading if not loaded.
        
        Returns:
            The loaded embedding model
        """
        if self._model is None:
            self.load_model()
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value
    
    @property
    def embedding_dim(self) -> int:
        """
        Gets the embedding dimension of the current model.
        
        Returns:
            Embedding dimension as integer
        """
        if self._embedding_dim is None:
            # Load model if necessary to get dimensions
            _ = self.model
        return self._embedding_dim
    
    def get_dimensions(self) -> int:
        """
        Gets the embedding dimension of the current model.
        
        Returns:
            Embedding dimension as integer
        """
        return self.embedding_dim
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generates the embedding for a given text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector as numpy array
        """
        # Ensure the model is loaded
        model = self.model
        
        # Generate embedding
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            Matrix of embeddings as numpy array
        """
        # Ensure the model is loaded
        model = self.model
        
        # Generate embeddings in batch
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        return embeddings
    
    def get_document_embedding(self, header: str = None, text: str = None) -> List[float]:
        """
        Generates embeddings for documents or chunks by combining header and text.
        
        Args:
            header: Document or chunk header (can be None)
            text: Main text of the document or chunk
            
        Returns:
            List of float values representing the embedding
        """
        # Combine header and text to generate the embedding
        if header and header.strip():
            full_text = f"{header} - {text}"
        else:
            full_text = text
        
        # Generate embedding
        embedding = self.model.encode(full_text)
        
        # Convert to list if it's a numpy array
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        
        return embedding
    
    def get_query_embedding(self, query: str) -> List[float]:
        """
        Generates embeddings for queries.
        
        Args:
            query: Query text
            
        Returns:
            List of float values representing the embedding
        """
        # Some models require special formatting for queries (e.g., E5)
        if self.model_type == "e5-small" and hasattr(self, 'model_config') and self.model_config.get("prefix_queries", False):
            query = f"query: {query}"
        
        # Generate embedding
        embedding = self.model.encode(query)
        
        # Convert to list if it's a numpy array
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        
        return embedding
