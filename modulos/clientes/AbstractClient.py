from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class IAClient(ABC):
    """
    Abstract base class for AI clients.
    All AI client implementations should inherit from this class.
    """
    
    @abstractmethod
    def __init__(self, api_key: str = None, **kwargs):
        """
        Initialize the AI client with an API key and additional parameters.
        
        Args:
            api_key (str, optional): API key for the service. If None, will try to load from environment variables.
            **kwargs: Additional parameters specific to the implementation.
        """
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, context: List[Dict[str, Any]] = None, **kwargs) -> str:
        """
        Generate a response based on the prompt and optional context.
        
        Args:
            prompt (str): The user's query or prompt.
            context (List[Dict[str, Any]], optional): List of chunks containing context information.
            **kwargs: Additional parameters specific to the implementation.
            
        Returns:
            str: The generated response.
        """
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding vector for the provided text.
        
        Args:
            text (str): The text to create an embedding for.
            
        Returns:
            List[float]: The embedding vector.
        """
        pass