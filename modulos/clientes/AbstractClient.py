from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class IAClient(ABC):
    """
    Abstract base class for AI clients.
    All AI client implementations should inherit from this class.
    """
    
    @abstractmethod
    def __init__(self, api_key: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initialize the AI client with an API key and additional parameters.
        
        Args:
            api_key (Optional[str]): API key for the service. If None, will try to load from environment variables.
            **kwargs: Additional parameters specific to the implementation.
                model_name (str, optional): Name of the model to use.
                timeout (int, optional): Timeout in seconds for API calls.
                temperature (float, optional): Default temperature for generation.
        """
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, context: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
        """
        Generate a response based on the prompt and optional context.
        
        Args:
            prompt (str): The user's query or prompt.
            context (Optional[List[Dict[str, Any]]]): List of chunks containing context information.
            **kwargs: Additional parameters specific to the implementation.
                temperature (float, optional): Controls randomness of output.
                max_tokens (int, optional): Maximum number of tokens to generate.
                top_p (float, optional): Nucleus sampling parameter.
                stream (bool, optional): Whether to stream the response.
            
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
            List[float]: The embedding vector as a list of floating point numbers.
            
        Raises:
            Exception: If the embedding generation fails or is not supported.
        """
        pass