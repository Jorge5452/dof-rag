from typing import Optional, Dict, Any
from .AbstractClient import IAClient
from config import Config

class ClientFactory:
    """
    Factory class for creating AI client instances.
    """
    
    @staticmethod
    def get_client(client_type: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> IAClient:
        """
        Get an instance of the specified AI client type.
        
        Args:
            client_type (str, optional): The type of AI client to create ('openai', 'gemini', 'ollama', etc.).
                                      If None, will use the type from config.
            api_key (str, optional): The API key for the service. If None, will use from config or env vars.
            **kwargs: Additional parameters specific to the client implementation.
            
        Returns:
            IAClient: An instance of the specified AI client.
            
        Raises:
            ValueError: If the specified client type is not supported.
        """
        # Get config singleton instance
        config = Config()
        
        # Get AI client configuration from config.yaml
        ai_config = config.get_ai_client_config()
        
        # If client_type not provided, use from config
        if client_type is None:
            client_type = ai_config.get("type", "openai")
        
        client_type = client_type.lower()
        
        # Get filtered configuration for the client type
        # This configuration already has general and specific parameters merged
        # and is filtered to only include parameters relevant to this client type
        params = config.get_specific_ai_config(client_type)
        
        # Override with kwargs (user provided parameters have highest priority)
        if kwargs:
            params.update(kwargs)
        
        # If api_key is provided, use it, otherwise it will be handled by the client classes
        if api_key:
            params["api_key"] = api_key
            
        # Get model name from config if not in kwargs
        if "model" in params and "model_name" not in params:
            params["model_name"] = params.pop("model")
        
        # Create and return the appropriate client instance
        if client_type == 'openai':
            from modulos.clientes.implementaciones.openai import OpenAIClient
            return OpenAIClient(**params)
        elif client_type == 'gemini':
            from modulos.clientes.implementaciones.gemini import GeminiClient
            return GeminiClient(**params)
        elif client_type == 'ollama':
            from modulos.clientes.implementaciones.ollama import OllamaClient
            return OllamaClient(**params)
        else:
            raise ValueError(f"Cliente de IA no soportado: {client_type}") 