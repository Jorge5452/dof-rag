import os
import threading
from google import genai
from google.genai import types
from typing import List, Dict, Any, Optional, Generator, Union
from ..AbstractClient import IAClient
from dotenv import load_dotenv

# Lock para prevenir problemas con llamadas concurrentes a la API
api_lock = threading.Lock()

class GeminiClient(IAClient):
    """
    Gemini AI client implementation using Google's official google-genai SDK.
    Optimizado para el flujo RAG: recibir contexto y pregunta, generar respuesta.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_key_env: Optional[str] = None,
        response_mime_type: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Gemini client using the official google-genai library.
        
        Args:
            api_key (str, optional): API key for the Gemini API. If None, will try to load from environment.
            model (str, optional): Name of the model to use.
            api_key_env (str, optional): Name of the environment variable to use for API key.
            response_mime_type (str, optional): MIME type for response format.
            **kwargs: Additional parameters for the Gemini client.
        """
        load_dotenv()
        
        # Get global and Gemini-specific configuration
        from config import config
        general_config = config.get_ai_client_config().get('general', {})
        gemini_config = config.get_ai_client_config().get('gemini', {})
        
        # Set API key environment variable name from config or kwargs
        self.api_key_env = api_key_env or gemini_config.get('api_key_env', 'GEMINI_API_KEY')
        
        # Get API key with correct priority: direct param > env var
        self.api_key = api_key
        if not self.api_key:
            self.api_key = os.getenv(self.api_key_env)
            if not self.api_key:
                raise ValueError(f"Gemini API key not found. Set {self.api_key_env} environment variable or pass api_key parameter.")
        
        # Clean the API key - remove quotes if present for accessibility
        self.api_key = self._clean_api_key(self.api_key)
        
        # Simplificación de log de API key
        print(f"Gemini API key configurada correctamente")
        
        # Get model name from params, Gemini config, or default
        default_model = gemini_config.get('model', 'gemini-2.0-flash')
        self.model_name = model or gemini_config.get('model', default_model)
        print(f"Modelo Gemini: {self.model_name}")
        
        # Set generation parameters using global config values
        self.temperature = kwargs.get("temperature", general_config.get('temperature', 0.3))
        self.max_tokens = kwargs.get("max_tokens", general_config.get('max_tokens', 2048))
        self.top_p = kwargs.get("top_p", general_config.get('top_p', 0.85))
        self.top_k = kwargs.get("top_k", general_config.get('top_k', 40))
        self.stream = kwargs.get("stream", general_config.get('stream', False))
        self.response_mime_type = response_mime_type or general_config.get('response_mime_type', 'text/plain')
        
        # Get formatting configurations
        self.context_format = general_config.get('context_format', 'fragments')
        self.instruction_style = general_config.get('instruction_style', 'detailed')
        
        # Store last used context for access
        self.last_used_context = None
        
        # Store last response text for backup
        self.last_response_text = None
        
        # Initialize Google's GenAI client using the new official library
        try:
            # Initialize the client with the new google-genai library
            self.client = genai.Client(api_key=self.api_key)
            print(f"Cliente Gemini inicializado correctamente")
            
        except ImportError as ie:
            raise ImportError("Google GenAI package not installed. Please install it with 'pip install google-genai'.")
        except Exception as e:
            print(f"Error al inicializar cliente Gemini: {str(e)}")
            raise ValueError(f"Error initializing Gemini client: {str(e)}")
    
    def _clean_api_key(self, api_key: str) -> str:
        """
        Clean the API key by removing quotes and whitespace for better accessibility.
        Supports both single and double quotes, and handles various edge cases.
        
        Args:
            api_key: Raw API key from environment or parameter
            
        Returns:
            str: Cleaned API key
        """
        if not api_key:
            return api_key
        
        # Remove leading and trailing whitespace
        cleaned_key = api_key.strip()
        
        # Remove surrounding quotes (both single and double)
        if len(cleaned_key) >= 2:
            # Check for double quotes
            if cleaned_key.startswith('"') and cleaned_key.endswith('"'):
                cleaned_key = cleaned_key[1:-1]
            # Check for single quotes
            elif cleaned_key.startswith("'") and cleaned_key.endswith("'"):
                cleaned_key = cleaned_key[1:-1]
        
        # Remove any remaining whitespace after quote removal
        cleaned_key = cleaned_key.strip()
        
        return cleaned_key

    def _build_unified_prompt(self, system_prompt: str, context: List[Dict[str, Any]], query: str) -> str:
        """
        Builds a unified prompt combining system prompt, context and user query.
        
        Args:
            system_prompt: System instructions for the model
            context: List of context chunks
            query: User's query
            
        Returns:
            str: Unified prompt for the model
        """
        # Start with system prompt
        unified_prompt = f"{system_prompt}\n\n"
        
        # Add context if available
        if context:
            unified_prompt += "CONTEXTO DE DOCUMENTOS:\n"
            for i, chunk in enumerate(context):
                chunk_text = chunk.get('text', '')
                chunk_header = chunk.get('header', '')
                
                if chunk_header:
                    unified_prompt += f"Fragmento {i+1} - {chunk_header}:\n{chunk_text}\n\n"
                else:
                    unified_prompt += f"Fragmento {i+1}:\n{chunk_text}\n\n"
            
            unified_prompt += "INSTRUCCIONES:\n"
            unified_prompt += "Utiliza únicamente la información del contexto anterior para responder la siguiente pregunta. "
            unified_prompt += "Si la información no es suficiente, indícalo claramente.\n\n"
        else:
            unified_prompt += "NOTA: No se ha proporcionado contexto específico para esta consulta.\n\n"
        
        # Add user query
        unified_prompt += f"PREGUNTA DEL USUARIO:\n{query}\n\n"
        unified_prompt += "RESPUESTA:"
        
        return unified_prompt
    
    def generate_response(
        self,
        prompt: str,
        context: Optional[Union[str, List[Dict[str, Any]]]] = None,
        stream: Optional[bool] = None,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response from the model using the new google-genai library.
        
        Args:
            prompt (str): The prompt to generate a response for.
            context (Union[str, List[Dict[str, Any]]], optional): Context information to include in the prompt.
            stream (bool, optional): Whether to stream the response.
            **kwargs: Additional parameters for the generation.
        
        Returns:
            Union[str, Generator[str, None, None]]: The generated response or a generator for streaming.
        """
        try:
            # Store the context for later access
            self.last_used_context = context
            
            # Get system prompt from config.yaml always
            from config import config
            ai_config = config.get_ai_client_config()
            general_config = ai_config.get('general', {})
            system_prompt = general_config.get('system_prompt', 'You are a helpful assistant.')
            
            print(f"Generando respuesta con modelo {self.model_name}...")
            
            # Set stream parameter
            stream = stream if stream is not None else self.stream
            
            # Build unified prompt combining system prompt, context and query
            unified_prompt = self._build_unified_prompt(system_prompt, context, prompt)
            
            # Prepare contents with the new structure - only user message
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=unified_prompt)]
                )
            ]
            
            # Get generation config, adjusting temperature if necessary
            use_temperature = kwargs.get("temperature", self.temperature)
                
            # Configure generation parameters using new types.GenerateContentConfig
            generation_config = types.GenerateContentConfig(
                temperature=use_temperature,
                top_p=kwargs.get("top_p", self.top_p),
                top_k=kwargs.get("top_k", self.top_k),
                max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
                response_mime_type=self.response_mime_type,
            )
            
            # Generate response with thread lock to prevent concurrent API issues
            with api_lock:
                if stream:
                    # Use streaming generation
                    response_chunks = self.client.models.generate_content_stream(
                        model=self.model_name,
                        contents=contents,
                        config=generation_config
                    )
                    
                    # Accumulate streaming response
                    full_response = ""
                    for chunk in response_chunks:
                        if hasattr(chunk, 'text') and chunk.text:
                            full_response += chunk.text
                    
                    return full_response
                else:
                    # Use non-streaming generation
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=generation_config
                    )
                    
                    if hasattr(response, 'text'):
                        return response.text
                    else:
                        return str(response)
                
        except Exception as e:
            error_msg = f"Error al generar respuesta: {str(e)}"
            return error_msg
    
    def get_last_context(self) -> Union[str, List[Dict[str, Any]], None]:
        """
        Get the context that was used for the last response generation.
        
        Returns:
            Union[str, List[Dict[str, Any]], None]: Context used in the last response.
        """
        return self.last_used_context
    
    # Implementación requerida por la interfaz IAClient
    # Método vacío ya que Gemini no necesita generar embeddings en este flujo
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings for a text. 
        
        Note: Esta implementación es un stub para cumplir con la interfaz IAClient.
        En el flujo RAG actual, este método no se utiliza para Gemini.
        
        Args:
            text (str): The text to get embeddings for.
            
        Returns:
            List[float]: The embedding vector (empty for Gemini implementation).
        """
        print("ADVERTENCIA: El método get_embedding no está implementado para GeminiClient en el flujo RAG actual.")
        return []