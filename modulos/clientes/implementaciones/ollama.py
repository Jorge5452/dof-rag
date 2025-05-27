import os
import json
import requests
from typing import List, Dict, Any, Optional
from ..AbstractClient import IAClient

class OllamaClient(IAClient):
    """
    Ollama AI client implementation for local LLM inference.
    """
    
    def __init__(self, base_url: str = None, model_name: str = None, **kwargs):
        """
        Initialize the Ollama client.
        
        Args:
            base_url (str, optional): Base URL for the Ollama API. If None, will try to load from environment variable.
            model_name (str, optional): Name of the model to use. If None, obtains from config.
            **kwargs: Additional parameters for the Ollama client.
                embedding_model (str, optional): Name of the embedding model to use.
                base_url_env (str, optional): Name of the environment variable to use for base URL.
                timeout (int, optional): Timeout in seconds for the API call.
        """
        # Obtener configuración
        from config import config
        ollama_config = config.get_ai_client_config().get('ollama', {})
        general_config = config.get_ai_client_config().get('general', {})
        
        # Set default environment variable name for API URL
        api_url_env = kwargs.get("api_url_env", ollama_config.get("api_url_env", "OLLAMA_API_URL"))
        
        # Get base URL, clean it if it's from environment variable
        if base_url:
            self.base_url = self._clean_value(base_url)
        else:
            # First try from api_url parameter, then from environment variable, then use default
            api_url = kwargs.get("api_url") or os.getenv(api_url_env)
            if api_url:
                self.base_url = self._clean_value(api_url)
            else:
                # Usar valor de configuración o el valor por defecto
                self.base_url = ollama_config.get("api_url", "http://localhost:11434")
        
        # Ensure URL doesn't end with a slash
        if self.base_url.endswith('/'):
            self.base_url = self.base_url[:-1]
        
        # Log base URL configuration - show the correct endpoint being tested
        print(f"Ollama base URL configured: {self.base_url}")
        
        # Get model name with appropriate fallbacks
        default_model = ollama_config.get("model", "llama2")
        self.model_name = model_name or default_model
        
        # Get embedding model or use default (same as model_name if not specified)
        self.embedding_model = kwargs.get("embedding_model", ollama_config.get("embedding_model", self.model_name))
        
        # Set generation parameters
        self.temperature = kwargs.get("temperature", ollama_config.get("temperature", general_config.get("temperature", 0.7)))
        self.max_tokens = kwargs.get("max_tokens", ollama_config.get("max_tokens", general_config.get("max_tokens", 1024)))
        self.top_p = kwargs.get("top_p", ollama_config.get("top_p", general_config.get("top_p", 0.95)))
        self.stream = kwargs.get("stream", ollama_config.get("stream", general_config.get("stream", False)))
        
        # Request timeout
        self.timeout = kwargs.get("timeout", ollama_config.get("timeout", 60))
        
        # Validate connection to Ollama server and check if model is available
        try:
            # First, check if Ollama server is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if response.status_code != 200:
                raise ConnectionError(f"Failed to connect to Ollama server: HTTP {response.status_code} - {response.text}")
            
            # Check if the specified model is available
            available_models = response.json().get('models', [])
            model_names = [model['name'] for model in available_models]
            
            if self.model_name not in model_names:
                print(f"Warning: Model '{self.model_name}' not found in Ollama.")
                print(f"Available models: {', '.join(model_names) if model_names else 'None'}")
                if model_names:
                    print(f"To install the model, run: ollama pull {self.model_name}")
                else:
                    print("No models are installed. Please install a model first.")
                raise ConnectionError(f"Model '{self.model_name}' is not available in Ollama")
            else:
                print(f"✓ Model '{self.model_name}' is available in Ollama")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Failed to connect to Ollama server. Please ensure Ollama is running on http://localhost:11434")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama server: {str(e)}")
    
    def _clean_value(self, value: str) -> str:
        """
        Clean a value by removing whitespace and quotes for better accessibility.
        Supports both single and double quotes, and handles various edge cases.
        
        Args:
            value (str): The value to clean.
            
        Returns:
            str: The cleaned value.
        """
        if not value:
            return None
            
        # Remove leading/trailing whitespace
        cleaned_value = value.strip()
        
        # Remove surrounding quotes (both single and double)
        if len(cleaned_value) >= 2:
            # Check for double quotes
            if cleaned_value.startswith('"') and cleaned_value.endswith('"'):
                cleaned_value = cleaned_value[1:-1]
            # Check for single quotes
            elif cleaned_value.startswith("'") and cleaned_value.endswith("'"):
                cleaned_value = cleaned_value[1:-1]
        
        # Remove any remaining whitespace after quote removal
        cleaned_value = cleaned_value.strip()
        
        return cleaned_value if cleaned_value else None
    
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
    
    def generate_response(self, prompt: str, context: List[Dict[str, Any]] = None, **kwargs) -> str:
        """
        Generate a response using the Ollama model.
        
        Args:
            prompt (str): The user's query or prompt.
            context (List[Dict[str, Any]], optional): List of chunks containing context information.
            **kwargs: Additional parameters for the generation.
                temperature (float, optional): Controls randomness of output. Defaults to instance default.
                max_tokens (int, optional): Maximum number of tokens to generate. Defaults to instance default.
                top_p (float, optional): Nucleus sampling parameter. Defaults to instance default.
                stream (bool, optional): Whether to stream the response. Defaults to instance default.
        
        Returns:
            str: The generated response.
        """
        # Override generation parameters with kwargs if provided
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens) or kwargs.get("num_predict", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", 40)
        stream = kwargs.get("stream", self.stream)
        
        # Get system prompt from config.yaml always
        from config import config
        ai_config = config.get_ai_client_config()
        general_config = ai_config.get('general', {})
        system_prompt = general_config.get('system_prompt', 'You are a helpful assistant.')
        
        # Build unified prompt combining system prompt, context and query
        unified_prompt = self._build_unified_prompt(system_prompt, context, prompt)
        
        # Use chat API with only user message
        endpoint = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": unified_prompt}
            ],
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_predict": max_tokens
            }
        }
        
        # Send request to Ollama
        try:
            response = requests.post(endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise Exception("Connection error: Ollama server is not reachable. Please ensure Ollama is running.")
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                raise Exception(f"Model '{self.model_name}' not found on Ollama server. Please install it with: ollama pull {self.model_name}")
            else:
                raise Exception(f"Ollama API HTTP error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API error: {str(e)}")
        
        # Handle streaming or non-streaming response
        if stream:
            # For streaming, we need to accumulate the response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    if "message" in json_response and "content" in json_response["message"]:
                        full_response += json_response["message"]["content"]
                    elif "response" in json_response:
                        full_response += json_response["response"]
                    # Stop if done
                    if json_response.get("done", False):
                        break
            return full_response
        else:
            # For non-streaming, extract response from JSON
            json_response = response.json()
            if "message" in json_response and "content" in json_response["message"]:
                return json_response["message"]["content"]
            elif "response" in json_response:  # Generate API response
                return json_response["response"]
            else:
                raise Exception(f"Unexpected Ollama API response: {json_response}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding vector for the provided text using Ollama's embedding model.
        
        Args:
            text (str): The text to create an embedding for.
            
        Returns:
            List[float]: The embedding vector.
        """
        endpoint = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.embedding_model,
            "prompt": text
        }
        
        try:
            response = requests.post(endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise Exception("Connection error: Ollama server is not reachable. Please ensure Ollama is running.")
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                raise Exception(f"Embedding model '{self.embedding_model}' not found on Ollama server. Please install it with: ollama pull {self.embedding_model}")
            else:
                raise Exception(f"Ollama embeddings API HTTP error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama embeddings API error: {str(e)}")
        
        json_response = response.json()
        if "embedding" in json_response:
            return json_response["embedding"]
        else:
            raise Exception(f"Unexpected Ollama embeddings API response: {json_response}")