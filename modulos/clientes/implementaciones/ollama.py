import os
import json
import requests
from typing import List, Dict, Any, Optional
from ..AbstractClient import IAClient

class OllamaClient(IAClient):
    """
    Ollama AI client implementation for local LLM inference.
    """
    
    def __init__(self, base_url: str = None, model_name: str = "llama2", **kwargs):
        """
        Initialize the Ollama client.
        
        Args:
            base_url (str, optional): Base URL for the Ollama API. If None, will try to load from environment variable.
            model_name (str, optional): Name of the model to use. Defaults to "llama2".
            **kwargs: Additional parameters for the Ollama client.
                embedding_model (str, optional): Name of the embedding model to use.
                base_url_env (str, optional): Name of the environment variable to use for base URL.
                timeout (int, optional): Timeout in seconds for the API call.
                system_prompt (str, optional): Default system prompt.
        """
        # Set default environment variable name for API URL
        api_url_env = kwargs.get("api_url_env", "OLLAMA_API_URL")
        
        # Get base URL, clean it if it's from environment variable
        if base_url:
            self.base_url = self._clean_value(base_url)
        else:
            # First try from api_url parameter, then from environment variable, then use default
            api_url = kwargs.get("api_url") or os.getenv(api_url_env)
            if api_url:
                self.base_url = self._clean_value(api_url)
            else:
                self.base_url = "http://localhost:11434"  # Default Ollama URL
        
        # Ensure URL doesn't end with a slash
        if self.base_url.endswith('/'):
            self.base_url = self.base_url[:-1]
        
        # Log base URL configuration
        print(f"Ollama base URL configured: {self.base_url}")
        
        # Get model name with appropriate fallbacks
        self.model_name = model_name
        
        # Get embedding model or use default (same as model_name if not specified)
        self.embedding_model = kwargs.get("embedding_model", self.model_name)
        
        # Set generation parameters
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.top_p = kwargs.get("top_p", 0.95)
        self.stream = kwargs.get("stream", False)
        self.system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
        
        # Request timeout
        self.timeout = kwargs.get("timeout", 60)
        
        # Validate connection to Ollama server
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if response.status_code != 200:
                raise ConnectionError(f"Failed to connect to Ollama server: {response.text}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama server: {str(e)}")
    
    def _clean_value(self, value: str) -> str:
        """
        Clean a value by removing whitespace and quotes.
        
        Args:
            value (str): The value to clean.
            
        Returns:
            str: The cleaned value.
        """
        if not value:
            return None
            
        # Remove leading/trailing whitespace
        cleaned_value = value.strip()
        
        # Remove quotes if present (both single and double quotes)
        if (cleaned_value.startswith('"') and cleaned_value.endswith('"')) or \
           (cleaned_value.startswith("'") and cleaned_value.endswith("'")):
            cleaned_value = cleaned_value[1:-1]
            
        # Additional cleaning - remove any remaining quotes and whitespace
        cleaned_value = cleaned_value.replace('"', '').replace("'", '').strip()
        
        return cleaned_value if cleaned_value else None
    
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
                system_prompt (str, optional): System prompt to prepend. Defaults to instance default.
                stream (bool, optional): Whether to stream the response. Defaults to instance default.
        
        Returns:
            str: The generated response.
        """
        # Override generation parameters with kwargs if provided
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens) or kwargs.get("num_predict", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", 40)
        system_prompt = kwargs.get("system_prompt", self.system_prompt)
        stream = kwargs.get("stream", self.stream)
        
        # Prepare system prompt with context if provided
        if context:
            context_text = "\n\n".join([f"Context {i+1}:\n{chunk['text']}" for i, chunk in enumerate(context)])
            custom_system_prompt = (
                f"{system_prompt}\n\n"
                f"Use the following contexts to answer the question at the end. "
                f"If you don't know the answer or the context doesn't contain relevant information, say so.\n\n"
                f"{context_text}"
            )
        else:
            custom_system_prompt = system_prompt
        
        # Use chat API 
        endpoint = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": custom_system_prompt},
                {"role": "user", "content": prompt}
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
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama embeddings API error: {str(e)}")
        
        json_response = response.json()
        if "embedding" in json_response:
            return json_response["embedding"]
        else:
            raise Exception(f"Unexpected Ollama embeddings API response: {json_response}") 