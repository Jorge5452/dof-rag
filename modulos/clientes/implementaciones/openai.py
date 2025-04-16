import os
import openai
from typing import List, Dict, Any, Optional
from ..AbstractClient import IAClient

class OpenAIClient(IAClient):
    """
    OpenAI client implementation using OpenAI's Python SDK.
    """
    
    def __init__(self, api_key: str = None, model_name: str = "gpt-3.5-turbo", **kwargs):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key (str, optional): OpenAI API key. If None, will try to load from environment variable.
            model_name (str, optional): Name of the model to use. Defaults to "gpt-3.5-turbo".
            **kwargs: Additional parameters for the OpenAI client.
                embedding_model (str, optional): Name of the embedding model to use.
                api_key_env (str, optional): Name of the environment variable to use for API key.
                org_env (str, optional): Name of the environment variable to use for organization.
                api_base_env (str, optional): Name of the environment variable to use for API base.
                timeout (int, optional): Timeout in seconds for the API call.
        """
        # Set default environment variable names
        api_key_env = kwargs.get("api_key_env", "OPENAI_API_KEY")
        org_env = kwargs.get("org_env", "OPENAI_ORGANIZATION")
        api_base_env = kwargs.get("api_base_env", "OPENAI_API_BASE")
        
        # Get API key, clean it if it's from environment variable
        if api_key:
            self.api_key = self._clean_api_key(api_key)
        else:
            env_key = os.getenv(api_key_env, "")
            # Clean the key
            self.api_key = self._clean_api_key(env_key)
        
        # Verify API key is valid
        if not self.api_key:
            raise ValueError(f"OpenAI API key not provided and not found in environment variable {api_key_env}")
        
        # Log API key configuration (masked for security)
        masked_key = self.api_key[:4] + "*" * 10 + self.api_key[-4:] if len(self.api_key) > 8 else "***"
        print(f"OpenAI API key configured: {masked_key}")
        
        # Get model name with appropriate fallbacks
        self.model_name = model_name
        
        # Get embedding model or use default
        self.embedding_model = kwargs.get("embedding_model", "text-embedding-3-small")
        
        # Get organization if available
        org = kwargs.get("organization", os.getenv(org_env))
        self.organization = self._clean_api_key(org)
        
        # Get API base URL if available
        api_base = kwargs.get("api_base", os.getenv(api_base_env))
        self.api_base = self._clean_api_key(api_base)
        
        # Get timeout if available
        self.timeout = kwargs.get("timeout", 30)
        
        # Initialize OpenAI client
        client_args = {"api_key": self.api_key, "timeout": self.timeout}
        
        if self.organization:
            client_args["organization"] = self.organization
        
        if self.api_base:
            client_args["base_url"] = self.api_base
        
        self.client = openai.OpenAI(**client_args)
        
        # Store generation parameters
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.top_p = kwargs.get("top_p", 0.95)
        self.stream = kwargs.get("stream", False)
        self.system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
    
    def _clean_api_key(self, api_key: str) -> str:
        """
        Clean an API key by removing whitespace and quotes.
        
        Args:
            api_key (str): The API key to clean.
            
        Returns:
            str: The cleaned API key.
        """
        if not api_key:
            return None
            
        # Remove leading/trailing whitespace
        cleaned_key = api_key.strip()
        
        # Remove quotes if present (both single and double quotes)
        if (cleaned_key.startswith('"') and cleaned_key.endswith('"')) or \
           (cleaned_key.startswith("'") and cleaned_key.endswith("'")):
            cleaned_key = cleaned_key[1:-1]
            
        # Additional cleaning - remove any remaining quotes and whitespace
        cleaned_key = cleaned_key.replace('"', '').replace("'", '').strip()
        
        return cleaned_key if cleaned_key else None
    
    def generate_response(self, prompt: str, context: List[Dict[str, Any]] = None, **kwargs) -> str:
        """
        Generate a response using the OpenAI model.
        
        Args:
            prompt (str): The user's query or prompt.
            context (List[Dict[str, Any]], optional): List of chunks containing context information.
            **kwargs: Additional parameters for the generation.
                temperature (float, optional): Controls randomness of output. Defaults to instance default.
                max_tokens (int, optional): Maximum number of tokens to generate. Defaults to instance default.
                top_p (float, optional): Nucleus sampling parameter. Defaults to instance default.
                system_message (str, optional): Custom system message. Defaults to instance default.
                stream (bool, optional): Whether to stream the response. Defaults to False.
        
        Returns:
            str: The generated response.
        """
        # Override generation parameters with kwargs if provided
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        frequency_penalty = kwargs.get("frequency_penalty", 0.0)
        presence_penalty = kwargs.get("presence_penalty", 0.0)
        system_message = kwargs.get("system_message") or kwargs.get("system_prompt", self.system_prompt)
        stream = kwargs.get("stream", False)
        
        # Prepare messages with context if provided
        messages = []
        
        # Add system message
        if context:
            context_text = "\n\n".join([f"Context {i+1}:\n{chunk['text']}" for i, chunk in enumerate(context)])
            messages.append({
                "role": "system", 
                "content": system_message
            })
            messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {prompt}"})
        else:
            messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=stream
        )
        
        # Handle streaming or non-streaming response
        if stream:
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
            return full_response
        else:
            return response.choices[0].message.content
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding vector for the provided text using OpenAI's embedding model.
        
        Args:
            text (str): The text to create an embedding for.
            
        Returns:
            List[float]: The embedding vector.
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        
        return response.data[0].embedding 