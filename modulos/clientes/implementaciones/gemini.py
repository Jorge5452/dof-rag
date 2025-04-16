import os
from google import genai
from google.genai import types
from typing import List, Dict, Any, Optional, Generator, Union
from ..AbstractClient import IAClient
from config import config
from dotenv import load_dotenv
import threading

# Lock para prevenir problemas con llamadas concurrentes a la API
api_lock = threading.Lock()

class GeminiClient(IAClient):
    """
    Gemini AI client implementation using Google's Generative AI SDK.
    Optimizado para el flujo RAG: recibir contexto y pregunta, generar respuesta.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_key_env: Optional[str] = "GEMINI_API_KEY",
        response_mime_type: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Gemini client, optimized for RAG workflow.
        
        Args:
            api_key (str, optional): API key for the Gemini API. If None, will try to load from environment.
            model (str, optional): Name of the model to use.
            api_key_env (str, optional): Name of the environment variable to use for API key.
            response_mime_type (str, optional): MIME type for response format.
            **kwargs: Additional parameters for the Gemini client.
                temperature (float, optional): Temperature for generation.
                max_tokens (int, optional): Maximum number of tokens to generate.
                top_p (float, optional): Top p sampling parameter.
                top_k (int, optional): Top k sampling parameter.
                stream (bool, optional): Whether to stream the response.
        """
        load_dotenv()
        
        # Get global and Gemini-specific configuration
        general_config = config.get_ai_client_config().get('general', {})
        gemini_config = config.get_ai_client_config().get('gemini', {})
        
        # Set API key environment variable name from config or kwargs
        self.api_key_env = api_key_env or gemini_config.get('api_key_env', 'GEMINI_API_KEY')
        
        # Get API key with correct priority: direct param > env var
        self.api_key = api_key
        if not self.api_key:
            self.api_key = os.getenv(self.api_key_env)
            if not self.api_key:
                raise ValueError(f"API key not provided. Please provide an API key or set the {self.api_key_env} environment variable.")
        
        # Remove quotes if present
        if self.api_key:
            if self.api_key.startswith('"') and self.api_key.endswith('"'):
                self.api_key = self.api_key[1:-1]
            elif self.api_key.startswith("'") and self.api_key.endswith("'"):
                self.api_key = self.api_key[1:-1]
        
        # Simplificación de log de API key
        print(f"Gemini API key configurada correctamente")
        
        # Get model name from params, Gemini config, or default
        self.model_name = model or gemini_config.get('model', 'gemini-1.5-flash')
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
        
        # Get system prompt from kwargs or global config
        system_prompt_default = "Eres un asistente amigable y conversacional especializado en responder preguntas sobre documentos. Tu objetivo es analizar el contexto proporcionado y extraer respuestas relevantes. Prioriza la información del contexto, pero puedes ofrecer respuestas útiles y conversacionales cuando la información específica no esté disponible. Responde siempre en español, usando un tono amable, cercano y profesional."
        
        # Prioridad: 1) kwargs, 2) config específica, 3) config general, 4) default
        self.system_prompt = kwargs.get("system_prompt") or gemini_config.get("system_prompt") or general_config.get("system_prompt", system_prompt_default)
        
        # Store last used context for access
        self.last_used_context = None
        
        # Store last response text for backup
        self.last_response_text = None
        
        # Initialize Google's GenerativeAI client
        try:
            # Initialize the client
            self.client = genai.Client(api_key=self.api_key)
            print(f"Cliente Gemini inicializado correctamente")
            
        except ImportError as ie:
            raise ImportError("Google Gemini API package not installed. Please install it with 'pip install google-generativeai'.")
        except Exception as e:
            import traceback
            print(f"Error al inicializar cliente Gemini: {str(e)}")
            raise ValueError(f"Error initializing Gemini client: {str(e)}")
    
    def get_generate_config(self, **kwargs) -> types.GenerateContentConfig:
        """
        Creates a configuration object for content generation.
        
        Args:
            **kwargs: Override parameters for this specific generation.
                
        Returns:
            Configuration object for Gemini content generation.
        """
        return types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            top_k=kwargs.get("top_k", self.top_k),
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
            response_mime_type=self.response_mime_type,
        )
    
    def generate_response(
        self,
        prompt: str,
        context: Optional[Union[str, List[Dict[str, Any]]]] = None,
        stream: Optional[bool] = None,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response from the model.
        
        Args:
            prompt (str): The prompt to generate a response for.
            context (Union[str, List[Dict[str, Any]]], optional): Context information to include in the prompt.
                Can be a string or a list of dictionaries with 'header' and 'text' keys.
            stream (bool, optional): Whether to stream the response.
            **kwargs: Additional parameters for the generation.
                temperature (float, optional): Temperature for generation.
                max_tokens (int, optional): Maximum number of tokens to generate.
                top_p (float, optional): Top p sampling parameter.
                top_k (int, optional): Top k sampling parameter.
        
        Returns:
            Union[str, Generator[str, None, None]]: The generated response or a generator for streaming.
        """
        try:
            # Store the context for later access
            self.last_used_context = context
            
            # Get system prompt - allow override through kwargs
            system_prompt = kwargs.get("system_prompt", self.system_prompt)
            
            print(f"Generando respuesta con modelo {self.model_name}...")
            
            # Set stream parameter
            stream = stream if stream is not None else self.stream
            
            # Format context according to the configured style
            formatted_context = self._format_context(context)
            
            # Format instructions according to the configured style
            user_text = self._format_instructions(system_prompt, formatted_context, prompt)
            
            # Prepare contents with appropriate structure - only user message
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_text)]
                )
            ]
            
            # Get generation config, adjusting temperature if necessary
            use_temperature = kwargs.get("temperature", self.temperature)
                
            # Configure generation parameters
            generation_config = types.GenerateContentConfig(
                temperature=use_temperature,
                top_p=kwargs.get("top_p", self.top_p),
                top_k=kwargs.get("top_k", self.top_k),
                max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
                response_mime_type=self.response_mime_type,
            )
            
            # Generate response with thread lock to prevent concurrent API issues
            with api_lock:
                # Generate response based on streaming preference
                if stream:
                    # For streaming, ensure we collect all chunks safely
                    response_text = ""
                    
                    # Definir el generador como una función segura
                    def safe_streaming_generator():
                        nonlocal response_text
                        try:
                            # Usar la API de streaming
                            stream_response = self.client.models.generate_content_stream(
                                model=self.model_name,
                                contents=contents,
                                config=generation_config,
                            )
                            
                            # Procesar cada chunk de manera segura
                            for chunk in stream_response:
                                try:
                                    # Si el chunk tiene texto directo
                                    if hasattr(chunk, 'text'):
                                        text_chunk = chunk.text
                                    # Si el chunk tiene estructura diferente
                                    elif hasattr(chunk, 'parts') and chunk.parts:
                                        text_chunk = chunk.parts[0].text
                                    # Para cualquier otro formato de respuesta
                                    elif hasattr(chunk, 'candidates') and chunk.candidates:
                                        candidate = chunk.candidates[0]
                                        if hasattr(candidate, 'content') and candidate.content:
                                            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                                text_chunk = candidate.content.parts[0].text
                                            else:
                                                text_chunk = ""
                                        else:
                                            text_chunk = ""
                                    else:
                                        # Si no podemos extraer texto de forma estándar
                                        text_chunk = str(chunk)
                                    
                                    # Solo procesar si hay texto real
                                    if text_chunk:
                                        response_text += text_chunk
                                        yield text_chunk
                                        
                                except Exception as chunk_error:
                                    # Manejar errores de procesamiento de chunks individuales
                                    error_msg = f"Error procesando chunk: {str(chunk_error)}"
                                    yield error_msg
                                    
                        except Exception as stream_error:
                            # Manejar errores generales del streaming
                            error_msg = f"Error en streaming: {str(stream_error)}"
                            yield error_msg
                    
                    # Guardar el texto completo como respaldo
                    self.last_response_text = response_text
                    
                    # Si se necesita mostrar contexto, envolverlo en una función generadora
                    if kwargs.get("show_context", False):
                        def context_wrapper():
                            # Recolectar todo el texto del generador principal
                            full_text = ""
                            for chunk in safe_streaming_generator():
                                full_text += chunk
                                yield chunk
                            
                            # Una vez terminado, guardar el texto completo
                            self.last_response_text = full_text
                        
                        # Devolver el generador envuelto
                        return context_wrapper()
                    
                    # Devolver el generador directo
                    return safe_streaming_generator()
                    
                else:
                    # Non-streaming response
                    try:
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=contents,
                            config=generation_config,
                        )
                        
                        if hasattr(response, 'text'):
                            response_text = response.text
                        elif hasattr(response, 'candidates') and response.candidates:
                            if hasattr(response.candidates[0], 'content') and response.candidates[0].content:
                                if hasattr(response.candidates[0].content, 'parts') and response.candidates[0].content.parts:
                                    response_text = response.candidates[0].content.parts[0].text
                                else:
                                    return "Error: No se pudo extraer texto de las partes de la respuesta"
                            else:
                                return "Error: No se pudo extraer contenido del candidato"
                        else:
                            # Si llegamos aquí, no pudimos extraer el texto 
                            return "Error: No se pudo extraer texto de la respuesta del modelo"
                        
                        # Guardar la respuesta original de texto para respaldo
                        self.last_response_text = response_text
                        
                        # Format the response with context if needed
                        if kwargs.get("show_context", False):
                            return self._format_response_with_context(response_text, prompt)
                        return response_text
                    except Exception as e:
                        error_msg = f"Error al generar contenido: {str(e)}"
                        return error_msg
                
        except Exception as e:
            error_msg = f"Error al generar respuesta: {str(e)}"
            return error_msg
    
    def _format_context(self, context: Optional[Union[str, List[Dict[str, Any]]]]) -> str:
        """
        Format context based on the configured style.
        
        Args:
            context: Raw context information.
            
        Returns:
            str: Formatted context string.
        """
        if not context:
            return ""
            
        # Format based on context_format configuration
        if isinstance(context, str):
            return context
            
        elif isinstance(context, list):
            format_style = self.context_format
            
            if format_style == "simple":
                # Simple format: just concatenate all texts
                return "\n\n".join([item.get('text', '').strip() for item in context])
                
            elif format_style == "numbered":
                # Numbered format: simple numbered list
                return "\n\n".join([f"{i+1}. {item.get('header', '')}: {item.get('text', '')}" 
                                  for i, item in enumerate(context)])
                
            else:  # Default "fragments" format or any other value
                # Fragments format: detailed with headers
                fragments = []
                for i, item in enumerate(context):
                    header = item.get('header', '').strip()
                    text = item.get('text', '').strip()
                    
                    if header:
                        fragments.append(f"FRAGMENTO {i+1}: {header}\n{text}")
                    else:
                        fragments.append(f"FRAGMENTO {i+1}:\n{text}")
                
                return "\n\n".join(fragments)
        
        return ""
        
    def _format_instructions(self, system_prompt: str, formatted_context: str, prompt: str) -> str:
        """
        Format instructions based on the configured style.
        
        Args:
            system_prompt: System prompt to include.
            formatted_context: Pre-formatted context.
            prompt: User's query.
            
        Returns:
            str: Formatted instructions for the model.
        """
        instruction_style = self.instruction_style
        
        if not formatted_context:
            # No context case
            return (
                f"{system_prompt}\n\n"
                f"NOTA: No se ha proporcionado ningún contexto específico para responder a esta pregunta.\n\n"
                f"PREGUNTA DEL USUARIO:\n{prompt}\n\n"
                f"RESPUESTA (en español, amigable y conversacional):"
            )
        
        if instruction_style == "minimal":
            # Minimal instructions
            return (
                f"{system_prompt}\n\n"
                f"CONTEXTO:\n{formatted_context}\n\n"
                f"PREGUNTA: {prompt}\n\n"
                f"RESPUESTA (prioriza la información del contexto, sé amigable y conversacional):"
            )
            
        elif instruction_style == "standard":
            # Standard instructions
            return (
                f"{system_prompt}\n\n"
                f"INSTRUCCIONES:\n"
                f"- Responde basándote principalmente en la información proporcionada en el contexto.\n"
                f"- Si la información no es suficiente para responder completamente, ofrece una respuesta amigable y alternativa.\n"
                f"- Mantén un tono conversacional y útil en todo momento.\n"
                f"- Evita respuestas secas del tipo 'No hay información sobre X'.\n"
                f"- Responde siempre en español.\n\n"
                f"CONTEXTO:\n{formatted_context}\n\n"
                f"PREGUNTA: {prompt}\n\n"
                f"RESPUESTA (en español, amigable y conversacional):"
            )
            
        else:  # Default "detailed" format or any other value
            # Detailed instructions
            return (
                f"{system_prompt}\n\n"
                f"INSTRUCCIONES IMPORTANTES:\n"
                f"1. Prioriza la información proporcionada en los FRAGMENTOS de contexto para responder.\n"
                f"2. Si la información en los FRAGMENTOS no es suficiente, ofrece una respuesta amigable y útil.\n"
                f"3. Mantén un tono conversacional y cercano, evitando respuestas secas como 'No hay información sobre X'.\n"
                f"4. Si no encuentras información específica, ofrece ayuda alternativa o haz preguntas para entender mejor la necesidad.\n"
                f"5. Proporciona respuestas claras y completas en español.\n"
                f"6. Sé empático y útil, incluso cuando no puedas responder directamente la pregunta.\n\n"
                f"FRAGMENTOS DE CONTEXTO:\n"
                f"----------------\n"
                f"{formatted_context}\n"
                f"----------------\n\n"
                f"PREGUNTA DEL USUARIO:\n{prompt}\n\n"
                f"RESPUESTA (en español, amigable y conversacional):"
            )
    
    def _format_response_with_context(self, response_text: str, prompt: str) -> str:
        """
        Format the response with context information.
        
        Args:
            response_text (str): The generated response text.
            prompt (str): The original prompt.
            
        Returns:
            str: Formatted response with context information.
        """
        # Encabezado de la sección de respuesta
        formatted_output = "\n=======================  RESPUESTA  =======================\n\n"
        formatted_output += response_text.strip()
        
        # Encabezado de la sección de contexto
        formatted_output += "\n\n=======================  CONTEXTO  =======================\n\n"
        formatted_output += f"PREGUNTA: {prompt}\n\n"
        
        if self.last_used_context:
            if isinstance(self.last_used_context, list):
                for i, chunk in enumerate(self.last_used_context):
                    # Extraer metadata relevante
                    header = chunk.get('header', '').strip()
                    similarity = chunk.get('similarity', 0.0)
                    page = chunk.get('page', 'N/A')
                    text = chunk.get('text', '').strip()
                    document = chunk.get('document', '').strip()
                    
                    # Construir encabezado del fragmento
                    chunk_header = f"FRAGMENTO #{i+1}"
                    
                    # Añadir información de origen si está disponible
                    if header:
                        chunk_header += f" - {header}"
                    if document and document != header:
                        chunk_header += f" | Documento: {document}"
                    
                    # Añadir información de página y similitud
                    chunk_header += f" | PÁGINA: {page}"
                    if similarity > 0:
                        chunk_header += f" | Similitud: {similarity:.2f}"
                    
                    # Añadir información completa del fragmento
                    formatted_output += f"{chunk_header}\n"
                    formatted_output += f"{text}\n\n"
            elif isinstance(self.last_used_context, str):
                formatted_output += "CONTEXTO COMPLETO:\n"
                formatted_output += self.last_used_context + "\n\n"
        else:
            formatted_output += "No se utilizó contexto para esta respuesta.\n"
        
        return formatted_output
            
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