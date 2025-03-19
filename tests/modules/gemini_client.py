import os
from google import genai
from google.genai import types
import time
from PIL import Image
import threading

from .AbstractClient import AbstractAIClient

api_lock = threading.Lock()

class GeminiClient(AbstractAIClient):
    """
    Implementación concreta para interactuar con la API de Gemini.

    Esta clase se encarga de gestionar la configuración del modelo, 
    el cliente de la API, y de implementar los métodos definidos en la clase abstracta.
    """
    def __init__(self, 
                 model="gemini-2.0-flash", 
                 max_tokens=256, 
                 temperature=0.6,
                 top_p=0.6,
                 top_k=20,
                 response_mime_type="text/plain",
                 api_key=None):
        """
        Inicializa la configuración del cliente Gemini.

        Args:
            model (str): Modelo de Gemini a utilizar.
            max_tokens (int): Número máximo de tokens en la salida.
            temperature (float): Controla la creatividad de la generación.
            top_p (float): Valor top_p para la generación.
            top_k (int): Valor top_k para la generación.
            response_mime_type (str): Tipo MIME de la respuesta.
            api_key (str, opcional): API key para acceder a Gemini. Si no se proporciona,
                                     se toma de la variable de entorno 'GEMINI_API_KEY'.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.response_mime_type = response_mime_type
        self.question = "¿Qué se observa en esta imagen?, respóndelo en español, por favor."
        
        
    def set_question(self, question: str):
        """
        Configura la pregunta (prompt) en español que se utilizará para la generación.

        Args:
            question (str): Pregunta o prompt en español.
        """
        self.question = question


    def set_api_key(self, api_key: str):
        """
        Actualiza la API key y reinicializa el cliente Gemini.

        Args:
            api_key (str): Nueva API key.
        """
        if not api_key:
            raise ValueError("No se encontró la variable de entorno GEMINI_API_KEY. Por favor, configúlela.")
        self.api_key = api_key
        self._client = genai.Client(api_key=self.api_key)

    def process_imagen(self, image_path, force_overwrite=False):
        """
        Procesa una imagen usando Gemini.
        
        Si ya existe un archivo de descripción y force_overwrite es False,
        se retorna el resultado ya procesado. En caso contrario, se genera
        la descripción a través de la API, se valida la salida y, si es correcta,
        se escribe un archivo TXT con la descripción en la misma ubicación que la imagen.
        
        Args:
            image_path (str): Ruta de la imagen a procesar.
            force_overwrite (bool): Si es True, se reprocesa la imagen aun existiendo una descripción.
        
        Returns:
            dict: Diccionario con información del procesamiento (descripción, tiempo, estado, error, etc.).
        """
        result = {"image_path": image_path}
        output_file = f"{os.path.splitext(image_path)[0]}.txt"

        # Si ya existe la descripción y no se forza el reprocesamiento, se retorna el resultado.
        if os.path.exists(output_file) and not force_overwrite:
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content and not content.startswith("ERROR:"):
                        result["description"] = content
                        result["status"] = "already_processed"
                        return result
            except Exception:
                pass

        start = time.time()

        # Abrir la imagen
        try:
            image = Image.open(image_path)
        except Exception as e:
            result["error"] = f"Error abriendo imagen: {str(e)}"
            self.log_error(image_path)
            return result

        try:
            with api_lock:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=[
                        self.question,
                        image
                    ],
                    config=self.get_generate_config()
                )
                description = response.text if hasattr(response, 'text') else str(response)
            process_time = time.time() - start
            result["description"] = description
            result["process_time"] = process_time
            result["status"] = "processed"
        except Exception as e:
            error_msg = str(e)
            result["error"] = f"Error procesando imagen con Gemini: {error_msg}"
            if "429" in error_msg:
                result["error_type"] = "rate_limit"
            elif "404" in error_msg:
                result["error_type"] = "not_found"
            else:
                result["error_type"] = "other"
            self.log_error(image_path)
            return result

        # Verificar la salida generada.
        if not self.verify_output(result):
            return result

        # Si no hay errores, crear el archivo con la descripción generada.
        self.create_output_description(image_path, result["description"])

        return result

    def verify_output(self, result: dict) -> bool:
        """
        Verifica que la salida generada cumpla con los criterios:
        - La descripción no debe estar vacía.
        - No debe haberse producido ningún error durante el procesamiento.
        
        En caso de encontrar un error, se registra la ruta de la imagen en un log.
        
        Args:
            result (dict): Diccionario resultado del procesamiento.
        
        Returns:
            bool: True si la salida es válida; False en caso contrario.
        """
        if "error" in result:
            self.log_error(result["image_path"])
            return False
        if "description" not in result or not result["description"].strip():
            self.log_error(result["image_path"])
            return False
        return True

    def create_output_description(self, image_path: str, description: str):
        """
        Crea un archivo TXT con la descripción generada por el modelo.
        
        El archivo se crea en la misma ruta que la imagen, utilizando el mismo nombre base.
        
        Args:
            image_path (str): Ruta de la imagen procesada.
            description (str): Descripción generada por la API.
        """
        output_file = f"{os.path.splitext(image_path)[0]}.txt"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(description)
        except Exception as e:
            # En caso de error al escribir el archivo, se registra en el log de errores.
            self.log_error(image_path, extra_info=f"Error al escribir descripción: {str(e)}")

    def log_error(self, image_path: str, extra_info: str = ""):
        """
        Registra la ruta de la imagen que produjo un error en el archivo 'error_images.txt'.
        
        Args:
            image_path (str): Ruta de la imagen con error.
            extra_info (str, opcional): Información adicional sobre el error.
        """
        log_file = "error_images.txt"
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                log_entry = image_path
                if extra_info:
                    log_entry += f" | {extra_info}"
                f.write(log_entry + "\n")
        except Exception:
            # Si falla el registro del error, se omite para no interrumpir el flujo.
            pass


    def get_generate_config(self):
        """
        Configura y retorna los parámetros para la generación de contenido mediante la API.

        Returns:
            GenerateContentConfig: Configuración para la generación.
        """
        return types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_tokens,
            response_mime_type=self.response_mime_type,
        )

    def get_client(self):
        """
        Retorna el cliente de la API de Gemini.

        Returns:
            Client: Instancia del cliente de Gemini.
        """
        return self._client

    def update_config(self, max_tokens=None, model=None, temperature=None, 
                      top_p=None, top_k=None, response_mime_type=None):
        """
        Permite actualizar la configuración del cliente.

        Args:
            max_tokens (int, opcional): Nuevo número máximo de tokens.
            model (str, opcional): Nuevo modelo a utilizar.
            temperature (float, opcional): Nueva temperatura.
            top_p (float, opcional): Nuevo valor top_p.
            top_k (int, opcional): Nuevo valor top_k.
            response_mime_type (str, opcional): Nuevo tipo MIME para la respuesta.

        Returns:
            self: Instancia actual para permitir el encadenamiento de métodos.
        """
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if model is not None:
            self.model = model
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if top_k is not None:
            self.top_k = top_k
        if response_mime_type is not None:
            self.response_mime_type = response_mime_type
            
        return self
