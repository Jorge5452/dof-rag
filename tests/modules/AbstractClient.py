from abc import ABC, abstractmethod

class AbstractAIClient(ABC):
    """
    Clase abstracta que define la interfaz para clientes de diferentes modelos de IA.
    
    Los métodos que deben implementarse en cada modelo son:
        - set_api_key: Configurar o actualizar la API key.
        - process_imagen: Procesar una imagen de entrada.
        - verify_output: Validar la salida generada.
        - create_output_description: Generar una descripción de la salida.
        - set_question: Configurar la pregunta o prompt en español.

    """

    @abstractmethod
    def set_api_key(self, api_key: str):
        """
        Configura la API key para el cliente.
        """
        pass

    @abstractmethod
    def process_imagen(self, image_path, force_overwrite=False):
        """
        Procesa la imagen de entrada.
        
        Args:
            imagen: Objeto o ruta de la imagen.
        """
        pass

    @abstractmethod
    def verify_output(self, output):
        """
        Verifica y valida la salida generada.
        
        Args:
            output: Resultado generado por el modelo.
        """
        pass

    @abstractmethod
    def create_output_description(self, output) -> str:
        """
        Genera una descripción a partir de la salida generada.
        
        Args:
            output: Resultado generado por el modelo.
        
        Returns:
            str: Descripción de la salida.
        """
        pass
    
    @abstractmethod
    def set_question(self, question: str):
        """
        Configura la pregunta (prompt) en español que se utilizará para la generación.
        
        Args:
            question (str): Pregunta o prompt en español.
        """
        pass