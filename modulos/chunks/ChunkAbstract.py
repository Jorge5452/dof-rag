import re
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from config import config
from modulos.embeddings.embeddings_factory import EmbeddingFactory

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChunkAbstract(ABC):
    """
    Clase abstracta para la creación de chunks a partir de texto.
    
    Principios:
      - El chunking se realiza sobre el contenido ya proporcionado, eliminando la responsabilidad
        de leer archivos.
      - Se extraen de forma inteligente los encabezados con expresiones regulares respetando la jerarquía,
        lo que permite asignar un contexto completo a cada chunk.
      - Cada chunk contendrá:
            • text: El contenido del chunk.
            • header: El encabezado asociado (o jerarquía de encabezados) obtenido mediante la extracción.
            • page: Un valor numérico o etiquetado que indique la "página" del chunk, obtenido de forma inteligente.
      - Todas las configuraciones (tamaños, solapamientos, extracción de headers, etc.) se pasan mediante un diccionario
        cargado en la inicialización (idealmente desde config.py).
    """

    def __init__(self, embedding_model=None) -> None:
        """
        Constructor que inicializa el chunker con un modelo de embeddings y la configuración.
        
        Parámetros:
            embedding_model: Modelo de embeddings inicializado. Si es None, se asigna posteriormente.
        """
        self.chunks_config = config.get_chunks_config()
        self.method = self.chunks_config.get("method", "context")
        self.model = embedding_model
        
    def set_embedding_model(self, model):
        """
        Establece el modelo de embeddings para calcular las representaciones vectoriales.
        
        Parámetros:
            model: Modelo de embeddings inicializado.
        """
        self.model = model
    
    @abstractmethod
    def extract_headers(self, content: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Extrae los encabezados del contenido Markdown utilizando expresiones regulares y respetando la jerarquía.
        
        Parámetros:
            content: Contenido del texto.
            **kwargs: Parámetros adicionales, por ejemplo, niveles a considerar, longitudes mínimas/máximas, etc.
            
        Retorna:
            Lista de diccionarios con la información de cada encabezado:
              {
                  "header_text": str,
                  "level": int,           # 1 para h1, 2 para h2, etc.
                  "start_index": int,
                  "end_index": int
              }
        """
        pass

    @abstractmethod
    def chunk(self, content: str, headers: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Realiza el particionado del texto en chunks según la estrategia (caracteres, tokens, contexto, etc.).
        
        Parámetros:
            content: Contenido del texto.
            headers: Lista de encabezados extraídos con extract_headers.
            **kwargs: Parámetros opcionales que pueden sobrescribir o complementar la configuración.
            
        Retorna:
            Lista de diccionarios con la siguiente estructura:
              {
                  "text": str,             # Contenido del chunk.
                  "header": str,           # Encabezado o jerarquía de encabezados asociados.
                  "page": str,             # Número o etiqueta de página, asignado de forma inteligente.
              }
        """
        pass

    def process_content(self, content: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Procesa el contenido para generar chunks.
        Ya no tiene la responsabilidad de leer archivos, solo recibe el contenido.
        
        Parámetros:
            content: Contenido de texto a procesar.
            **kwargs: Parámetros adicionales para ajustar el proceso de chunking.
            
        Retorna:
            Lista de diccionarios con los chunks generados.
        """
        try:
            # 1. Extraer los encabezados
            headers = self.extract_headers(content, **kwargs)
            
            # 2. Dividir en chunks según la estrategia implementada
            raw_chunks = self.chunk(content, headers, **kwargs)
            
            # 3. Devolver los chunks generados
            logger.info(f"Generados {len(raw_chunks)} chunks")
            return raw_chunks
            
        except Exception as e:
            logger.error(f"Error en el procesamiento del contenido: {e}")
            raise

    def find_header_for_position(self, position: int, headers: List[Dict[str, Any]]) -> str:
        """
        Encuentra el encabezado más relevante para una posición dada en el texto.
        Considera la jerarquía de encabezados para construir un contexto completo.
        
        Parámetros:
            position: Posición en el texto para la que queremos encontrar el encabezado.
            headers: Lista de encabezados extraídos previamente.
            
        Retorna:
            String con el encabezado construido (puede incluir jerarquía).
        """
        relevant_headers = {}
        
        # Encontrar todos los encabezados aplicables hasta la posición
        for header in headers:
            if header["start_index"] <= position:
                level = header["level"]
                # Guardar el encabezado más reciente para cada nivel
                if level not in relevant_headers or header["start_index"] > relevant_headers[level]["start_index"]:
                    relevant_headers[level] = header
        
        # Si no hay encabezados aplicables, retornar cadena vacía
        if not relevant_headers:
            return ""
        
        # Construir la jerarquía de encabezados
        header_hierarchy = []
        for level in sorted(relevant_headers.keys()):
            header_hierarchy.append(relevant_headers[level]["header_text"])
        
        # Unir los encabezados con un separador
        return " > ".join(header_hierarchy)
