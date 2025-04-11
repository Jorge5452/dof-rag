import re
import logging
import math
from typing import List, Dict, Any, Optional

from modulos.chunks.ChunkAbstract import ChunkAbstract
from config import config

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CharacterChunker(ChunkAbstract):
    """
    Implementación de chunker basado en caracteres.
    Divide el texto en chunks de tamaño fijo con solapamiento configurable.
    """
    
    def __init__(self, embedding_model=None) -> None:
        """
        Constructor que inicializa el chunker con la configuración específica para chunking por caracteres.
        
        Parámetros:
            embedding_model: Modelo de embeddings inicializado. Si es None, se debe asignar posteriormente.
        """
        super().__init__(embedding_model)
        
        # Obtener configuración específica para chunking por caracteres
        self.character_config = self.chunks_config.get("character", {})
        
        # Parámetros de configuración con valores por defecto
        self.chunk_size = self.character_config.get("chunk_size", 1000)
        self.chunk_overlap = self.character_config.get("chunk_overlap", 200)
        self.header_extraction_enabled = self.character_config.get("header_extraction_enabled", True)
        self.min_header_length = self.character_config.get("min_header_length", 1)
        self.max_header_length = self.character_config.get("max_header_length", 3)
        
        logger.info(f"CharacterChunker inicializado con tamaño={self.chunk_size}, solapamiento={self.chunk_overlap}")
    
    def extract_headers(self, content: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Extrae los encabezados del contenido Markdown utilizando expresiones regulares.
        
        Parámetros:
            content: Contenido del archivo Markdown.
            **kwargs: Parámetros adicionales (opcional).
            
        Retorna:
            Lista de diccionarios con información de cada encabezado.
        """
        # Si la extracción de encabezados está deshabilitada, retornar lista vacía
        if not self.header_extraction_enabled and not kwargs.get("force_header_extraction", False):
            return []
        
        # Obtener parámetros de kwargs o usar valores por defecto
        min_header_level = kwargs.get("min_header_level", self.min_header_length)
        max_header_level = kwargs.get("max_header_level", self.max_header_length)
        
        headers = []
        
        # Expresión regular para buscar encabezados en Markdown (# Título, ## Subtítulo, etc.)
        header_pattern = r'^(#{1,6})\s+(.+?)(?:\s+#+)?$'
        
        # Buscar encabezados en cada línea
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            level = len(match.group(1))  # Número de # determina el nivel
            
            # Verificar si el nivel está dentro del rango deseado
            if min_header_level <= level <= max_header_level:
                header_text = match.group(2).strip()
                
                # Posición de inicio y fin en el contenido
                start_index = match.start()
                end_index = match.end()
                
                headers.append({
                    "header_text": header_text,
                    "level": level,
                    "start_index": start_index,
                    "end_index": end_index
                })
        
        logger.debug(f"Extraídos {len(headers)} encabezados")
        return headers
    
    def chunk(self, content: str, headers: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Divide el contenido en chunks de tamaño fijo con solapamiento.
        
        Parámetros:
            content: Contenido del archivo Markdown.
            headers: Lista de encabezados extraídos previamente.
            **kwargs: Parámetros adicionales que pueden incluir:
                      - chunk_size: Tamaño de cada chunk en caracteres.
                      - chunk_overlap: Solapamiento entre chunks en caracteres.
            
        Retorna:
            Lista de diccionarios con los chunks generados.
        """
        # Obtener parámetros de kwargs o usar valores por defecto
        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        chunk_overlap = kwargs.get("chunk_overlap", self.chunk_overlap)
        
        chunks = []
        content_length = len(content)
        
        # Si el contenido es más pequeño que el tamaño de chunk, crear un único chunk
        if content_length <= chunk_size:
            # Encontrar el encabezado adecuado (si hay)
            header = self.find_header_for_position(0, headers)
            
            chunks.append({
                "text": content,
                "header": header,
                "page": "1"
            })
            
            return chunks
        
        # Calcular número estimado de páginas para dividir el contenido
        total_pages = max(1, math.ceil(content_length / (chunk_size * 2)))
        chars_per_page = math.ceil(content_length / total_pages)
        
        # Calcular los puntos de inicio para cada chunk con solapamiento
        start_positions = list(range(0, content_length, chunk_size - chunk_overlap))
        
        # Si el último chunk está más allá del fin del contenido, ajustar
        if start_positions[-1] >= content_length:
            start_positions.pop()
        
        # Generar chunks
        for i, start in enumerate(start_positions):
            # Calcular el fin del chunk (no exceder el contenido)
            end = min(start + chunk_size, content_length)
            
            chunk_text = content[start:end]
            
            # Encontrar el encabezado más relevante para este chunk
            header = self.find_header_for_position(start, headers)
            
            # Calcular número de página (basado en la posición relativa en el contenido)
            page_num = min(total_pages, 1 + math.floor((start / content_length) * total_pages))
            
            chunks.append({
                "text": chunk_text,
                "header": header,
                "page": str(page_num)
            })
        
        logger.info(f"Generados {len(chunks)} chunks por caracteres")
        return chunks
