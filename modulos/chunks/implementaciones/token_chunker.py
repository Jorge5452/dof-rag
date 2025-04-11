import re
import logging
import math
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer

from modulos.chunks.ChunkAbstract import ChunkAbstract
from config import config

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TokenChunker(ChunkAbstract):
    """
    Implementación de chunker basado en tokens.
    Divide el texto en chunks con un número máximo de tokens con solapamiento configurable.
    """
    
    def __init__(self, embedding_model=None) -> None:
        """
        Constructor que inicializa el chunker con la configuración específica para chunking por tokens.
        
        Parámetros:
            embedding_model: Modelo de embeddings inicializado. Si es None, se debe asignar posteriormente.
        """
        super().__init__(embedding_model)
        
        # Obtener configuración específica para chunking por tokens
        self.token_config = self.chunks_config.get("token", {})
        
        # Parámetros de configuración con valores por defecto
        self.max_tokens = self.token_config.get("max_tokens", 512)
        self.token_overlap = self.token_config.get("token_overlap", 100)
        self.tokenizer_name = self.token_config.get("tokenizer", "intfloat/multilingual-e5-small")
        
        # Inicializar el tokenizador
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            logger.info(f"Tokenizador {self.tokenizer_name} cargado correctamente")
        except Exception as e:
            logger.error(f"Error al cargar el tokenizador {self.tokenizer_name}: {e}")
            self.tokenizer = None
            
        logger.info(f"TokenChunker inicializado con max_tokens={self.max_tokens}, overlap={self.token_overlap}")
    
    def extract_headers(self, content: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Extrae los encabezados del contenido Markdown utilizando expresiones regulares.
        
        Parámetros:
            content: Contenido del archivo Markdown.
            **kwargs: Parámetros adicionales (opcional).
            
        Retorna:
            Lista de diccionarios con información de cada encabezado.
        """
        # Obtener parámetros de kwargs o usar valores por defecto
        min_header_level = kwargs.get("min_header_level", 1)
        max_header_level = kwargs.get("max_header_level", 3)
        
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
        Divide el contenido en chunks basados en un número máximo de tokens con solapamiento.
        
        Parámetros:
            content: Contenido del archivo Markdown.
            headers: Lista de encabezados extraídos previamente.
            **kwargs: Parámetros adicionales que pueden incluir:
                      - max_tokens: Número máximo de tokens por chunk.
                      - token_overlap: Solapamiento entre chunks en tokens.
            
        Retorna:
            Lista de diccionarios con los chunks generados.
        """
        if self.tokenizer is None:
            raise ValueError("El tokenizador no está inicializado correctamente")
            
        # Obtener parámetros de kwargs o usar valores por defecto
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        token_overlap = kwargs.get("token_overlap", self.token_overlap)
        
        chunks = []
        
        # Tokenizar el contenido completo
        tokens = self.tokenizer.encode(content)
        total_tokens = len(tokens)
        
        # Si el contenido tiene menos tokens que el máximo, crear un único chunk
        if total_tokens <= max_tokens:
            # Encontrar el encabezado adecuado
            header = self.find_header_for_position(0, headers)
            
            chunks.append({
                "text": content,
                "header": header,
                "page": "1"
            })
            
            return chunks
        
        # Calcular número estimado de páginas
        total_pages = max(1, math.ceil(total_tokens / (max_tokens * 2)))
        
        # Calcular los puntos de inicio para cada chunk con solapamiento
        start_positions = list(range(0, total_tokens, max_tokens - token_overlap))
        
        # Si el último chunk está más allá del fin, ajustar
        if start_positions[-1] >= total_tokens:
            start_positions.pop()
        
        # Función auxiliar para convertir índices de tokens a índices de caracteres
        def token_idx_to_char_idx(token_idx):
            if token_idx >= total_tokens:
                return len(content)
            # Decodificar hasta el token para obtener el texto correspondiente
            text_until_token = self.tokenizer.decode(tokens[:token_idx])
            return len(text_until_token)
        
        # Generar chunks
        for i, start_token in enumerate(start_positions):
            # Calcular el fin del chunk en tokens
            end_token = min(start_token + max_tokens, total_tokens)
            
            # Convertir a índices de caracteres
            start_char = token_idx_to_char_idx(start_token)
            end_char = token_idx_to_char_idx(end_token)
            
            chunk_text = content[start_char:end_char]
            
            # Encontrar el encabezado más relevante
            header = self.find_header_for_position(start_char, headers)
            
            # Calcular número de página
            page_num = min(total_pages, 1 + math.floor((start_token / total_tokens) * total_pages))
            
            chunks.append({
                "text": chunk_text,
                "header": header,
                "page": str(page_num)
            })
        
        logger.info(f"Generados {len(chunks)} chunks por tokens")
        return chunks
