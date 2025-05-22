import re
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from config import config
from modulos.embeddings.embeddings_factory import EmbeddingFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChunkAbstract(ABC):
    """
    Abstract class for creating chunks from text.
    
    Principles:
      - Chunking is performed on the content already provided, removing the responsibility
        of reading files.
      - Headers are intelligently extracted using regular expressions respecting the hierarchy,
        which allows assigning a complete context to each chunk.
      - Each chunk will contain:
            • text: The content of the chunk.
            • header: The associated header (or hierarchy of headers) obtained through extraction.
            • page: A numerical or labeled value indicating the chunk's "page", intelligently obtained.
      - All configurations (sizes, overlaps, header extraction, etc.) are passed through a dictionary
        loaded during initialization (ideally from config.py).
    """

    def __init__(self, embedding_model=None) -> None:
        """
        Constructor that initializes the chunker with an embedding model and configuration.
        
        Args:
            embedding_model: Initialized embedding model. If None, it's assigned later.
        """
        self.chunks_config = config.get_chunks_config()
        self.method = self.chunks_config.get("method", "context")
        self.model = embedding_model
        
        # Configuration for header format
        self.header_format = self.chunks_config.get("header_format", "standard")
        
    def set_embedding_model(self, model):
        """
        Establece el modelo de embeddings para calcular las representaciones vectoriales.
        
        Parámetros:
            model: Modelo de embeddings inicializado.
        """
        self.model = model
    
    # Patrón para detectar encabezados en Markdown
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.*)$')
    
    def get_heading_level(self, line: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Obtiene el nivel y texto de un encabezado, o (None, None) si la línea no es un encabezado.
        
        Parámetros:
            line: Línea de texto a analizar.
            
        Retorna:
            Tupla (nivel, texto) o (None, None) si no es un encabezado.
        """
        match = self.HEADING_PATTERN.match(line.strip())
        if match:
            hashes = match.group(1)
            heading_text = match.group(2).strip()
            level = len(hashes)
            return level, heading_text
        return None, None
    
    def update_open_headings(self, open_headings: List[Tuple[int, str]], line: str) -> List[Tuple[int, str]]:
        """
        Actualiza la lista de encabezados abiertos según la línea actual.
        
        La estrategia es:
        - Si se encuentra un H1, se reinicia la lista.
        - Si la línea es un encabezado de nivel >1:
            * Si la lista está vacía, se añade.
            * Si el último encabezado abierto tiene un nivel menor o igual, 
              se añade sin eliminar el anterior (para preservar hermanos).
            * Si el nuevo encabezado es de nivel superior (número menor) 
              al último, se preservan los de nivel inferior al nuevo y se añade.
              
        Parámetros:
            open_headings: Lista actual de encabezados abiertos [(nivel, texto), ...].
            line: Línea de texto a analizar.
            
        Retorna:
            Lista actualizada de encabezados abiertos.
        """
        lvl, txt = self.get_heading_level(line)
        if lvl is None:
            # Línea sin encabezado, no se modifica el estado
            return open_headings

        if lvl == 1:
            # Un H1 cierra todo el contexto anterior
            return [(1, txt)]
        else:
            if not open_headings:
                return [(lvl, txt)]
            else:
                # Si el último encabezado tiene un nivel menor o igual, se añade el actual
                if open_headings[-1][0] <= lvl:
                    open_headings.append((lvl, txt))
                else:
                    # Si el nuevo encabezado es de nivel superior (más importante)
                    # solo se preservan los encabezados de nivel inferior (mayor) al nuevo
                    new_chain = [item for item in open_headings if item[0] < lvl]
                    new_chain.append((lvl, txt))
                    open_headings = new_chain
            return open_headings
    
    def build_header_from_open_headings(self, doc_title: str, page: str, open_headings: List[Tuple[int, str]], chunk_number: int) -> str:
        """
        Construye el encabezado del chunk según el formato configurado.
        Soporta múltiples formatos para su uso por diferentes implementaciones de chunkers.
        
        Formato "standard" (por defecto):
          # Document: <Nombre del Documento> | page: <Número de Página>
          {Lista de encabezados en formato Markdown}
        
        Formato "simple":
          <Nombre del Documento> - Página <Número de Página> - <Jerarquía de encabezados>
        
        Parámetros:
            doc_title: Título del documento.
            page: Número o etiqueta de página.
            open_headings: Lista de encabezados abiertos [(nivel, texto), ...].
            chunk_number: Número del chunk actual.
            
        Retorna:
            String con el encabezado construido según el formato configurado.
        """
        # Determinar formato específico para el método actual
        method_config = self.chunks_config.get(self.method, {})
        header_format = method_config.get("header_format", self.header_format)
        
        # Formato simple (estilo PageChunker)
        if header_format == "simple":
            if chunk_number == 1:
                return f"{doc_title} - Página {page}"
            else:
                if open_headings:
                    sorted_headings = sorted(open_headings, key=lambda x: x[0])
                    headers_text = " > ".join([h[1] for h in sorted_headings])
                    return f"{doc_title} - Página {page} - {headers_text}"
                else:
                    return f"{doc_title} - Página {page}"
        
        # Formato estándar (original)
        else:
            header_lines = [f"# Document: {doc_title} | page: {page}"]

            if chunk_number == 1:
                if open_headings:
                    # Para el primer chunk, solo se incluye el primer encabezado detectado
                    top_level, top_text = open_headings[0]
                    hashes = "#" * top_level
                    header_lines.append(f"{hashes} {top_text}")
            else:
                for (lvl, txt) in open_headings:
                    hashes = "#" * lvl
                    header_lines.append(f"{hashes} {txt}")

            return "\n".join(header_lines)
    
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
                      Puede incluir doc_title para el título del documento.
            
        Retorna:
            Lista de diccionarios con los chunks generados.
        """
        try:
            # 1. Extraer los encabezados
            headers = self.extract_headers(content, **kwargs)
            
            # 2. Dividir en chunks según la estrategia implementada
            raw_chunks = self.chunk(content, headers, **kwargs)
            
            # 3. Devolver los chunks generados
            logger.info(f"Generated {len(raw_chunks)} chunks")
            return raw_chunks
            
        except Exception as e:
            logger.error(f"Error in content processing: {e}")
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

    def process_content_stream(self, content: str, **kwargs):
        """
        Versión streaming del proceso de generación de chunks.
        Procesa el contenido y devuelve chunks de forma iterativa con un generador.
        Esta implementación es óptima para documentos grandes, ya que no mantiene
        todos los chunks en memoria simultáneamente.
        
        Parámetros:
            content: Contenido de texto a procesar.
            **kwargs: Parámetros adicionales para ajustar el proceso de chunking.
                      Puede incluir doc_title para el título del documento.
            
        Retorna:
            Generador que produce diccionarios con los chunks uno por uno.
        """
        try:
            # 1. Extraer los encabezados
            headers = self.extract_headers(content, **kwargs)
            
            # 2. Dividir en chunks según la estrategia implementada
            raw_chunks = self.chunk(content, headers, **kwargs)
            
            # 3. Devolver los chunks generados uno por uno
            logger.info("Starting streaming chunk generation")
            total_chunks = len(raw_chunks)
            
            for i, chunk in enumerate(raw_chunks):
                if i % 10 == 0:  # Log cada 10 chunks para no saturar logs
                    logger.debug(f"Processing chunk {i+1}/{total_chunks}")
                
                yield chunk
                
                # Liberar la referencia para ayudar al recolector de basura
                raw_chunks[i] = None
                
            logger.info(f"Completed streaming generation of {total_chunks} chunks")
            
        except Exception as e:
            logger.error(f"Error in streaming content processing: {e}")
            raise
