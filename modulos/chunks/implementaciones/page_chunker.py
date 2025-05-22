import re
import logging
from typing import List, Dict, Any, Tuple

from modulos.chunks.ChunkAbstract import ChunkAbstract
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PageChunker(ChunkAbstract):
    """
    Page-based chunker implementation.
    Divides text into chunks based on specific page markers,
    such as {number} followed by several dashes.
    """
    
    def __init__(self, embedding_model=None) -> None:
        """
        Constructor initializing the chunker with page-specific configuration.
        
        Args:
            embedding_model: Initialized embedding model. If None, it must be assigned later.
        """
        super().__init__(embedding_model)
        
        # Get specific configuration for page chunking
        self.page_config = self.chunks_config.get("page", {})
        
        # Configuration parameters with default values
        self.use_headers = self.page_config.get("use_headers", True)
        self.max_header_level = self.page_config.get("max_header_level", 6)
        self.page_pattern = self.page_config.get("page_pattern", r'\{(\d+)\}\s*-{5,}')
        
        logger.info(f"PageChunker initialized with max_header_level={self.max_header_level}")
    
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
        max_header_level = kwargs.get("max_header_level", self.max_header_level)
        
        headers = []
        
        # Expresión regular para buscar encabezados en Markdown (# Título, ## Subtítulo, etc.)
        header_pattern = r'^(#{1,6})\s+(.+?)(?:\s+#+)?$'
        
        # Buscar encabezados en cada línea
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            level = len(match.group(1))  # Número de # determina el nivel
            
            # Verificar si el nivel está dentro del rango deseado
            if level <= max_header_level:
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
    
    def split_text_by_page_break(self, text: str, page_pattern: str) -> List[Dict[str, Any]]:
        """
        Divide el texto en chunks basados en el patrón de página:
        {número} seguido de al menos 5 guiones.
        
        Parámetros:
            text: Texto a dividir.
            page_pattern: Patrón de expresión regular para identificar marcadores de página.
            
        Retorna:
            Lista de diccionarios con texto y número de página de cada chunk.
        """
        page_regex = re.compile(page_pattern)
        chunks = []
        last_index = 0
        last_page = None

        for match in page_regex.finditer(text):
            page_num = match.group(1)
            chunk_text = text[last_index:match.start()].strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "page": page_num})
            last_index = match.end()
            last_page = page_num

        # Último fragmento después del último marcador de página
        remaining = text[last_index:].strip()
        if remaining:
            final_page = last_page if last_page else "1"
            chunks.append({"text": remaining, "page": final_page})

        return chunks
    
    def chunk(self, content: str, headers: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Divide el contenido en chunks basados únicamente en marcadores de página
        y mantiene un sistema de encabezados abiertos para preservar el contexto.
        
        Parámetros:
            content: Contenido del archivo Markdown.
            headers: Lista de encabezados extraídos previamente.
            **kwargs: Parámetros adicionales (opcional).
            
        Retorna:
            Lista de diccionarios con los chunks generados.
        """
        # Obtener parámetros de kwargs o usar valores por defecto
        page_pattern = kwargs.get("page_pattern", self.page_pattern)
        doc_title = kwargs.get("doc_title", "Documento")
        
        # Lista final de chunks a retornar
        final_chunks = []
        
        # Verificar si el contenido tiene marcadores de página
        if re.search(page_pattern, content):
            page_chunks = self.split_text_by_page_break(content, page_pattern)
        else:
            # Si no hay marcadores de página, tratar todo el documento como una sola página
            page_chunks = [{"text": content, "page": "1"}]
        
        # Lista de encabezados abiertos (nivel, texto)
        open_headings = []
        chunk_counter = 0
        
        for chunk in page_chunks:
            chunk_counter += 1
            chunk_text = chunk["text"]
            page_number = chunk["page"]
            
            # Para el primer chunk, pre-leer las líneas hasta obtener el primer encabezado
            if chunk_counter == 1:
                lines = chunk_text.splitlines()
                initial_headings = []
                for line in lines:
                    lvl, txt = self.get_heading_level(line)
                    if lvl is not None:
                        initial_headings.append((lvl, txt))
                        # Si el primer encabezado es H1, detenemos la pre-lectura
                        if lvl == 1:
                            break
                    else:
                        # Detenerse si se encuentra la primera línea que no es un encabezado
                        break
                if initial_headings:
                    open_headings = initial_headings.copy()
            
            # Construir el encabezado usando los encabezados "abiertos" al inicio del chunk
            header = self.build_header_from_open_headings(doc_title, page_number, open_headings, chunk_counter)
            
            final_chunks.append({
                "text": chunk_text,
                "header": header,
                "page": page_number
            })
            
            # Actualizar el estado de encabezados abiertos para los siguientes chunks
            lines = chunk_text.splitlines()
            for line in lines:
                open_headings = self.update_open_headings(open_headings, line)
        
        logger.info(f"Generados {len(final_chunks)} chunks por página")
        return final_chunks 