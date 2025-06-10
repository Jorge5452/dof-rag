import re
import logging
import math
from typing import List, Dict, Any
from modulos.chunks.ChunkAbstract import ChunkAbstract

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CharacterChunker(ChunkAbstract):
    """
    Character-based chunker implementation.
    Divides text into fixed-size chunks with configurable overlap.
    """
    
    def __init__(self, embedding_model=None) -> None:
        """
        Constructor initializing the chunker with character-specific configuration.
        
        Args:
            embedding_model: Initialized embedding model. If None, it must be assigned later.
        """
        super().__init__(embedding_model)
        
        # Get specific configuration for character chunking
        self.character_config = self.chunks_config.get("character", {})
        
        # Configuration parameters with default values
        self.chunk_size = self.character_config.get("chunk_size", 1000)
        self.chunk_overlap = self.character_config.get("chunk_overlap", 200)
        self.header_extraction_enabled = self.character_config.get("header_extraction_enabled", True)
        self.min_header_length = self.character_config.get("min_header_length", 1)
        self.max_header_length = self.character_config.get("max_header_length", 3)
        
        logger.info(f"CharacterChunker initialized with size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def extract_headers(self, content: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Extract headers from Markdown content using regular expressions.
        
        Args:
            content: Markdown file content.
            **kwargs: Additional parameters (optional).
            
        Returns:
            List of dictionaries with information about each header.
        """
        # If header extraction is disabled, return empty list
        if not self.header_extraction_enabled and not kwargs.get("force_header_extraction", False):
            return []
        
        # Get parameters from kwargs or use default values
        min_header_level = kwargs.get("min_header_level", self.min_header_length)
        max_header_level = kwargs.get("max_header_level", self.max_header_length)
        
        headers = []
        
        # Regular expression to search for headers in Markdown (# Title, ## Subtitle, etc.)
        header_pattern = r'^(#{1,6})\s+(.+?)(?:\s+#+)?$'
        
        # Search for headers in each line
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            level = len(match.group(1))  # Number of # determines the level
            
            # Verify if the level is within the desired range
            if min_header_level <= level <= max_header_level:
                header_text = match.group(2).strip()
                
                # Start and end positions in the content
                start_index = match.start()
                end_index = match.end()
                
                headers.append({
                    "header_text": header_text,
                    "level": level,
                    "start_index": start_index,
                    "end_index": end_index
                })
        
        logger.debug(f"Extracted {len(headers)} headers")
        return headers
    
    def chunk(self, content: str, headers: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Divide the content into fixed-size chunks with overlap.
        
        Args:
            content: Markdown file content.
            headers: List of headers extracted previously.
            **kwargs: Additional parameters that may include:
                      - chunk_size: Size of each chunk in characters.
                      - chunk_overlap: Overlap between chunks in characters.
            
        Returns:
            List of dictionaries with the generated chunks.
        """
        # Get parameters from kwargs or use default values
        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        chunk_overlap = kwargs.get("chunk_overlap", self.chunk_overlap)
        
        chunks = []
        content_length = len(content)
        
        # If the content is smaller than the chunk size, create a single chunk
        if content_length <= chunk_size:
            # Find the appropriate header (if any)
            header = self.find_header_for_position(0, headers)
            
            chunks.append({
                "text": content,
                "header": header,
                "page": "1"
            })
            
            return chunks
        
        # Calculate estimated number of pages to divide the content
        total_pages = max(1, math.ceil(content_length / (chunk_size * 2)))
        
        # Calculate start points for each chunk with overlap
        start_positions = list(range(0, content_length, chunk_size - chunk_overlap))
        
        # If the last chunk is beyond the end of the content, adjust
        if start_positions[-1] >= content_length:
            start_positions.pop()
        
        # Generate chunks
        for i, start in enumerate(start_positions):
            # Calculate the end of the chunk (do not exceed the content)
            end = min(start + chunk_size, content_length)
            
            chunk_text = content[start:end]
            
            # Find the most relevant header for this chunk
            header = self.find_header_for_position(start, headers)
            
            # Calculate page number (based on position relative to the content)
            page_num = min(total_pages, 1 + math.floor((start / content_length) * total_pages))
            
            chunks.append({
                "text": chunk_text,
                "header": header,
                "page": str(page_num)
            })
        
        logger.info(f"Generated {len(chunks)} chunks by characters")
        return chunks
