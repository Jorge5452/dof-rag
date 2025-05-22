import re
import logging
import math
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer

from modulos.chunks.ChunkAbstract import ChunkAbstract
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TokenChunker(ChunkAbstract):
    """
    Token-based chunker implementation.
    Divides text into chunks with a maximum number of tokens with configurable overlap.
    """
    
    def __init__(self, embedding_model=None) -> None:
        """
        Constructor initializing the chunker with token-specific configuration.
        
        Args:
            embedding_model: Initialized embedding model. If None, it must be assigned later.
        """
        super().__init__(embedding_model)
        
        # Get specific configuration for token chunking
        self.token_config = self.chunks_config.get("token", {})
        
        # Configuration parameters with default values
        self.max_tokens = self.token_config.get("max_tokens", 512)
        self.token_overlap = self.token_config.get("token_overlap", 100)
        self.tokenizer_name = self.token_config.get("tokenizer", "intfloat/multilingual-e5-small")
        
        # Initialize the tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            logger.info(f"Tokenizer {self.tokenizer_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading tokenizer {self.tokenizer_name}: {e}")
            self.tokenizer = None
            
        logger.info(f"TokenChunker initialized with max_tokens={self.max_tokens}, overlap={self.token_overlap}")
    
    def extract_headers(self, content: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Extracts headers from Markdown content using regular expressions.
        
        Args:
            content: Markdown file content.
            **kwargs: Additional parameters (optional).
            
        Returns:
            List of dictionaries with information about each header.
        """
        # Get parameters from kwargs or use default values
        min_header_level = kwargs.get("min_header_level", 1)
        max_header_level = kwargs.get("max_header_level", 3)
        
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
        Divides content into chunks based on a maximum number of tokens with overlap.
        
        Args:
            content: Markdown file content.
            headers: List of headers extracted previously.
            **kwargs: Additional parameters that may include:
                      - max_tokens: Maximum number of tokens per chunk.
                      - token_overlap: Overlap between chunks in tokens.
            
        Returns:
            List of dictionaries with the generated chunks.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized correctly")
            
        # Get parameters from kwargs or use default values
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        token_overlap = kwargs.get("token_overlap", self.token_overlap)
        
        chunks = []
        
        # Tokenize the entire content
        tokens = self.tokenizer.encode(content)
        total_tokens = len(tokens)
        
        # If the content has fewer tokens than the maximum, create a single chunk
        if total_tokens <= max_tokens:
            # Find the appropriate header
            header = self.find_header_for_position(0, headers)
            
            chunks.append({
                "text": content,
                "header": header,
                "page": "1"
            })
            
            return chunks
        
        # Calculate estimated number of pages
        total_pages = max(1, math.ceil(total_tokens / (max_tokens * 2)))
        
        # Calculate start points for each chunk with overlap
        start_positions = list(range(0, total_tokens, max_tokens - token_overlap))
        
        # If the last chunk is beyond the end, adjust
        if start_positions[-1] >= total_tokens:
            start_positions.pop()
        
        # Helper function to convert token indices to character indices
        def token_idx_to_char_idx(token_idx):
            if token_idx >= total_tokens:
                return len(content)
            # Decode until the token to get the corresponding text
            text_until_token = self.tokenizer.decode(tokens[:token_idx])
            return len(text_until_token)
        
        # Generate chunks
        for i, start_token in enumerate(start_positions):
            # Calculate the end of the chunk in tokens
            end_token = min(start_token + max_tokens, total_tokens)
            
            # Convert to character indices
            start_char = token_idx_to_char_idx(start_token)
            end_char = token_idx_to_char_idx(end_token)
            
            chunk_text = content[start_char:end_char]
            
            # Find the most relevant header
            header = self.find_header_for_position(start_char, headers)
            
            # Calculate page number
            page_num = min(total_pages, 1 + math.floor((start_token / total_tokens) * total_pages))
            
            chunks.append({
                "text": chunk_text,
                "header": header,
                "page": str(page_num)
            })
        
        logger.info(f"Generated {len(chunks)} chunks by tokens")
        return chunks
