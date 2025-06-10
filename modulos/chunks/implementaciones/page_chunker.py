import re
import logging
from typing import List, Dict, Any

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
        Extract headers from Markdown content using regular expressions.
        
        Args:
            content: Markdown file content.
            **kwargs: Additional parameters (optional).
            
        Returns:
            List of dictionaries with information about each header.
        """
        # Get parameters from kwargs or use default values
        max_header_level = kwargs.get("max_header_level", self.max_header_level)
        
        headers = []
        
        # Regular expression to search for headers in Markdown (# Title, ## Subtitle, etc.)
        header_pattern = r'^(#{1,6})\s+(.+?)(?:\s+#+)?$'
        
        # Search for headers in each line
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            level = len(match.group(1))  # Number of # determines the level
            
            # Verify if the level is within the desired range
            if level <= max_header_level:
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
    
    def split_text_by_page_break(self, text: str, page_pattern: str) -> List[Dict[str, Any]]:
        """
        Splits text into chunks based on page pattern:
        {number} followed by at least 5 dashes.
        
        Args:
            text: Text to split.
            page_pattern: Regular expression pattern to identify page markers.
            
        Returns:
            List of dictionaries with text and page number for each chunk.
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

        # Last fragment after the last page marker
        remaining = text[last_index:].strip()
        if remaining:
            final_page = last_page if last_page else "1"
            chunks.append({"text": remaining, "page": final_page})

        return chunks
    
    def chunk(self, content: str, headers: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Divides content into chunks based solely on page markers
        and maintains an open headings system to preserve context.
        
        Args:
            content: Markdown file content.
            headers: List of headers extracted previously.
            **kwargs: Additional parameters (optional).
            
        Returns:
            List of dictionaries with the generated chunks.
        """
        # Get parameters from kwargs or use default values
        page_pattern = kwargs.get("page_pattern", self.page_pattern)
        doc_title = kwargs.get("doc_title", "Document")
        
        # Final list of chunks to return
        final_chunks = []
        
        # Check if content has page markers
        if re.search(page_pattern, content):
            page_chunks = self.split_text_by_page_break(content, page_pattern)
        else:
            # If there are no page markers, treat the entire document as a single page
            page_chunks = [{"text": content, "page": "1"}]
        
        # List of open headings (level, text)
        open_headings = []
        chunk_counter = 0
        
        for chunk in page_chunks:
            chunk_counter += 1
            chunk_text = chunk["text"]
            page_number = chunk["page"]
            
            # For the first chunk, pre-read lines until getting the first heading
            if chunk_counter == 1:
                lines = chunk_text.splitlines()
                initial_headings = []
                for line in lines:
                    lvl, txt = self.get_heading_level(line)
                    if lvl is not None:
                        initial_headings.append((lvl, txt))
                        # If the first heading is H1, stop pre-reading
                        if lvl == 1:
                            break
                    else:
                        # Stop if the first non-heading line is found
                        break
                if initial_headings:
                    open_headings = initial_headings.copy()
            
            # Build the header using the "open" headings at the beginning of the chunk
            header = self.build_header_from_open_headings(doc_title, page_number, open_headings, chunk_counter)
            
            final_chunks.append({
                "text": chunk_text,
                "header": header,
                "page": page_number
            })
            
            # Update the state of open headings for the following chunks
            lines = chunk_text.splitlines()
            for line in lines:
                open_headings = self.update_open_headings(open_headings, line)
        
        logger.info(f"Generated {len(final_chunks)} chunks by page")
        return final_chunks