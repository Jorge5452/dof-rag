import re
import logging
from typing import List, Dict, Any, Optional

from modulos.chunks.ChunkAbstract import ChunkAbstract
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextChunker(ChunkAbstract):
    """
    Context-based chunker implementation.
    Divides text into chunks based on the semantic structure of the content,
    using headings and other contextual signals.
    """
    
    def __init__(self, embedding_model: Optional[Any] = None, max_chunk_size: Optional[int] = None, max_header_level: Optional[int] = None) -> None:
        """
        Initializes the ContextChunker with configurable parameters.
        
        Args:
            embedding_model: Embedding model to use
            max_chunk_size: Maximum chunk size in characters
            max_header_level: Maximum header level to consider
        """
        super().__init__(embedding_model)
        
        # Load configuration
        chunks_config = config.get_chunks_config()
        context_config = chunks_config.get("context", {})
        
        # Set parameters with default values from configuration
        self.max_chunk_size = max_chunk_size or context_config.get("max_chunk_size", 1500)
        self.max_header_level = max_header_level or context_config.get("max_header_level", 6)
        self.use_headers = context_config.get("use_headers", True)
    
    def extract_headers(self, content: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Extracts headings from Markdown content using regular expressions.
        
        Args:
            content: Markdown file content.
            **kwargs: Additional parameters (optional).
            
        Returns:
            List of dictionaries with information about each heading.
        """
        # Get parameters from kwargs or use default values
        max_header_level = kwargs.get("max_header_level", self.max_header_level)
        
        headers = []
        
        # Regular expression to search for headings in Markdown (# Title, ## Subtitle, etc.)
        header_pattern = r'^(#{1,6})\s+(.+?)(?:\s+#+)?$'
        
        # Search for headings in each line
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            level = len(match.group(1))  # Number of # determines the level
            
            # Check if the level is within the desired range
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
        
        logger.debug(f"Extracted {len(headers)} headings")
        return headers
    
    def chunk(self, content: str, headers: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Divides content into chunks using the 'open headings' system.
        
        Args:
            content: Markdown file content.
            headers: List of headings extracted previously.
            **kwargs: Additional parameters (optional).
            
        Returns:
            List of dictionaries with the generated chunks.
        """
        # Get parameters from kwargs or use default values
        use_headers = kwargs.get("use_headers", self.use_headers)
        max_chunk_size = kwargs.get("max_chunk_size", self.max_chunk_size)
        doc_title = kwargs.get("doc_title", "Document")
        
        chunks = []
        
        # If there are headings and they are used
        if headers and use_headers:
            # Divide content into blocks based on headings
            # Sort headings by position
            ordered_headers = sorted(headers, key=lambda h: h["start_index"])
            
            # Add a last placeholder for easier definition of chunks
            if content:
                ordered_headers.append({
                    "header_text": "",
                    "level": 0,
                    "start_index": len(content),
                    "end_index": len(content)
                })
            
            # If there are no ordered headings, return a single chunk
            if not ordered_headers:
                chunks.append({
                    "text": content,
                    "header": "",
                    "page": "1"
                })
                return chunks
            
            # List of open headings (level, text)
            open_headings = []
            
            # First analysis to detect initial headings
            # For the first chunk, we examine the first lines until we find headings
            if ordered_headers:
                # Extract text from the initial until the first heading
                initial_text = content[:ordered_headers[0]["start_index"]]
                
                # We examine the lines to identify initial headings
                for line in initial_text.splitlines():
                    open_headings = self.update_open_headings(open_headings, line)
            
            # Generate chunks based on headings
            for i in range(len(ordered_headers) - 1):
                header = ordered_headers[i]
                next_header = ordered_headers[i + 1]
                
                # Update the list of open headings
                header_line = content[header["start_index"]:header["end_index"]]
                open_headings = self.update_open_headings(open_headings, header_line)
                
                # Determine the chunk text from the end of the current heading to the start of the next
                chunk_start = header["end_index"]
                chunk_end = next_header["start_index"]
                chunk_text = content[chunk_start:chunk_end].strip()
                
                # If the chunk is not empty
                if chunk_text:
                    # Build the header using the open_headings system
                    chunk_number = i + 1
                    page_num = str(i + 1)  # Based on the heading number
                    
                    header_content = self.build_header_from_open_headings(
                        doc_title, 
                        page_num, 
                        open_headings, 
                        chunk_number
                    )
                    
                    # If the chunk is too large, subdivide it
                    if len(chunk_text) > max_chunk_size:
                        sub_chunks = self._subdivide_large_chunk(chunk_text, max_chunk_size)
                        for j, sub_chunk in enumerate(sub_chunks):
                            chunks.append({
                                "text": sub_chunk,
                                "header": header_content,
                                "page": f"{page_num}.{j+1}"
                            })
                    else:
                        chunks.append({
                            "text": chunk_text,
                            "header": header_content,
                            "page": page_num
                        })
        else:
            # If there are no headings or they are not used, divide by size
            # Divide content into paragraphs
            paragraphs = re.split(r'\n\s*\n', content)
            
            current_chunk = ""
            current_page = 1
            
            # List of open headings
            open_headings = []
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Update open headings with each paragraph
                for line in paragraph.splitlines():
                    open_headings = self.update_open_headings(open_headings, line)
                
                # If adding the paragraph exceeds the maximum size, create a new chunk
                if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                    # Build the header for the chunk
                    header_content = self.build_header_from_open_headings(
                        doc_title, 
                        str(current_page), 
                        open_headings, 
                        current_page
                    )
                    
                    chunks.append({
                        "text": current_chunk,
                        "header": header_content,
                        "page": str(current_page)
                    })
                    
                    current_chunk = paragraph
                    current_page += 1
                else:
                    # Add space if it's not the first paragraph of the chunk
                    if current_chunk:
                        current_chunk += "\n\n"
                    current_chunk += paragraph
            
            # Add the last chunk if there's content left
            if current_chunk:
                # Build the header for the last chunk
                header_content = self.build_header_from_open_headings(
                    doc_title, 
                    str(current_page), 
                    open_headings, 
                    current_page
                )
                
                chunks.append({
                    "text": current_chunk,
                    "header": header_content,
                    "page": str(current_page)
                })
        
        logger.info(f"Generated {len(chunks)} chunks by context")
        return chunks
    
    # Legacy method for compatibility
    def build_hierarchical_header(self, current_header: Dict[str, Any], previous_headers: List[Dict[str, Any]]) -> str:
        """
        [LEGACY] Builds a hierarchical header based on the current header and previous ones.
        Kept for compatibility with existing code.
        
        Args:
            current_header: Current header.
            previous_headers: List of previous headers.
            
        Returns:
            String with the hierarchical header.
        """
        # Level of the current header
        current_level = current_header["level"]
        current_text = current_header["header_text"]
        
        # Search for higher level headings to construct the hierarchy
        relevant_headers = {}
        
        # Search among previous headings
        for header in previous_headers:
            level = header["level"]
            # Only consider headings of higher level than the current
            if level < current_level:
                # Save the most recent heading for each level
                if level not in relevant_headers or header["start_index"] > relevant_headers[level]["start_index"]:
                    relevant_headers[level] = header
        
        # Construct the hierarchy
        hierarchy = []
        
        # Add higher level headings in order
        for level in sorted(relevant_headers.keys()):
            hierarchy.append(relevant_headers[level]["header_text"])
        
        # Add the current heading
        hierarchy.append(current_text)
        
        # Join the hierarchy
        return " > ".join(hierarchy)
    
    def _subdivide_large_chunk(self, text: str, max_size: int) -> List[str]:
        """
        Subdivides a large chunk into smaller sub-chunks.
        
        Args:
            text: Text of the chunk to subdivide.
            max_size: Maximum size of each sub-chunk.
            
        Returns:
            List of sub-chunks.
        """
        sub_chunks = []
        
        # Divide by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If a paragraph is too large by itself, divide it by sentences
            if len(paragraph) > max_size:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    if len(sentence) > max_size:
                        # If a sentence is too long, divide it by size
                        for i in range(0, len(sentence), max_size):
                            sub_chunks.append(sentence[i:i+max_size])
                    else:
                        # If adding the sentence exceeds the maximum size, create a new sub-chunk
                        if len(current_chunk) + len(sentence) > max_size and current_chunk:
                            sub_chunks.append(current_chunk)
                            current_chunk = sentence
                        else:
                            # Add space if it's not the first sentence of the chunk
                            if current_chunk:
                                current_chunk += " "
                            current_chunk += sentence
            else:
                # If adding the paragraph exceeds the maximum size, create a new sub-chunk
                if len(current_chunk) + len(paragraph) > max_size and current_chunk:
                    sub_chunks.append(current_chunk)
                    current_chunk = paragraph
                else:
                    # Add space if it's not the first paragraph of the chunk
                    if current_chunk:
                        current_chunk += "\n\n"
                    current_chunk += paragraph
        
        # Add the last sub-chunk if there's content left
        if current_chunk:
            sub_chunks.append(current_chunk)
        
        return sub_chunks
