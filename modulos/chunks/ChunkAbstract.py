import re
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from config import config

# Configure logging for chunk processing
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
        
    def set_embedding_model(self, model: Any) -> None:
        """
        Sets the embedding model for calculating vector representations.
        
        Args:
            model: Initialized embedding model.
        """
        self.model = model
    
    # Regex pattern for detecting Markdown headers
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.*)$')
    
    def get_heading_level(self, line: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Extracts heading level and text from a line, or returns (None, None) if not a heading.
        
        Args:
            line: Text line to analyze.
            
        Returns:
            Tuple (level, text) or (None, None) if not a heading.
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
        Updates the list of open headings based on the current line.
        
        Strategy:
        - If H1 is found, reset the list.
        - If line is a heading with level >1:
            * If list is empty, add it.
            * If last open heading has lower or equal level, 
              add without removing previous (to preserve siblings).
            * If new heading is higher level (lower number) than last,
              preserve only headings with lower level numbers and add new one.
              
        Args:
            open_headings: Current list of open headings [(level, text), ...].
            line: Text line to analyze.
            
        Returns:
            Updated list of open headings.
        """
        lvl, txt = self.get_heading_level(line)
        if lvl is None:
            # Non-heading line, state remains unchanged
            return open_headings

        if lvl == 1:
            # H1 closes all previous context
            return [(1, txt)]
        else:
            if not open_headings:
                return [(lvl, txt)]
            else:
                # If last heading has lower or equal level, append current
                if open_headings[-1][0] <= lvl:
                    open_headings.append((lvl, txt))
                else:
                    # If new heading is higher level (more important),
                    # preserve only headings with lower level numbers
                    new_chain = [item for item in open_headings if item[0] < lvl]
                    new_chain.append((lvl, txt))
                    open_headings = new_chain
            return open_headings
    
    def build_header_from_open_headings(self, doc_title: str, page: str, open_headings: List[Tuple[int, str]], chunk_number: int) -> str:
        """
        Builds chunk header according to configured format.
        Supports multiple formats for use by different chunker implementations.
        
        "standard" format (default):
          # Document: <Document Name> | page: <Page Number>
          {List of headings in Markdown format}
        
        "simple" format:
          <Document Name> - Page <Page Number> - <Heading Hierarchy>
        
        Args:
            doc_title: Document title.
            page: Page number or label.
            open_headings: List of open headings [(level, text), ...].
            chunk_number: Current chunk number.
            
        Returns:
            String with header built according to configured format.
        """
        # Determine specific format for current method
        method_config = self.chunks_config.get(self.method, {})
        header_format = method_config.get("header_format", self.header_format)
        
        # Simple format (PageChunker style)
        if header_format == "simple":
            if chunk_number == 1:
                return f"{doc_title} - Page {page}"
            else:
                if open_headings:
                    sorted_headings = sorted(open_headings, key=lambda x: x[0])
                    headers_text = " > ".join([h[1] for h in sorted_headings])
                    return f"{doc_title} - Page {page} - {headers_text}"
                else:
                    return f"{doc_title} - Page {page}"
        
        # Standard format (original)
        else:
            header_lines = [f"# Document: {doc_title} | page: {page}"]

            if chunk_number == 1:
                if open_headings:
                    # For first chunk, only include first detected heading
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
        Extracts headers from Markdown content using regex while respecting hierarchy.
        
        Args:
            content: Text content.
            **kwargs: Additional parameters, e.g., levels to consider, min/max lengths, etc.
            
        Returns:
            List of dictionaries with header information:
              {
                  "header_text": str,
                  "level": int,           # 1 for h1, 2 for h2, etc.
                  "start_index": int,
                  "end_index": int
              }
        """
        pass

    @abstractmethod
    def chunk(self, content: str, headers: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Performs text partitioning into chunks according to strategy (characters, tokens, context, etc.).
        
        Args:
            content: Text content.
            headers: List of headers extracted with extract_headers.
            **kwargs: Optional parameters that can override or complement configuration.
            
        Returns:
            List of dictionaries with the following structure:
              {
                  "text": str,             # Chunk content.
                  "header": str,           # Associated header or header hierarchy.
                  "page": str,             # Page number or label, intelligently assigned.
              }
        """
        pass

    def process_content(self, content: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Processes content to generate chunks.
        No longer responsible for reading files, only receives content.
        
        Args:
            content: Text content to process.
            **kwargs: Additional parameters to adjust chunking process.
                      May include doc_title for document title.
            
        Returns:
            List of dictionaries with generated chunks.
        """
        try:
            # Extract headers from content
            headers = self.extract_headers(content, **kwargs)
            
            # Split into chunks according to implemented strategy
            raw_chunks = self.chunk(content, headers, **kwargs)
            
            # Return generated chunks
            logger.info(f"Generated {len(raw_chunks)} chunks")
            return raw_chunks
            
        except Exception as e:
            logger.error(f"Error in content processing: {e}")
            raise

    def find_header_for_position(self, position: int, headers: List[Dict[str, Any]]) -> str:
        """
        Finds the most relevant header for a given position in text.
        Considers header hierarchy to build complete context.
        
        Args:
            position: Text position for which to find the header.
            headers: List of previously extracted headers.
            
        Returns:
            String with built header (may include hierarchy).
        """
        relevant_headers = {}
        
        # Find all applicable headers up to position
        for header in headers:
            if header["start_index"] <= position:
                level = header["level"]
                # Save most recent header for each level
                if level not in relevant_headers or header["start_index"] > relevant_headers[level]["start_index"]:
                    relevant_headers[level] = header
        
        # If no applicable headers, return empty string
        if not relevant_headers:
            return ""
        
        # Build header hierarchy
        header_hierarchy = []
        for level in sorted(relevant_headers.keys()):
            header_hierarchy.append(relevant_headers[level]["header_text"])
        
        # Join headers with separator
        return " > ".join(header_hierarchy)

    def process_content_stream(self, content: str, **kwargs) -> Any:
        """
        Streaming version of chunk generation process.
        Processes content and returns chunks iteratively with a generator.
        This implementation is optimal for large documents as it doesn't keep
        all chunks in memory simultaneously.
        
        Args:
            content: Text content to process.
            **kwargs: Additional parameters to adjust chunking process.
                      May include doc_title for document title.
            
        Returns:
            Generator that produces chunk dictionaries one by one.
        """
        try:
            # Extract headers from content
            headers = self.extract_headers(content, **kwargs)
            
            # Split into chunks according to implemented strategy
            raw_chunks = self.chunk(content, headers, **kwargs)
            
            # Return chunks one by one
            logger.info("Starting streaming chunk generation")
            total_chunks = len(raw_chunks)
            
            for i, chunk in enumerate(raw_chunks):
                if i % 10 == 0:  # Log every 10 chunks to avoid log saturation
                    logger.debug(f"Processing chunk {i+1}/{total_chunks}")
                
                yield chunk
                
                # Release reference to help garbage collector
                raw_chunks[i] = None
                
            logger.info(f"Completed streaming generation of {total_chunks} chunks")
            
        except Exception as e:
            logger.error(f"Error in streaming content processing: {e}")
            raise
