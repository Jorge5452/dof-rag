import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import time
import re

from config import config

# Configure logging
logger = logging.getLogger(__name__)

class MarkdownProcessor:
    """
    Class for processing Markdown files.
    
    This class is responsible for:
    1. Reading Markdown files
    2. Extracting metadata
    3. Preparing the document for chunking
    """
    
    def __init__(self):
        """
        Initializes the Markdown file processor.
        """
        self.processing_config = config.get_processing_config()
        # Pattern to detect headers in Markdown
        self.HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.*)$')
    
    def read_markdown_file(self, document_path: str) -> str:
        """
        Reads a Markdown file and returns its content.
        
        Args:
            document_path: Path to the Markdown file.
            
        Returns:
            File content as a string.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        try:
            path = Path(document_path)
            if not path.exists():
                raise FileNotFoundError(f"File {document_path} does not exist")
                
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            logger.info(f"Markdown file read successfully: {document_path}")
            return content
        except Exception as e:
            logger.error(f"Error reading Markdown file {document_path}: {e}")
            raise
    
    def extract_metadata(self, document_path: str, content: Optional[str] = None) -> Dict[str, Any]:
        """
        Extracts basic metadata from a Markdown document.
        
        Args:
            document_path: Path to the Markdown file.
            content: File content (optional, if already read).
        
        Returns:
            Dictionary with document metadata:
            {
                'title': str,       # Extracted title or file name
                'url': str,         # URL constructed as 'file://' + absolute path
                'file_path': str,   # Complete file path
                'file_name': str,   # File name without extension
                'file_size': int,   # Size in bytes
                'created_at': str,  # Creation date (ISO format)
                'modified_at': str, # Modification date (ISO format)
            }
        """
        path = Path(document_path)
        
        # Read content if not provided
        if content is None:
            content = self.read_markdown_file(document_path)
        
        # Basic metadata from filesystem
        stat = path.stat()
        
        # Try to extract title from content
        title = self._extract_title_from_content(content) or path.stem
        
        # Build metadata
        metadata = {
            'title': title,
            'url': f"file://{path.absolute()}",
            'file_path': str(path.absolute()),
            'file_name': path.stem,
            'file_size': stat.st_size,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_ctime)),
            'modified_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
            'content_length': len(content)
        }
        
        return metadata
    
    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """
        Attempts to extract a title from the Markdown file content.
        Looks for level 1 headers (# Title) or YAML frontmatter metadata.
        
        Args:
            content: Markdown file content.
            
        Returns:
            Extracted title or None if not found.
        """
        # Look for H1 header (# Title)
        h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()
        
        # Look for title in YAML frontmatter
        frontmatter_match = re.search(r'^---\s+(?:.|\n)+?title:\s*"?([^"\n]+)"?(?:.|\n)+?---', content)
        if frontmatter_match:
            return frontmatter_match.group(1).strip()
        
        return None
    
    def get_heading_level(self, line: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Gets the level and text of a header, or (None, None) if the line is not a header.
        
        Args:
            line: Text line to analyze.
            
        Returns:
            Tuple (level, text) or (None, None) if not a header.
        """
        match = self.HEADING_PATTERN.match(line)
        if match:
            hashes = match.group(1)
            heading_text = match.group(2).strip()
            level = len(hashes)
            return level, heading_text
        return None, None
        
    def update_open_headings(self, open_headings: List[Tuple[int, str]], line: str) -> List[Tuple[int, str]]:
        """
        Updates the list of open headers according to the current line.
        
        The strategy is:
        - If an H1 is found, the list is reset.
        - If the line is a header of level >1:
            * If the list is empty, it's added.
            * If the last open header has a level less than or equal to the current,
              it's added without removing the previous one (to preserve siblings).
            * If the new header is of a higher level (smaller number) 
              than the last one, those of a level lower than the new one are preserved and the new one is added.
              
        Args:
            open_headings: Current list of open headers [(level, text), ...].
            line: Text line to analyze.
            
        Returns:
            Updated list of open headers.
        """
        lvl, txt = self.get_heading_level(line)
        if lvl is None:
            # Line without header, state is not modified
            return open_headings

        if lvl == 1:
            # An H1 closes all previous context
            return [(1, txt)]
        else:
            if not open_headings:
                return [(lvl, txt)]
            else:
                # If the last header has a level less than or equal, the current is added
                if open_headings[-1][0] <= lvl:
                    open_headings.append((lvl, txt))
                else:
                    # If the new header is of a higher level (more important)
                    # only headers of a higher level (lower number) than the new one are preserved
                    new_chain = [item for item in open_headings if item[0] < lvl]
                    new_chain.append((lvl, txt))
                    open_headings = new_chain
            return open_headings
    
    def process_document(self, document_path: str) -> Tuple[Dict[str, Any], str]:
        """
        Processes a complete Markdown document.
        
        Args:
            document_path: Path to the Markdown file.
            
        Returns:
            Tuple with (metadata, content) of the document.
        """
        # Read content
        content = self.read_markdown_file(document_path)
        
        # Extract metadata
        metadata = self.extract_metadata(document_path, content)
        
        return metadata, content
    
    def process_batch(self, document_paths: List[str]) -> List[Tuple[Dict[str, Any], str]]:
        """
        Processes a batch of Markdown documents.
        
        Args:
            document_paths: List of paths to Markdown files.
            
        Returns:
            List of tuples (metadata, content) for each document.
        """
        results = []
        
        for path in document_paths:
            try:
                result = self.process_document(path)
                results.append(result)
                logger.info(f"Document processed successfully: {path}")
            except Exception as e:
                logger.error(f"Error processing document {path}: {e}")
                # Continue with the next document
        
        return results
    
    def find_markdown_files(self, directory_path: str) -> List[str]:
        """
        Searches for all Markdown files in a directory.
        
        Args:
            directory_path: Path of the directory to explore.
            
        Returns:
            List of paths to found Markdown files.
        """
        markdown_files = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.warning(f"Directory {directory_path} does not exist")
            return []
            
        if not directory.is_dir():
            logger.warning(f"{directory_path} is not a directory")
            return []
        
        # Search for .md and .markdown files
        for ext in ["*.md", "*.markdown"]:
            markdown_files.extend([str(f) for f in directory.glob(ext)])
        
        logger.info(f"Found {len(markdown_files)} Markdown files in {directory_path}")
        return markdown_files
