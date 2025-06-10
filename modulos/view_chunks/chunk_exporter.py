"""Chunk exporter to text files for visualization and analysis.

This module allows exporting chunks stored in the database
to plain text files located in the same path as the original
Markdown files. Includes document metadata and details of each chunk.
"""

import os
import logging
import gc
from typing import Dict, Any, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class ChunkExporter:
    """
    Class for exporting document chunks to text files.
    
    Extracts chunk information from the database and generates
    text files with readable format for visualization and analysis.
    """
    
    def __init__(self, db_instance):
        """
        Initializes the exporter with a database instance.
        
        Args:
            db_instance: Vector database instance
        """
        self.db = db_instance
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # Base directory for exports if none is specified
        self.output_base_dir = "exported_chunks"
        # Format options
        self.include_separators = True
        self.include_chunk_number = True
    
    def export_document_chunks(self, document_path: str, output_path: Optional[str] = None) -> bool:
        """Exports the chunks of a specific document to a text file.
        
        Args:
            document_path: Path to the original document
            output_path: Optional path for the output file
            
        Returns:
            True if the export was successful, False otherwise
        """
        try:
            # Normalize the path for database search
            normalized_path = os.path.normpath(document_path)
            
            # Search for the document in the database
            document = self.find_document_by_path(normalized_path)
            if not document:
                self.logger.warning(f"Document not found for: {document_path}")
                return False
            
            # Determine output path if not provided
            if output_path is None:
                output_path = document_path + ".txt"
            
            self.logger.info(f"Exporting document chunks: {document_path} -> {output_path}")
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Get and process chunks in batches to handle large documents
            offset = 0
            limit = 100  # Get chunks in batches for large documents
            total_chunks = 0
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header and metadata
                f.write(self.format_document_metadata(document))
                
                # Write chunks in batches
                while True:
                    self.logger.debug(f"Getting chunk batch: offset={offset}, limit={limit}")
                    batch = self.db.get_chunks_by_document(document['id'], offset, limit)
                    
                    if not batch:
                        break
                    
                    total_chunks += len(batch)
                    
                    # Write the chunks from this batch
                    for i, chunk in enumerate(batch, offset + 1):
                        chunk_text = self.format_chunk(chunk, i)
                        f.write(chunk_text)
                    
                    # Free memory
                    del batch
                    if offset % 500 == 0:  # Force GC periodically for large documents
                        gc.collect()
                    
                    offset += limit
                
                # Write summary at the end
                f.write("\n\n================ SUMMARY ================\n")
                f.write(f"Total chunks: {total_chunks}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            self.logger.info(f"Export completed: {output_path} with {total_chunks} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting chunks for {document_path}: {e}")
            return False
    
    def find_document_by_path(self, document_path: str) -> Optional[Dict[str, Any]]:
        """Searches for a document in the database by its file path.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Dictionary with document information or None if not found
        """
        try:
            self.logger.debug(f"Searching for document with path: {document_path}")
            
            # Execute query in the database
            # Most of our implementations have a cursor
            cursor = self.db._cursor
            
            # Search for the document by exact path
            cursor.execute(
                "SELECT id, title, url, file_path, created_at FROM documents WHERE file_path = ?", 
                (document_path,)
            )
            doc = cursor.fetchone()
            
            if doc:
                # Convert to dictionary if it's a SQLite row
                if hasattr(doc, 'keys'):
                    return dict(doc)
                else:
                    # Create dictionary manually
                    return {
                        'id': doc[0],
                        'title': doc[1],
                        'url': doc[2],
                        'file_path': doc[3],
                        'created_at': doc[4]
                    }
            else:
                # Try alternative search - with partial path matching
                cursor.execute(
                    "SELECT id, title, url, file_path, created_at FROM documents WHERE file_path LIKE ?", 
                    (f"%{os.path.basename(document_path)}%",)
                )
                doc = cursor.fetchone()
                
                if doc:
                    if hasattr(doc, 'keys'):
                        return dict(doc)
                    else:
                        return {
                            'id': doc[0],
                            'title': doc[1],
                            'url': doc[2],
                            'file_path': doc[3],
                            'created_at': doc[4]
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error searching for document: {e}")
            return None
    
    def format_document_metadata(self, document: Dict[str, Any]) -> str:
        """Formats the document metadata for the output file.
        
        Args:
            document: Dictionary with document information
            
        Returns:
            Formatted string with metadata
        """
        lines = []
        lines.append("================ DOCUMENT METADATA ================\n")
        lines.append(f"ID: {document.get('id', 'N/A')}")
        lines.append(f"Title: {document.get('title', 'No title')}")
        lines.append(f"Path: {document.get('file_path', 'N/A')}")
        
        # Format date if available
        created_at = document.get('created_at')
        if created_at:
            # Try to convert to readable format if it's a timestamp
            try:
                if isinstance(created_at, (int, float)):
                    created_at = datetime.fromtimestamp(created_at).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                pass
            lines.append(f"Processing date: {created_at}")
        
        # Get the number of chunks in the document
        try:
            doc_id = document.get('id')
            if doc_id:
                # Count the document chunks
                cursor = self.db._cursor
                cursor.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,))
                count_result = cursor.fetchone()
                total_chunks = count_result[0] if count_result else 0
                lines.append(f"Total chunks: {total_chunks}")
        except Exception as e:
            self.logger.warning(f"Could not get total number of chunks: {e}")
        
        lines.append("\n=================== GENERATED CHUNKS ===================\n")
        
        return "\n".join(lines)
    
    def format_chunk(self, chunk: Dict[str, Any], chunk_num: int) -> str:
        """Formats a chunk for the output file.
        
        Args:
            chunk: Dictionary with chunk information
            chunk_num: Sequential number of the chunk
            
        Returns:
            Formatted string with chunk information
        """
        lines = []
        lines.append(f"\n----- CHUNK #{chunk_num} (ID: {chunk.get('id', 'N/A')}) -----\n")
        
        # Chunk information
        if 'page' in chunk and chunk['page']:
            lines.append(f"Page: {chunk['page']}")
        
        if 'header' in chunk and chunk['header']:
            lines.append(f"Header: {chunk['header']}")
        
        # Chunk content
        lines.append(f"\nContent:\n{'-' * 50}")
        lines.append(chunk.get('text', ''))
        lines.append(f"{'-' * 50}\n")
        
        return "\n".join(lines)
    
    def export_all_chunks_from_db(self) -> Dict[str, bool]:
        """
        Exports chunks from all documents in the database.
        
        Returns:
            Dictionary with document paths and whether their export was successful
        """
        try:
            # Get all documents from the database
            cursor = self.db._cursor
            cursor.execute("SELECT id, title, url, file_path, created_at FROM documents")
            
            documents = []
            for row in cursor.fetchall():
                if hasattr(row, 'keys'):
                    documents.append(dict(row))
                else:
                    documents.append({
                        'id': row[0],
                        'title': row[1],
                        'url': row[2],
                        'file_path': row[3],
                        'created_at': row[4]
                    })
            
            results = {}
            
            # Export chunks for each document
            for doc in documents:
                file_path = doc.get('file_path')
                if file_path:
                    result = self.export_document_chunks(file_path)
                    results[file_path] = result
                
                # Free resources every few documents
                if len(results) % 10 == 0:
                    gc.collect()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error exporting all chunks: {e}")
            return {}

    def export_chunk(self, chunk, filename_prefix, metadata=None):
        """Exports an individual chunk to a text file.
        
        Args:
            chunk: The chunk to export
            filename_prefix: Prefix for the filename
            metadata: Optional additional metadata
            
        Returns:
            Path to the generated file
        """
        try:
            # Ensure the directory exists
            os.makedirs(self.output_base_dir, exist_ok=True)
            
            # Create filename
            filename = os.path.join(self.output_base_dir, f"{filename_prefix}_{chunk.id if hasattr(chunk, 'id') else 'unnamed'}.txt")
            
            with open(filename, 'w', encoding='utf-8') as f:
                # Write metadata if it exists
                if metadata:
                    f.write("================ METADATA ================\n")
                    for key, value in metadata.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                # Write top separator if enabled
                if self.include_separators:
                    f.write("="*40 + "\n")
                
                # Write chunk number if enabled
                if self.include_chunk_number:
                    f.write(f"Chunk #{getattr(chunk, 'id', 1)}\n\n")
                
                # Write header if it exists
                if hasattr(chunk, 'header') and chunk.header:
                    f.write(f"Header: {chunk.header}\n\n")
                
                # Write chunk text
                f.write(chunk.text)
                
                # Write bottom separator if enabled
                if self.include_separators:
                    f.write("\n" + "="*40)
            
            self.logger.info(f"Chunk exported to: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error exporting chunk: {e}")
            return None
    
    def export_chunks(self, chunks, base_filename, metadata=None):
        """
        Exports multiple chunks to text files.
        
        Args:
            chunks: List of chunks to export
            base_filename: Base name for the files
            metadata: Optional additional metadata
            
        Returns:
            List of paths to the generated files
        """
        filenames = []
        
        for i, chunk in enumerate(chunks):
            filename = self.export_chunk(
                chunk, 
                f"{base_filename}_{i+1}",
                metadata
            )
            if filename:
                filenames.append(filename)
        
        return filenames


def export_chunks_for_files(file_paths: str, db_instance) -> Dict[str, bool]:
    """
    Exports chunks for all specified Markdown files.
    
    Args:
        file_paths: Path to a directory or individual file
        db_instance: Vector database instance
        
    Returns:
        Dictionary with processed paths and whether their export was successful
    """
    exporter = ChunkExporter(db_instance)
    results = {}
    
    try:
        if os.path.isdir(file_paths):
            # Recursively traverse the directory
            logger.info(f"Exporting chunks for all Markdown files in: {file_paths}")
            
            for root, _, files in os.walk(file_paths):
                for file in files:
                    if file.lower().endswith('.md'):
                        md_path = os.path.join(root, file)
                        result = exporter.export_document_chunks(md_path)
                        results[md_path] = result
                
                # Free resources periodically
                if len(results) % 10 == 0:
                    gc.collect()
                    
        elif os.path.isfile(file_paths) and file_paths.lower().endswith('.md'):
            # Export a single file
            logger.info(f"Exporting chunks for Markdown file: {file_paths}")
            result = exporter.export_document_chunks(file_paths)
            results[file_paths] = result
        else:
            logger.warning(f"The provided path is not a valid Markdown file or directory: {file_paths}")
    
    except Exception as e:
        logger.error(f"Error exporting chunks: {e}")
    
    # Free resources
    del exporter
    gc.collect()
    
    # Now generate t-SNE visualizations for the same files
    try:
        logger.info("Generating t-SNE visualizations for processed files...")
        from .tsne_visualizer import visualize_tsne_for_files
        
        # Get documents for the processed files and generate visualizations
        for file_path in file_paths if isinstance(file_paths, list) else [file_paths]:
            visualize_tsne_for_files(file_path)
    except Exception as e:
        logger.error(f"Error generating t-SNE visualizations: {e}")
    
    return results