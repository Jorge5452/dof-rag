from abc import ABC, abstractmethod
import json
import logging
import struct
from typing import Any, Dict, List, Optional

import numpy as np

class VectorialDatabase(ABC):
    """
    Abstract class that defines the interface for vectorial databases.
    
    Defines methods for:
    - Connecting/disconnecting to the database
    - Creating the schema
    - Inserting documents and chunks
    - Searching documents by embedding
    - And more
    
    Classes implementing this interface must provide concrete implementations
    for these methods.
    """
    
    def __init__(self, embedding_dim: Optional[int] = None) -> None:
        """Initialize the logger and other common attributes.
        
        Args:
            embedding_dim: Dimension of the embedding vectors (optional for base class)
        """
        self._logger: logging.Logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._conn: Optional[Any] = None
        self._cursor: Optional[Any] = None
        self._db_path: Optional[str] = None
        self._metadata: Dict[str, Any] = {}  # Local metadata cache
        self._in_transaction: bool = False  # Flag to track if there's an active transaction
        self._embedding_dim: Optional[int] = embedding_dim  # Store embedding dimension
    
    @abstractmethod
    def connect(self, db_path: str) -> bool:
        """
        Connect to the database.
        
        Args:
            db_path: Path to the database
            
        Returns:
            True if connection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def close_connection(self) -> bool:
        """
        Close the database connection.
        
        Returns:
            True if closing was successful, False otherwise
        """
        pass
    
    def close(self) -> bool:
        """
        Alias for close_connection() for compatibility.
        
        Returns:
            True if closing was successful, False otherwise
        """
        return self.close_connection()
    
    @abstractmethod
    def create_schema(self) -> bool:
        """
        Create the database schema if it doesn't exist.
        
        Returns:
            True if creation was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def insert_document(self, document: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Optional[int]:
        """
        Insert a document and its chunks into the database.
        
        Args:
            document: Dictionary with document data
            chunks: List of dictionaries with chunk data
            
        Returns:
            ID of the inserted document, None if it fails
        """
        pass
    
    @abstractmethod
    def get_chunks_by_document(self, document_id: int, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get chunks from a document.
        
        Args:
            document_id: Document ID
            offset: Offset for pagination (optional)
            limit: Limit for pagination (optional)
            
        Returns:
            List of chunks
        """
        pass
    
    @abstractmethod
    def vector_search(self, query_embedding: List[float], filters: Optional[Dict[str, Any]] = None, 
                     n_results: int = 5, include_neighbors: bool = False) -> List[Dict[str, Any]]:
        """
        Perform a vector similarity search.
        
        Args:
            query_embedding: Query embedding vector
            filters: Search filters (optional)
            n_results: Maximum number of results (optional)
            include_neighbors: Whether to include neighboring chunks in results (optional)
            
        Returns:
            List of chunks ordered by similarity
        """
        pass
    
    def document_exists(self, file_path: str) -> bool:
        """
        Check if a document already exists in the database.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if the document exists, False otherwise
        """
        try:
            self._cursor.execute(
                "SELECT COUNT(*) FROM documents WHERE file_path = ?", 
                (file_path,)
            )
            count = self._cursor.fetchone()[0]
            return count > 0
        except Exception as e:
            self._logger.error(f"Error checking document existence: {str(e)}")
            return False
    
    def serialize_vector(self, vector: List[float]) -> bytes:
        """
        Serialize a vector to bytes for storage.
        
        Args:
            vector: Embedding vector
            
        Returns:
            Serialized vector
        """
        return struct.pack(f'{len(vector)}f', *vector)
    
    def deserialize_vector(self, serialized: bytes, vector_dim: Optional[int] = None) -> List[float]:
        """
        Deserialize a vector from bytes.
        
        Args:
            serialized: Serialized vector
            vector_dim: Vector dimension (optional)
            
        Returns:
            Deserialized vector
        """
        if vector_dim is None:
            vector_dim = len(serialized) // 4  # 4 bytes per float
        
        return list(struct.unpack(f'{vector_dim}f', serialized))
    
    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Insert multiple chunks into the database.
        
        Args:
            chunks: List of dictionaries with chunk data.
                Each chunk must contain at least 'text', 'embedding', 'metadata'
        
        Returns:
            True if insertion was successful, False otherwise.
        """
        # Default implementation that should be overridden
        return False
    
    def insert_chunk(self, text: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Insert a single chunk into the database.
        
        Args:
            text: Chunk text
            embedding: Embedding vector
            metadata: Associated chunk metadata (optional)
            
        Returns:
            ID of the inserted chunk, None if it fails
        """
        # Default implementation that should be overridden
        return None

    @abstractmethod
    def insert_document_metadata(self, document: Dict[str, Any]) -> Optional[int]:
        """
        Insert only document metadata into the database, without chunks.
        Useful for streaming processing of large documents.
        
        Args:
            document: Dictionary with document data
            
        Returns:
            ID of the inserted document, None if it fails
        """
        pass
    
    @abstractmethod
    def insert_single_chunk(self, document_id: int, chunk_data: Dict[str, Any]) -> Optional[int]:
        """
        Insert a single chunk associated with a specific document.
        
        Args:
            document_id: ID of the document the chunk belongs to
            chunk_data: Dictionary with chunk data:
                - text (str): Chunk text
                - header (str, optional): Chunk header
                - page (str, optional): Page number or identifier
                - embedding (list): Chunk embedding vector
                - embedding_dim (int): Embedding dimension
                
        Returns:
            ID of the inserted chunk, None if it fails
        """
        pass

    def get_db_path(self) -> Optional[str]:
        """
        Return the database path.
        
        Returns:
            The database path or None if not connected yet.
        """
        return self._db_path
    
    def store_metadata(self, key: str, value: Any) -> bool:
        """
        Store metadata in the database.
        
        Args:
            key: Metadata key
            value: Metadata value (must be JSON serializable)
            
        Returns:
            True if stored correctly
        """
        try:
            # Create metadata table if it doesn't exist
            self._cursor.execute("""
                CREATE TABLE IF NOT EXISTS db_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Serialize value to JSON if necessary
            if not isinstance(value, (str, int, float, bool, type(None))):
                value = json.dumps(value)
                
            # Insert or update the metadata
            self._cursor.execute("""
                INSERT OR REPLACE INTO db_metadata (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP);
            """, (key, value))
            
            self._conn.commit()
            self._metadata[key] = value  # Update local cache
            return True
            
        except Exception as e:
            self._logger.error(f"Error storing metadata {key}: {e}")
            return False
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Retrieve metadata from the database.
        
        Args:
            key: Metadata key
            default: Default value if key doesn't exist
            
        Returns:
            The metadata value, or default value if it doesn't exist
        """
        # First try to get from memory cache
        if key in self._metadata:
            return self._metadata[key]
            
        try:
            # Check if table exists
            self._cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='db_metadata';
            """)
            if not self._cursor.fetchone():
                return {}
                
            # Get all metadata
            self._cursor.execute("SELECT key, value FROM db_metadata;")
            results = self._cursor.fetchall()
            
            metadata: Dict[str, Any] = {}
            for key, value in results:
                try:
                    # Try to deserialize JSON
                    metadata[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, use the value as is
                    metadata[key] = value
            
            # Update in-memory cache
            self._metadata.update(metadata)
            return metadata
            
        except Exception as e:
            self._logger.error(f"Error listing metadata: {e}")
            return {}
    
    @abstractmethod
    def optimize_database(self) -> bool:
        """
        Optimize the database (compaction, index recreation, etc.)
        
        Returns:
            True if optimization was successful, False otherwise
        """
        pass
    
    # --- TRANSACTION HANDLING METHODS ---
    
    def begin_transaction(self) -> bool:
        """
        Start a manual transaction for bulk insertion.
        Useful for improving performance with many insertions.
        
        Returns:
            True if transaction was started correctly
        """
        try:
            if hasattr(self, "_conn") and self._conn:
                # Check if there's already an active transaction
                if hasattr(self, "_in_transaction") and self._in_transaction:
                    self._logger.debug("There's already an active transaction, ignoring begin_transaction")
                    return True  # Return True because conceptually we're already in a transaction
                
                self._conn.execute("BEGIN TRANSACTION;")
                self._in_transaction = True
                self._logger.info("Transaction started")
                return True
            return False
        except Exception as e:
            self._logger.error(f"Error starting transaction: {e}")
            return False
    
    def commit_transaction(self) -> bool:
        """
        Commit an ongoing transaction.
        
        Returns:
            True if transaction was committed correctly
        """
        try:
            if hasattr(self, "_conn") and self._conn:
                # Check if there's an active transaction
                if not hasattr(self, "_in_transaction") or not self._in_transaction:
                    self._logger.debug("No active transaction to commit")
                    return True  # No transaction to commit, but it's not an error
                
                self._conn.commit()
                self._in_transaction = False
                self._logger.info("Transaction committed")
                return True
            return False
        except Exception as e:
            self._logger.error(f"Error committing transaction: {e}")
            return False
    
    def rollback_transaction(self) -> bool:
        """
        Rollback an ongoing transaction.
        
        Returns:
            True if transaction was rolled back correctly
        """
        try:
            if hasattr(self, "_conn") and self._conn:
                # Check if there's an active transaction
                if not hasattr(self, "_in_transaction") or not self._in_transaction:
                    self._logger.debug("No active transaction to revert")
                    return True  # No transaction to revert, but no error
                
                self._conn.rollback()
                self._in_transaction = False
                self._logger.info("Transaction rolled back")
                return True
            return False
        except Exception as e:
            self._logger.error(f"Error rolling back transaction: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics (documents, chunks, etc.)
        
        Returns:
            Dict[str, Any]: Database statistics including document count, chunk count,
                          latest document info, creation date, and database size
        """
        stats: Dict[str, Any] = {}
        
        try:
            # Total documents count
            self._cursor.execute("SELECT COUNT(*) FROM documents;")
            stats["total_documents"] = self._cursor.fetchone()[0]
            
            # Total chunks count
            self._cursor.execute("SELECT COUNT(*) FROM chunks;")
            stats["total_chunks"] = self._cursor.fetchone()[0]
            
            # Most recent document
            self._cursor.execute("""
                SELECT id, title, created_at FROM documents
                ORDER BY created_at DESC LIMIT 1;
            """)
            doc = self._cursor.fetchone()
            if doc:
                stats["latest_document"] = {
                    "id": doc[0],
                    "title": doc[1],
                    "created_at": doc[2]
                }
            
            # Database creation date
            stats["db_created"] = self.get_metadata("db_created", "unknown")
            
            # Database size in MB
            try:
                if self._db_path and self._db_path != ":memory:":
                    import os
                    stats["db_size_mb"] = os.path.getsize(self._db_path) / (1024 * 1024)
            except Exception:
                pass
            
            return stats
            
        except Exception as e:
            self._logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
            
    def convert_embedding_dimension(self, embedding: List[float], target_dim: int) -> List[float]:
        """
        Convert an embedding to a specific dimension (truncating or padding).
        Useful for compatibility between different models.
        
        Args:
            embedding: Original embedding vector
            target_dim: Target dimension
            
        Returns:
            Converted embedding vector
        """
        if len(embedding) == target_dim:
            return embedding
            
        if len(embedding) > target_dim:
            # Truncate if larger
            return embedding[:target_dim]
        else:
            # Pad with zeros
            return embedding + [0.0] * (target_dim - len(embedding))
    
    def _vector_search_manual(self, query_embedding: List[float], filters: Optional[Dict[str, Any]] = None, n_results: int = 5, include_neighbors: bool = False) -> List[Dict[str, Any]]:
        """
        Manual implementation of vector search when there's no native support.
        
        This implementation loads all embeddings from the database and calculates 
        cosine similarity manually using numpy operations. It's a fallback method
        for databases that don't have built-in vector search capabilities.
        
        Args:
            query_embedding: Query vector as a list of floats
            filters: Optional dictionary with search filters:
                   - 'document_id': Filter by specific document ID
                   - 'min_similarity': Minimum similarity threshold
            n_results: Maximum number of results to return
            include_neighbors: Whether to include neighboring chunks in results
            
        Returns:
            List[Dict[str, Any]]: List of chunks ordered by similarity score (descending).
                                Each chunk contains: id, document_id, text, header, page,
                                title, url, file_path, and similarity score
        """
        try:
            # Convert query embedding to numpy array for efficient calculations
            query_vector: np.ndarray = np.array(query_embedding, dtype=np.float32)
            
            # Normalize query vector for cosine similarity
            query_norm: float = np.linalg.norm(query_vector)
            if query_norm > 0:
                query_vector = query_vector / query_norm
            
            # Get all chunks with their embeddings
            self._cursor.execute("""
                SELECT c.id, c.document_id, c.text, c.embedding, c.embedding_dim, 
                       c.header, c.page, d.title, d.url, d.file_path
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
            """)
            
            rows = self._cursor.fetchall()
            
            # Calculate similarity for each chunk
            similarities: List[Dict[str, Any]] = []
            
            for row in rows:
                # Extract fields according to query order
                chunk_id: int = row[0]
                doc_id: int = row[1]
                text: str = row[2]
                embedding_blob: bytes = row[3]
                embedding_dim: Optional[int] = row[4] if row[4] else self._embedding_dim
                header: Optional[str] = row[5]
                page: Optional[str] = row[6]
                title: str = row[7]
                url: Optional[str] = row[8]
                file_path: Optional[str] = row[9]
                
                if not embedding_blob:
                    continue
                
                # Deserialize embedding
                try:
                    chunk_vector: np.ndarray = np.array(self.deserialize_vector(embedding_blob, embedding_dim), dtype=np.float32)
                except Exception as e:
                    self._logger.warning(f"Error deserializing vector from chunk {chunk_id}: {str(e)}")
                    continue
                
                # Normalize chunk vector
                chunk_norm: float = np.linalg.norm(chunk_vector)
                if chunk_norm > 0:
                    chunk_vector = chunk_vector / chunk_norm
                
                # Calculate cosine similarity
                similarity: float = np.dot(query_vector, chunk_vector)
                
                # Apply filters if they exist
                if filters:
                    if 'document_id' in filters and doc_id != filters['document_id']:
                        continue
                    if 'min_similarity' in filters and similarity < filters['min_similarity']:
                        continue
                
                # Only include results above threshold
                similarity_threshold: float = filters.get('min_similarity', self._similarity_threshold) if filters else self._similarity_threshold
                if similarity >= similarity_threshold:
                    similarities.append({
                        "id": chunk_id,
                        "document_id": doc_id,
                        "text": text,
                        "header": header,
                        "page": page,
                        "title": title,
                        "url": url,
                        "file_path": file_path,
                        "similarity": float(similarity)
                    })
            
            # Sort by descending similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Limit to n_results
            top_results: List[Dict[str, Any]] = similarities[:n_results]
            
            # Include neighboring chunks if requested
            if include_neighbors and top_results:
                best_match = top_results[0]
                neighbors = self._get_adjacent_chunks(best_match["document_id"], best_match["id"])
                
                if neighbors:
                    # Add neighbors at the beginning of results
                    return neighbors + top_results
            
            return top_results
            
        except Exception as e:
            self._logger.error(f"Error in manual vector search: {str(e)}")
            return []
            
    def _get_adjacent_chunks(self, document_id: int, chunk_id: int) -> List[Dict[str, Any]]:
        """
        Get chunks adjacent to the specified chunk.
        
        This method retrieves the previous and next chunks relative to the given chunk ID
        within the same document. Basic implementation that should be customized in 
        concrete classes if needed for better performance or specific requirements.
        
        Args:
            document_id: ID of the document containing the chunks
            chunk_id: ID of the reference chunk to find neighbors for
            
        Returns:
            List[Dict[str, Any]]: List of adjacent chunks with their metadata.
                                Each chunk dict contains: id, document_id, text, header,
                                page, title, url, file_path, and similarity (set to 0.0)
        """
        adjacent_chunks: List[Dict[str, Any]] = []
        
        try:
            # Get previous chunk (closest smaller ID)
            self._cursor.execute("""
                SELECT c.id, c.document_id, c.text, c.header, c.page, d.title, d.url, d.file_path
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = ? AND c.id < ?
                ORDER BY c.id DESC
                LIMIT 1
            """, [document_id, chunk_id])
            
            prev_chunk = self._cursor.fetchone()
            if prev_chunk:
                chunk_id, doc_id, text, header, page, title, url, file_path = prev_chunk
                adjacent_chunks.append({
                    "id": chunk_id,
                    "document_id": doc_id,
                    "text": text,
                    "header": header,
                    "page": page,
                    "title": title,
                    "url": url,
                    "file_path": file_path,
                    "similarity": 0.0  # Mark as neighbor with similarity 0
                })
            
            # Get next chunk (closest larger ID)
            self._cursor.execute("""
                SELECT c.id, c.document_id, c.text, c.header, c.page, d.title, d.url, d.file_path
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = ? AND c.id > ?
                ORDER BY c.id ASC
                LIMIT 1
            """, [document_id, chunk_id])
            
            next_chunk = self._cursor.fetchone()
            if next_chunk:
                chunk_id, doc_id, text, header, page, title, url, file_path = next_chunk
                adjacent_chunks.append({
                    "id": chunk_id,
                    "document_id": doc_id,
                    "text": text,
                    "header": header,
                    "page": page,
                    "title": title,
                    "url": url,
                    "file_path": file_path,
                    "similarity": 0.0  # Mark as neighbor with similarity 0
                })
            
            return adjacent_chunks
        except Exception as e:
            self._logger.error(f"Error getting adjacent chunks: {str(e)}")
            return []

    def insert_chunks_batch(self, document_id: int, chunks_data: List[Dict[str, Any]]) -> Optional[List[int]]:
        """
        Insert a batch of chunks associated with a specific document.
        
        This method allows efficient insertion of multiple chunks in a single operation,
        which can substantially improve performance by reducing the number of
        individual operations and leveraging transactions.
        
        Concrete implementations can optimize this process using
        specific features of the underlying database engine.
        
        Args:
            document_id: ID of the document to which the chunks belong
            chunks_data: List of dictionaries with data for each chunk:
                - text (str): Chunk text
                - header (str, optional): Chunk header
                - page (str, optional): Page number or identifier
                - embedding (list): Chunk embedding vector
                - embedding_dim (int): Embedding dimension
                
        Returns:
            List of IDs of inserted chunks, or None if it fails
        """
        # Default implementation that inserts each chunk individually
        # Derived classes should override this with a more efficient implementation
        self._logger.debug(f"Inserting batch of {len(chunks_data)} chunks using default method")
        chunk_ids: List[int] = []
        try:
            # Ensure we're in a transaction
            in_transaction: bool = getattr(self, '_in_transaction', False)
            if not in_transaction:
                self.begin_transaction()
                transaction_started = True
            else:
                transaction_started = False
                
            # Process each chunk
            for chunk_data in chunks_data:
                chunk_id = self.insert_single_chunk(document_id, chunk_data)
                if chunk_id:
                    chunk_ids.append(chunk_id)
                else:
                    self._logger.warning("Failed to insert individual chunk in batch")
            
            # Commit only if we started the transaction
            if transaction_started:
                self.commit_transaction()
                
            return chunk_ids
        except Exception as e:
            self._logger.error(f"Error inserting chunk batch: {e}")
            # Rollback only if we started the transaction
            if 'transaction_started' in locals() and transaction_started:
                self.rollback_transaction()
            return None