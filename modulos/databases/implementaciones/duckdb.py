import os
import logging
import json
from typing import List, Dict, Any
import numpy as np

try:
    import duckdb
except ImportError:
    duckdb = None

from modulos.databases.VectorialDatabase import VectorialDatabase

logger = logging.getLogger(__name__)

class DuckDBVectorialDatabase(VectorialDatabase):
    """
    DuckDB implementation of VectorialDatabase for vector similarity search.
    
    Embedding dimensions are fixed during initialization and used for all subsequent
    vector operations, ensuring consistency and optimal performance.
    """
    
    def __init__(self, embedding_dim: int):
        """
        Initialize DuckDB database with a specific embedding dimension.
        
        Args:
            embedding_dim: Fixed dimension for embeddings to be used.
                          This value must come from the embedding model and is required.
        """
        super().__init__(embedding_dim)  # Initialize superclass with embedding dimension
        
        if duckdb is None:
            raise ImportError("DuckDB is not installed. Install it with 'pip install duckdb'.")
        
        if embedding_dim is None or embedding_dim <= 0:
            raise ValueError("Embedding dimension must be a positive number")
            
        self._conn = None
        self._ext_loaded = False
        self._schema_created = False
        
        # Load similarity threshold from configuration
        try:
            from config import config, DefaultValues
            db_config = config.get_database_config()
            duckdb_config = db_config.get('duckdb', {})
            self._similarity_threshold = duckdb_config.get('similarity_threshold', DefaultValues.SIMILARITY_THRESHOLD)
        except Exception as e:
            from config import DefaultValues
            logger.warning(f"Could not load similarity_threshold from config: {e}, using default {DefaultValues.SIMILARITY_THRESHOLD}")
            self._similarity_threshold = DefaultValues.SIMILARITY_THRESHOLD
            
        self._db_path = None  # Store database path for later use
        self._in_transaction = False  # Track active transaction state
        self._metadata = {}  # Local metadata cache
        
        # Set fixed embedding dimension
        self._embedding_dim = embedding_dim
        logger.info(f"Embedding dimension set to: {self._embedding_dim}")
    
    def connect(self, db_path: str) -> bool:
        """
        Connect to DuckDB database.
        
        Args:
            db_path: Path to database file
            
        Returns:
            True if connection was successful, False otherwise
        """
        try:
            # Store database path for later use
            self._db_path = db_path
            
            # Read DuckDB-specific configuration
            from config import config, DefaultValues
            db_config = config.get_database_config()
            duckdb_config = db_config.get('duckdb', {})
            
            # Get configuration parameters
            memory_limit = duckdb_config.get('memory_limit', DefaultValues.DUCKDB_MEMORY_LIMIT)
            threads_config = duckdb_config.get('threads', DefaultValues.DUCKDB_THREADS)
            self._similarity_threshold = duckdb_config.get('similarity_threshold', DefaultValues.SIMILARITY_THRESHOLD)
            
            # Process threads value: DuckDB requires at least 1 thread
            import multiprocessing
            if threads_config == 'auto' or threads_config == 0:
                # Use available CPU cores, minimum 1
                threads = max(1, multiprocessing.cpu_count())
            else:
                # Convert to integer, ensuring minimum value of 1
                try:
                    threads = max(1, int(threads_config))
                except (ValueError, TypeError):
                    # Use safe default if conversion fails
                    logger.warning(f"Invalid threads value: '{threads_config}'. Using default value 1.")
                    threads = 1
            
            # Additional validation to ensure valid number
            if not isinstance(threads, int) or threads < 1:
                threads = 1
                logger.warning(f"Threads value corrected to {threads} to meet DuckDB requirements")
            
            # DuckDB connection configuration
            connection_config = {
                'memory_limit': memory_limit,
                'threads': threads  # Must be integer, not string
            }
            
            logger.debug(f"DuckDB connection configuration: {connection_config}")
            
            if db_path == ':memory:':
                self._conn = duckdb.connect(database=':memory:', config=connection_config)
            else:
                # Ensure directory exists
                db_dir = os.path.dirname(os.path.abspath(db_path))
                if not os.path.exists(db_dir):
                    try:
                        os.makedirs(db_dir, exist_ok=True)
                        logger.info(f"Database directory created: {db_dir}")
                    except Exception as e:
                        logger.error(f"Error creating database directory: {e}")
                        return False
                
                # Enhanced DuckDB configuration
                self._conn = duckdb.connect(database=db_path, config=connection_config)
            
            # Verify connection was established correctly
            if self._conn is None:
                logger.error("DuckDB connection created as None")
                return False
                
            # Execute simple query to verify connection works
            try:
                self._conn.execute("SELECT 1")
                self._conn.fetchone()
            except Exception as e:
                logger.error(f"DuckDB connection created but not functional: {e}")
                return False
            
            # Initialize cursor for abstract class compatibility
            self._cursor = self._conn
            
            # Load necessary extensions immediately after connecting
            # Don't fail if extensions cannot be loaded
            self.load_extensions()
            
            logger.info(f"Successful DuckDB connection: {db_path} (memory: {memory_limit}, threads: {threads})")
            
            # Create schema after successful connection
            if not self.create_schema():
                logger.warning("Could not create complete schema in DuckDB, but process will continue")
            
            return True
        except Exception as e:
            logger.error(f"Error connecting to DuckDB: {e}")
            self._conn = None
            self._cursor = None
            return False
    
    def close_connection(self) -> bool:
        """
        Close database connection.
        
        Returns:
            True if closed correctly, False otherwise
        """
        try:
            if self._conn:
                self._conn.close()
                self._conn = None
                logger.info("DuckDB connection closed successfully")
            return True
        except Exception as e:
            logger.error(f"Error closing DuckDB connection: {e}")
            return False
    
    def create_schema(self) -> bool:
        """
        Create database schema if it doesn't exist.
        
        Returns:
            True if created correctly, False otherwise
        """
        if not self._conn:
            logger.error("No database connection available")
            return False
        
        try:
            # Check if tables already exist to avoid errors
            try:
                self._conn.execute("CREATE SCHEMA IF NOT EXISTS main")
            except Exception as e:
                logger.warning(f"Could not create main schema (can be ignored): {e}")
            
            # Check if main tables already exist
            try:
                self._conn.execute("SELECT 1 FROM documents LIMIT 1")
                self._conn.execute("SELECT 1 FROM chunks LIMIT 1")
                logger.info("Tables already exist in database, skipping schema creation")
                self._schema_created = True
                return True
            except Exception:
                # If error querying tables, they don't exist and must be created
                pass
            
            # Create documents table with auto-increment ID
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    url TEXT,
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create chunks table with auto-increment ID
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    document_id INTEGER,
                    text TEXT NOT NULL,
                    header TEXT,
                    page TEXT,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                )
            """)
            
            # Create metadata table if it doesn't exist
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS db_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create index for fast searches by document_id
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
            
            # Store metadata about embedding dimensions
            try:
                self.store_metadata("embedding_dim", str(self._embedding_dim))
            except Exception as e:
                logger.warning(f"Could not save embedding_dim metadata: {e}")
            
            self._schema_created = True
            logger.info("DuckDB schema created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating DuckDB schema: {e}")
            return False
    
    def load_extensions(self) -> bool:
        """
        Load extensions for vector search if available.
        
        Returns:
            True if extensions loaded correctly, False otherwise.
        """
        # Avoid loading extensions if already loaded
        if self._ext_loaded:
            logger.debug("DuckDB extensions already loaded")
            return True
            
        # Avoid trying to load extensions if no connection
        if not self._conn:
            logger.error("No database connection available to load extensions")
            return False
            
        try:
            # Read configured extensions
            from config import config
            db_config = config.get_database_config()
            duckdb_config = db_config.get('duckdb', {})
            extensions = duckdb_config.get('extensions', [])  # Empty list by default
            
            # Explicitly filter unwanted extensions
            excluded_extensions = ['httpfs', 'json']
            extensions = [ext for ext in extensions if ext not in excluded_extensions]
            
            logger.info(f"Extensions to load: {extensions}")
            
            # Install and load configured extensions
            for extension in extensions:
                try:
                    # Check if extension is already loaded
                    try:
                        self._conn.execute(f"SELECT 1 FROM pragma_installed_extensions() WHERE extension_name = '{extension}'")
                        already_installed = len(self._conn.fetchall()) > 0
                    except:
                        already_installed = False
                    
                    if not already_installed:
                        logger.debug(f"Installing DuckDB extension '{extension}'...")
                        self._conn.execute(f"INSTALL {extension};")
                    
                    logger.debug(f"Loading DuckDB extension '{extension}'...")
                    self._conn.execute(f"LOAD {extension};")
                    logger.info(f"DuckDB extension '{extension}' loaded successfully")
                except Exception as e:
                    logger.debug(f"Could not load extension '{extension}' (not critical): {e}")
            
            self._ext_loaded = True
            logger.info("DuckDB extension loading process completed")
            return True
        except Exception as e:
            logger.warning(f"Could not load all DuckDB extensions: {e}")
            # Don't fail completely if extensions don't load
            self._ext_loaded = True
            return True  # Return True to allow process to continue even with extension issues
    
    def serialize_vector(self, vector: List[float]) -> bytes:
        """
        Convert vector for DuckDB storage as BLOB.
        
        Args:
            vector: List of float values representing the vector
            
        Returns:
            Vector as bytes (BLOB format for DuckDB)
        """
        import struct
        
        # Adapt vector to configured dimension
        if len(vector) != self._embedding_dim:
            if len(vector) > self._embedding_dim:
                # Truncate if larger
                vector = vector[:self._embedding_dim]
                logger.debug(f"Vector truncated to {self._embedding_dim} dimensions")
            else:
                # Pad with zeros if smaller
                vector = vector + [0.0] * (self._embedding_dim - len(vector))
                logger.debug(f"Vector padded to {self._embedding_dim} dimensions")
        
        # Return as bytes for DuckDB BLOB
        return struct.pack(f"{self._embedding_dim}f", *vector)
    
    def deserialize_vector(self, vector_data, dim: int = None) -> List[float]:
        """
        Get vector from DuckDB BLOB.
        
        Args:
            vector_data: Vector data (BLOB bytes)
            dim: Vector dimension (uses configured fixed dimension if None)
            
        Returns:
            List of float values representing the vector
        """
        import struct
        
        if vector_data is None:
            return []
        
        # Use configured dimension if not provided
        if dim is None:
            dim = self._embedding_dim
        
        # Handle BLOB format (bytes)
        if isinstance(vector_data, bytes):
            try:
                # Unpack binary data as floats
                return list(struct.unpack(f"{dim}f", vector_data))
            except struct.error as e:
                logger.error(f"Cannot deserialize BLOB vector data: {e}")
                return []
        
        # Handle legacy formats for backward compatibility
        elif isinstance(vector_data, (list, tuple)):
            # Already a list/tuple, return as list of floats
            return [float(x) for x in vector_data]
        elif isinstance(vector_data, np.ndarray):
            # NumPy array, convert to list
            return vector_data.astype(float).tolist()
        else:
            # Try to convert directly
            try:
                return [float(x) for x in vector_data]
            except (TypeError, ValueError) as e:
                logger.error(f"Cannot deserialize vector data of type {type(vector_data)}: {e}")
                return []

    def insert_document(self, document: Dict[str, Any], chunks: List[Dict[str, Any]]) -> int:
        """
        Insert document and its chunks into the database.
        
        Embeddings are automatically adapted to the configured fixed dimension.
        
        Args:
            document: Dictionary with document information
            chunks: List of chunks generated from the document
            
        Returns:
            ID of the inserted document
            
        Raises:
            Exception: If there's an error during insertion
        """
        if not self._conn:
            raise ValueError("No database connection available")
        
        if not self._schema_created:
            self.create_schema()
        
        try:
            # Begin transaction to ensure consistency
            self.begin_transaction()
            
            # Insert document and get its ID using RETURNING clause
            insert_query = """
                INSERT INTO documents (title, url, file_path, created_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                RETURNING id
            """
            
            result = self._conn.execute(insert_query, (
                document.get('title', ''),
                document.get('url', ''),
                document.get('file_path', ''),
            )).fetchone()
            
            document_id = result[0] if result else None
            
            if document_id is None:
                raise Exception("Could not retrieve inserted document ID")
            
            # Insert each chunk
            for chunk in chunks:
                # Process embedding
                embedding = chunk.get('embedding')
                
                # Convert embedding for DuckDB BLOB
                processed_embedding = None
                if embedding is not None:
                    processed_embedding = self.serialize_vector(embedding)
                
                # Insert chunk with processed embedding
                self._conn.execute("""
                    INSERT INTO chunks (document_id, text, header, page, embedding, embedding_dim, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    document_id,
                    chunk.get('text', ''),
                    chunk.get('header', None),
                    chunk.get('page', None),
                    processed_embedding,
                    self._embedding_dim if processed_embedding else None
                ))
            
            # Commit transaction
            self.commit_transaction()
            
            logger.info(f"Document inserted with ID {document_id} and {len(chunks)} chunks")
            
            return document_id
            
        except Exception as e:
            # Rollback changes on error
            self.rollback_transaction()
            logger.error(f"Error inserting document: {e}")
            raise
    
    def insert_document_metadata(self, document: Dict[str, Any]) -> int:
        """
        Insert only document metadata into the database.
        Implementation for streaming processing of large documents.
        
        Args:
            document: Dictionary with document data
            
        Returns:
            int: ID of inserted document, None if fails
        """
        if not self._conn:
            logger.error("No database connection available")
            return None
        
        if not self._schema_created:
            self.create_schema()
        
        try:
            # Start explicit transaction
            self.begin_transaction()
            
            # Insert document without RETURNING clause (not supported in DuckDB for SQLite tables)
            insert_query = """
                INSERT INTO documents (title, url, file_path, created_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """
            
            self._conn.execute(insert_query, (
                document.get('title', ''),
                document.get('url', ''),
                document.get('file_path', ''),
            ))
            
            # Get the ID of the last inserted document
            result = self._conn.execute("SELECT MAX(id) FROM documents").fetchone()
            document_id = result[0] if result else None
            
            if document_id is None:
                raise Exception("Could not retrieve inserted document ID")
            
            # Commit using parent class method instead of direct SQL command
            self.commit_transaction()
            
            logger.debug(f"Document (metadata only) inserted with ID: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error inserting document metadata: {e}")
            # Rollback on error using parent class method
            self.rollback_transaction()
            return None
    
    def insert_single_chunk(self, document_id: int, chunk: Dict[str, Any]) -> int:
        """
        Insert single chunk associated with a document into the database.
        Designed for streaming processing of large documents.
        
        Args:
            document_id (int): ID of the document the chunk belongs to
            chunk (dict): Dictionary with chunk data
                Must contain: 'text', 'header', 'page', 'embedding', 'embedding_dim'
            
        Returns:
            int: ID of inserted chunk, None if fails
        """
        if not self._conn:
            logger.error("No database connection available")
            return None
        
        try:
            # Process embedding
            embedding = chunk.get('embedding')
            
            # Convert embedding for DuckDB BLOB
            processed_embedding = None
            if embedding is not None:
                processed_embedding = self.serialize_vector(embedding)
            
            # Insert individual chunk without RETURNING clause (not supported in DuckDB for SQLite tables)
            insert_query = """
                INSERT INTO chunks (document_id, text, header, page, embedding, embedding_dim, created_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """
            
            self._conn.execute(insert_query, (
                document_id,
                chunk.get('text', ''),
                chunk.get('header', None),
                chunk.get('page', None),
                processed_embedding,
                self._embedding_dim if processed_embedding else None
            ))
            
            # Get the ID of the last inserted chunk
            result = self._conn.execute("SELECT MAX(id) FROM chunks").fetchone()
            chunk_id = result[0] if result else None
            
            if chunk_id is None:
                raise Exception("Could not retrieve inserted chunk ID")
            
            return chunk_id
            
        except Exception as e:
            logger.error(f"Error inserting individual chunk: {e}")
            return None
    
    def optimize_database(self) -> bool:
        """
        Optimize database by performing maintenance operations.
        
        Returns:
            True if optimization was successful, False otherwise
        """
        if not self._conn:
            logger.error("No database connection available")
            return False
        
        try:
            # Only use ANALYZE for DuckDB optimization
            # VACUUM has limited support and can cause internal errors
            self._conn.execute("ANALYZE")
            
            logger.info("DuckDB database optimized successfully")
            return True
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics (documents, chunks, etc.)
        
        Returns:
            Dict[str, Any]: Database statistics
        """
        stats = {}
        
        if not self._conn:
            logger.error("No database connection available")
            return {"error": "No database connection"}
        
        try:
            # Total documents
            result = self._conn.execute("SELECT COUNT(*) FROM documents;").fetchone()
            stats["total_documents"] = result[0] if result else 0
            
            # Total chunks
            result = self._conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()
            stats["total_chunks"] = result[0] if result else 0
            
            # Latest document
            result = self._conn.execute("""
                SELECT id, title, created_at FROM documents
                ORDER BY created_at DESC LIMIT 1;
            """).fetchone()
            if result:
                stats["latest_document"] = {
                    "id": result[0],
                    "title": result[1],
                    "created_at": result[2]
                }
            
            # Database creation date
            stats["db_created"] = self.get_metadata("db_created", "unknown")
            
            # Database size
            try:
                if self._db_path and self._db_path != ":memory:":
                    import os
                    stats["db_size_mb"] = os.path.getsize(self._db_path) / (1024 * 1024)
            except Exception:
                pass
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    def vector_search(self, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a vector search using cosine similarity with DuckDB BLOB embeddings.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            
        Returns:
            List of chunks with similarity scores
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"🔍 Starting vector search - query embedding dims: {len(query_embedding)}, requesting {n_results} results")
        
        if not query_embedding:
            logger.error("❌ Empty query embedding provided")
            return []
        
        try:
            # Check if we have any chunks
            count_query = "SELECT COUNT(*) as total FROM chunks WHERE embedding IS NOT NULL"
            result = self._conn.execute(count_query).fetchone()
            total_chunks = result[0] if result else 0
            
            logger.info(f"📊 Database contains {total_chunks} total chunks with embeddings")
            
            if total_chunks == 0:
                logger.warning("⚠️ No chunks with embeddings found in database")
                return []
            
            # Get embedding dimension from existing data using BLOB format
            try:
                # For BLOB format, use the stored embedding_dim column
                dim_query = "SELECT embedding_dim FROM chunks WHERE embedding IS NOT NULL AND embedding_dim IS NOT NULL LIMIT 1"
                dim_result = self._conn.execute(dim_query).fetchone()
                if not dim_result:
                    logger.error("❌ No valid embeddings found to determine dimension")
                    return []
                stored_dim = dim_result[0]
                logger.info(f"📏 Detected BLOB format embeddings with dimension: {stored_dim}")
            except Exception as e:
                logger.error(f"❌ Cannot determine embedding dimension from BLOB format: {e}")
                return []
            
            logger.info(f"📏 Stored embedding dimensions: {stored_dim}, Query dimensions: {len(query_embedding)}")
            
            if stored_dim != len(query_embedding):
                logger.error(f"❌ Dimension mismatch: stored={stored_dim}, query={len(query_embedding)}")
                return []
            
            # For BLOB format, we need to use manual similarity calculation
            # Get all chunks with embeddings and calculate similarity manually
            logger.info("🔄 Using manual similarity calculation for BLOB embeddings...")
            
            # Get all chunks with their embeddings
            search_query = """
            SELECT 
                c.id,
                c.document_id,
                c.text,
                c.header,
                c.page,
                c.embedding,
                c.embedding_dim,
                d.title as document_title,
                d.url,
                d.file_path
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
            """
            
            results = self._conn.execute(search_query).fetchall()
            
            # Calculate similarities manually
            import numpy as np
            query_vector = np.array(query_embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query_vector)
            if query_norm > 0:
                query_vector = query_vector / query_norm
            
            similarities = []
            for row in results:
                chunk_id, doc_id, text, header, page, embedding_blob, embedding_dim, doc_title, url, file_path = row
                
                if not embedding_blob:
                    continue
                
                # Deserialize embedding
                try:
                    chunk_vector = np.array(self.deserialize_vector(embedding_blob, embedding_dim), dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Error deserializing vector from chunk {chunk_id}: {str(e)}")
                    continue
                
                # Normalize chunk vector
                chunk_norm = np.linalg.norm(chunk_vector)
                if chunk_norm > 0:
                    chunk_vector = chunk_vector / chunk_norm
                
                # Calculate cosine similarity
                similarity = float(np.dot(query_vector, chunk_vector))
                
                # Only include results above threshold
                if similarity >= self._similarity_threshold:
                    similarities.append({
                        "id": chunk_id,
                        "text": text,
                        "header": header,
                        "page": page,
                        "document_title": doc_title,
                        "similarity": similarity
                    })
            
            # Sort by similarity and limit results
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            results = similarities[:n_results]
            
            logger.info(f"📋 Manual similarity calculation returned {len(results)} results")
            
            # Format results
            chunks = []
            for result in results:
                chunk_id = result["id"]
                text = result["text"]
                header = result["header"]
                page = result["page"]
                doc_title = result["document_title"]
                similarity = result["similarity"]
                
                logger.info(f"📄 Result {len(chunks)+1}: ID={chunk_id}, similarity={similarity:.3f}, text_preview='{text[:50]}...'")
                
                chunks.append({
                    'id': chunk_id,
                    'text': text,
                    'header': header or '',
                    'page': page or 'N/A',
                    'document_title': doc_title or '',
                    'similarity': float(similarity)
                })
            
            # Apply similarity threshold filtering
            from config import DefaultValues
            threshold = getattr(self, '_similarity_threshold', DefaultValues.SIMILARITY_THRESHOLD)
            logger.info(f"🎯 Applying similarity threshold: {threshold}")
            
            filtered_chunks = [chunk for chunk in chunks if chunk['similarity'] >= threshold]
            
            if len(filtered_chunks) != len(chunks):
                logger.info(f"🔽 Filtered from {len(chunks)} to {len(filtered_chunks)} chunks based on threshold")
            
            logger.info(f"✅ Vector search completed - returning {len(filtered_chunks)} chunks")
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"❌ Error in vector search: {e}", exc_info=True)
            return []

    def get_total_chunks_count(self) -> int:
        """
        Gets the total number of chunks in the database.
        
        Returns:
            Total number of chunks
        """
        try:
            result = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error getting chunk count: {e}")
            return 0
    
    def _vector_search_manual(self, query_embedding: List[float], filters=None, n_results: int = 5, include_neighbors: bool = False):
        """
        Manual vector search implementation when no specialized extensions are available.
        
        Args:
            query_embedding: Query embedding vector
            filters: Search filters
            n_results: Maximum number of results
            include_neighbors: Whether to include neighboring chunks in results
        
        Returns:
            List of chunks ordered by similarity
        """
        results = []
        
        try:
            # Convert query embedding to the format expected by DuckDB
            query_vector = self.serialize_vector(query_embedding)
            
            # Build base query
            base_query = """
                SELECT c.id, c.document_id, c.text, c.header, c.page, c.embedding,
                       d.title, d.url, d.file_path
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
            """
            
            # Add filter conditions if they exist
            params = []
            if filters:
                for key, value in filters.items():
                    if key == 'document_id':
                        base_query += " AND c.document_id = ?"
                        params.append(value)
                    elif key == 'header':
                        base_query += " AND c.header LIKE ?"
                        params.append(f"%{value}%")
                    # Add more filters as needed
            
            # Execute query
            self._cursor.execute(base_query, params)
            all_chunks = self._cursor.fetchall()
            
            # Calculate similarity with each chunk
            chunk_similarities = []
            
            for row in all_chunks:
                if row is None or len(row) < 6:
                    continue
                    
                chunk_id, doc_id, text, header, page, embedding = row[:6]
                title, url, file_path = row[6:9] if len(row) >= 9 else (None, None, None)
                
                if embedding:
                    try:
                        # Calculate similarity directly with BLOB data
                        similarity = self._cosine_similarity(
                            np.array(query_vector, dtype=np.float32), 
                            np.array(embedding, dtype=np.float32)
                        )
                                
                        
                        # Add to results
                        chunk_similarities.append({
                            'id': chunk_id,
                            'document_id': doc_id,
                            'text': text,
                            'header': header,
                            'page': page,
                            'title': title,
                            'url': url,
                            'file_path': file_path,
                            'similarity': float(similarity)
                        })
                    except Exception as vec_error:
                        logger.debug(f"Error procesando vector para chunk {chunk_id}: {vec_error}")
                        continue
            
            # Sort by similarity and limit results
            chunk_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            results = chunk_similarities[:n_results]
            
            # Include neighbors if requested
            if include_neighbors and results:
                results = self._include_neighboring_chunks(results)
            
            return results
                
        except Exception as e:
            logger.error(f"Error in fast vector search: {e}")
            return []
    
    def store_metadata(self, key: str, value: Any) -> bool:
        """
        Store metadata in the database.
        
        Args:
            key: Metadata key
            value: Metadata value
            
        Returns:
            True if stored correctly, False otherwise
        """
        if not self._conn:
            logger.warning(f"Cannot store metadata '{key}': no database connection")
            
            # If we have a saved database path, try to reconnect
            if self._db_path:
                logger.info(f"Attempting to reconnect to database {self._db_path} to save metadata")
                if self.connect(self._db_path):
                    logger.info("Reconnection successful, continuing with metadata storage")
                else:
                    logger.error(f"Could not reconnect to database {self._db_path}")
                    return False
            else:
                return False
            
        try:
            # Check if table exists
            try:
                self._conn.execute("SELECT 1 FROM db_metadata LIMIT 1")
            except Exception:
                # If error, create table
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS db_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            
            # Convert value to string if necessary
            if not isinstance(value, str):
                if isinstance(value, (int, float, bool, type(None))):
                    value = str(value)
                else:
                    value = json.dumps(value)
            
            # Insert or update metadata
            self._conn.execute("""
                INSERT OR REPLACE INTO db_metadata (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP);
            """, (key, value))
            
            self._conn.commit()
            self._metadata[key] = value  # Update local cache
            return True
        except Exception as e:
            logger.error(f"Error storing metadata '{key}': {e}")
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
            
        if not self._conn:
            logger.warning(f"Cannot retrieve metadata '{key}': no database connection")
            return default
            
        try:
            # Check if table exists
            try:
                self._conn.execute("SELECT 1 FROM db_metadata LIMIT 1")
            except Exception:
                # Table doesn't exist
                return default
                
            # Get specific metadata value
            result = self._conn.execute(
                "SELECT value FROM db_metadata WHERE key = ?", 
                (key,)
            ).fetchone()
            
            if result is None:
                return default
                
            value = result[0]
            
            # Try to deserialize JSON
            try:
                deserialized_value = json.loads(value)
                self._metadata[key] = deserialized_value  # Update local cache
                return deserialized_value
            except (json.JSONDecodeError, TypeError):
                # If not JSON, use the value as is
                self._metadata[key] = value  # Update local cache
                return value
                
        except Exception as e:
            logger.error(f"Error retrieving metadata '{key}': {e}")
            return default
    
    def get_chunks_by_document(self, document_id: int, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieves chunks associated with a document.
        
        Args:
            document_id: Document ID
            offset: Starting offset for pagination
            limit: Maximum number of chunks to return
        
        Returns:
            List of chunks with their metadata and embeddings
        """
        try:
            # Query to get chunks with document information
            sql = """
            SELECT c.id, c.text, c.document_id, c.header, c.page, c.embedding
            FROM chunks c
            WHERE c.document_id = ?
            ORDER BY c.id
            LIMIT ? OFFSET ?
            """
            
            self._cursor.execute(sql, (document_id, limit, offset))
            rows = self._cursor.fetchall()
            
            chunks = []
            for row in rows:
                chunk = {
                    "id": row[0],
                    "text": row[1],
                    "document_id": row[2],
                    "header": row[3],
                    "page": row[4],
                    "embedding": self.deserialize_vector(row[5])  # Use deserialize_vector for consistency
                }
                chunks.append(chunk)
            
            logger.info(f"Retrieved {len(chunks)} chunks for document ID: {document_id} (offset: {offset}, limit: {limit})")
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks for document {document_id}: {e}")
            return []
    
    def _add_neighbors_to_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add neighboring chunks to search results.
        
        Args:
            results: List of search results
            
        Returns:
            Augmented list with neighboring chunks
        """
        if not results:
            return results
            
        final_results = results.copy()
        
        for result in results:
            neighbors = self._get_adjacent_chunks(result['document_id'], result['id'])
            
            # Add neighbors to results, marking them as neighbors
            for neighbor in neighbors:
                neighbor['is_neighbor'] = True
                neighbor['neighbor_of'] = result['id']
                # Ensure we're not duplicating chunks
                if not any(r['id'] == neighbor['id'] for r in final_results):
                    final_results.append(neighbor)
        
        return final_results
    
    def begin_transaction(self) -> bool:
        """
        Start a transaction in DuckDB.
        
        Returns:
            bool: True if started correctly, False otherwise
        """
        try:
            if not self._conn:
                logger.error("No database connection to start transaction")
                return False
            
            # Check if there's already an active transaction
            if self._in_transaction:
                logger.debug("There's already an active transaction in DuckDB, ignoring begin_transaction")
                return True  # We're already in a transaction, not an error
                
            # Start transaction in DuckDB
            self._conn.execute("BEGIN TRANSACTION")
            self._in_transaction = True
            logger.debug("Transaction started in DuckDB")
            return True
        except Exception as e:
            logger.error(f"Error starting transaction in DuckDB: {e}")
            return False
    
    def commit_transaction(self) -> bool:
        """
        Commit an active transaction in DuckDB.
        
        Returns:
            bool: True if committed correctly, False otherwise
        """
        try:
            if not self._conn:
                logger.error("No database connection to commit transaction")
                return False
            
            # Check if there's an active transaction to commit
            if not self._in_transaction:
                logger.debug("No active transaction in DuckDB to commit")
                return True  # No transaction to commit, not an error
                
            # Commit transaction in DuckDB
            self._conn.execute("COMMIT")
            self._in_transaction = False
            logger.debug("Transaction committed in DuckDB")
            return True
        except Exception as e:
            logger.error(f"Error committing transaction in DuckDB: {e}")
            # Try rollback in case of error
            try:
                self._conn.execute("ROLLBACK")
                self._in_transaction = False
            except:
                pass
            return False
    
    def rollback_transaction(self) -> bool:
        """
        Rollback an active transaction in DuckDB.
        
        Returns:
            bool: True if rolled back correctly, False otherwise
        """
        try:
            if not self._conn:
                logger.error("No database connection to rollback transaction")
                return False
            
            # Check if there's an active transaction to rollback
            if not self._in_transaction:
                logger.debug("No active transaction in DuckDB to rollback")
                return True  # No transaction to rollback, not an error
                
            # Rollback transaction in DuckDB
            self._conn.execute("ROLLBACK")
            self._in_transaction = False
            logger.debug("Transaction rolled back in DuckDB")
            return True
        except Exception as e:
            logger.error(f"Error rolling back transaction in DuckDB: {e}")
            return False