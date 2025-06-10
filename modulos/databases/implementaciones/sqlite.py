import sqlite3
import numpy as np
import struct
import os
import logging
from typing import List, Dict, Any, Optional

# Try to import sqlite-vec extension
try:
    from sqlite_vec import load
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

from modulos.databases.VectorialDatabase import VectorialDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable to store sqlite-vec version
SQLITE_VEC_VERSION = None

class SQLiteVectorialDatabase(VectorialDatabase):
    """
    SQLite implementation of the vectorial database.
    
    This class provides a concrete implementation of VectorialDatabase using SQLite
    as the backend storage. It supports both the sqlite-vec extension for efficient
    vector operations and manual vector functions as fallback.
    
    Attributes:
        _conn: SQLite database connection
        _cursor: Database cursor for executing queries
        _use_vector_extension: Flag indicating if sqlite-vec extension is being used
        _vector_table_name: Name of the table used for vector operations
    """
    
    def __init__(self, embedding_dim: int = 384, use_vector_extension: bool = True):
        """
        Initializes the SQLite vectorial database.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            use_vector_extension: Whether to attempt using sqlite-vec extension
        """
        super().__init__(embedding_dim)
        self._conn: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None
        self._use_vector_extension: bool = use_vector_extension and SQLITE_VEC_AVAILABLE
        
        # Load vector table name from configuration
        try:
            from config import DefaultValues
            self._vector_table_name: str = DefaultValues.SQLITE_VECTOR_TABLE_NAME
        except Exception:
            self._vector_table_name: str = "vec_chunks"  # Fallback if import fails
        
        # Load similarity threshold from configuration
        try:
            from config import config, DefaultValues
            db_config = config.get_database_config()
            sqlite_config = db_config.get('sqlite', {})
            self._similarity_threshold: float = sqlite_config.get('similarity_threshold', DefaultValues.SIMILARITY_THRESHOLD)
        except Exception as e:
            logger.warning(f"Could not load similarity_threshold from config: {e}, using default {DefaultValues.SIMILARITY_THRESHOLD}")
            self._similarity_threshold: float = DefaultValues.SIMILARITY_THRESHOLD
        
        # Internal state for batch operations
        self._batch_mode: bool = False
        self._batch_chunks: List[Dict[str, Any]] = []
        
        # Initialize extension loaded flag
        self._extension_loaded: bool = False
        
        logger.info(f"SQLite vectorial database initialized with dimension {embedding_dim}")
        if not SQLITE_VEC_AVAILABLE:
            logger.warning("sqlite-vec extension not available. Manual vector functions will be used.")
    
    def connect(self, db_path: str) -> None:
        """
        Establishes connection to the SQLite database.
        
        Args:
            db_path: Path to the SQLite database file
            
        Raises:
            sqlite3.Error: If connection fails
        """
        try:
            # Ensure the directory exists
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                logger.info(f"Created directory: {db_dir}")
            
            # Store database path
            self._db_path = db_path
            
            # Establish connection
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row  # Enable column access by name
            self._cursor = self._conn.cursor()
            
            # Enable foreign keys
            self._cursor.execute("PRAGMA foreign_keys = ON")
            
            # Try to load vector extensions
            if self._use_vector_extension:
                extension_loaded = self.load_extensions()
                if not extension_loaded:
                    logger.info("Vector extension not loaded, using manual functions")
                    self._extension_loaded = False
                    self._create_manual_vector_functions()
                else:
                    # Even if extension is loaded, register manual functions as fallback
                    # This ensures compatibility if extension functions are not available
                    self._create_manual_vector_functions()
            else:
                logger.info("Using manual vector functions (extension disabled)")
                self._extension_loaded = False
                self._create_manual_vector_functions()
            
            # Create schema if it doesn't exist
            self.create_schema()
            
            logger.info(f"Connection established successfully to SQLite database: {db_path}")
        
        except sqlite3.Error as e:
            logger.error(f"Error connecting to SQLite database: {e}")
            if self._conn:
                self._conn.close()
            raise
    
    def _create_manual_vector_functions(self) -> None:
        """
        Creates custom functions in SQLite for vector operations when
        the sqlite-vec extension is not available.
        """
        logger.info("Creating manual vector functions in SQLite...")
        
        def unpack_vector(blob: bytes) -> Optional[np.ndarray]:
            """Unpacks a BLOB into a numpy vector"""
            if not blob:
                return None
            try:
                # Determine dimension from blob size
                dim = len(blob) // 4  # 4 bytes per float (float32)
                fmt = f"{dim}f"  # format for struct.unpack
                return np.array(struct.unpack(fmt, blob))
            except Exception as e:
                logger.error(f"Error unpacking vector: {e}")
                return None
        
        def cosine_similarity(blob1: bytes, blob2: bytes) -> float:
            """Calculates cosine similarity between two vectors stored as BLOBs"""
            vec1 = unpack_vector(blob1)
            vec2 = unpack_vector(blob2)
            
            if vec1 is None or vec2 is None:
                return 0.0
                
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(dot_product / (norm1 * norm2))
            
        # Register functions in SQLite
        self._conn.create_function("vec_cosine_similarity", 2, cosine_similarity)
        logger.info("Manual vector functions created successfully")
    
    def load_extensions(self) -> bool:
        """
        Loads the sqlite-vec extension for efficient vector operations.
        
        Returns:
            bool: True if the extension was loaded successfully, False otherwise.
        """
        global SQLITE_VEC_VERSION
        
        if not self._use_vector_extension:
            logger.info("Vector extension disabled by configuration.")
            return False
            
        if not SQLITE_VEC_AVAILABLE:
            logger.warning("sqlite-vec package not available. Try installing it with: pip install sqlite-vec")
            self._use_vector_extension = False
            return False
        
        try:
            # Enable extension loading
            self._conn.enable_load_extension(True)
            
            # Method 1: Using sqlite_vec module to load the extension
            try:
                load(self._conn)
                # Verify loading
                try:
                    vec_version = self._conn.execute("SELECT vec_version()").fetchone()[0]
                    SQLITE_VEC_VERSION = vec_version
                    logger.info(f"sqlite-vec extension loaded successfully with method 1. Version: {vec_version}")
                    
                    # Disable extension loading for security
                    self._conn.enable_load_extension(False)
                    
                    # Store extension information
                    self.store_metadata("vector_extension_version", vec_version)
                    self.store_metadata("vector_extension_enabled", "true")
                    
                    # Set extension loaded flag
                    self._extension_loaded = True
                    
                    # If version is v0.1.6, set specific table and functions
                    if vec_version == "v0.1.6":
                        self._vector_table_name = "vec_index"  # Simple name for the table
                        
                    return True
                except sqlite3.OperationalError:
                    # If we can't get the version, the extension might not have loaded completely
                    logger.warning("sqlite-vec extension loaded but vec_version() function is not available")
                    
                    # Check if other functions like vec_cosine_similarity are available
                    try:
                        # Create test vectors
                        test_vec1 = np.ones(5, dtype=np.float32).tobytes()
                        test_vec2 = np.ones(5, dtype=np.float32).tobytes()
                        
                        # Try to use vec_cosine_similarity
                        result = self._conn.execute("SELECT vec_cosine_similarity(?, ?)", (test_vec1, test_vec2)).fetchone()[0]
                        logger.info(f"vec_cosine_similarity function available, test result: {result}")
                        self.store_metadata("vector_extension_enabled", "true")
                        self.store_metadata("vector_extension_version", "unknown")
                        self._extension_loaded = True
                        self._conn.enable_load_extension(False)
                        return True
                    except sqlite3.OperationalError as e:
                        logger.warning(f"vec_cosine_similarity function is not available: {e}")
            except Exception as e:
                logger.warning(f"Error loading sqlite-vec with method 1: {e}")
            
            # Method 2: Try another way to load the extension (direct with .so/.dll)
            try:
                # Get the library path from the module
                import sqlite_vec
                ext_path = os.path.dirname(os.path.abspath(sqlite_vec.__file__))
                
                # Try to load from different possible locations
                possible_paths = [
                    os.path.join(ext_path, "sqlite_vec"),
                    os.path.join(ext_path, "sqlite_vec.so"),
                    os.path.join(ext_path, "sqlite_vec.dll"),
                    "sqlite_vec",
                ]
                
                for path in possible_paths:
                    try:
                        self._conn.enable_load_extension(True)
                        self._conn.load_extension(path)
                        logger.info(f"sqlite-vec extension loaded from: {path}")
                        self.store_metadata("vector_extension_enabled", "true")
                        self.store_metadata("vector_extension_path", path)
                        self._extension_loaded = True
                        self._conn.enable_load_extension(False)
                        return True
                    except Exception as e:
                        logger.debug(f"Could not load extension from {path}: {e}")
            except Exception as e:
                logger.warning(f"Error in alternative loading method: {e}")
                
            # If we get here, the extension could not be loaded
            logger.warning("Could not load sqlite-vec extension. Manual vector functions will be used.")
            self._use_vector_extension = False
            self.store_metadata("vector_extension_enabled", "false")
            
            # Disable extension loading for security
            try:
                self._conn.enable_load_extension(False)
            except sqlite3.Error:
                pass
                
            return False
            
        except sqlite3.Error as e:
            logger.error(f"Error managing extensions in SQLite: {e}")
            self._use_vector_extension = False
            return False
    
    def _check_vector_table_exists(self) -> bool:
        """Check if the vector table exists.
        
        Returns:
            bool: True if the vector table exists, False otherwise
        """
        try:
            self._cursor.execute(f"""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='{self._vector_table_name}';
            """)
            return self._cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Error checking vector table existence: {e}")
            return False
    
    def _create_vector_table(self) -> bool:
        """Create the vector table if it doesn't exist.
        
        Returns:
            bool: True if the table was created successfully, False otherwise
        """
        try:
            if not self._extension_loaded:
                logger.warning("Cannot create vector table without sqlite-vec extension")
                return False
                
            sql = f"CREATE VIRTUAL TABLE {self._vector_table_name} USING vec0(embedding float[{self._embedding_dim}])"
            logger.info(f"Creating vector table with: {sql}")
            self._cursor.execute(sql)
            self._conn.commit()
            logger.info(f"Vector table '{self._vector_table_name}' created successfully")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error creating vector table: {e}")
            return False
    
    def close_connection(self) -> None:
        """
        Closes the connection to the SQLite database.
        """
        if self._conn:
            try:
                self._conn.close()
                self._conn = None
                self._cursor = None
                logger.info("SQLite database connection closed successfully.")
            except sqlite3.Error as e:
                logger.error(f"Error closing SQLite database connection: {e}")
    
    def create_schema(self) -> None:
        """
        Creates the database schema if it doesn't exist.
        
        Creates the documents and chunks tables with their corresponding indexes.
        """
        try:
            # Documents table
            self._cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Index for URL search
            self._cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);
            """)
            
            # Chunks table
            self._cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    document_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    header TEXT,
                    page TEXT,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );
            """)
            
            # Index for searching chunks by document
            self._cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
            """)
            
            # Commit changes
            self._conn.commit()
            logger.info("Database schema created successfully.")
            
            # Create vector indexes using fixed dimension
            if self._extension_loaded:
                self.create_vector_index()
        
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error(f"Error creating database schema: {e}")
            raise
    
    def create_vector_index(self, force_rebuild: bool = False) -> bool:
        """
        Creates or rebuilds the vector index to optimize searches.
        
        Uses the fixed embedding dimension established during initialization.
        Compatible with sqlite-vec v0.1.6.
        
        Args:
            force_rebuild: If True, forces index reconstruction even if it already exists.
                                  
        Returns:
            bool: True if the index was created/rebuilt successfully, False otherwise.
        """
        if not self._extension_loaded:
            logger.warning("Cannot create vector index without sqlite-vec extension")
            return False
            
        try:
            # Check if vector index already exists
            self._cursor.execute(f"""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='{self._vector_table_name}';
            """)
            table_exists = self._cursor.fetchone() is not None
            
            if table_exists and not force_rebuild:
                logger.info(f"Vector index '{self._vector_table_name}' already exists. Will not rebuild.")
                return True
                
            # If it exists and we want to rebuild, drop it first
            if table_exists:
                self._cursor.execute(f"DROP TABLE IF EXISTS {self._vector_table_name}")
                logger.info(f"Vector index '{self._vector_table_name}' dropped for reconstruction.")
            
            # Use exactly the syntax provided in the official documentation
            sql = f"CREATE VIRTUAL TABLE {self._vector_table_name} USING vec0(embedding float[{self._embedding_dim}])"
            
            try:
                logger.info(f"Attempting to create vector index with: {sql}")
                self._cursor.execute(sql)
                
                # Verify if it was created correctly
                self._cursor.execute(f"""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='{self._vector_table_name}';
                """)
                
                if self._cursor.fetchone() is not None:
                    logger.info(f"Vector index '{self._vector_table_name}' created successfully.")
                    self._conn.commit()
                    return True
                else:
                    logger.warning(f"Could not verify creation of vector index '{self._vector_table_name}'.")
                    return False
            except sqlite3.Error as e:
                logger.warning(f"Error creating vector index: {e}")
                # Disable vector functionality if index cannot be created
                self._use_vector_extension = False
                self.store_metadata("vector_extension_enabled", "false")
                return False
                
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error(f"Error creating vector index: {e}")
            return False
    
    def serialize_vector(self, vector: List[float]) -> bytes:
        """
        Serializes a float vector to a binary blob.
        
        Uses the fixed dimension established during initialization. If the provided
        vector has a different dimension, it is truncated or padded with zeros.
        
        Args:
            vector: Vector to serialize.
            
        Returns:
            bytes: Binary representation of the vector.
        """
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
                
        return struct.pack(f"{self._embedding_dim}f", *vector)
    
    def deserialize_vector(self, blob: bytes, dim: int = None) -> List[float]:
        """
        Deserializes a vector from its binary blob to a list of floats.
        
        Args:
            blob: Blob with the vector serialized
            dim: Dimension of the vector to deserialize. If it is None, it uses the
                 configured dimension of the instance.
                 
        Returns:
            List[float]: Vector deserialized.
        """
        # If no dim, use the configured dimension of the instance
        embedding_dim = dim if dim is not None else self._embedding_dim
        return list(struct.unpack(f"{embedding_dim}f", blob))
    
    def insert_document(self, document: Dict[str, Any], chunks: List[Dict[str, Any]]) -> int:
        """
        Inserts a document and its chunks embedded in the database.
        
        The embeddings are adapted automatically to the fixed dimension established.
        
        Args:
            document: Contains fields as title, url, file_path.
            chunks: List of dictionaries for each chunk with fields as text, header, page, embedding.
        
        Returns:
            int: ID of the document inserted
            
        Raises:
            sqlite3.Error: If an error occurred during insertion.
        """
        try:
            # Init transaction
            self._conn.execute("BEGIN TRANSACTION;")
            
            # Insert document
            self._cursor.execute("""
                INSERT INTO documents (title, url, file_path)
                VALUES (?, ?, ?)
            """, (
                document.get('title', 'Untitled'),
                document.get('url', f"local://{document.get('file_path', 'unknown')}"),
                document.get('file_path', '')
            ))
            
            # Get the ID of the document inserted
            self._cursor.execute("SELECT MAX(id) FROM documents")
            result = self._cursor.fetchone()
            document_id = result[0] if result and result[0] is not None else None
            
            if document_id is None:
                logger.error("Failed to retrieve document ID after insertion")
                raise Exception("Could not retrieve document ID after insertion")
            
            logger.debug(f"Document inserted with ID: {document_id}")
            
            # Verify the status of the vectorial extension
            vector_extension_enabled = self._extension_loaded and self._use_vector_extension
            vector_table_exists = False
            
            if vector_extension_enabled:
                # Verify if the table exists
                self._cursor.execute(f"""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='{self._vector_table_name}';
                """)
                vector_table_exists = self._cursor.fetchone() is not None
                
                if not vector_table_exists:
                    # Try to create the table if it doesn't exist
                    vector_table_exists = self.create_vector_index()
                    if not vector_table_exists:
                        logger.warning("Could not create vector table during insertion.")
            
            # Insert the chunks associated with the document
            for chunk in chunks:
                # Convert the embedding to bytes for efficient storage
                embedding = chunk.get('embedding')
                embedding_bytes = None
                
                if embedding is not None:
                    if isinstance(embedding, list):
                        embedding_bytes = self.serialize_vector(embedding)
                        logger.debug(f"Embedding serialized: {len(embedding)} dimensions -> {len(embedding_bytes)} bytes")
                    elif isinstance(embedding, np.ndarray):
                        embedding_bytes = self.serialize_vector(embedding.tolist())
                        logger.debug(f"Embedding (numpy) serialized: {embedding.shape} -> {len(embedding_bytes)} bytes")
                    else:
                        logger.warning(f"Embedding is None for chunk: {chunk.get('header', 'No header')[:50]}")
            
            # Commit this operation to ensure the document exists
            self._conn.commit()
            logger.info(f"Document inserted successfully with ID: {document_id}, with {len(chunks)} chunks.")
            return document_id
            
        except Exception as e:
            # If there's an error, rollback
            if self._conn:
                self._conn.rollback()
            logger.error(f"Error inserting document and chunks: {e}")
            raise
    
    def get_chunks_by_document(self, document_id, offset=0, limit=100):
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
                    "embedding": self.deserialize_vector(row[5])
                }
                chunks.append(chunk)
            
            logger.info(f"Retrieved {len(chunks)} chunks for document ID: {document_id} (offset: {offset}, limit: {limit})")
            return chunks
        
        except Exception as e:
            logger.error(f"Error retrieving chunks by document: {e}")
            return []

    def _vector_search_with_extension(self, query_embedding, filters=None, n_results=5, include_neighbors=False):
        """
        Performs vector search using the sqlite-vec extension.
        
        Args:
            query_embedding: Query vector
            filters: Search filters
            n_results: Number of results to return
            include_neighbors: Whether to include neighboring chunks
        
        Returns:
            List of chunks ordered by similarity
        """
        # Check if we have the extension and table
        self._cursor.execute(f"""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='{self._vector_table_name}';
        """)
        vector_table_exists = self._cursor.fetchone() is not None
        
        if not vector_table_exists:
            logger.warning(f"Vector table {self._vector_table_name} does not exist. Using manual search.")
            return self._vector_search_manual(query_embedding, filters, n_results, include_neighbors)
            
        # Serialize the query embedding
        query_blob = self.serialize_vector(query_embedding)
        
        # Build SQL query adapted to v0.1.6
        # In v0.1.6, the vector table has implicit rowid and embedding columns
        base_query = """
        SELECT c.id, c.document_id, c.text, c.header, c.page, c.embedding_dim,
            d.title, d.url, d.file_path,
            vec_cosine_similarity(c.embedding, ?) AS similarity
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        """
        
        # Add filters if they exist
        where_clauses = []
        params = [query_blob]
        
        if filters:
            if 'document_id' in filters:
                where_clauses.append("c.document_id = ?")
                params.append(filters['document_id'])
            
            if 'min_similarity' in filters:
                where_clauses.append("similarity >= ?")
                params.append(filters['min_similarity'])
            else:
                # Use default similarity threshold
                where_clauses.append("similarity >= ?")
                params.append(self._similarity_threshold)
        else:
            # Use default similarity threshold
            where_clauses.append("similarity >= ?")
            params.append(self._similarity_threshold)
        
        # Complete the query
        if where_clauses:
            base_query += " WHERE " + " AND ".join(where_clauses)
        
        base_query += " ORDER BY similarity DESC LIMIT ?"
        params.append(n_results)
        
        try:
            # Execute the query
            cursor = self._conn.cursor()
            cursor.execute(base_query, params)
            
            # Process results
            results = []
            for row in cursor.fetchall():
                chunk_id, doc_id, text, header, page, embedding_dim, title, url, file_path, similarity = row
                
                # Create result object
                chunk_result = {
                    'id': chunk_id,
                    'document_id': doc_id,
                    'text': text,
                    'header': header,
                    'page': page,
                    'title': title,
                    'url': url,
                    'file_path': file_path,
                    'similarity': float(similarity)
                }
                results.append(chunk_result)
            
            # Include neighboring chunks if requested
            if include_neighbors and results:
                best_match = results[0]
                neighbors = self._get_adjacent_chunks(best_match["document_id"], best_match["id"])
                
                if neighbors:
                    # Insert neighbors at the beginning
                    results = neighbors + results
            
            return results
        except sqlite3.Error as e:
            logger.error(f"Error during vector search: {e}")
            # Fallback to manual search
            return self._vector_search_manual(query_embedding, filters, n_results, include_neighbors)
    
    def optimize_database(self) -> bool:
        """
        Optimizes the SQLite database by executing VACUUM and ANALYZE.
        
        Returns:
            bool: True if optimization was successful, False otherwise
        """
        try:
            logger.info(f"Optimizing SQLite database: {self._db_path}")
            # Execute VACUUM to compact the database
            self._conn.execute("VACUUM;")
            # Execute ANALYZE to update statistics
            self._conn.execute("ANALYZE;")
            # Execute PRAGMA optimize for additional optimizations
            self._conn.execute("PRAGMA optimize;")
            logger.info("SQLite optimization completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error optimizing SQLite database: {e}")
            return False

    def vector_search(self, query_embedding, filters=None, n_results=5, include_neighbors=False):
        """
        Performs a vector search.
        
        Args:
            query_embedding: Query embedding vector
            filters: Additional search filters
            n_results: Maximum number of results
            include_neighbors: Whether to include neighboring chunks
            
        Returns:
            list: List of chunks ordered by similarity
        """
        # Check if we can use the vector extension
        if self._extension_loaded and self._use_vector_extension:
            return self._vector_search_with_extension(
                query_embedding, filters, n_results, include_neighbors
            )
        else:
            # Fallback to manual search if extension is not available
            return self._vector_search_manual(
                query_embedding, filters, n_results, include_neighbors
            )

    def insert_document_metadata(self, document: Dict[str, Any]) -> int:
        """
        Inserts only document metadata without chunks.
        Designed for streaming processing where chunks will be inserted later.
        
        Args:
            document: Dictionary with document data (title, url, file_path, etc.)
        
        Returns:
            int: ID of the inserted document, None if it fails
        """
        try:
            # Insert the document
            self._cursor.execute("""
                INSERT INTO documents (title, url, file_path)
                VALUES (?, ?, ?)
            """, (
                document.get('title', 'Untitled'),
                document.get('url', f"local://{document.get('file_path', 'unknown')}"),
                document.get('file_path', '')
            ))
            
            # Get the inserted document ID
            self._cursor.execute("SELECT MAX(id) FROM documents")
            result = self._cursor.fetchone()
            document_id = result[0] if result and result[0] is not None else None
            
            if document_id is None:
                logger.error("Failed to retrieve document ID after insertion")
                return None
            
            # Commit this operation to ensure the document exists
            self._conn.commit()
            logger.debug(f"Document (metadata only) inserted with ID: {document_id}")
            return document_id
        
        except Exception as e:
            logger.error(f"Error inserting document metadata: {e}")
            return None

    def insert_single_chunk(self, document_id: int, chunk: Dict[str, Any]) -> int:
        """
        Inserts a single chunk associated with a document in the database.
        Designed for streaming processing of large documents.
        
        Args:
            document_id (int): ID of the document this chunk belongs to
            chunk (dict): Dictionary with chunk data
            Must contain: 'text', 'header', 'page', 'embedding', 'embedding_dim'
        
        Returns:
            int: ID of the inserted chunk, None if it fails
        """
        try:
            logger.debug(f"Starting chunk insertion for document ID {document_id}")
            
            # Check if vector extension is enabled
            vector_extension_enabled = self._extension_loaded and self._use_vector_extension
            
            # Convert the embedding to bytes for efficient storage
            embedding = chunk.get('embedding')
            embedding_bytes = None
            
            if embedding is not None:
                if isinstance(embedding, list):
                    embedding_bytes = self.serialize_vector(embedding)
                    logger.debug(f"Embedding serialized: {len(embedding)} dimensions -> {len(embedding_bytes)} bytes")
                elif isinstance(embedding, np.ndarray):
                    embedding_bytes = self.serialize_vector(embedding.tolist())
                    logger.debug(f"Embedding (numpy) serialized: {embedding.shape} -> {len(embedding_bytes)} bytes")
            else:
                logger.warning(f"Embedding is None for chunk: {chunk.get('header', 'No header')[:50]}")
            
            # Insert the chunk into the main table
            sql_insert = """
                INSERT INTO chunks (document_id, text, header, page, embedding, embedding_dim)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            
            params = (
                int(document_id),  # Ensure it's an integer
                chunk.get('text', ''),
                chunk.get('header', None),
                chunk.get('page', None),
                embedding_bytes,
                chunk.get('embedding_dim', self._embedding_dim)  # Use the proposed dimension or the fixed one
            )
            
            logger.debug(f"Executing SQL: {sql_insert}")
            logger.debug(f"Parameters: doc_id={document_id}, header_len={len(chunk.get('header', '')) if chunk.get('header') else 0}, text_len={len(chunk.get('text', ''))}, embedding_dim={chunk.get('embedding_dim', self._embedding_dim)}")
            
            self._cursor.execute(sql_insert, params)
            
            # Get the inserted chunk ID
            self._cursor.execute("SELECT MAX(id) FROM chunks")
            result = self._cursor.fetchone()
            chunk_id = result[0] if result and result[0] is not None else None
            
            if chunk_id is None:
                logger.error("Failed to retrieve chunk ID after insertion")
                return None
                
            logger.debug(f"Chunk inserted with ID: {chunk_id}")
            
            # Update vector index if extension is enabled
            if vector_extension_enabled and chunk_id and embedding_bytes:
                try:
                    # Check if the table exists
                    vector_table_exists = self._check_vector_table_exists()
                    logger.debug(f"Vector table exists: {vector_table_exists}")
                    
                    if vector_table_exists:
                        # Insert into vector table
                        sql_vector = f"INSERT INTO {self._vector_table_name}(rowid, embedding) VALUES (?, ?)"
                        logger.debug(f"Updating vector index with SQL: {sql_vector}")
                        self._cursor.execute(sql_vector, (chunk_id, embedding_bytes))
                    else:
                        # If it doesn't exist, try to create it
                        logger.warning(f"Vector table {self._vector_table_name} does not exist. Attempting to create...")
                        self._create_vector_table()
                        logger.debug("Vector table created. Inserting data...")
                        self._cursor.execute(
                            f"INSERT INTO {self._vector_table_name}(rowid, embedding) VALUES (?, ?)",
                            (chunk_id, embedding_bytes)
                        )
                        
                except Exception as e:
                    # Non-critical error, continue
                    logger.warning(f"Error updating vector index for chunk {chunk_id} (non-critical): {e}")
            
            return chunk_id
        
        except Exception as e:
            logger.error(f"Error inserting individual chunk: {e}", exc_info=True)
            return None

    def begin_transaction(self) -> bool:
        """
        Starts a manual transaction for bulk insertion.
        
        Returns:
            bool: True if transaction started correctly
        """
        try:
            # Check if there's already an active transaction
            if self._in_transaction:
                logger.debug("There's already an active transaction, ignoring begin_transaction")
                return True  # Return True because conceptually we're already in a transaction
            
            logger.debug("Starting transaction in SQLite")
            self._conn.execute("BEGIN")
            self._in_transaction = True
            return True
        
        except AttributeError:
            logger.error("Cannot start transaction: connection not available")
            return False
        except Exception as e:
            logger.error(f"Error starting transaction in SQLite: {e}", exc_info=True)
            return False
    
    def rollback_transaction(self) -> bool:
        """
        Reverts an ongoing transaction.
        
        Returns:
            bool: True if the transaction was reverted correctly
        """
        try:
            if self._conn:
                # Only rollback if there's an active transaction
                if not self._in_transaction:
                    logger.debug("No active transaction to revert")
                    return True
                
                logger.debug("Reverting transaction in SQLite")
                self._conn.rollback()
                self._in_transaction = False
                return True
            else:
                logger.error("Cannot revert transaction: connection not available")
                return False
        except Exception as e:
            logger.error(f"Error reverting transaction in SQLite: {e}", exc_info=True)
            return False