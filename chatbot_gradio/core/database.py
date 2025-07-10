"""Database operations for RAG chat system using DuckDB with VSS extension."""

import logging
from typing import Any, Dict, List, Optional
import duckdb
from .embeddings import embedding_manager


logger = logging.getLogger(__name__)


def connect_duckdb(path: str) -> duckdb.DuckDBPyConnection:
    """Connect to DuckDB database with VSS extension for vector operations.
    
    Args:
        path: Database file path
        
    Returns:
        DuckDB connection with VSS extension loaded
        
    Raises:
        duckdb.Error: Connection or extension loading failure
    """
    try:
        conn = duckdb.connect(path)
        
        # Install VSS extension for vector similarity operations
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        
        initialize_database(conn)
        
        logger.info(f"Connected to DuckDB at {path} with VSS extension")
        return conn
        
    except duckdb.Error as e:
        logger.error(f"Failed to connect to DuckDB or load VSS extension: {e}")
        raise


def initialize_database(conn: duckdb.DuckDBPyConnection) -> None:
    """Create database tables with vector support but no HNSW indexes.
    
    Args:
        conn: DuckDB connection with VSS extension loaded
    """
    try:
        embedding_dim = embedding_manager.get_dimension()
        
        # Create auto-increment sequences
        conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS documents_id_seq START 1
        """)
        
        conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS chunks_id_seq START 1
        """)
        
        conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS image_descriptions_id_seq START 1
        """)
        
        # Create documents table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY DEFAULT nextval('documents_id_seq'),
                title VARCHAR,
                url VARCHAR UNIQUE,
                file_path VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create chunks table with vector embeddings
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY DEFAULT nextval('chunks_id_seq'),
                document_id INTEGER,
                text VARCHAR,
                header VARCHAR,
                embedding FLOAT[{embedding_dim}],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        
        # Create image_descriptions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS image_descriptions (
                id INTEGER PRIMARY KEY DEFAULT nextval('image_descriptions_id_seq'),
                document_name VARCHAR,
                page_number INTEGER,
                image_filename VARCHAR,
                description VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(document_name, page_number, image_filename)
            )
        """)
        
        # Create performance indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
            ON chunks(document_id)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_url 
            ON documents(url)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_image_descriptions_document 
            ON image_descriptions(document_name, page_number)
        """)
        
        conn.commit()
        logger.info(f"Database tables initialized successfully with embedding dimension {embedding_dim}")
        
    except duckdb.Error as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def query_context(
    conn: duckdb.DuckDBPyConnection, 
    embedding: List[float], 
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Retrieve chunks most similar to query embedding.
    
    Args:
        conn: DuckDB connection
        embedding: Query vector
        top_k: Maximum results to return
        
    Returns:
        Chunks ranked by similarity score
        
    Raises:
        duckdb.Error: Query execution failure
    """
    try:
        # Execute vector similarity search using array_distance
        expected_dim = embedding_manager.get_dimension()
        query = f"""
            SELECT 
                c.id,
                c.text,
                c.header,
                c.document_id,
                d.title,
                d.url,
                d.file_path,
                array_distance(c.embedding, ?::FLOAT[{expected_dim}]) as distance
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT ?
        """
        
        result = conn.execute(query, [embedding, top_k]).fetchall()
        
        chunks = []
        for row in result:
            # Convert distance to similarity score
            distance = row[7]
            similarity = 1.0 / (1.0 + distance) if distance >= 0 else 1.0
            
            chunk = {
                "id": row[0],
                "text": row[1],
                "header": row[2],
                "document_id": row[3],
                "title": row[4],
                "url": row[5],
                "file_path": row[6],
                "distance": distance,
                "similarity": similarity
            }
            chunks.append(chunk)
        
        logger.info(f"Retrieved {len(chunks)} chunks for context")
        return chunks
        
    except duckdb.Error as e:
        logger.error(f"Failed to query context: {e}")
        raise


def get_image_descriptions(
    conn: duckdb.DuckDBPyConnection,
    document_name: str,
    page_number: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Get image descriptions for a document and optionally a specific page.
    
    Args:
        conn: DuckDB connection object
        document_name: Name of the document
        page_number: Optional page number to filter by
        
    Returns:
        List of image descriptions
        
    Raises:
        duckdb.Error: If query fails
    """
    try:
        if page_number is not None:
            query = """
                SELECT id, document_name, page_number, image_filename, 
                       description, created_at, updated_at
                FROM image_descriptions
                WHERE document_name = ? AND page_number = ?
                ORDER BY page_number, image_filename
            """
            result = conn.execute(query, [document_name, page_number]).fetchall()
        else:
            query = """
                SELECT id, document_name, page_number, image_filename, 
                       description, created_at, updated_at
                FROM image_descriptions
                WHERE document_name = ?
                ORDER BY page_number, image_filename
            """
            result = conn.execute(query, [document_name]).fetchall()
        
        images = []
        for row in result:
            images.append({
                "id": row[0],
                "document_name": row[1],
                "page_number": row[2],
                "image_filename": row[3],
                "description": row[4],
                "created_at": row[5],
                "updated_at": row[6]
            })
        
        return images
        
    except duckdb.Error as e:
        logger.error(f"Failed to get image descriptions: {e}")
        raise


def close_connection(conn: duckdb.DuckDBPyConnection) -> None:
    """Close DuckDB connection safely.
    
    Args:
        conn: DuckDB connection object to close
    """
    try:
        conn.close()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database connection: {e}")