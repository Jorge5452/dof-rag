import sqlite3
import os
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
import logging

class DatabaseManager:
    """
    SQLite database manager for storing image descriptions.
    
    This class handles all database operations for the caption extraction system,
    including table creation, data insertion, querying, and transaction management.
    It replaces the file-based storage system with a more robust SQLite approach.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Ensure database directory exists if path contains a directory
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        # Initialize database schema
        self._init_database()
    
    def _init_database(self) -> None:
        """
        Initialize the database schema if it doesn't exist.
        
        Creates the image_descriptions table with the schema specified
        in the instructions: document_name, page_number, image_filename, description.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create image_descriptions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_descriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_name TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    image_filename TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(document_name, page_number, image_filename)
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_name 
                ON image_descriptions(document_name)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_page 
                ON image_descriptions(document_name, page_number)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_filename 
                ON image_descriptions(image_filename)
            """)
            
            conn.commit()
            self.logger.info(f"Database initialized at {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Yields:
            sqlite3.Connection: Database connection with proper error handling.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def insert_description(self, 
                          document_name: str, 
                          page_number: int, 
                          image_filename: str, 
                          description: str) -> bool:
        """
        Insert or update an image description in the database.
        
        Args:
            document_name: Name of the document containing the image.
            page_number: Page number where the image appears.
            image_filename: Filename of the image.
            description: Generated description text.
            
        Returns:
            bool: True if operation was successful, False otherwise.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO image_descriptions 
                    (document_name, page_number, image_filename, description, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (document_name, page_number, image_filename, description))
                
                conn.commit()
                self.logger.debug(f"Inserted description for {image_filename} in {document_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error inserting description: {e}")
            return False
    
    def get_description(self, 
                       document_name: str, 
                       page_number: int, 
                       image_filename: str) -> Optional[str]:
        """
        Retrieve a specific image description from the database.
        
        Args:
            document_name: Name of the document.
            page_number: Page number.
            image_filename: Image filename.
            
        Returns:
            Optional[str]: Description text if found, None otherwise.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT description FROM image_descriptions 
                    WHERE document_name = ? AND page_number = ? AND image_filename = ?
                """, (document_name, page_number, image_filename))
                
                result = cursor.fetchone()
                return result['description'] if result else None
                
        except Exception as e:
            self.logger.error(f"Error retrieving description: {e}")
            return None
    
    def description_exists(self, 
                          document_name: str, 
                          page_number: int, 
                          image_filename: str) -> bool:
        """
        Check if a description already exists for the given image.
        
        Args:
            document_name: Name of the document.
            page_number: Page number.
            image_filename: Image filename.
            
        Returns:
            bool: True if description exists, False otherwise.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 1 FROM image_descriptions 
                    WHERE document_name = ? AND page_number = ? AND image_filename = ?
                """, (document_name, page_number, image_filename))
                
                return cursor.fetchone() is not None
                
        except Exception as e:
            self.logger.error(f"Error checking description existence: {e}")
            return False
    
    def get_document_descriptions(self, document_name: str) -> List[Dict[str, Any]]:
        """
        Get all descriptions for a specific document.
        
        Args:
            document_name: Name of the document.
            
        Returns:
            List[Dict]: List of description records.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM image_descriptions 
                    WHERE document_name = ? 
                    ORDER BY page_number, image_filename
                """, (document_name,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"Error retrieving document descriptions: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict with statistics about stored descriptions.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Total descriptions
                cursor.execute("SELECT COUNT(*) as total FROM image_descriptions")
                total = cursor.fetchone()['total']
                
                # Unique documents
                cursor.execute("SELECT COUNT(DISTINCT document_name) as docs FROM image_descriptions")
                docs = cursor.fetchone()['docs']
                
                # Recent descriptions (last 24 hours)
                cursor.execute("""
                    SELECT COUNT(*) as recent FROM image_descriptions 
                    WHERE created_at >= datetime('now', '-1 day')
                """)
                recent = cursor.fetchone()['recent']
                
                return {
                    'total_descriptions': total,
                    'unique_documents': docs,
                    'recent_descriptions': recent
                }
                
        except Exception as e:
            self.logger.error(f"Error retrieving statistics: {e}")
            return {'total_descriptions': 0, 'unique_documents': 0, 'recent_descriptions': 0}
    
    def delete_document_descriptions(self, document_name: str) -> bool:
        """
        Delete all descriptions for a specific document.
        
        Args:
            document_name: Name of the document to delete.
            
        Returns:
            bool: True if operation was successful, False otherwise.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM image_descriptions WHERE document_name = ?
                """, (document_name,))
                
                conn.commit()
                deleted_count = cursor.rowcount
                self.logger.info(f"Deleted {deleted_count} descriptions for document {document_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error deleting document descriptions: {e}")
            return False
    
    def batch_insert_descriptions(self, descriptions: List[Tuple[str, int, str, str]]) -> int:
        """
        Insert multiple descriptions in a single transaction.
        
        Args:
            descriptions: List of tuples (document_name, page_number, image_filename, description).
            
        Returns:
            int: Number of successfully inserted descriptions.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.executemany("""
                    INSERT OR REPLACE INTO image_descriptions 
                    (document_name, page_number, image_filename, description, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, descriptions)
                
                conn.commit()
                inserted_count = cursor.rowcount
                self.logger.info(f"Batch inserted {inserted_count} descriptions")
                return inserted_count
                
        except Exception as e:
            self.logger.error(f"Error in batch insert: {e}")
            return 0