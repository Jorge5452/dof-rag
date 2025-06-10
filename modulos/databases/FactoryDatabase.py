import os
import logging
from typing import Dict, Any, Optional
import json
import time
from pathlib import Path
import uuid
from datetime import datetime

from modulos.databases.VectorialDatabase import VectorialDatabase
from config import config

logger = logging.getLogger(__name__)

class DatabaseFactory:
    """
    Factory for creating vectorial database instances.
    Implements the Singleton pattern for database instances.
    
    This class now incorporates an intelligent naming system to avoid
    conflicts when using different embedding and chunking configurations.
    """
    
    # Dictionary to store created instances (Singleton)
    _instances = {}
    
    @classmethod
    def get_database_instance(cls, 
                             db_type: Optional[str] = None, 
                             embedding_dim: Optional[int] = None,
                             embedding_model: Optional[str] = None,
                             chunking_method: Optional[str] = None,
                             session_id: Optional[str] = None,
                             custom_name: Optional[str] = None,
                             load_existing: bool = False,
                             db_path: Optional[str] = None) -> VectorialDatabase:
        """
        Gets a database instance of the specified type with specific configuration.
        
        Args:
            db_type: Database type ('sqlite', 'duckdb', etc.). If None, uses config value.
            embedding_dim: Embedding dimension. This value must come from the embeddings model.
            embedding_model: Name of the embedding model used (for tracking).
            chunking_method: Chunking method used (for tracking).
            session_id: Unique session identifier (if provided, used to identify the database).
            custom_name: Custom name for the database (takes priority over generated one).
            load_existing: If True, will load an existing database at db_path without creating a new one.
            db_path: Path to an existing database that will be loaded if load_existing=True.
            
        Returns:
            VectorialDatabase instance.
            
        Raises:
            ValueError: If the database type is not valid or if embedding_dim is not provided.
        """
        # Validate that we have an embedding dimension
        if embedding_dim is None:
            raise ValueError("Embedding dimension is required to initialize the vectorial database")
        
        # Load configuration if arguments are not provided
        if db_type is None:
            db_config = cls._load_config()
            db_type = db_config.get("type", "sqlite")
            
        # Get current configuration information if not provided
        if embedding_model is None:
            embedding_config = config.get_embedding_config()
            embedding_model = embedding_config.get("model", "modernbert")
            
        if chunking_method is None:
            chunks_config = config.get_chunks_config()
            chunking_method = chunks_config.get("method", "context")
        
        # Normalize the database type
        db_type = db_type.lower()
        
        # Verify that the implementation exists
        db_class = cls._get_db_class(db_type)
        
        # If we are loading an existing database and the path is provided
        if load_existing and db_path:
            # Verify that the file exists
            if not os.path.exists(db_path):
                raise ValueError(f"The database at {db_path} does not exist")
            
            # Create unique key for this instance
            instance_key = f"{db_type}:{db_path}:{embedding_dim}"
            
            # If an instance already exists for this database, return it
            if instance_key in cls._instances:
                logger.debug(f"Reusing existing instance for {db_path}")
                return cls._instances[instance_key]
            
            # Create new instance connected to the existing database
            logger.info(f"Loading existing {db_type} database from {db_path}")
            db_instance = db_class(embedding_dim=embedding_dim)
            
            # Configure tracking attributes
            db_instance._embedding_model = embedding_model
            db_instance._chunking_method = chunking_method
            db_instance._session_id = session_id
            
            # Connect to the existing database
            try:
                db_instance.connect(db_path)
                logger.info(f"Successful connection to existing database: {db_path}")
                
                # Database connection successful
                
                # Store the instance in the cache
                cls._instances[instance_key] = db_instance
                return db_instance
            except Exception as e:
                logger.error(f"Error connecting to existing database {db_path}: {e}")
                raise ValueError(f"Could not load the database: {e}")
        
        # Existing code for creating new database or reusing instance
        # Generate a unique name for the database based on configuration
        if custom_name:
            db_name = custom_name
        elif session_id:
            db_name = f"rag_db_{session_id}"
        else:
            db_name = cls._generate_db_name(embedding_model, chunking_method, db_type)
        
        # Determine the database path
        if not db_path:
            db_path = cls._get_db_path(db_type, db_name)
        
        # Create unique key that includes the dimension
        instance_key = f"{db_type}:{db_path}:{embedding_dim}"
        
        # Check if an instance already exists for this key
        if instance_key not in cls._instances:
            # Create new instance with the specific dimension
            logger.info(
                f"Creating new {db_type} database instance "
                f"at {db_path} with dimension {embedding_dim}, "
                f"model {embedding_model}, chunking {chunking_method}"
            )
            db_instance = db_class(embedding_dim=embedding_dim)
            
            # Add attributes for better traceability
            db_instance._embedding_model = embedding_model
            db_instance._chunking_method = chunking_method
            db_instance._session_id = session_id
            
            # Save configuration metadata (useful for session_manager)
            metadata = {
                "db_type": db_type,
                "db_path": db_path,
                "embedding_dim": embedding_dim,
                "embedding_model": embedding_model,
                "chunking_method": chunking_method,
                "session_id": session_id,
                "custom_name": custom_name,
                "created_at": time.time(),
                "last_used": time.time()
            }
            
            # Database created successfully
            
            # Try to connect to the database
            try:
                db_instance.connect(db_path)
                # If it's a new database, save metadata inside it as well
                if hasattr(db_instance, "store_metadata"):
                    for key, value in metadata.items():
                        db_instance.store_metadata(key, value)
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
                
            cls._instances[instance_key] = db_instance
        else:
            logger.debug(f"Reusing existing {db_type} database instance with dimension {embedding_dim}")
            
            # Database instance ready
        
        return cls._instances[instance_key]
    
    @classmethod
    def _get_db_class(cls, db_type: str):
        """
        Gets the appropriate database class according to the type.
        
        Args:
            db_type: Database type ('sqlite', 'duckdb', etc.)
            
        Returns:
            Database class.
            
        Raises:
            ValueError: If the database type is not valid.
        """
        if db_type == "sqlite":
            from modulos.databases.implementaciones.sqlite import SQLiteVectorialDatabase
            return SQLiteVectorialDatabase
        elif db_type == "duckdb":
            try:
                from modulos.databases.implementaciones.duckdb import DuckDBVectorialDatabase
                return DuckDBVectorialDatabase
            except ImportError as e:
                logger.error(f"Could not import DuckDBVectorialDatabase: {e}")
                raise ValueError(f"DuckDB is not available: {e}")
        elif db_type == "postgresql":
            try:
                # from modulos.databases.implementaciones.postgresql import PostgreSQLVectorialDatabase
                # return PostgreSQLVectorialDatabase
                pass
            except ImportError as e:
                logger.error(f"Could not import PostgreSQLVectorialDatabase: {e}")
                raise ValueError(f"PostgreSQL is not available: {e}")
        else:
            raise ValueError(f"Invalid database type: {db_type}")
    
    @classmethod
    def _generate_db_name(cls, embedding_model: str, chunking_method: str, db_type: str) -> str:
        """
        Generates a unique name for the database based on configuration.
        
        Args:
            embedding_model: Embedding model used.
            chunking_method: Chunking method used.
            db_type: Database type.
            
        Returns:
            Database name.
        """
        # Extract short name from models
        if '/' in embedding_model:
            embedding_model = embedding_model.split('/')[1]
        
        # Generate a random identifier
        random_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
        
        # Create safe name
        safe_name = f"rag_{embedding_model}_{chunking_method}_{db_type}_{random_id}"
        
        # Replace characters not allowed in file names
        safe_name = safe_name.replace('-', '_').replace('.', '_').lower()
        
        return safe_name
    
    @classmethod
    def _get_db_path(cls, db_type: str, db_name: str) -> str:
        """
        Gets the database path for the specific type and name.
        
        Args:
            db_type: Database type.
            db_name: Database name.
            
        Returns:
            Path to the database.
        """
        # For file databases (SQLite, DuckDB)
        if db_type in ["sqlite", "duckdb"]:
            db_config = cls._load_config()
            db_type_config = db_config.get(db_type, {})
            
            # Get base directory and ensure it's absolute
            db_dir = db_type_config.get("db_dir", "modulos/databases/db")
            
            # Ensure the path is absolute
            if not os.path.isabs(db_dir):
                db_dir = os.path.abspath(db_dir)
            
            # If name is empty, generate a name based on timestamp
            if not db_name:
                import time
                timestamp = int(time.time())
                db_name = f"rag_db_{timestamp}"
            
            # Ensure correct extension
            extension = ".db"
            if db_type == "sqlite":
                extension = ".sqlite"  # Important change: .db -> .sqlite for SQLite
            elif db_type == "duckdb":
                extension = ".duckdb"
                
            if not db_name.endswith(extension):
                db_name = f"{db_name}{extension}"
            
            # Create directory if it doesn't exist - IMPORTANT: This must always run!
            os.makedirs(db_dir, exist_ok=True)
            
            # Build the complete path
            db_path = os.path.join(db_dir, db_name)
            
            logger.info(f"Generated database path: {db_path}")
            return db_path
            
        elif db_type == "postgresql":
            # For PostgreSQL, the "path" is actually a connection string
            db_config = cls._load_config()
            pg_config = db_config.get("postgresql", {})
            
            # Build connection string
            conn_string = f"postgresql://{pg_config.get('user', 'postgres')}:{pg_config.get('password', '')}@"
            conn_string += f"{pg_config.get('host', 'localhost')}:{pg_config.get('port', 5432)}/{pg_config.get('database', 'rag_db')}"
            
            return conn_string
        else:
            # Unsupported database type
            logger.warning(f"Unsupported database type: {db_type}, using SQLite")
            return cls._get_db_path("sqlite", db_name)
    

    
    @classmethod
    def get_available_databases(cls) -> Dict[str, Dict[str, Any]]:
        """
        Scans and returns all available databases with their metadata.
        
        Returns:
            Dictionary with database names and their metadata.
        """
        available_dbs = {}
        
        # Search in the databases directory
        db_config = cls._load_config()
        
        # Load directly from configuration the main directory
        db_dir = Path(db_config.get("sqlite", {}).get("db_dir", "modulos/databases/db"))
        
        # Ensure it's an absolute path
        if not db_dir.is_absolute():
            db_dir = Path(os.path.abspath(db_dir))
        
        logger.info(f"Searching for databases in: {db_dir}")
        
        # Verify that the directory exists
        if not db_dir.exists():
            logger.warning(f"Database directory does not exist: {db_dir}")
            return available_dbs
        
        # Search for files with common database extensions
        db_files_by_ext = {}
        total_files = 0
        
        for extension in [".db", ".sqlite", ".duckdb"]:
            db_files = list(db_dir.glob(f"*{extension}"))
            db_files_by_ext[extension] = db_files
            total_files += len(db_files)
            logger.debug(f"Found {len(db_files)} files with extension {extension} in {db_dir}")
        
        if total_files == 0:
            logger.warning(f"No database files found in {db_dir}")
            return available_dbs
            
        logger.info(f"Total databases found: {total_files}")
        
        # Process each database by extension
        for extension, db_files in db_files_by_ext.items():
            for db_file in db_files:
                # Use filename without extension as key
                key = db_file.stem
                
                # Create basic database info
                available_dbs[key] = cls._create_default_metadata_for_file(db_file, extension)
                logger.debug(f"Found database: {key} (type: {available_dbs[key]['db_type']})")
        
        # Sort by creation/usage date
        for key in available_dbs:
            if "last_used" not in available_dbs[key] and "created_at" in available_dbs[key]:
                available_dbs[key]["last_used"] = available_dbs[key]["created_at"]
        
        # Show summary by type
        db_types = {}
        for key, metadata in available_dbs.items():
            db_type = metadata.get("db_type", "unknown")
            db_types[db_type] = db_types.get(db_type, 0) + 1
            
        for db_type, count in db_types.items():
            logger.info(f"Databases of type {db_type}: {count}")
            
        logger.info(f"Total available databases: {len(available_dbs)}")
        return available_dbs
    
    @classmethod
    def _create_default_metadata_for_file(cls, db_file: Path, extension: str) -> Dict[str, Any]:
        """
        Creates default metadata for a database file without metadata.
        
        Args:
            db_file: Path to the database file
            extension: File extension (.db, .sqlite, .duckdb)
            
        Returns:
            Dictionary with basic metadata
        """
        # Infer database type from extension
        if extension in [".db", ".sqlite"]:
            db_type = "sqlite"
        elif extension == ".duckdb":
            db_type = "duckdb"
        else:
            db_type = "unknown"
        
        # Get file statistics
        stat = db_file.stat()
        created_at = datetime.fromtimestamp(stat.st_ctime).isoformat()
        
        return {
            "db_type": db_type,
            "db_path": str(db_file),
            "created_at": created_at,
            "last_used": created_at,
            "embedding_model": "unknown",
            "chunking_method": "unknown",
            "embedding_dimension": None,
            "description": f"Database inferred from file {db_file.name}"
        }
    
    @classmethod
    def _load_config(cls) -> Dict[str, Any]:
        """
        Loads the database configuration from the config file.
        
        Returns:
            Configuration dictionary
        """
        try:
            config_path = Path("config/config.json")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading configuration: {e}")
        
        # Return default configuration
        return {
            "sqlite": {
                "db_dir": "modulos/databases/db"
            }
        }
    
    @classmethod
    def close_all_instances(cls) -> None:
        """
        Closes all active database instances.
        """
        for instance in cls._instances.values():
            try:
                if hasattr(instance, 'close'):
                    instance.close()
                    logger.debug(f"Closed database instance: {type(instance).__name__}")
            except Exception as e:
                logger.error(f"Error closing database instance: {e}")
        
        cls._instances.clear()
        logger.info("All database instances have been closed")
    

    
    @classmethod
    def get_db_statistics(cls) -> Dict[str, Any]:
        """
        Collects and aggregates statistics from all available databases.
        
        Returns:
            Dictionary with aggregated statistics
        """
        available_dbs = cls.get_available_databases()
        total_stats = {
            "total_databases": len(available_dbs),
            "by_type": {},
            "by_embedding_model": {},
            "total_chunks": 0,
            "total_size_mb": 0.0
        }
        
        for db_name, metadata in available_dbs.items():
            db_type = metadata.get("db_type", "unknown")
            embedding_model = metadata.get("embedding_model", "unknown")
            
            # Count by type
            total_stats["by_type"][db_type] = total_stats["by_type"].get(db_type, 0) + 1
            
            # Count by embedding model
            total_stats["by_embedding_model"][embedding_model] = total_stats["by_embedding_model"].get(embedding_model, 0) + 1
            
            # Try to get individual statistics
            try:
                db_instance = cls.get_database_instance(
                    embedding_model=embedding_model,
                    chunking_method=metadata.get("chunking_method", "unknown"),
                    db_type=db_type
                )
                
                if hasattr(db_instance, 'get_statistics'):
                    db_stats = db_instance.get_statistics()
                    total_stats["total_chunks"] += db_stats.get("total_chunks", 0)
                    total_stats["total_size_mb"] += db_stats.get("size_mb", 0.0)
                    
            except Exception as e:
                logger.warning(f"Error getting statistics for database {db_name}: {e}")
        
        return total_stats
    
    @classmethod
    def optimize_all_databases(cls) -> Dict[str, bool]:
        """
        Iterates through all available databases and optimizes them.
        
        Returns:
            Dictionary with optimization results for each database
        """
        available_dbs = cls.get_available_databases()
        results = {}
        
        for db_name, metadata in available_dbs.items():
            try:
                db_instance = cls.get_database_instance(
                    embedding_model=metadata.get("embedding_model", "unknown"),
                    chunking_method=metadata.get("chunking_method", "unknown"),
                    db_type=metadata.get("db_type", "sqlite")
                )
                
                if hasattr(db_instance, 'optimize'):
                    db_instance.optimize()
                    results[db_name] = True
                    logger.info(f"Successfully optimized database: {db_name}")
                else:
                    results[db_name] = False
                    logger.warning(f"Database {db_name} does not support optimization")
                    
            except Exception as e:
                results[db_name] = False
                logger.error(f"Error optimizing database {db_name}: {e}")
        
        return results
    
    @classmethod
    def create_database(cls, db_type: str, embedding_dim: int, **kwargs) -> 'VectorialDatabase':
        """
        Creates a new database instance of the specified type.
        
        Args:
            db_type: Type of database ('sqlite', 'duckdb', etc.)
            embedding_dim: Dimension of the embeddings
            **kwargs: Additional arguments for the database
            
        Returns:
            Instance of the vectorial database
        """
        db_class = cls._get_db_class(db_type)
        return db_class(embedding_dim=embedding_dim, **kwargs)
    
    @classmethod
    def load_database(cls, db_path: str) -> 'VectorialDatabase':
        """
        Loads an existing database from the specified path.
        
        Args:
            db_path: Path to the database file
            
        Returns:
            Instance of the vectorial database
            
        Raises:
            FileNotFoundError: If the database file doesn't exist
            ValueError: If the database type cannot be determined
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        # Try to get metadata from session that uses this database path
        metadata = None
        try:
            from modulos.session_manager.session_manager import SessionManager
            session_manager = SessionManager()
            sessions = session_manager.list_sessions()
            
            # Look for a session that uses this database path
            for session in sessions:
                if session.get("db_path") == db_path:
                    metadata = session
                    break
        except Exception as e:
            logger.debug(f"Could not load metadata from sessions for {db_path}: {e}")
        
        # Determine database type from session metadata or file extension
        db_type = None
        if metadata:
            db_type = metadata.get("db_type")
        
        if not db_type:
            # Try to infer from file extension
            extension = os.path.splitext(db_path)[1].lower()
            if extension in [".db", ".sqlite"]:
                db_type = "sqlite"
            elif extension == ".duckdb":
                db_type = "duckdb"
            else:
                raise ValueError(f"Cannot determine database type for: {db_path}")
        
        # Get embedding dimension from metadata or use default
        embedding_dim = 384  # Default value
        if metadata:
            embedding_dim = metadata.get("embedding_dim", 384)
        
        # Create database instance
        db_class = cls._get_db_class(db_type)
        db_instance = db_class(embedding_dim=embedding_dim)
        
        # Connect to the existing database
        db_instance.connect(db_path)
        
        return db_instance