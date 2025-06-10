"""
Session management for the RAG system.

This module handles RAG sessions, keeping track of used configurations
and facilitating access to existing databases.
"""

import os
import json
import logging
import time
import threading

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from config import Config

logger = logging.getLogger(__name__)

# Instantiate Config class correctly
config = Config()

# Custom get_value method for backwards compatibility
def get_value(section, key, default=None):
    """
    Gets a specific configuration value.
    
    Args:
        section: Configuration section
        key: Value key
        default: Default value if not found
        
    Returns:
        Configuration value or default value
    """
    try:
        # Try to get the corresponding getter method
        getter_method = getattr(config, f"get_{section}_config", None)
        if getter_method is None:
            # If the method doesn't exist, look in general configuration
            general_config = config.get_general_config() or {}
            if section in general_config:
                return general_config[section].get(key, default)
            return default
            
        # Get section configuration
        section_config = getter_method()
        # Verify that section_config is not None before calling get()
        if section_config is None:
            return default
            
        return section_config.get(key, default)
    except (AttributeError, KeyError, Exception) as e:
        logger.warning(f"Error getting configuration value {section}.{key}: {e}")
        return default

# Configuration limits and timeouts for sessions
try:
    # Try to get session configuration with error handling
    MAX_SESSIONS = get_value("sessions", "max_sessions", 50)  
    SESSION_TIMEOUT = get_value("sessions", "timeout", 3600)  # 1 hour by default
    CLEANUP_INTERVAL = get_value("sessions", "cleanup_interval", 300)  # 5 minutes by default
    MAX_CONTEXTS_PER_SESSION = get_value("sessions", "max_contexts", 50)  # Maximum number of contexts per session
except Exception as e:
    # If it fails, use default values and log the error
    logger.debug(f"Using default values for session configuration: {e}")
    MAX_SESSIONS = 50
    SESSION_TIMEOUT = 3600
    CLEANUP_INTERVAL = 300
    MAX_CONTEXTS_PER_SESSION = 50

class SessionManager:
    """
    Manages user sessions and their associated data.
    
    Implements a singleton pattern to ensure a single global instance.
    Includes functionality to:
    - Create and maintain sessions
    - Store context data by message
    - Clean up inactive sessions
    - Monitor resource usage
    
    In the new implementation, each session is stored in an individual JSON file
    in the sessions directory, and represents a unified concept of session and database.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    # Default configuration values
    MAX_SESSIONS = 100            # Maximum limit of active sessions
    SESSION_TIMEOUT = 3600 * 24 * 7  # 1 week by default, could be configurable
    CLEANUP_INTERVAL = 60         # Interval (in seconds) between automatic cleanups
    MAX_CONTEXTS_PER_SESSION = 30 # Maximum number of contexts stored per session
    
    def __new__(cls):
        """
        Implementation of the singleton pattern.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SessionManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """
        Initializes the session manager.
        """
        # Avoid reinitializing the singleton
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized: # Double check in case another thread initialized
                return
            
            self.config = Config()
            self.general_config = self.config.get_general_config()
            # Ensure that the sessions path is absolute
            sessions_dir_config = self.general_config.get("sessions_dir", "sessions")
            if os.path.isabs(sessions_dir_config):
                self.sessions_dir = Path(sessions_dir_config)
            else:
                # If it's a relative path, make it absolute from the project root directory
                root_dir = Path(__file__).parent.parent.parent
                self.sessions_dir = root_dir / sessions_dir_config
            
            # Ensure the directory exists
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Sessions directory configured at: {self.sessions_dir}")
            
            # Cache for sessions (to avoid reading from disk every time)
            self._sessions_cache = {}
            # Store last modify time for each session file to know when to reload
            self._sessions_last_modified = {}
            # Initialize context space (in-memory only, not persisted)
            self.contexts = {}

            # Initialize reference to ResourceManager (lazy initialization)
            self._resource_manager = None
            
            # Migrate existing sessions if needed
            self._migrate_existing_sessions()

            self._initialized = True
            logger.info("SessionManager initialized. Cleanup is now coordinated by ResourceManager.")
    
    def _migrate_existing_sessions(self):
        """
        Migrates existing sessions from the old format (sessions.json and db_metadata.json)
        to the new format (one file per session).
        """
        try:
            old_sessions_file = self.sessions_dir / "sessions.json"
            old_db_metadata_file = self.sessions_dir / "db_metadata.json"
            
            if old_sessions_file.exists() and old_db_metadata_file.exists():
                logger.info("Found old session format files. Starting migration...")
                
                # Load old data
                old_sessions = self._load_from_file(old_sessions_file)
                old_db_metadata = self._load_from_file(old_db_metadata_file)
                
                # Create new format sessions
                migrated_count = 0
                for session_id, session_data in old_sessions.items():
                    # Find corresponding DB metadata
                    db_name = session_data.get("db_name")
                    if not db_name or db_name not in old_db_metadata:
                        continue
                    
                    db_metadata = old_db_metadata[db_name]
                    
                    # Create unified session data
                    unified_data = {
                        "id": session_id,
                        "name": session_id,  # Use the same ID as name for now
                        "created_at": session_data.get("created_at", time.time()),
                        "last_modified": session_data.get("last_activity", time.time()),
                        "db_path": db_metadata.get("db_path", ""),
                        "db_type": db_metadata.get("db_type", "sqlite"),
                        "embedding_model": db_metadata.get("embedding_model", "modernbert"),
                        "embedding_dim": db_metadata.get("embedding_dim", 768),
                        "chunking_method": db_metadata.get("chunking_method", "character"),
                        "files": session_data.get("files", []),
                        "total_chunks": db_metadata.get("total_chunks", 0),  # At root level for consistency
                        "stats": {
                            "total_documents": db_metadata.get("total_files", 0)
                        }
                    }
                    
                    # Save new format file
                    session_file = self.sessions_dir / f"{session_id}.json"
                    with open(session_file, 'w', encoding='utf-8') as f:
                        json.dump(unified_data, f, indent=2, ensure_ascii=False)
                    
                    migrated_count += 1
                
                # Create backup of old files
                backup_dir = self.sessions_dir / "backup"
                backup_dir.mkdir(exist_ok=True)
                
                # Move old files to backup
                import shutil
                shutil.copy2(old_sessions_file, backup_dir / "sessions.json.bak")
                shutil.copy2(old_db_metadata_file, backup_dir / "db_metadata.json.bak")
                
                logger.info(f"Migration completed. {migrated_count} sessions migrated to new format.")
                
        except Exception as e:
            logger.error(f"Error during session migration: {e}")
    
    def _load_from_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Loads data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary with loaded data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading data from {file_path}: {e}")
            return {}
    
    def _load_session(self, session_id: str, force_reload: bool = False) -> Optional[Dict[str, Any]]:
        """
        Loads a session from its individual JSON file.
        
        Args:
            session_id: ID of the session to load
            force_reload: If True, forces reloading from disk
            
        Returns:
            Session data or None if not found
        """
        try:
            # Check if already in cache and up to date
            if not force_reload and session_id in self._sessions_cache:
                # Check if file has been modified since last load
                session_file = self.sessions_dir / f"{session_id}.json"
                if session_file.exists():
                    last_modified = session_file.stat().st_mtime
                    if session_id in self._sessions_last_modified:
                        if last_modified <= self._sessions_last_modified[session_id]:
                            # Return cached version if file hasn't changed
                            return self._sessions_cache[session_id]
                
            # Load from file
            session_file = self.sessions_dir / f"{session_id}.json"
            if not session_file.exists():
                return None
                
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                
            # Update cache
            self._sessions_cache[session_id] = session_data
            self._sessions_last_modified[session_id] = session_file.stat().st_mtime
                
            return session_data
            
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None
    
    def _save_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """
        Saves a session to its individual JSON file.
        
        Args:
            session_id: ID of the session
            session_data: Session data to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure sessions directory exists
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            
            # Update last_modified timestamp
            session_data["last_modified"] = time.time()
            
            # Save to file
            session_file = self.sessions_dir / f"{session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
                
            # Update cache
            self._sessions_cache[session_id] = session_data
            self._sessions_last_modified[session_id] = session_file.stat().st_mtime
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {e}")
            return False
    
    def create_session(self, session_id: Optional[str] = None, 
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Creates a new session.
        
        Args:
            session_id: Custom session ID (optional)
            metadata: Additional data for the session
        
        Returns:
            ID of the created session
        
        Raises:
            ValueError: If maximum session limit is reached
        """
        with self._lock:
            # Check session limit
            existing_sessions = len(list(self.sessions_dir.glob("*.json")))
            if existing_sessions >= self.MAX_SESSIONS:
                # Try to clean up first
                self.clean_expired_sessions(aggressive=True)
                
                # If still too many sessions, reject creation
                if len(list(self.sessions_dir.glob("*.json"))) >= self.MAX_SESSIONS:
                    logger.error(f"Session limit reached: {self.MAX_SESSIONS}")
                    raise ValueError(f"Maximum session limit reached ({self.MAX_SESSIONS})")
            
            # Generate ID if not provided
            if not session_id:
                # Simple UUID based on timestamp
                session_id = f"session_{int(time.time() * 1000)}"
            
            # Check if already exists
            if (self.sessions_dir / f"{session_id}.json").exists():
                logger.warning(f"Session {session_id} already exists, updating")
                return self.update_session_metadata(session_id, metadata)
            
            # Create session
            current_time = time.time()
            session_data = {
                "id": session_id,
                "name": session_id,  # Default name is the same as ID
                "created_at": current_time,
                "last_modified": current_time,
                "files": [],  # Empty list to store processed files
                "stats": {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "creation_date": datetime.now().isoformat()
                }
            }
            
            # Add metadata if provided
            if metadata:
                session_data.update(metadata)
            
            # Save session
            self._save_session(session_id, session_data)
            
            # Initialize context space
            self.contexts[session_id] = {}
            
            logger.info(f"Session {session_id} created successfully")
            return session_id
    
    def update_session_metadata(self, session_id: str, 
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Updates metadata of an existing session.
        
        Args:
            session_id: ID of the session
            metadata: Updated data for the session
        
        Returns:
            ID of the updated session
        
        Raises:
            KeyError: If the session doesn't exist
        """
        with self._lock:
            # Check if session exists
            session_data = self._load_session(session_id)
            if not session_data:
                logger.warning(f"Attempt to update non-existent session: {session_id}, creating new")
                return self.create_session(session_id, metadata)
            
            # Update last_modified timestamp
            session_data["last_modified"] = time.time()
            
            # Add or update metadata if provided
            if metadata:
                session_data.update(metadata)
            
            # Save changes
            self._save_session(session_id, session_data)
            
            return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Gets data for a session.
        
        Args:
            session_id: ID of the session
        
        Returns:
            Session data or None if doesn't exist
        """
        with self._lock:
            session_data = self._load_session(session_id)
            if not session_data:
                return None
            
            # Update last_modified time when accessed
            session_data["last_modified"] = time.time()
            self._save_session(session_id, session_data)
            
            # Return a copy to prevent uncontrolled modifications
            return dict(session_data)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Deletes a session and its associated data.
        
        Args:
            session_id: ID of the session to delete
        
        Returns:
            True if deleted successfully, False if didn't exist
        """
        with self._lock:
            session_file = self.sessions_dir / f"{session_id}.json"
            if not session_file.exists():
                return False
            
            # Delete context if it exists
            if session_id in self.contexts:
                del self.contexts[session_id]
            
            # Delete from cache
            if session_id in self._sessions_cache:
                del self._sessions_cache[session_id]
            if session_id in self._sessions_last_modified:
                del self._sessions_last_modified[session_id]
            
            # Delete file
            try:
                session_file.unlink()
                logger.info(f"Session deleted: {session_id}")
                return True
            except Exception as e:
                logger.error(f"Error deleting session file {session_id}: {e}")
                return False
    
    def list_sessions(self, sort_by: str = "last_modified", reverse: bool = True) -> List[Dict[str, Any]]:
        """
        Lists all available sessions with detailed information.
        
        Args:
            sort_by: Field to sort by ("last_modified", "created_at", "name")
            reverse: If True, sort in descending order
            
        Returns:
            List of session data dictionaries
        """
        with self._lock:
            sessions = []
            
            # Find all session files
            session_files = list(self.sessions_dir.glob("*.json"))
            
            for session_file in session_files:
                try:
                    # Load session data
                    with open(session_file, 'r', encoding='utf-8') as f:
                        try:
                            session_data = json.load(f)
                        except json.JSONDecodeError as json_err:
                            # Try to repair the corrupted JSON file
                            logger.error(f"Corrupted JSON in {session_file}: {json_err}. Attempting repair...")
                            # Read the content as text
                            f.seek(0)
                            content = f.read()
                            # Buscar el primer cierre de llave
                            first_closing = content.find('}')
                            if first_closing > 0:
                                # Use only up to the first closing brace
                                fixed_content = content[:first_closing+1]                                
                                try:
                                    session_data = json.loads(fixed_content)
                                    logger.info(f"JSON file repaired: {session_file}")
                                except (json.JSONDecodeError, ValueError):
                                    logger.error(f"Could not repair JSON file: {session_file}")
                                    continue
                            else:
                                logger.error(f"Could not repair JSON file: {session_file}")
                                continue
                    

                    
                    # Add to list
                    sessions.append(session_data)
                except Exception as e:
                    logger.error(f"Error loading session from {session_file}: {e}")
                    # Continuar con el siguiente archivo
            
            # Sort sessions
            if sort_by in ["last_modified", "created_at", "name"]:
                sessions.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)
            
            return sessions
    
    def get_session_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Gets a session by its index in the sorted list (0 is most recent).
        
        Args:
            index: Index of the session in the sorted list
            
        Returns:
            Session data or None if index out of range
        """
        with self._lock:
            sessions = self.list_sessions()
            
            if index < 0 or index >= len(sessions):
                logger.warning(f"Session index out of range: {index}")
                return None
                
            return sessions[index]
    
    def get_database_by_index(self, index: int) -> Tuple[Any, Dict[str, Any]]:
        """
        Gets a database instance and its metadata by index.
        
        Args:
            index: Index of the database in the sorted list
            
        Returns:
            Tuple (database instance, metadata)
            
        Raises:
            IndexError: If the index is out of range
            ValueError: If database cannot be loaded
        """
        with self._lock:
            # Get session by index
            session = self.get_session_by_index(index)
            if not session:
                raise IndexError(f"Database index out of range: {index}")
                
            # Extract necessary information
            session_id = session.get("id")
            db_path = session.get("db_path")
            db_type = session.get("db_type", "sqlite")
            embedding_dim = session.get("embedding_dim", 768)
            embedding_model = session.get("embedding_model", "modernbert")
            chunking_method = session.get("chunking_method", "character")
            
            # Validate required fields
            if not session_id:
                raise ValueError("Missing session ID")
            if not db_path:
                raise ValueError(f"Missing database path for session {session_id}")
                
            # Load database
            try:
                from modulos.databases.FactoryDatabase import DatabaseFactory
                db = DatabaseFactory.get_database_instance(
                    db_type=db_type,
                    embedding_dim=embedding_dim,
                    embedding_model=embedding_model,
                    chunking_method=chunking_method,
                    load_existing=True,
                    db_path=db_path
                )
                
                # Update last_modified timestamp
                session["last_modified"] = time.time()
                self._save_session(session_id, session)
                
                return db, session
            except Exception as e:
                logger.error(f"Error loading database for session {session_id}: {e}")
                raise ValueError(f"Could not load database: {e}")
    
    def store_message_context(self, session_id: str, message_id: str, 
                            context_data: List[Dict[str, Any]]) -> bool:
        """
        Stores context used to generate a response.
        
        Args:
            session_id: ID of the session
            message_id: ID of the message
            context_data: Context data to store
        
        Returns:
            True if stored successfully, False otherwise
        """
        with self._lock:
            # Check if session exists
            if not self._load_session(session_id):
                logger.warning(f"Attempt to store context in non-existent session: {session_id}")
                return False
            
            # Initialize context dictionary if doesn't exist
            if session_id not in self.contexts:
                self.contexts[session_id] = {}
            
            # Check context limit per session
            contexts = self.contexts[session_id]
            if len(contexts) >= self.MAX_CONTEXTS_PER_SESSION:
                # Remove oldest context
                oldest_message_id = min(contexts.keys())
                del contexts[oldest_message_id]
                logger.debug(f"Context limit reached for session {session_id}, removing oldest")
            
            # Store context
            self.contexts[session_id][message_id] = context_data
            
            # Update last_modified timestamp
            session_data = self._load_session(session_id)
            if session_data:
                session_data["last_modified"] = time.time()
                self._save_session(session_id, session_data)
            
            logger.debug(f"Context stored: session={session_id}, message={message_id}, fragments={len(context_data)}")
            return True
    
    def get_message_context(self, session_id: str, message_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Gets context used for a specific message.
        
        Args:
            session_id: ID of the session
            message_id: ID of the message
        
        Returns:
            Context data or None if doesn't exist
        """
        with self._lock:
            # Check if session exists
            if not self._load_session(session_id) or session_id not in self.contexts:
                return None
            
            # Update last_modified timestamp when accessed
            session_data = self._load_session(session_id)
            if session_data:
                session_data["last_modified"] = time.time()
                self._save_session(session_id, session_data)
            
            # Get context
            if message_id not in self.contexts[session_id]:
                return None
            
            # Return a copy to prevent uncontrolled modifications
            return list(self.contexts[session_id][message_id])
    
    def clean_expired_sessions(self, aggressive: bool = False, cleanup_reason: str = "routine") -> Dict[str, Any]:
        """
        Cleans expired sessions and associated resources.
        
        This function is typically invoked by `ResourceManager.request_cleanup`.
        The decision whether cleaning is `aggressive` is made by `ResourceManager`
        based on global resource thresholds.
        
        Args:
            aggressive (bool): Indicates if more aggressive cleaning should be performed
                               (delete older sessions if still too many after
                               deleting expired ones).
                               Defaults to False.
            cleanup_reason (str): Reason for executing the cleanup.
                                  Defaults to "routine".
                                  
        Returns:
            Dict[str, Any]: Cleanup results (deleted sessions, errors).
        """
        results = {
            "status": "success", 
            "timeout_removed": 0,
            "aggressive_removed": 0,
            "remaining_sessions": 0
        }
        logger.info(f"Starting clean_expired_sessions. Aggressive: {aggressive}, Reason: {cleanup_reason}")
        
        with self._lock:
            try:
                current_time = time.time()
                
                # Get all session files
                session_files = list(self.sessions_dir.glob("*.json"))
                sessions_to_remove_by_timeout = []
                
                # Identify expired sessions by timeout
                for session_file in session_files:
                    try:
                        # Get basic metadata without loading full session
                        if session_file.exists():
                            stat = session_file.stat()
                            # Use file modification time as a proxy for last activity
                            last_modified = stat.st_mtime
                            
                            if current_time - last_modified > self.SESSION_TIMEOUT:
                                sessions_to_remove_by_timeout.append(session_file)
                    except Exception as e:
                        logger.error(f"Error checking session timeout {session_file}: {e}")
                
                # Delete expired sessions
                for session_file in sessions_to_remove_by_timeout:
                    try:
                        session_id = session_file.stem
                        # Delete from memory if loaded
                        if session_id in self._sessions_cache:
                            del self._sessions_cache[session_id]
                        if session_id in self._sessions_last_modified:
                            del self._sessions_last_modified[session_id]
                        if session_id in self.contexts:
                            del self.contexts[session_id]
                            
                        # Delete file
                        session_file.unlink()
                        results["timeout_removed"] += 1
                    except Exception as e:
                        logger.error(f"Error deleting expired session {session_file}: {e}")
                
                if results["timeout_removed"] > 0:
                    logger.info(f"{results['timeout_removed']} sessions removed due to timeout.")
                
                # If aggressive cleanup and still too many sessions
                if aggressive:
                    # Get remaining files after timeout cleanup
                    remaining_files = list(self.sessions_dir.glob("*.json"))
                    
                    if len(remaining_files) > self.MAX_SESSIONS * 0.8:
                        logger.warning(f"Aggressive session cleanup activated due to high pressure and {len(remaining_files)}/{self.MAX_SESSIONS} sessions.")
                        # Sort by modification time (oldest first)
                        remaining_files.sort(key=lambda f: f.stat().st_mtime)
                        
                        # Calculate how many to remove (to reduce to 70%)
                        num_to_remove = len(remaining_files) - int(self.MAX_SESSIONS * 0.7)
                        num_to_remove = max(0, num_to_remove)
                        
                        # Remove oldest sessions
                        for i in range(min(num_to_remove, len(remaining_files))):
                            try:
                                session_file = remaining_files[i]
                                session_id = session_file.stem
                                
                                # Skip if modified recently (last minute)
                                if current_time - session_file.stat().st_mtime <= 60:
                                    continue
                                    
                                # Delete from memory if loaded
                                if session_id in self._sessions_cache:
                                    del self._sessions_cache[session_id]
                                if session_id in self._sessions_last_modified:
                                    del self._sessions_last_modified[session_id]
                                if session_id in self.contexts:
                                    del self.contexts[session_id]
                                    
                                # Delete file
                                session_file.unlink()
                                results["aggressive_removed"] += 1
                                logger.warning(f"Aggressively removed session {session_id} due to resource constraints.")
                            except Exception as e:
                                logger.error(f"Error during aggressive removal of session {remaining_files[i]}: {e}")
                        
                        if results["aggressive_removed"] > 0:
                            logger.info(f"{results['aggressive_removed']} sessions removed aggressively.")
                
            except Exception as e:
                logger.error(f"Error during clean_expired_sessions: {e}", exc_info=True)
                results["status"] = "error"
                results["error"] = str(e)
            
            # Count remaining sessions
            try:
                results["remaining_sessions"] = len(list(self.sessions_dir.glob("*.json")))
            except Exception:
                results["remaining_sessions"] = -1
        
        logger.info(f"clean_expired_sessions completed. Results: {results}")
        return results
    
    def get_active_sessions_count(self) -> int:
        """Returns the current number of active sessions."""
        with self._lock:
            return len(list(self.sessions_dir.glob("*.json")))
    
    # Property for lazy initialization of ResourceManager
    @property
    def resource_manager(self):
        """Gets a ResourceManager instance with lazy initialization and cycle prevention."""
        if self._resource_manager is None:
            # Detect initialization cycle
            if getattr(self, '_initializing_resource_manager', False):
                logger.warning("Initialization cycle detected between SessionManager and ResourceManager")
                return None
            
            self._initializing_resource_manager = True
            try:
                # Import within method to avoid circular dependency
                from modulos.resource_management.resource_manager import ResourceManager
                self._resource_manager = ResourceManager()
                logger.debug("ResourceManager retrieved for SessionManager.")
            except Exception as e:
                logger.error(f"Error accessing ResourceManager: {e}")
            finally:
                self._initializing_resource_manager = False
        return self._resource_manager

    def update_session_file_list(self, session_id: str, new_files: List[str], 
                           file_metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> bool:
        """
        Updates the list of files in a session with a simple list of paths.
        
        Args:
            session_id: ID of the session to update
            new_files: List of file paths (str) to add
            file_metadata: Dictionary with additional metadata (optional)
                          
        Returns:
            bool: True if updated successfully, False otherwise
        """
        with self._lock:
            try:
                # Check if session exists
                session_data = self._load_session(session_id)
                if not session_data:
                    logger.error(f"Cannot update file list: session {session_id} does not exist")
                    return False
                
                # Initialize or get existing files list
                if "files" not in session_data:
                    session_data["files"] = []
                
                # Add new paths, avoiding duplicates
                for file_path in new_files:
                    abs_path = os.path.abspath(file_path)
                    # Check if file already exists in list
                    exists = False
                    for existing_file in session_data["files"]:
                        if isinstance(existing_file, str) and os.path.abspath(existing_file) == abs_path:
                            exists = True
                            break
                        elif isinstance(existing_file, dict) and os.path.abspath(existing_file.get("path", "")) == abs_path:
                            exists = True
                            break
                    
                    # If file doesn't exist in list, add it with metadata
                    if not exists:
                        if file_metadata and abs_path in file_metadata:
                            # Add as dictionary with metadata
                            meta = file_metadata[abs_path]
                            file_entry = {
                                "path": abs_path,
                                "name": os.path.basename(abs_path),
                                "processed_at": time.time()
                            }
                            # Add additional metadata
                            file_entry.update(meta)
                            session_data["files"].append(file_entry)
                        else:
                            # Add as simple path
                            session_data["files"].append(abs_path)
                
                # Update stats
                if "stats" not in session_data:
                    session_data["stats"] = {}
                
                session_data["stats"]["total_documents"] = len(session_data["files"])
                
                # Save changes
                self._save_session(session_id, session_data)
                
                logger.info(f"File list updated for session {session_id}: {len(session_data['files'])} files")
                return True
                
            except Exception as e:
                logger.error(f"Error updating file list for session {session_id}: {e}")
                return False
    
    def create_unified_session(self, 
                          database_metadata: Dict[str, Any],
                          files_list: Optional[List[str]] = None) -> str:
        """
        Creates a unified session that integrates session and database metadata with a common ID.
        
        This method replaces previous separate methods for creating sessions and registering databases,
        ensuring everything is created at once after processing.
        
        Args:
            database_metadata: Complete database metadata
            files_list: Optional list of processed file paths
            
        Returns:
            ID of the unified session/database
        """
        # Make sure time module is imported
        import time
        import hashlib
        from datetime import datetime
        
        with self._lock:
            try:
                # Check session limit
                existing_sessions = len(list(self.sessions_dir.glob("*.json")))
                if existing_sessions >= self.MAX_SESSIONS:
                    # Try to clean up first
                    self.clean_expired_sessions(aggressive=True)
                    
                    # If still too many sessions, reject creation
                    if len(list(self.sessions_dir.glob("*.json"))) >= self.MAX_SESSIONS:
                        logger.error(f"Session limit reached: {self.MAX_SESSIONS}")
                        raise ValueError(f"Maximum session limit reached ({self.MAX_SESSIONS})")
                
                # Get or generate unique ID for session/database
                db_name = database_metadata.get("name", "")
                session_id = db_name
                
                # If no name, generate one based on metadata
                if not session_id:
                    # Generate name based on configuration (e.g., rag_modernbert_page_sqlite_xxx)
                    db_type = database_metadata.get("db_type", "sqlite")
                    embedding_model = database_metadata.get("embedding_model", "default")
                    chunking_method = database_metadata.get("chunking_method", "default")
                    
                    # Generate short unique hash
                    hash_input = f"{embedding_model}_{chunking_method}_{time.time()}"
                    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
                    
                    # Build name with consistent format
                    session_id = f"rag_{embedding_model}_{chunking_method}_{db_type}_{short_hash}"
                    database_metadata["name"] = session_id
                
                # Ensure ID and name are the same
                database_metadata["id"] = session_id
                
                # Create session with current timestamp
                current_time = time.time()
                session_data = {
                    "id": session_id,
                    "name": session_id,
                    "created_at": current_time,
                    "last_modified": current_time,
                    "files": files_list or [],  # Simplified file list
                    "stats": {
                        "total_documents": len(files_list) if files_list else 0,
                        "creation_date": datetime.now().isoformat()
                    }
                }
                
                # Merge with database metadata
                session_data.update(database_metadata)
                
                # Check required fields
                required_fields = ["db_type", "db_path", "embedding_dim", "embedding_model", 
                                  "chunking_method"]
                
                # Complete missing fields
                for field in required_fields:
                    if field not in session_data:
                        if field in ["created_at", "last_modified"]:
                            session_data[field] = current_time
                
                # Save session
                self._save_session(session_id, session_data)
                
                # Initialize context space
                self.contexts[session_id] = {}
                
                logger.info(f"Unified session {session_id} created successfully")
                return session_id
                
            except Exception as e:
                logger.error(f"Error creating unified session: {e}")
                raise
