import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """
    Class to manage the centralized configuration of the RAG system.
    Loads configuration from a YAML file and provides methods
    to access the different configuration sections.
    """
    _instance = None
    
    def __new__(cls, config_path: str = "config.yaml"):
        """
        Implements the Singleton pattern to ensure a single instance
        of configuration throughout the application.
        """
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes the configuration from a YAML file.
        
        Args:
            config_path: Path to YAML configuration file.
        """
        # Avoid reinitializing if already initialized (part of Singleton pattern)
        if self.initialized:
            return
            
        # Load environment variables from .env if it exists
        load_dotenv()
        
        self.config_path = config_path
        self.config = self._load_config()
        self.initialized = True
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Loads configuration from YAML file.
        
        Returns:
            Dictionary with configuration.
            
        Raises:
            FileNotFoundError: If configuration file is not found.
        """
        config_path = Path(self.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def _process_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes environment variables in configuration.
        
        Args:
            config: Configuration as dictionary.
            
        Returns:
            Configuration with processed environment variables.
        """
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, (dict, list)):
                    config[key] = self._process_env_vars(value)
                elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    config[key] = os.environ.get(env_var, "")
        elif isinstance(config, list):
            for i, item in enumerate(config):
                config[i] = self._process_env_vars(item)
        
        return config
    
    def get_config(self) -> Dict[str, Any]:
        """
        Gets the complete configuration.
        
        Returns:
            Configuration as dictionary.
        """
        # Process environment variables before returning
        return self._process_env_vars(self.config.copy())
    
    def get_general_config(self) -> Dict[str, Any]:
        """
        Gets the general system configuration.
        
        Returns:
            Dictionary with general configuration.
        """
        general_config = self.config.get("general", {})
        return self._process_env_vars(general_config)
    
    def get_sessions_config(self) -> Dict[str, Any]:
        """Get sessions configuration."""
        return self.config.get("sessions", {})
    
    def get_chunks_config(self) -> Dict[str, Any]:
        """
        Gets the document chunking configuration.
        
        Returns:
            Dictionary with chunks configuration.
        """
        chunks_config = self.config.get("chunks", {})
        return self._process_env_vars(chunks_config)
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Gets the embeddings model configuration.
        
        Returns:
            Dictionary with embeddings configuration.
        """
        embedding_config = self.config.get("embeddings", {})
        return self._process_env_vars(embedding_config)
    
    def get_database_config(self) -> Dict[str, Any]:
        """
        Gets the database configuration.
        
        Returns:
            Dictionary with database configuration.
        """
        # Get database configuration or empty dictionary if it doesn't exist
        db_config = self.config.get("database", {})
        
        # If db_config is None (which shouldn't happen, but for safety)
        if db_config is None:
            db_config = {}
            
        # Ensure database type entry exists
        if "type" not in db_config:
            db_config["type"] = "sqlite"  # Default value
            
        # Ensure entries for each database type exist
        if "sqlite" not in db_config:
            db_config["sqlite"] = {"db_dir": "modulos/databases/db", "db_name": ""}
            
        if "duckdb" not in db_config:
            db_config["duckdb"] = {"db_dir": "modulos/databases/db", "db_name": ""}
        
        # Update paths and process environment variables
        db_config = self._update_db_paths(db_config)
        return self._process_env_vars(db_config)
    
    def _update_db_paths(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates database paths to ensure they are correct
        and directories exist.
        
        Args:
            db_config: Original database configuration.
            
        Returns:
            Updated configuration with absolute paths.
        """
        # Verify db_config is a valid dictionary
        if not isinstance(db_config, dict):
            db_config = {}
            
        # Ensure database type exists
        db_type = db_config.get("type", "sqlite")
        
        # Get specific configuration for database type
        type_config = db_config.get(db_type, {})
        
        # Verify type_config is a valid dictionary
        if not isinstance(type_config, dict):
            type_config = {}
            db_config[db_type] = type_config
        
        # Check and provide default values if missing
        if "db_dir" not in type_config:
            type_config["db_dir"] = "modulos/databases/db"
            
        if "db_name" not in type_config:
            type_config["db_name"] = ""
        
        # Create directory if it doesn't exist
        try:
            db_dir = Path(type_config["db_dir"])
            # Ensure directory exists
            os.makedirs(db_dir, exist_ok=True)
            
            # Calculate and add full path
            db_name = type_config.get("db_name", "")
            db_path = db_dir / db_name
            type_config["db_path"] = str(db_path)
            db_config[db_type] = type_config
        except Exception as e:
            # If error, use default paths
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error configuring DB paths: {e}, using default values")
            
            # Set safe default values
            type_config["db_dir"] = "modulos/databases/db"
            type_config["db_name"] = ""
            type_config["db_path"] = "modulos/databases/db/"
            db_config[db_type] = type_config
        
        return db_config
    
    def get_ai_client_config(self) -> Dict[str, Any]:
        """
        Gets the AI client configuration.
        
        Returns:
            Dictionary with AI client configuration.
        """
        ai_client_config = self.config.get("ai_client", {})
        return self._process_env_vars(ai_client_config)
    
    def get_processing_config(self) -> Dict[str, Any]:
        """
        Gets the document processing configuration.
        
        Returns:
            Dictionary with processing configuration.
        """
        processing_config = self.config.get("processing", {})
        return self._process_env_vars(processing_config)
    
    def get_resource_management_config(self) -> Dict[str, Any]:
        """
        Gets the resource manager configuration.
        
        This method processes the 'resource_management' section of the configuration,
        which includes subsections like 'monitoring', 'memory' and 'concurrency'.
        
        Returns:
            Dictionary with resource management configuration, transformed
            to a flat structure for easier use in ResourceManager.
        """
        resource_config = self.config.get("resource_management", {})
        
        # Process environment variables in the configuration
        processed_config = self._process_env_vars(resource_config)
        
        # Convert hierarchical structure to flat structure for easier use
        # in ResourceManager, keeping the original section keys
        flat_config = {}
        
        # Include top-level properties
        for key, value in processed_config.items():
            if not isinstance(value, dict):
                flat_config[key] = value
        
        # Include subsections with their original keys
        for section_name, section_values in processed_config.items():
            if isinstance(section_values, dict):
                for key, value in section_values.items():
                    flat_config[f"{key}"] = value
        
        return flat_config
    
    def get_specific_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Gets the specific configuration for an embedding model.
        
        Args:
            model_type: Model type (e.g. 'modernbert', 'cde-small', 'e5-small')
        
        Returns:
            Specific model configuration.
        """
        embedding_config = self.get_embedding_config()
        model_config = embedding_config.get(model_type, {})
        return model_config
    
    def get_specific_ai_config(self, ai_type: str) -> Dict[str, Any]:
        """
        Gets the specific configuration for an AI client, combining
        general parameters with specific ones and filtering according to
        parameters supported by each client.
        
        Args:
            ai_type: Client type (e.g. 'openai', 'gemini', 'ollama')
        
        Returns:
            Specific AI client configuration with filtered parameters.
        """
        ai_config = self.get_ai_client_config()
        specific_config = ai_config.get(ai_type, {})
        
        # Combine with general parameters if they exist
        general_params = ai_config.get("general", {})
        
        # Ensure system_prompt is always available
        if "system_prompt" not in specific_config and "system_prompt" in general_params:
            specific_config["system_prompt"] = general_params["system_prompt"]
        
        # Specific parameters have priority over general ones
        combined_config = {**general_params, **specific_config}
        
        # Define parameters supported by each client
        client_params = {
            # Parameters common to all clients
            "common": [
                "model", "temperature", "max_tokens", "top_p", "top_k", "system_prompt", 
                "stream", "response_mime_type", "context_format", "instruction_style"
            ],
            
            # Parameters specific to OpenAI
            "openai": ["api_key", "api_key_env", "organization", "api_base", "api_base_env", 
                      "frequency_penalty", "presence_penalty", "timeout", "embedding_model"],
            
            # Parameters specific to Gemini
            "gemini": ["api_key", "api_key_env", "max_output_tokens", 
                      "embedding_model"],
            
            # Parameters specific to Ollama
            "ollama": ["base_url", "api_url", "api_url_env", "timeout", 
                      "num_predict", "embedding_model"]
        }
        
        # Get valid parameters for this client type
        valid_params = client_params["common"] + client_params.get(ai_type, [])
        
        # Filter configuration only for valid parameters
        filtered_config = {k: v for k, v in combined_config.items() if k in valid_params}
        
        # Handle alternative parameter names
        if ai_type == "gemini" and "max_tokens" in filtered_config and "max_output_tokens" not in filtered_config:
            filtered_config["max_output_tokens"] = filtered_config.pop("max_tokens")
        
        if ai_type == "ollama" and "max_tokens" in filtered_config and "num_predict" not in filtered_config:
            filtered_config["num_predict"] = filtered_config.get("max_tokens")
        
        return filtered_config
    
    def get_chunker_method_config(self, method: Optional[str] = None) -> Dict[str, Any]:
        """
        Gets the configuration for a specific chunking method.
        
        Args:
            method: Chunking method ('character', 'token', 'context').
                   If None, the default configured method is used.
        
        Returns:
            Configuration for the chunking method.
        """
        chunks_config = self.get_chunks_config()
        if method is None:
            method = chunks_config.get("method", "context")
        
        method_config = chunks_config.get(method, {})
        return method_config
    
    def save_config(self) -> None:
        """
        Saves the current configuration to the YAML file.
        """
        with open(self.config_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.config, file, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    def update_config(self, section_path: list, value: Any) -> None:
        """
        Updates a specific value in the configuration.
        
        Args:
            section_path: Path to the section (e.g. ['ai_client', 'parameters', 'stream'])
            value: New value
        """
        if not isinstance(section_path, list) or not section_path:
            raise ValueError("section_path must be a non-empty list")
        
        # Navigate to the correct section
        current = self.config
        for key in section_path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Update the value
        current[section_path[-1]] = value
    
    def get_database_instance_config(self, db_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Gets the specific configuration to initialize a database instance.
        
        Args:
            db_type: Database type. If None, the configured one is used.
            
        Returns:
            Configuration to initialize the database.
        """
        db_config = self.get_database_config()
        
        if db_type is None:
            db_type = db_config.get("type", "sqlite")
        
        type_config = db_config.get(db_type, {})
        
        # Add type to configuration
        instance_config = {"type": db_type, **type_config}
        
        return instance_config

# Global instance for use as singleton
config = Config()