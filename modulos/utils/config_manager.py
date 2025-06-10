"""
Centralized configuration management.

This module provides functions to load and access the system
configuration from a single point.
"""

import logging
from config import config  # Global configuration singleton
from typing import Dict, Any

logger = logging.getLogger(__name__)

def get_config() -> Any:
    """
    Gets the configuration singleton instance.
    
    Returns:
        Configuration singleton instance
    """
    return config

def get_processing_config() -> Dict[str, Any]:
    """
    Gets the processing configuration.
    
    Returns:
        Dictionary with processing configuration
    """
    return config.get_processing_config()

def get_embedding_config() -> Dict[str, Any]:
    """
    Gets the embeddings configuration.
    
    Returns:
        Dictionary with embeddings configuration
    """
    return config.get_embedding_config()

def get_chunking_config() -> Dict[str, Any]:
    """
    Gets the chunking configuration.
    
    Returns:
        Dictionary with chunking configuration
    """
    return config.get_chunks_config()

def get_database_config() -> Dict[str, Any]:
    """
    Gets the database configuration.
    
    Returns:
        Dictionary with database configuration
    """
    return config.get_database_config()

def get_ai_client_config() -> Dict[str, Any]:
    """
    Gets the AI client configuration.
    
    Returns:
        Dictionary with AI client configuration
    """
    return config.get_ai_client_config()

def get_general_config() -> Dict[str, Any]:
    """
    Gets the general configuration.
    
    Returns:
        Dictionary with general configuration
    """
    return config.get_general_config()

def is_debug_enabled() -> bool:
    """
    Verifies if debug mode is enabled.
    
    Returns:
        True if debug mode is enabled, False otherwise
    """
    return config.get_general_config().get("debug", False)