"""
Gestión centralizada de la configuración.

Este módulo proporciona funciones para cargar y acceder a la configuración
del sistema desde un único punto.
"""

import os
import logging
from config import config  # Singleton global de configuración
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def get_config() -> Any:
    """
    Obtiene la instancia del singleton de configuración.
    
    Returns:
        Instancia del singleton de configuración
    """
    return config

def get_processing_config() -> Dict[str, Any]:
    """
    Obtiene la configuración de procesamiento.
    
    Returns:
        Diccionario con la configuración de procesamiento
    """
    return config.get_processing_config()

def get_embedding_config() -> Dict[str, Any]:
    """
    Obtiene la configuración de embeddings.
    
    Returns:
        Diccionario con la configuración de embeddings
    """
    return config.get_embedding_config()

def get_chunking_config() -> Dict[str, Any]:
    """
    Obtiene la configuración de chunking.
    
    Returns:
        Diccionario con la configuración de chunking
    """
    return config.get_chunks_config()

def get_database_config() -> Dict[str, Any]:
    """
    Obtiene la configuración de base de datos.
    
    Returns:
        Diccionario con la configuración de la base de datos
    """
    return config.get_database_config()

def get_ai_client_config() -> Dict[str, Any]:
    """
    Obtiene la configuración del cliente de IA.
    
    Returns:
        Diccionario con la configuración del cliente de IA
    """
    return config.get_ai_client_config()

def get_general_config() -> Dict[str, Any]:
    """
    Obtiene la configuración general.
    
    Returns:
        Diccionario con la configuración general
    """
    return config.get_general_config()

def is_debug_enabled() -> bool:
    """
    Verifica si el modo debug está habilitado.
    
    Returns:
        True si el modo debug está habilitado, False en caso contrario
    """
    return config.get_general_config().get("debug", False) 