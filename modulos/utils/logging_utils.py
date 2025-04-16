#!/usr/bin/env python
"""
Módulo de utilidades para configurar y personalizar el logging del sistema RAG.
"""

import logging
import os
import sys
from datetime import datetime

def setup_logging(level=logging.INFO, log_file=None):
    """
    Configura el sistema de logging con el nivel especificado y opcionalmente un archivo de log.
    
    Args:
        level: Nivel de logging (logging.DEBUG, logging.INFO, etc.)
        log_file: Ruta opcional al archivo donde se guardarán los logs
    
    Returns:
        El logger configurado
    """
    # Formato del log: timestamp, nivel, mensaje
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Crear el formateador
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Configurar el logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Limpiar handlers existentes para evitar duplicados
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Añadir handler para la consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Si se especificó un archivo de log, añadir un handler para él
    if log_file:
        # Asegurar que el directorio existe
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # Añadir handler para el archivo
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

def silence_verbose_loggers(verbose_mode: bool = False) -> None:
    """
    Silencia los loggers excesivamente verbosos que no son relevantes para el usuario.
    
    Args:
        verbose_mode: Si es True, no silencia los loggers (modo debug/verbose)
    """
    if verbose_mode:
        return
    
    # Lista de módulos a silenciar (ERROR = solo errores graves, WARNING = solo warnings y errores)
    modules_to_silence = {
        'modulos.databases': logging.ERROR,
        'modulos.databases.implementaciones': logging.ERROR,
        'modulos.databases.implementaciones.sqlite': logging.ERROR,
        'modulos.databases.implementaciones.duckdb': logging.ERROR,
        'modulos.session_manager': logging.WARNING,
        'sentence_transformers': logging.WARNING,
        'transformers': logging.ERROR,
        'filelock': logging.ERROR,
        'huggingface_hub': logging.ERROR,
    }
    
    # Aplicar niveles de logging
    for module_name, level in modules_to_silence.items():
        logging.getLogger(module_name).setLevel(level)

def get_timestamp_str():
    """
    Obtiene un string con el timestamp actual en formato adecuado para nombres de archivo.
    
    Returns:
        String con el timestamp actual
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S") 