#!/usr/bin/env python
"""
Utility module for configuring and customizing the logging of the RAG system.
"""

import logging
import os
import sys
from datetime import datetime

def setup_logging(level=logging.INFO, log_file=None):
    """
    Configures the logging system with the specified level and optionally a log file.
    
    Args:
        level: Logging level (logging.DEBUG, logging.INFO, etc.)
        log_file: Optional path to the file where logs will be stored
    
    Returns:
        The configured logger
    """
    # Log format: timestamp, level, message
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Create the formatter
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add handler for the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # If a log file was specified, add a handler for it
    if log_file:
        # Ensure the directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # Add handler for the file
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

def silence_verbose_loggers(verbose_mode: bool = False) -> None:
    """
    Silences excessively verbose loggers that are not relevant to the user.
    
    Args:
        verbose_mode: If True, does not silence loggers (debug/verbose mode)
    """
    if verbose_mode:
        return
    
    # List of modules to silence (ERROR = only severe errors, WARNING = only warnings and errors)
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
    
    # Apply logging levels
    for module_name, level in modules_to_silence.items():
        logging.getLogger(module_name).setLevel(level)

def get_timestamp_str():
    """
    Gets a string with the current timestamp in a format suitable for filenames.
    
    Returns:
        String with the current timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S") 