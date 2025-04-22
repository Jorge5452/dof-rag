"""
Funciones para preparar el entorno de pruebas.

Este módulo contiene funciones para configurar directorios, logging,
y otras preparaciones necesarias para ejecutar pruebas.
"""
import os
import sys
import logging
from pathlib import Path
import time
from datetime import datetime

from test.utils.constants import (
    RESULTS_DIR, DATABASE_RESULTS_DIR, CHUNKER_RESULTS_DIR,
    CLIENT_RESULTS_DIR, EMBEDDING_RESULTS_DIR, INTEGRATION_RESULTS_DIR,
    DOC_PROCESSOR_RESULTS_DIR, RAG_RESULTS_DIR, SESSION_MANAGER_RESULTS_DIR,
    VIEW_CHUNKS_RESULTS_DIR, ANALYSIS_DIR, DATETIME_FORMAT, PROJECT_ROOT
)

def ensure_dir_exists(dir_path):
    """
    Asegura que un directorio exista, creándolo si es necesario.
    
    Args:
        dir_path: Ruta del directorio a crear
        
    Returns:
        Path: Objeto Path con la ruta del directorio
    """
    dir_path = Path(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def get_test_result_path(test_type, custom_dir=None):
    """
    Obtiene la ruta para los resultados de un tipo de prueba específico.
    
    Args:
        test_type: Tipo de prueba (databases, chunkers, etc.)
        custom_dir: Directorio personalizado (opcional)
        
    Returns:
        Path: Ruta al directorio de resultados
    """
    if custom_dir:
        result_dir = Path(custom_dir)
    else:
        # Mapeo de tipos de prueba a directorios de resultados
        result_dirs = {
            "databases": DATABASE_RESULTS_DIR,
            "chunkers": CHUNKER_RESULTS_DIR,
            "clients": CLIENT_RESULTS_DIR,
            "embeddings": EMBEDDING_RESULTS_DIR,
            "integration": INTEGRATION_RESULTS_DIR,
            "doc_processor": DOC_PROCESSOR_RESULTS_DIR,
            "rag": RAG_RESULTS_DIR,
            "session_manager": SESSION_MANAGER_RESULTS_DIR,
            "view_chunks": VIEW_CHUNKS_RESULTS_DIR,
            "analysis": ANALYSIS_DIR
        }
        
        result_dir = result_dirs.get(test_type, RESULTS_DIR / test_type)
    
    return ensure_dir_exists(result_dir)

def setup_logging(log_file=None, console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Configura el sistema de logging para las pruebas.
    
    Args:
        log_file: Ruta al archivo de log (opcional)
        console_level: Nivel de logging para la consola
        file_level: Nivel de logging para el archivo
        
    Returns:
        Logger: Objeto logger configurado
    """
    # Configurar logger raíz
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Nivel base, los handlers pueden filtrar
    
    # Limpiar handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Crear formato consistente
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configurar handler de consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Configurar handler de archivo si se especifica
    if log_file:
        log_file = Path(log_file)
        ensure_dir_exists(log_file.parent)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def prepare_test_environment(test_type, output_dir=None, log_file=None):
    """
    Prepara el entorno para las pruebas, incluyendo directorios y logging.
    
    Args:
        test_type: Tipo de prueba (databases, chunkers, etc.)
        output_dir: Directorio donde guardar los resultados (opcional)
        log_file: Archivo de log específico (opcional)
        
    Returns:
        tuple: (directorio de resultados, logger configurado)
    """
    # Asegurar que el directorio de resultados exista
    result_dir = get_test_result_path(test_type, output_dir)
    
    # Determinar archivo de log
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = result_dir / f"{test_type}_test_{timestamp}.log"
    
    # Configurar logging
    logger = setup_logging(log_file)
    
    # Asegurar que el directorio raíz del proyecto esté en el path
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    # Establecer variables de entorno
    os.environ["TEST_RESULTS_DIR"] = str(result_dir)
    os.environ["TEST_TYPE"] = test_type
    
    logger.info(f"Entorno de pruebas preparado para: {test_type}")
    logger.info(f"Directorio de resultados: {result_dir}")
    
    return result_dir, logger 