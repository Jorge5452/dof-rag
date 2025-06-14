"""Enhanced Image Caption Extraction System

This module provides an improved version of the caption extraction system
with SQLite database storage, better error handling, and enhanced processing
capabilities. It replaces the file-based approach with a more robust
database-driven solution.

Main components:
- DatabaseManager: SQLite database operations
- OpenAIClient: AI client for image description generation with support for multiple providers
- FileProcessor: Batch image processing with checkpoints
- ErrorHandler: Comprehensive error handling and logging
- CaptionExtractor: Main orchestrator class

Usage:
    from modules_captions import CaptionExtractor
    
    config = {
        'provider': 'openai',
        'db_path': 'captions.db',
        'root_dir': '/path/to/images'
    }
    
    extractor = CaptionExtractor(config)
    results = extractor.extract_captions()
"""

__version__ = "2.0.0"
__author__ = "DOF-RAG Project"
__description__ = "Enhanced Image Caption Extraction System with SQLite Storage"

# Import main classes for easy access
from .db.manager import DatabaseManager
from .clients import create_client, OpenAIClient
from .utils.file_processor import FileProcessor
from .utils.error_handler import ErrorHandler

__all__ = [
    'DatabaseManager',
    'create_client',
    'OpenAIClient',
    'FileProcessor',
    'ErrorHandler',
]