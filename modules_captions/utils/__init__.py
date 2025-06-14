"""Utilities module for image processing and error handling.

This module provides utility classes for file processing, error handling,
and other common operations needed by the caption extraction system.
"""

from .file_processor import FileProcessor
from .error_handler import ErrorHandler

__all__ = [
    'FileProcessor',
    'ErrorHandler',
]