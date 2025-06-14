"""Database module for image descriptions storage.

This module provides SQLite-based storage for image descriptions,
replacing the file-based approach with a more robust database solution.
"""

from .manager import DatabaseManager

__all__ = ['DatabaseManager']