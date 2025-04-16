"""
Módulo principal del sistema RAG.

Este módulo proporciona las clases y funcionalidades principales
del sistema de Recuperación Aumentada de Generación (RAG).
"""

from modulos.rag.app import RagApp
from modulos.rag.api import run_api

__all__ = ["RagApp", "run_api"] 