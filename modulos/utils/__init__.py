"""
Módulo de utilidades para el sistema RAG.

Este paquete contiene utilidades comunes usadas por varios componentes del sistema,
como formateo de salidas, manejo de logs, y funciones auxiliares.
"""

from modulos.utils.formatting import *
from modulos.utils.logging_utils import *

__all__ = [
    # Formatting
    'print_header', 'print_separator', 'print_status', 
    'print_formatted_response', 'print_command_help', 'print_useful_commands',
    # Logging
    'setup_logging', 'silence_verbose_loggers'
] 