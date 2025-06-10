"""
Utilities module for the RAG system.

This package contains common utilities used by various components of the system,
such as output formatting, log handling, and auxiliary functions.
"""

from modulos.utils.formatting import (
    print_header, print_separator, print_status, 
    print_formatted_response, print_command_help, print_useful_commands,
    C_TITLE, C_SUBTITLE, C_SUCCESS, C_ERROR, C_WARNING,
    C_HIGHLIGHT, C_COMMAND, C_PARAM, C_INFO, C_VALUE,
    C_PROMPT, C_RESET, C_SEPARATOR
)
from modulos.utils.logging_utils import (
    setup_logging, silence_verbose_loggers, get_timestamp_str
)
from modulos.utils.config_manager import (
    get_config, get_processing_config, get_embedding_config,
    get_chunking_config, get_database_config, get_ai_client_config,
    get_general_config, is_debug_enabled
)

__all__ = [
    # Formatting
    'print_header', 'print_separator', 'print_status', 
    'print_formatted_response', 'print_command_help', 'print_useful_commands',
    # Color constants
    'C_TITLE', 'C_SUBTITLE', 'C_SUCCESS', 'C_ERROR', 'C_WARNING',
    'C_HIGHLIGHT', 'C_COMMAND', 'C_PARAM', 'C_INFO', 'C_VALUE',
    'C_PROMPT', 'C_RESET', 'C_SEPARATOR',
    # Logging
    'setup_logging', 'silence_verbose_loggers', 'get_timestamp_str',
    # Configuration
    'get_config', 'get_processing_config', 'get_embedding_config',
    'get_chunking_config', 'get_database_config', 'get_ai_client_config',
    'get_general_config', 'is_debug_enabled'
]