"""AI Clients module for image description generation.

This module provides implementations of AI clients for different providers
that can generate descriptions of images extracted from DOF documents.

Available clients:
- OpenAIClient: OpenAI GPT models with support for custom base URLs
"""

from .openai import OpenAIClient

__all__ = [
    'OpenAIClient',
]

# Client factory function for easy instantiation
def create_client(provider='openai', **kwargs):
    """
    Factory function to create AI client instances.
    
    Args:
        provider (str): The AI provider (all providers use OpenAI client)
        **kwargs: Additional configuration parameters
        
    Returns:
        OpenAIClient: Configured client instance
        
    Note:
        All providers (openai, gemini, ollama) now use the OpenAI client
        with different base_url configurations.
    """
    from .openai import OpenAIClient
    return OpenAIClient(**kwargs)