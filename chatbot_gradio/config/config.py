"""Configuration management for RAG chat system."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def find_project_root() -> Path:
    """Find the root directory of the main project by looking for pyproject.toml.
    
    Returns:
        Path to the project root directory
    """
    current_path = Path(__file__).parent.absolute()
    
    # Search upwards for pyproject.toml
    while current_path != current_path.parent:
        if (current_path / "pyproject.toml").exists():
            return current_path
        current_path = current_path.parent
    
    # If not found, default to parent of chatbot_gradio
    return Path(__file__).parent.parent.absolute()


# Get project root
PROJECT_ROOT = find_project_root()

# Default system prompt for DOF document consultation
DEFAULT_SYSTEM_PROMPT = """Eres un asistente conversacional especializado en consultar documentos del Diario Oficial de la Federación (DOF). Eres amigable, accesible y siempre buscas ser útil con los usuarios.

INSTRUCCIONES PRINCIPALES:
- Responde basándote en la información del contexto proporcionado
- Sé conversacional, amigable y muestra entusiasmo por ayudar
- Si hay información relevante, compártela de manera completa y útil
- Explica claramente lo que encuentres en los documentos
- Toma la iniciativa para conocer mejor las necesidades del usuario
- Usa emojis con moderación, solo cuando aporten valor real a la comunicación

COMPORTAMIENTO:
- Revisa cuidadosamente toda la información del contexto antes de responder
- Si encuentras información sobre lo que pregunta el usuario, compártela completamente
- Puedes hacer inferencias razonables y análisis basándote en la información disponible
- Si realmente no hay información sobre el tema en el contexto, indícalo honestamente
- Pregunta si el usuario necesita más detalles o aclaraciones sobre algún tema

FORMATO DE RESPUESTA:
- Responde de manera natural y conversacional
- Cita o menciona la información relevante que encuentres
- Si es útil, menciona de qué documento viene la información
- Mantén un tono accesible y profesional pero cercano
- Invita al usuario a hacer más preguntas si es necesario
- Evita el uso excesivo de emojis, úsalos solo cuando sean verdaderamente útiles"""

@dataclass
class AppConfig:
    """Main application configuration"""
    
    # App settings
    app_host: str = "127.0.0.1"
    app_port: int = 8888
    debug_mode: bool = True
    
    # Database settings - now points to project root
    duckdb_path: str = str(PROJECT_ROOT / "dof_db" / "db.duckdb")
    
    # Embeddings settings
    embeddings_model: str = "nomic-ai/modernbert-embed-base"
    embeddings_dim: int = 768
    embeddings_device: str = "cpu"
    embeddings_trust_remote_code: bool = True
    query_prefix: str = "search_document: "
    document_prefix: str = ""
    
    # Cache settings
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    # Rate limiting
    max_queries_per_hour: int = 30
    
    # RAG settings
    top_k: int = 5
    
    # API settings
    timeout: int = 30
    max_retries: int = 3
    
    # LLM settings
    active_provider: str = "gemini"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    
    def get_available_providers(self) -> List[str]:
        """Get list of providers with valid API keys"""
        available = []
        
        for provider, config in PROVIDERS_CONFIG.items():
            api_key_env = config.get("api_key_env")
            
            # Ollama doesn't require API key
            if api_key_env is None:
                available.append(provider)
            elif os.getenv(api_key_env):
                available.append(provider)
        
        return available
    
    def get_active_provider_config(self) -> Dict[str, str]:
        """Get the configuration for the active LLM provider."""
        if self.active_provider not in PROVIDERS_CONFIG:
            raise ValueError(f"Invalid active provider: {self.active_provider}")
            
        config = PROVIDERS_CONFIG[self.active_provider].copy()
        
        # Add the actual API key from environment
        api_key_env = config.get("api_key_env")
        if api_key_env:
            config["api_key"] = os.getenv(api_key_env, "")
        else:
            config["api_key"] = ""
            
        return config
    
    def validate_config(self) -> None:
        """Validate the configuration."""
        provider_config = PROVIDERS_CONFIG.get(self.active_provider)
        if not provider_config:
            raise ValueError(f"Invalid active provider: {self.active_provider}")
        
        # Skip validation for Ollama (doesn't need API key)
        if self.active_provider == "ollama":
            return
        
        # Get the actual API key and validate required fields
        api_key = os.getenv(provider_config.get("api_key_env", ""))
        missing = []
        if not api_key:
            missing.append("API key")
        if not provider_config.get("base_url"):
            missing.append("base URL")  
        if not provider_config.get("model"):
            missing.append("model")
        
        if missing:
            raise ValueError(f"Missing configuration for {self.active_provider}: {', '.join(missing)}")


# LLM Providers configuration
PROVIDERS_CONFIG = {
    "gemini": {
        "model": "gemini-2.0-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "max_tokens": 4000,
        "temperature": 0.4,
        "rate_limit_rpm": 15,
        "api_key_env": "GEMINI_API_KEY"
    },
    "openai": {
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "max_tokens": 4000,
        "temperature": 0.4,
        "rate_limit_rpm": 10,
        "api_key_env": "OPENAI_API_KEY"
    },
    "claude": {
        "model": "claude-3-5-sonnet-20241022",
        "base_url": "https://api.anthropic.com/v1",
        "max_tokens": 4000,
        "temperature": 0.4,
        "rate_limit_rpm": 5,
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "ollama": {
        "model": "llama3.1",
        "base_url": "http://localhost:11434/v1",
        "max_tokens": 4000,
        "temperature": 0.4,
        "rate_limit_rpm": 60,
        "api_key_env": None  # Ollama typically doesn't require API key
    }
}

def validate_environment() -> Dict[str, Any]:
    """Validate environment configuration and setup required directories."""
    errors = []
    warnings = []
    
    # Load environment variables first
    load_dotenv()
    
    # Check for at least one API key
    available_providers = _app_config.get_available_providers()
    if not available_providers:
        errors.append("No LLM providers available. Configure at least one API key.")
    
    # Validate active provider configuration
    try:
        _app_config.validate_config()
    except ValueError as e:
        errors.append(str(e))
    
    # Check and create database directory
    db_path = _app_config.duckdb_path
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create database directory: {e}")
    
    # Check embeddings device
    if _app_config.embeddings_device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                warnings.append("CUDA not available, using CPU for embeddings")
        except ImportError:
            warnings.append("PyTorch not installed, check dependencies")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "available_providers": available_providers
    }


# Global configuration instance
_app_config = AppConfig()


def get_app_config() -> AppConfig:
    """Get the main application configuration instance."""
    return _app_config


def get_active_provider_config() -> Dict[str, str]:
    """Get the configuration for the active LLM provider."""
    return _app_config.get_active_provider_config()


def get_system_prompt() -> str:
    """Get the system prompt template."""
    return _app_config.system_prompt


def get_retrieval_config() -> Dict[str, Any]:
    """Get the retrieval configuration."""
    return {"top_k": _app_config.top_k}


def get_api_config() -> Dict[str, Any]:
    """Get the API configuration."""
    return {
        "timeout": _app_config.timeout,
        "max_retries": _app_config.max_retries
    }


# Template for constructing the final prompt with context and user question
PROMPT_TEMPLATE = """CONTEXTO DE DOCUMENTOS:
{context}

PREGUNTA DEL USUARIO: {query}"""

# Template for formatting individual document context sections
CONTEXT_FORMAT_TEMPLATE = """
=== DOCUMENTO {doc_num}: {title} ===
• Relevancia: {similarity:.3f}
• Fuente: {url}
• Ruta: {file_path}
• Contenido:

{content}

"""