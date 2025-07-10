#!/usr/bin/env python3
"""
RAG Chat Application - Main Entry Point

This is the primary entry point for the RAG Chat system.
Run this file to start the chat interface with RAG capabilities.

Usage:
    python app.py
    
Or with UV:
    uv run python app.py

Environment Variables:
    See .env.example for configuration options
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

# Add both the project root and chatbot_gradio to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config.config import (
        get_active_provider_config,
        get_api_config,
        get_app_config,
        get_retrieval_config,
        get_system_prompt,
        PROJECT_ROOT,
        validate_environment
    )
    from core.database import connect_duckdb, close_connection
    from core.embeddings import initialize_embeddings, embedding_manager
    from core.llm_client import create_llm_client
    from core.rag_pipeline import RAGPipeline
    from interface.chat_ui import launch_ui
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    sys.exit(1)


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('chatbot_gradio.log', mode='a')
        ]
    )
    
    # Reduce verbosity of some modules
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)


def _initialize_components() -> Tuple[RAGPipeline, Optional[object]]:
    """Initialize all system components.
    
    Returns:
        Tuple of (RAGPipeline, database_connection)
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing system components...")
        
        # Get app configuration
        app_config = get_app_config()
        
        # Initialize embedding model FIRST (required by database initialization)
        logger.info("Initializing embedding model...")
        initialize_embeddings()
        
        # Initialize database connection
        logger.info("Connecting to database...")
        db_path = app_config.duckdb_path
        db_conn = connect_duckdb(db_path)
        logger.info(f"Database connected: {db_path}")
        
        # Initialize LLM client
        logger.info("Initializing LLM client...")
        provider_config = get_active_provider_config()
        api_config = get_api_config()
        
        llm_client = create_llm_client(
            provider_config,
            timeout=api_config["timeout"],
            max_retries=api_config["max_retries"]
        )
        
        # Test LLM connection
        if llm_client.test_connection():
            logger.info("LLM client initialized and tested successfully")
        else:
            logger.warning("LLM client initialized but connection test failed")
        
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")
        system_prompt = get_system_prompt()
        retrieval_config = get_retrieval_config()
        
        rag_pipeline = RAGPipeline(
            llm_client=llm_client,
            db_conn=db_conn,
            embedding_manager=embedding_manager,
            system_prompt=system_prompt,
            top_k=retrieval_config["top_k"]
        )
        
        logger.info("All components initialized successfully")
        return rag_pipeline, db_conn
        
    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        raise


def main() -> None:
    """Main application entry point."""
    logger = logging.getLogger(__name__)
    db_conn = None
    
    try:
        # Setup logging
        setup_logging()
        
        print("🚀 Starting RAG Chat System...")
        logger.info("Starting RAG chat application...")
        
        # Validate environment and configuration
        try:
            validation_result = validate_environment()
            if not validation_result["valid"]:
                print("❌ Environment validation failed:")
                for error in validation_result["errors"]:
                    print(f"   - {error}")
                return
            
            available_providers = validation_result["available_providers"]
            if available_providers:
                print(f"🤖 Available providers: {', '.join(available_providers)}")
            else:
                print("❌ No LLM providers available")
                return
                
        except Exception as e:
            print(f"⚠️  Environment validation failed: {e}")
            return
        
        # Initialize components
        rag_pipeline, db_conn = _initialize_components()
        
        # Check .env file
        env_file = PROJECT_ROOT / ".env"
        if not env_file.exists():
            print("⚠️  No .env file found. Using environment variables and defaults.")
        
        # Get configuration and start application
        config = get_app_config()
        print(f"🌐 Launching web interface at http://{config.app_host}:{config.app_port}")
        print("   Press Ctrl+C to stop the application")
        print()
        
        # Launch UI
        logger.info("Launching chat interface...")
        launch_ui(
            rag_pipeline=rag_pipeline,
            server_name=config.app_host,
            server_port=config.app_port,
            share=False
        )
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        logger.info("Application interrupted by user")
    except Exception as e:
        print(f"❌ Application failed: {e}")
        logger.error(f"Application failed to start: {e}")
    finally:
        # Cleanup resources
        if db_conn:
            close_connection(db_conn)


if __name__ == "__main__":
    main()
