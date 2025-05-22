#!/usr/bin/env python
"""
Script to run the RAG chatbot server.

This script starts a web server that provides a REST API to interact
with the RAG system through a chatbot with support for streaming responses.

Usage:
    python chatbot_server.py [--port PORT] [--host HOST] [--debug]
"""

import argparse
import logging
import os
import sys
from modulos.rag.api import run_api
from modulos.utils.logging_utils import setup_logging, silence_verbose_loggers
from config import Config

config = Config()

def main():
    """Main function that processes arguments and starts the server."""
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description="RAG chatbot server with real-time responses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Arguments for server configuration
    parser.add_argument("--port", type=int, default=5000, 
                        help="Port on which the server will listen")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="Host on which the server will listen")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug mode")
    parser.add_argument("--no-streaming", action="store_true",
                        help="Disable streaming (real-time responses)")
    
    # Process arguments
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    
    # Silence verbose loggers if not in debug mode
    if not args.debug:
        silence_verbose_loggers()
    
    # Force streaming mode in configuration
    ai_config = config.get_ai_client_config()
    
    # Override configuration to always use streaming unless --no-streaming is specified
    if not args.no_streaming:
        # Enable streaming if not specified otherwise
        if isinstance(ai_config, dict) and "parameters" in ai_config:
            ai_config["parameters"]["stream"] = True
            config.update_config(["ai_client", "parameters", "stream"], True)
            logging.info("Streaming mode enabled for chatbot")
    
    # Get project base directory path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(project_dir, "modulos", "rag", "static")
    
    # Verify necessary files exist
    if os.path.exists(static_dir):
        required_files = ["index.html", "app.js", "style.css"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(static_dir, f))]
        
        if missing_files:
            print(f"WARNING: Missing files in static directory: {', '.join(missing_files)}")
            for f in missing_files:
                print(f"  - Not found: {os.path.join(static_dir, f)}")
    else:
        print(f"WARNING: Static directory not found at {static_dir}")
        print("This may cause issues when serving the web interface.")
    
    # Check database availability
    try:
        from modulos.rag.app import RagApp
        databases = RagApp.list_available_databases()
        if databases:
            print(f"Available databases: {len(databases)}")
            for i, db in enumerate(databases[:5]):  # Show only first 5 to avoid console clutter
                print(f"  - {db['name']} ({db['type']}, {os.path.getsize(db['path']) / 1024 / 1024:.2f} MB)")
            if len(databases) > 5:
                print(f"  ... and {len(databases) - 5} more")
        else:
            print("WARNING: No available databases found.")
            print("You must first ingest documents using the ingest command.")
    except Exception as e:
        print(f"Error checking databases: {e}")
    
    # Print startup information
    print(f"Starting RAG chatbot server on {args.host}:{args.port}")
    print(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    print(f"Streaming mode: {'disabled' if args.no_streaming else 'enabled'}")
    print(f"Web interface available at: http://localhost:{args.port}/")
    print("Press Ctrl+C to stop the server")
    
    # Start the server
    try:
        run_api(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Error starting server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 