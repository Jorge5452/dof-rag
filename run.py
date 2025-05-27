"""
Entry point for the RAG system.

This script handles command-line arguments and executes the main system functions,
either for document ingestion or for queries.

Usage:
    Ingestion: python run.py --ingest --files [directory]
    Query: python run.py --query "Your question here?"
    Ingest and export: python run.py --ingest --export-chunks --files [directory]
    List sessions: python run.py --list-sessions
    List databases: python run.py --list-dbs
    Export chunks: python run.py --export-chunks --files [directory]
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

# Importación global necesaria para evitar errores de referencia
from config import config

# Importar utilidades del sistema
from modulos.utils.formatting import (
    C_TITLE, C_SUBTITLE, C_SUCCESS, C_ERROR, C_WARNING, C_HIGHLIGHT, 
    C_COMMAND, C_PARAM, C_INFO, C_VALUE, C_PROMPT, C_RESET, C_SEPARATOR, Style,
    print_header, print_separator, print_status, print_formatted_response,
    print_command_help, print_useful_commands
)
from modulos.utils.logging_utils import setup_logging, silence_verbose_loggers

# Configurar logging
setup_logging(level=logging.INFO, log_file="./logs/rag_system.log")

# Silenciar módulos verbosos
silence_verbose_loggers()

# Obtener logger para este módulo
logger = logging.getLogger(__name__)

def main() -> int:
    """
    Main function that processes arguments and executes the corresponding logic.
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description="RAG system for document ingestion and queries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Group for operation mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--ingest", action="store_true", help="Document ingestion mode")
    mode_group.add_argument("--query", type=str, nargs='?', const='', help="Query for the RAG system. Without argument starts interactive mode.")
    mode_group.add_argument("--list-sessions", action="store_true", help="List available sessions and databases")
    mode_group.add_argument("--optimize-db", type=int, help="Optimize a specific database by index")
    mode_group.add_argument("--optimize-all", action="store_true", help="Optimize all databases")
    mode_group.add_argument("--db-stats", action="store_true", help="Show database statistics")
    mode_group.add_argument("--resource-status", action="store_true", help="Show current status of resource manager")
    
    # Export chunks option (not mutually exclusive)
    parser.add_argument("--export-chunks", action="store_true", help="Export chunks to TXT files for specified files or directories")
    
    # Arguments for ingestion mode
    parser.add_argument("--files", type=str, help="Directory or Markdown file to process")
    parser.add_argument("--session-name", type=str, help="Custom name for the session")
    parser.add_argument("--db-index", type=int, help="Index of existing database to use for ingestion (adds files to it)")
    
    # Arguments for query mode
    processing_config = config.get_processing_config()
    default_chunks = processing_config.get("max_chunks_to_retrieve", 5)
    
    parser.add_argument("--chunks", type=int, default=default_chunks, 
                      help=f"Number of chunks to retrieve for the query (default: {default_chunks})")
    parser.add_argument("--model", type=str, help="AI model to use for the query")
    parser.add_argument("--session", type=str, help="Specific session ID to use")
    parser.add_argument("--show-dbs", action="store_true", help="Show available databases before the query")
    
    # General arguments
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (verbose logs)")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug or config.get_general_config().get("debug", False):
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug(C_SUCCESS + "Debug mode enabled")
    
    try:
        # Execute corresponding mode
        result = 0
        
        if args.ingest:
            result = handle_ingest_mode(args)
        elif args.query is not None:
            return handle_query_mode(args)
        elif args.list_sessions:
            return list_sessions()
        elif args.optimize_db is not None:
            optimize_database(args.optimize_db)
            return 0
        elif args.optimize_all:
            optimize_all_databases()
            return 0
        elif args.db_stats:
            show_database_statistics()
            return 0
        elif args.resource_status:
            return show_resource_status()
            
        # Export chunks if requested (can be combined with other modes)
        if args.export_chunks and (args.ingest is False or result == 0):
            export_result = handle_export_chunks_mode(args)
            if result == 0:  # Solo actualizar si no hubo errores previos
                result = export_result
                
        return result
    except Exception as e:
        logger.error(f"{C_ERROR}Unexpected error: {e}")
        return 1

def handle_ingest_mode(args: argparse.Namespace) -> int:
    """
    Handles document ingestion mode.
    
    Args:
        args: Command line arguments
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Load modules only when needed for ingestion
    from main import process_documents
    
    if not args.files:
        logger.error(f"{C_ERROR}The --files argument is required for --ingest mode")
        return 1
        
    # Verify directory or file exists
    if not os.path.exists(args.files):
        logger.error(f"{C_ERROR}The directory or file {args.files} does not exist")
        return 1
        
    # Process documents
    logger.info(f"{C_HIGHLIGHT}Starting document ingestion from: {C_VALUE}{args.files}")
    
    # Pass db_index if provided, so documents are added to an existing database
    db_index = args.db_index
    if db_index is not None:
        logger.info(f"{C_HIGHLIGHT}Using existing database with index: {C_VALUE}{db_index}")
        
    process_documents(args.files, session_name=args.session_name, db_index=db_index)
    return 0

def handle_query_mode(args: argparse.Namespace) -> int:
    """
    Handles query mode.
    
    Args:
        args: Command line arguments
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Load modules only when needed for query
    from main import process_query
    
    # Configure logging level based on debug mode
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # Reduce logging verbosity in normal mode
        logging.getLogger().setLevel(logging.WARNING)
        silence_verbose_loggers()
    
    # Check if we're in interactive mode (empty query)
    interactive_mode = args.query == ''
    
    # In query mode, check if we need to show the databases
    if args.show_dbs or args.db_index is None:
        # Get sessions list
        sessions = get_session_list(show_output=False)
        
        # Show a simple summary (only the most recent ones)
        print("\n" + C_TITLE + " AVAILABLE SESSIONS/DATABASES " + C_RESET)
        print_separator()
        
        for i, session in enumerate(sessions[:5]):  # Limit to 5
            created_date = datetime.fromtimestamp(session.get('created_at', 0)).strftime('%Y-%m-%d')
            model = session.get('embedding_model', 'unknown')
            
            if i == 0:  # Highlight the most recent
                print(f"{C_SUCCESS}[{i}] {C_HIGHLIGHT}{session['id']}{C_RESET} - {created_date} - Model: {C_VALUE}{model}{C_RESET}")
            else:
                print(f"[{i}] {C_HIGHLIGHT}{session['id']}{C_RESET} - {created_date} - Model: {model}")
        
        print_separator()
        
        # If we only wanted to show the databases and no index was provided, ask
        if args.db_index is None:
            try:
                db_index = input(f"\n{C_PROMPT}Select database index (Enter to use the most recent): ")
                if db_index.strip():
                    args.db_index = int(db_index)
            except ValueError:
                print_status("warning", "Invalid input. Using the most recent database.")
    
    # Show start message
    if interactive_mode:
        # Interactive mode
        run_interactive_mode(args.chunks, args.model, args.session, args.db_index)
    else:
        # Single query mode - Simplified interface
        print("\n" + C_SUBTITLE + " QUERY: " + C_VALUE + f"{args.query}" + C_RESET)
        
        # Timer to measure response time
        start_time = time.time()
        
        # Temporarily reduce log level during query if not in debug mode
        original_log_level = logging.getLogger().level
        if not args.debug:
            logging.getLogger().setLevel(logging.WARNING)
            
        try:
            # Process the query
            response = process_query(
                args.query, 
                n_chunks=args.chunks, 
                model=args.model, 
                session_id=args.session,
                db_index=args.db_index
            )
        except Exception as e:
            # In case of error, show clear message
            response = f"Error processing query: {str(e)}"
        finally:
            # Restore original log level
            logging.getLogger().setLevel(original_log_level)
        
        # Show response with better formatting
        print("\n" + C_TITLE + " RESPONSE " + C_RESET)
        print_separator()
        
        # Extract only the response section if it contains format separators
        if "=======================  RESPUESTA  =======================" in response:
            parts = response.split("=======================  RESPUESTA  =======================")
            if len(parts) > 1:
                # Extract the response part (without header)
                response_text = parts[1].split("=======================  CONTEXTO  =======================")[0].strip()
                context_text = response.split("=======================  CONTEXTO  =======================")
                
                # Print only the response first
                print(response_text)
                
                # Show response time after the main response
                elapsed_time = time.time() - start_time
                print_separator()
                print_status("info", f"Response time: {elapsed_time:.2f} seconds")
                
                # Print context if it exists - Now we always show the context
                if len(context_text) > 1:
                    print("\n" + C_TITLE + " CONTEXT USED " + C_RESET)
                    print_separator()
                    print(context_text[1].strip())
            else:
                print(response)
                
                # Show response time
                elapsed_time = time.time() - start_time
                print_separator()
                print_status("info", f"Response time: {elapsed_time:.2f} seconds")
        else:
            print(response)
            
            # Show response time
            elapsed_time = time.time() - start_time
            print_separator()
            print_status("info", f"Response time: {elapsed_time:.2f} seconds")
    
    return 0

def list_sessions() -> int:
    """
    Lists available sessions (which are the same as databases).
    
    Returns:
        int: Result code (0=success, 1=error).
    """
    # Import session manager
    from modulos.session_manager.session_manager import SessionManager
    
    # Use session manager to list sessions
    session_manager = SessionManager()
    sessions = session_manager.list_sessions()
    
    # Show title
    print("\n" + C_TITLE + " AVAILABLE SESSIONS/DATABASES " + C_RESET)
    print_separator()
    
    if not sessions:
        print_status("warning", "No available sessions")
        print()
        return 0
        
    # Sessions are already sorted by most recent activity in list_sessions
    
    for i, session in enumerate(sessions):
        # Simplify times for display
        created_readable = datetime.fromtimestamp(session.get('created_at', 0)).strftime('%Y-%m-%d %H:%M')
        last_modified_readable = datetime.fromtimestamp(session.get('last_modified', 0)).strftime('%Y-%m-%d %H:%M')
        
        # Get database info
        db_type = session.get('db_type', 'unknown')
        db_path = session.get('db_path', 'unknown')
        embedding_model = session.get('embedding_model', 'unknown')
        chunking_method = session.get('chunking_method', 'unknown')
        
        # Get file info
        files = session.get('files', [])
        file_count = len(files)
        
        # Get total chunks - puede estar en el nivel raíz o en stats
        total_chunks = session.get('total_chunks', 'unknown')
        if total_chunks == 'unknown':
            stats = session.get('stats', {})
            total_chunks = stats.get('total_chunks', 'unknown')
        
        # Format file size if available
        size_str = ""
        if os.path.exists(db_path):
            size_bytes = os.path.getsize(db_path)
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024 * 1024 * 1024:
                size_str = f"{size_bytes/(1024*1024):.1f} MB"
            else:
                size_str = f"{size_bytes/(1024*1024*1024):.2f} GB"
        
        # Display session/db information
        if i == 0:  # Highlight the most recent session
            print(f"{C_SUCCESS}[{i}] {C_HIGHLIGHT}{session['id']}{C_RESET}")
        else:
            print(f"[{i}] {C_HIGHLIGHT}{session['id']}{C_RESET}")
            
        print(f"{C_INFO}   Created: {created_readable} - Last modified: {last_modified_readable}")
        print(f"{C_INFO}   Type: {C_VALUE}{db_type}{C_RESET} - Model: {C_VALUE}{embedding_model}{C_RESET} - Chunking: {C_VALUE}{chunking_method}{C_RESET}")
        print(f"{C_INFO}   Files: {C_VALUE}{file_count}{C_RESET} - Chunks: {C_VALUE}{total_chunks}{C_RESET} - Size: {C_VALUE}{size_str}{C_RESET}")
        
        # Show file info (limit to first 3)
        if files:
            print(f"{C_INFO}   Processed files:")
            for j, file_item in enumerate(files[:3]):
                if isinstance(file_item, str):
                    file_name = os.path.basename(file_item)
                    print(f"{C_INFO}     {j+1}. {file_name}")
                elif isinstance(file_item, dict):
                    file_name = os.path.basename(file_item.get('path', 'unknown'))
                    chunks = file_item.get('chunks', 'unknown')
                    print(f"{C_INFO}     {j+1}. {file_name} - Chunks: {chunks}")
                    
            if len(files) > 3:
                print(f"{C_INFO}     ...and {len(files) - 3} more")
        
        print()
    
    print(Style.BRIGHT + "=" * 80 + "\n")
    print_status("info", "Use these indices with --db-index for queries or to add documents to existing databases")
    print_status("info", "Example: python run.py --ingest --files new_docs/ --db-index 0")
    print_status("info", "Example: python run.py --query \"My question\" --db-index 2")
    print()
    
    return 0

def get_session_list(show_output: bool = True) -> List[Dict[str, Any]]:
    """
    Gets the list of available sessions/databases.
    
    Args:
        show_output: Whether to display the list
        
    Returns:
        List of session/database data
    """
    # Import session manager
    from modulos.session_manager.session_manager import SessionManager
    
    # Use session manager to list sessions
    session_manager = SessionManager()
    sessions = session_manager.list_sessions()
    
    # Display if requested
    if show_output:
        list_sessions()
        
    return sessions

def run_interactive_mode(n_chunks: int = 5, model: Optional[str] = None, 
                        session_id: Optional[str] = None, db_index: Optional[int] = None) -> None:
    """
    Runs RAG system in interactive mode, allowing consecutive queries.
    
    Args:
        n_chunks: Number of chunks to retrieve for each query
        model: AI model to use (optional)
        session_id: Specific session ID to use (optional)
        db_index: Index of database to use (optional)
    """
    # Import only when needed for interactive mode
    from main import process_query
    from colorama import Style  # Ensure Style is available
    
    # Configure logging level for interactive mode
    logging.getLogger().setLevel(logging.WARNING)
    silence_verbose_loggers()
    
    print("\n" + C_TITLE + " INTERACTIVE MODE " + C_RESET)
    print_separator()
    
    # Show simplified instructions
    print_status("info", "Type your questions and press Enter to get answers.")
    print_status("info", f"To exit: {C_COMMAND}exit{C_RESET}, {C_COMMAND}quit{C_RESET} or {C_COMMAND}q{C_RESET}")
    print_status("info", f"To view databases: {C_COMMAND}dbs{C_RESET}")
    print_status("info", f"To change database: {C_COMMAND}change <n>{C_RESET}")
    print_status("info", f"For help: {C_COMMAND}help{C_RESET} or {C_COMMAND}?{C_RESET}")
    print_separator()
    
    # Variables to maintain state
    current_db_index = db_index
    current_session_id = session_id
    current_model = model
    history = []
    
    # Main loop for interactive mode
    while True:
        try:
            # Get query from user
            query = input(f"\n{C_PROMPT}> ")
            
            # Check special commands
            if query.lower() in ["exit", "quit", "q"]:
                print_status("success", "Session ended.")
                break
                
            elif query.lower() == "dbs":
                # Show available databases
                sorted_dbs = show_available_databases()
                continue
                
            elif query.lower().startswith("change "):
                # Change active database
                try:
                    new_index = int(query.split()[1])
                    current_db_index = new_index
                    print_status("success", f"Database changed to index {C_VALUE}{new_index}")
                except (IndexError, ValueError):
                    print_status("error", "Invalid format. Usage: change <number>")
                continue
                
            elif query.lower() in ["help", "?"]:
                # Show help commands
                print("\n" + C_TITLE + " AVAILABLE COMMANDS " + C_RESET)
                print_separator()
                print(f"{C_INFO}• {C_COMMAND}exit{C_RESET}, {C_COMMAND}quit{C_RESET}, {C_COMMAND}q{C_RESET} - Exit interactive mode")
                print(f"{C_INFO}• {C_COMMAND}dbs{C_RESET} - Show available databases")
                print(f"{C_INFO}• {C_COMMAND}change <n>{C_RESET} - Change to database with index <n>")
                print(f"{C_INFO}• {C_COMMAND}help{C_RESET}, {C_COMMAND}?{C_RESET} - Show this help")
                print_separator()
                continue
                
            elif not query.strip():
                # Ignore empty queries
                continue
            
            # Process normal query - Reduce verbosity
            # Don't show processing message for cleaner experience
            
            # Measure response time
            start_time = time.time()
            
            # Process query with error handling
            try:
                response = process_query(
                    query, 
                    n_chunks=n_chunks, 
                    model=current_model,
                    session_id=current_session_id,
                    db_index=current_db_index
                )
                
                # Add to history
                history.append((query, response))
                
                # Show response with improved formatting
                # Extract only response section if it contains format separators
                if "=======================  RESPUESTA  =======================" in response:
                    parts = response.split("=======================  RESPUESTA  =======================")
                    if len(parts) > 1:
                        # Extract response part (without header)
                        response_text = parts[1].split("=======================  CONTEXTO  =======================")[0].strip()
                        context_text = response.split("=======================  CONTEXTO  =======================")
                        
                        # Print only the response
                        print("\n" + C_TITLE + " RESPONSE " + C_RESET)
                        print_separator()
                        print(response_text)
                        
                        # Show response time discreetly
                        elapsed_time = time.time() - start_time
                        print_separator()
                        print_status("info", f"Time: {elapsed_time:.2f} seconds")
                        
                        # Print context if it exists
                        if len(context_text) > 1:
                            print("\n" + C_TITLE + " CONTEXT USED " + C_RESET)
                            print_separator()
                            print(context_text[1].strip())
                    else:
                        # If we can't separate the response, show everything
                        print("\n" + C_TITLE + " RESPONSE " + C_RESET)
                        print_separator()
                        print(response)
                        
                        # Show response time
                        elapsed_time = time.time() - start_time
                        print_separator()
                        print_status("info", f"Time: {elapsed_time:.2f} seconds")
                else:
                    # If not formatted, show complete response
                    print("\n" + C_TITLE + " RESPONSE " + C_RESET)
                    print_separator()
                    print(response)
                    
                    # Show response time
                    elapsed_time = time.time() - start_time
                    print_separator()
                    print_status("info", f"Time: {elapsed_time:.2f} seconds")
                
            except Exception as e:
                print_status("error", f"Error processing query: {str(e)}")
            
        except KeyboardInterrupt:
            print_status("info", "\nSession interrupted by user.")
            break
            
        except Exception as e:
            print_status("error", f"Error: {str(e)}")
            
    print_status("info", "Thank you for using the RAG system!")
    print()

def show_available_databases(show_output: bool = True) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Lists available databases.
    
    Args:
        show_output: Whether to display output on screen
        
    Returns:
        Sorted list of databases (name, metadata)
    """
    # Import session manager
    from modulos.session_manager.session_manager import SessionManager
    from colorama import Style  # Ensure Style is available
    import os
    
    # Use session manager to list databases
    session_manager = SessionManager()
    databases = session_manager.list_available_databases()
    
    # Convert dictionary to sorted list by last use (most recent first)
    sorted_dbs = []
    for name, metadata in databases.items():
        sorted_dbs.append((name, metadata))
    
    sorted_dbs.sort(key=lambda x: x[1].get('last_used', 0), reverse=True)
    
    # If requested to display output
    if show_output:
        # Show title
        print("\n" + C_TITLE + " AVAILABLE DATABASES " + C_RESET)
        print_separator()
        
        if not sorted_dbs:
            print_status("warning", "No databases available")
            return sorted_dbs
            
        # Show summarized information for each database
        for i, (name, db) in enumerate(sorted_dbs):
            # Format dates for human readability
            created_date = datetime.fromtimestamp(db.get('created_at', 0)).strftime('%Y-%m-%d')
            last_used_date = datetime.fromtimestamp(db.get('last_used', 0)).strftime('%Y-%m-%d')
            
            # Get basic information
            model = db.get('embedding_model', 'unknown')
            chunking = db.get('chunking_method', 'unknown')
            db_type = db.get('db_type', 'unknown')
            db_path = db.get('db_path', 'unknown')
            
            # Get file size if exists
            size_str = ""
            if db_path and os.path.exists(db_path):
                size_bytes = os.path.getsize(db_path)
                # Convert to KB, MB, GB as appropriate
                if size_bytes < 1024:
                    size_str = f"{size_bytes} bytes"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes/1024:.1f} KB"
                elif size_bytes < 1024 * 1024 * 1024:
                    size_str = f"{size_bytes/(1024*1024):.1f} MB"
                else:
                    size_str = f"{size_bytes/(1024*1024*1024):.2f} GB"
                size_str = f" - Size: {size_str}"
            else:
                size_str = ""
            
            # Get associated session information
            session_info = ""
            session_id = db.get('session_id')
            if session_id:
                session_info = f" - Session: {session_id}"
            
            # Get custom name if exists
            custom_name = db.get('custom_name', '')
            custom_name_info = f" - Name: {custom_name}" if custom_name else ""
            
            # Show information with improved formatting
            print(f"{C_HIGHLIGHT}{i}. {name}{custom_name_info}")
            print(f"{C_INFO}   Model: {C_VALUE}{model}{C_RESET} - Chunking: {C_VALUE}{chunking}{C_RESET} - Type: {C_VALUE}{db_type}{C_RESET}")
            print(f"{C_INFO}   Created: {C_VALUE}{created_date}{C_RESET} - Last use: {C_VALUE}{last_used_date}{C_RESET}{size_str}{session_info}")
            
            # Show file information if available
            files = db.get('files', [])
            if files:
                print(f"{C_INFO}   Files: {C_VALUE}{len(files)}{C_RESET}")
                # Show up to 2 example files
                for j, file_path in enumerate(files[:2]):
                    file_name = os.path.basename(file_path)
                    print(f"{C_INFO}     - {file_name}")
                if len(files) > 2:
                    print(f"{C_INFO}     - ...and {len(files) - 2} more")
            
            print()  # Separation between databases
        
        print_separator()
    
    return sorted_dbs

def optimize_database(db_index: int) -> None:
    """
    Optimizes a specific database.
    
    Args:
        db_index: Index of the database to optimize
    """
    # Import only when needed
    from modulos.session_manager.session_manager import SessionManager
    
    try:
        # Get list of databases
        sorted_dbs = show_available_databases(show_output=False)
        
        if not sorted_dbs or db_index >= len(sorted_dbs):
            print(f"{C_ERROR}Invalid database index: {db_index}")
            return
        
        # Get information about the database
        name, metadata = sorted_dbs[db_index]
        db_path = metadata.get('db_path')
        
        if not db_path or not os.path.exists(db_path):
            print(f"{C_ERROR}Could not locate the database file at: {db_path}")
            return
        
        print(f"\n{C_INFO}Optimizing database: {C_VALUE}{name}")
        
        # Get database instance
        session_manager = SessionManager()
        # Get database by index without session_id parameter
        db, metadata = session_manager.get_database_by_index(db_index)
        
        # Measure optimization time
        start_time = time.time()
        
        # Execute optimization
        success = db.optimize_database()
        
        # Calculate time
        elapsed_time = time.time() - start_time
        
        if success:
            print(f"{C_SUCCESS}✓ Optimization completed successfully in {elapsed_time:.2f} seconds")
        else:
            print(f"{C_ERROR}✗ Error during optimization")
        
    except Exception as e:
        print(f"{C_ERROR}Error: {str(e)}")

def optimize_all_databases() -> None:
    """
    Optimizes all available databases.
    """
    # Import only when needed
    from modulos.databases.FactoryDatabase import DatabaseFactory
    
    print_header("OPTIMIZATION OF ALL DATABASES")
    
    # Get list of databases
    sorted_dbs = show_available_databases(show_output=False)
    
    if not sorted_dbs:
        print(f"{C_WARNING}No databases available for optimization.")
        return
    
    # Extract database names for optimization
    db_names = [name for name, _ in sorted_dbs]
    
    print(f"{C_INFO}Optimizing {len(db_names)} databases...")
    
    # Start optimization
    start_time = time.time()
    results = DatabaseFactory.optimize_all_databases(db_names)  # Pass database names
    elapsed_time = time.time() - start_time
    
    # Show results
    success_count = sum(1 for success in results.values() if success)
    print(f"\n{C_SUBTITLE}OPTIMIZATION RESULTS:")
    print(f"{C_SUCCESS}✓ Successfully optimized databases: {success_count}")
    print(f"{C_ERROR}✗ Databases with errors: {len(results) - success_count}")
    print(f"{C_INFO}Total time: {elapsed_time:.2f} seconds")

def show_database_statistics() -> None:
    """
    Shows detailed statistics of all databases.
    """
    # Import only when needed
    from modulos.databases.FactoryDatabase import DatabaseFactory
    
    print_header("DATABASE STATISTICS")
    
    # Get statistics
    stats = DatabaseFactory.get_db_statistics()
    
    # Show global statistics
    print(f"{C_SUBTITLE}GLOBAL STATISTICS:")
    print(f"{C_INFO}• Total databases: {C_VALUE}{stats['total_databases']}")
    print(f"{C_INFO}• Total documents: {C_VALUE}{stats['total_documents']}")
    print(f"{C_INFO}• Total chunks: {C_VALUE}{stats['total_chunks']}")
    
    # Show database statistics
    if stats["databases"]:
        print(f"\n{C_SUBTITLE}DATABASE STATISTICS:")
        
        for name, db_stats in stats["databases"].items():
            if "error" in db_stats:
                print(f"\n{C_HIGHLIGHT}{name}: {C_ERROR}Error getting statistics: {db_stats['error']}")
                continue
                
            print(f"\n{C_HIGHLIGHT}{name}:")
            print(f"{C_INFO}• Documents: {C_VALUE}{db_stats.get('total_documents', 'N/A')}")
            print(f"{C_INFO}• Chunks: {C_VALUE}{db_stats.get('total_chunks', 'N/A')}")
            
            # Latest document information
            if "latest_document" in db_stats:
                doc = db_stats["latest_document"]
                print(f"{C_INFO}• Latest document: {C_VALUE}{doc.get('title', 'Untitled')} (ID: {doc.get('id', 'N/A')})")
            
            # Database size
            if "db_size_mb" in db_stats:
                print(f"{C_INFO}• Size: {C_VALUE}{db_stats['db_size_mb']:.2f} MB")
            
            # Creation date
            if "db_created" in db_stats and db_stats["db_created"] != "unknown":
                try:
                    created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(db_stats["db_created"]))
                    print(f"{C_INFO}• Created: {C_VALUE}{created_time}")
                except (TypeError, ValueError):
                    print(f"{C_INFO}• Created: {C_VALUE}{db_stats['db_created']}")
    
    print(Style.BRIGHT + "=" * 80)

def query_database(db_index: int, query: str, session_id: str = None) -> None:
    """
    Executes a query on a specific database.
    
    Args:
        db_index: Index of the database to query
        query: Query to execute
        session_id: Session ID (optional)
    """
    from modulos.rag.main import process_query
    
    try:
        # If we have a session ID, use it directly
        if session_id:
            print(f"{C_INFO}Querying in session: {C_VALUE}{session_id}")
        
        # Process the query
        response = process_query(
            query=query, 
            db_index=db_index,
            session_id=session_id,
            stream=True
        )
        
        # The response has already been shown to the user through streaming
        
    except Exception as e:
        print(f"{C_ERROR}Error querying: {str(e)}")

def handle_export_chunks_mode(args: argparse.Namespace) -> int:
    """
    Handles exporting chunks to TXT files.
    
    Args:
        args: Command line arguments
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    if not args.files:
        logger.error(f"{C_ERROR}The --files argument is required for --export-chunks mode")
        return 1
        
    # Verify that the directory or file exists
    if not os.path.exists(args.files):
        logger.error(f"{C_ERROR}The directory or file {args.files} does not exist")
        return 1
    
    try:
        # Import export module
        from modulos.view_chunks.chunk_exporter import export_chunks_for_files
        
        # Get most recent database or specified by index
        from modulos.session_manager.session_manager import SessionManager
        session_manager = SessionManager()
        
        if args.db_index is not None:
            db, session = session_manager.get_database_by_index(args.db_index)
        else:
            # Use most recent database
            db, session = session_manager.get_database_by_index(0)
        
        if not db:
            logger.error(f"{C_ERROR}Could not get a database connection")
            return 1
        
        print(f"\n{C_TITLE} EXPORTING CHUNKS {C_RESET}")
        print_separator()
        
        # Show information about the database being used
        print(f"{C_INFO}Database: {C_VALUE}{session.get('id', 'unknown')}")
        print(f"{C_INFO}Model: {C_VALUE}{session.get('embedding_model', 'unknown')}")
        print(f"{C_INFO}Chunking method: {C_VALUE}{session.get('chunking_method', 'unknown')}")
        print_separator()
        
        # Export chunks
        start_time = time.time()
        results = export_chunks_for_files(args.files, db)
        elapsed_time = time.time() - start_time
        
        # Show results
        successful = sum(1 for result in results.values() if result)
        failed = sum(1 for result in results.values() if not result)
        
        print_separator()
        print(f"{C_SUCCESS}Export completed in {elapsed_time:.2f} seconds")
        print(f"{C_SUCCESS}Successfully processed files: {successful}")
        if failed > 0:
            print(f"{C_ERROR}Files with errors: {failed}")
        print_separator()
        
        # Release resources
        db.close()
        import gc
        gc.collect()
        
        return 0
        
    except Exception as e:
        logger.error(f"{C_ERROR}Error exporting chunks: {e}")
        return 1

def show_resource_status() -> int:
    """
    Shows the current status of the ResourceManager and its metrics.
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        from modulos.resource_management.resource_manager import ResourceManager
        from modulos.utils.formatting import print_header, print_separator, C_HIGHLIGHT, C_VALUE, C_INFO, C_ERROR, C_RESET, C_SUCCESS
        import pprint

        print_header("RESOURCE MANAGER STATUS")

        # Get instance (must be initialized previously in normal flow)
        resource_manager = ResourceManager()

        # Force update of metrics to get latest values
        resource_manager.update_metrics()
        
        metrics = resource_manager.metrics
        static_info = resource_manager.get_system_static_info()

        print(f"{C_HIGHLIGHT}Static System Information:{C_RESET}")
        if "error" in static_info:
            print(f"{C_ERROR}  Error getting static information: {static_info['error']}{C_RESET}")
        else:
            print(f"{C_INFO}  OS: {C_VALUE}{static_info.get('os_platform', 'N/A')} {static_info.get('os_release', '')}{C_RESET}")
            print(f"{C_INFO}  Architecture: {C_VALUE}{static_info.get('architecture', 'N/A')}{C_RESET}")
            print(f"{C_INFO}  Python: {C_VALUE}{static_info.get('python_version', 'N/A')}{C_RESET}")
            print(f"{C_INFO}  CPU Cores (L/F): {C_VALUE}{static_info.get('cpu_cores_logical', 'N/A')} / {static_info.get('cpu_cores_physical', 'N/A')}{C_RESET}")
            print(f"{C_INFO}  Total RAM: {C_VALUE}{static_info.get('total_ram_gb', 'N/A')} GB{C_RESET}")
        print_separator()

        print(f"{C_HIGHLIGHT}Dynamic Metrics:{C_RESET}")
        print(f"{C_INFO}  Update: {C_VALUE}{datetime.fromtimestamp(metrics.get('last_metrics_update_ts', 0)).strftime('%Y-%m-%d %H:%M:%S')}{C_RESET}")
        print(f"{C_INFO}  Active Monitoring: {C_VALUE}{metrics.get('monitoring_thread_active', 'N/A')}{C_RESET}")
        print_separator()
        print(f"{C_INFO}  System Memory Usage:")
        print(f"{C_INFO}    Total: {C_VALUE}{metrics.get('system_memory_total_gb', 'N/A')} GB{C_RESET}")
        print(f"{C_INFO}    Available: {C_VALUE}{metrics.get('system_memory_available_gb', 'N/A')} GB{C_RESET}")
        print(f"{C_INFO}    Used: {C_VALUE}{metrics.get('system_memory_used_gb', 'N/A')} GB ({metrics.get('system_memory_percent', 'N/A')} %){C_RESET}")
        print_separator()
        print(f"{C_INFO}  RAG Process Memory Usage:")
        print(f"{C_INFO}    RSS: {C_VALUE}{metrics.get('process_memory_rss_mb', 'N/A')} MB{C_RESET}")
        print(f"{C_INFO}    VMS: {C_VALUE}{metrics.get('process_memory_vms_mb', 'N/A')} MB{C_RESET}")
        print(f"{C_INFO}    Percentage: {C_VALUE}{metrics.get('process_memory_percent', 'N/A')} %{C_RESET}")
        print_separator()
        print(f"{C_INFO}  CPU Usage:")
        print(f"{C_INFO}    System: {C_VALUE}{metrics.get('cpu_percent_system', 'N/A')} %{C_RESET}")
        print(f"{C_INFO}    RAG Process: {C_VALUE}{metrics.get('cpu_percent_process', 'N/A')} %{C_RESET}")
        print_separator()
        print(f"{C_HIGHLIGHT}RAG Components:{C_RESET}")
        print(f"{C_INFO}  Active Sessions: {C_VALUE}{metrics.get('active_sessions_rag', 'N/A')}{C_RESET}")
        print(f"{C_INFO}  Active Embedding Models: {C_VALUE}{metrics.get('active_embedding_models', 'N/A')}{C_RESET}")
        
        # Optional: Show ResourceManager loaded configuration
        print_separator()
        print(f"{C_HIGHLIGHT}Loaded Resource Management Configuration:{C_RESET}")
        print(f"{C_INFO}  Monitoring Interval: {C_VALUE}{getattr(resource_manager, 'monitoring_interval_sec', 'N/A')}s{C_RESET}")
        print(f"{C_INFO}  Aggressive Cleanup Threshold: {C_VALUE}{getattr(resource_manager, 'aggressive_cleanup_threshold_mem_pct', 'N/A')}%{C_RESET}")
        print(f"{C_INFO}  Warning Cleanup Threshold: {C_VALUE}{getattr(resource_manager, 'warning_cleanup_threshold_mem_pct', 'N/A')}%{C_RESET}")
        print(f"{C_INFO}  Warning CPU Threshold: {C_VALUE}{getattr(resource_manager, 'warning_threshold_cpu_pct', 'N/A')}%{C_RESET}")
        print(f"{C_INFO}  Monitoring Enabled: {C_VALUE}{getattr(resource_manager, 'monitoring_enabled', 'N/A')}{C_RESET}")
        print(f"{C_INFO}  CPU Workers: {C_VALUE}{getattr(resource_manager, 'default_cpu_workers', 'N/A')}{C_RESET}")
        print(f"{C_INFO}  IO Workers: {C_VALUE}{getattr(resource_manager, 'default_io_workers', 'N/A')}{C_RESET}")
        print(f"{C_INFO}  Max Total Workers: {C_VALUE}{getattr(resource_manager, 'max_total_workers', 'N/A')}{C_RESET}")

        print_separator()
        return 0

    except Exception as e:
        print(f"{C_ERROR}Error getting resource manager status: {e}{C_RESET}")
        logger.error("Detailed error in show_resource_status", exc_info=True)
        return 1

def print_separator():
    """Prints a separator line."""
    print(Style.BRIGHT + "-" * 80)
    
def print_header(title: str):
    """
    Prints a formatted header with title.
    
    Args:
        title: Title text to display
    """
    print("\n" + C_TITLE + f" {title} " + C_RESET)
    print_separator()
    
def print_status(status_type: str, message: str):
    """
    Prints a formatted status message.
    
    Args:
        status_type: Type of status ('success', 'error', 'warning', 'info')
        message: Message to display
    """
    prefix = ""
    color = C_INFO
    
    if status_type == "success":
        prefix = "✓ "
        color = C_SUCCESS
    elif status_type == "error":
        prefix = "✗ "
        color = C_ERROR
    elif status_type == "warning":
        prefix = "! "
        color = C_WARNING
    elif status_type == "info":
        prefix = "• "
        color = C_INFO
        
    print(f"{color}{prefix}{message}{C_RESET}")
    
def print_help():
    """Prints usage examples and tips."""
    print(C_TITLE + "\n RAG SYSTEM - HELP " + C_RESET)
    print_separator()
    
    print(C_SUBTITLE + "BASIC USAGE:" + C_RESET)
    print(f"  {C_VALUE}Ingest documents (new session):{C_RESET}")
    print(f"    {C_COMMAND}python run.py --ingest --files documents/{C_RESET}")
    print()
    print(f"  {C_VALUE}Add documents to existing database:{C_RESET}")
    print(f"    {C_COMMAND}python run.py --ingest --files new_docs/ --db-index 0{C_RESET}")
    print()
    print(f"  {C_VALUE}Query the system:{C_RESET}")
    print(f"    {C_COMMAND}python run.py --query \"What is the main topic of this document?\"{C_RESET}")
    print()
    print(f"  {C_VALUE}Start interactive mode:{C_RESET}")
    print(f"    {C_COMMAND}python run.py --query{C_RESET}")
    print()
    print(f"  {C_VALUE}List available sessions/databases:{C_RESET}")
    print(f"    {C_COMMAND}python run.py --list-sessions{C_RESET}")
    print()
    print(f"  {C_VALUE}Show database statistics:{C_RESET}")
    print(f"    {C_COMMAND}python run.py --db-stats{C_RESET}")
    
    print_separator()
    print(C_SUBTITLE + "ADDITIONAL OPTIONS:" + C_RESET)
    print(f"  {C_VALUE}Use specific database:{C_RESET}")
    print(f"    {C_COMMAND}python run.py --query \"My question\" --db-index 2{C_RESET}")
    print()
    print(f"  {C_VALUE}Export chunks to text files:{C_RESET}")
    print(f"    {C_COMMAND}python run.py --export-chunks --files documents/{C_RESET}")
    print()
    print(f"  {C_VALUE}Ingest documents and export chunks in one step:{C_RESET}")
    print(f"    {C_COMMAND}python run.py --ingest --export-chunks --files documents/{C_RESET}")
    print()
    print(f"  {C_VALUE}Show resource manager status:{C_RESET}")
    print(f"    {C_COMMAND}python run.py --resource-status{C_RESET}")
    print()
    print(f"  {C_VALUE}Debug mode (more verbose):{C_RESET}")
    print(f"    {C_COMMAND}python run.py --query \"My question\" --debug{C_RESET}")
    print()
    print(f"  {C_VALUE}Database optimization:{C_RESET}")
    print(f"    {C_COMMAND}python run.py --optimize-db 0{C_RESET}")
    print(f"    {C_COMMAND}python run.py --optimize-all{C_RESET}")
    
    print_separator()

if __name__ == "__main__":
    sys.exit(main())
