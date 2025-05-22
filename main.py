"""
Main logic of the RAG system.

This module contains functions for:
1. Processing documents (ingestion)
2. Processing queries (search)

It orchestrates all components of the RAG system to work together.
"""
import logging
import time
import os
import sys
from pathlib import Path
from typing import Optional, Any
import concurrent.futures
import modulos.session_manager.session_manager
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Import colorama for terminal formatting
from colorama import init, Fore, Style, Back
import gc

# Import DatabaseFactory
from modulos.databases.FactoryDatabase import DatabaseFactory

# Initialize colorama for cross-platform compatibility
init(autoreset=True)

# Define colors and styles to improve readability
C_TITLE = Back.BLUE + Fore.WHITE + Style.BRIGHT
C_SUBTITLE = Fore.BLUE + Style.BRIGHT
C_SUCCESS = Fore.GREEN + Style.BRIGHT
C_ERROR = Fore.RED + Style.BRIGHT
C_WARNING = Fore.YELLOW
C_HIGHLIGHT = Fore.MAGENTA + Style.BRIGHT
C_COMMAND = Fore.YELLOW
C_INFO = Style.RESET_ALL
C_VALUE = Fore.CYAN + Style.BRIGHT
C_PROMPT = Style.BRIGHT + Fore.GREEN
C_RESET = Style.RESET_ALL
C_SEPARATOR = Style.DIM + Fore.BLUE

# Reduce logging verbosity for certain modules
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('modulos.databases').setLevel(logging.ERROR)
logging.getLogger('modulos.databases.implementaciones').setLevel(logging.ERROR)
logging.getLogger('modulos.databases.implementaciones.sqlite').setLevel(logging.ERROR)
logging.getLogger('modulos.databases.implementaciones.duckdb').setLevel(logging.ERROR)
logging.getLogger('modulos.session_manager').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('filelock').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

def process_documents(file_path: str, session_name: Optional[str] = None) -> None:
    """
    Process documents from a path and ingest them into the database.
    
    Args:
        file_path: Path to file or directory to process
        session_name: Optional session name (used for database configuration)
    """
    start_time = time.time()
    
    # Get the embeddings model first to know dimensions
    try:
        from modulos.embeddings.embeddings_factory import EmbeddingFactory
        embedding_manager = EmbeddingFactory().get_embedding_manager()
        embedding_dim = embedding_manager.get_dimensions()
        
        # Model name will be obtained directly from configuration later
        # when importing config for the other configurations
        
        logger.info(f"Embedding dimensions: {C_VALUE}{embedding_dim}{C_RESET}")
    except Exception as e:
        logger.error(f"{C_ERROR}Error initializing embeddings model: {e}")
        return
    
    # Create or update database
    db = None
    db_metadata = {}  # Store database metadata here
    try:
        # Get database configuration
        from config import config
        database_config = config.get_database_config()
        db_type = database_config.get("type", "sqlite")
        
        # Get chunking configuration
        chunks_config = config.get_chunks_config()
        chunking_method = chunks_config.get("method", "character")
        
        # Get model name directly from configuration
        embedding_config = config.get_embedding_config()
        embedding_model = embedding_config.get("model", "modernbert")
        
        # Create database instance
        db = DatabaseFactory().get_database_instance(embedding_dim=embedding_dim)
        
        # Save relevant metadata for unified session
        db_path = getattr(db, "_db_path", "")
        db_metadata = {
            "db_type": db_type,
            "db_path": db_path,
            "embedding_dim": embedding_dim,
            "embedding_model": embedding_model,
            "chunking_method": chunking_method,
            "created_at": time.time(),
            "last_used": time.time()
        }
    except Exception as e:
        logger.error(f"{C_ERROR}Error creating or connecting to database: {e}")
        return
    
    # Get chunker and Markdown processor
    chunker = None
    try:
        from modulos.chunks.ChunkerFactory import ChunkerFactory
        chunker = ChunkerFactory().get_chunker(embedding_model=embedding_manager)
    except Exception as e:
        logger.error(f"{C_ERROR}Error initializing chunker: {e}")
        return
    
    try:
        from modulos.doc_processor.markdown_processor import MarkdownProcessor
        markdown_processor = MarkdownProcessor()
    except Exception as e:
        logger.error(f"{C_ERROR}Error initializing Markdown processor: {e}")
        return
    
    # Find Markdown files for processing
    md_files = []
    file_path_obj = Path(file_path).resolve()
    
    if file_path_obj.is_dir():
        # If directory, find all .md files recursively
        logger.info(f"Finding Markdown files in {file_path_obj}...")
        md_files = list(file_path_obj.glob("**/*.md"))
    elif file_path_obj.suffix.lower() == '.md':
        # If specific .md file
        md_files = [file_path_obj]
    else:
        if not md_files:
            logger.error(f"The specified file is not a valid Markdown file: {file_path}")
            return
    
    # Get ResourceManager and ConcurrencyManager - Single instance for entire process
    from modulos.resource_management.resource_manager import ResourceManager
    resource_manager = ResourceManager()
    concurrency_manager = resource_manager.concurrency_manager
    
    # Update ResourceManager on operation in progress
    resource_manager.metrics["operation_in_progress"] = "document_processing"
    resource_manager.metrics["total_documents"] = len(md_files)
    resource_manager.update_metrics() # Force initial metrics update
 
    # Process each document
    successful_docs = 0
    failed_docs = 0
    total_files = len(md_files)
    logger.info(f"Preparing to process {total_files} documents...")

    # List to store successfully processed files
    processed_files = []
    file_metadata = {}  # Dictionary to store rich metadata for each file

    # Use concurrency only if manager exists and more than 1 file
    if concurrency_manager and total_files > 1:
        logger.info(f"Using ConcurrencyManager for parallel processing of {total_files} documents.")
        
        # Determine optimal chunksize to distribute files among workers
        iterable_length = len(md_files)
        chunksize = concurrency_manager.get_optimal_chunksize(
            task_type="default", 
            iterable_length=iterable_length
        )
        logger.info(f"Calculated batch size (chunksize) for processing: {chunksize}")
        
        # Create arguments iterator
        args_iterator = ((str(doc_path), markdown_processor, chunker, db, resource_manager) 
                        for doc_path in md_files)
        
        # Use map_tasks method that automatically selects the best pool
        results = concurrency_manager.map_tasks(
            process_single_document_wrapper,
            args_iterator,
            chunksize=chunksize,
            task_type="default",  # Task type will determine optimal pool
            # Tip: use ProcessPool for more than 10 files
            prefer_process=total_files >= 10
        )
        
        # Process results from parallel execution
        if results:
            processed_count = 0
            for doc_path, result in zip(md_files, results):
                processed_count += 1
                if isinstance(result, dict) and result.get("document_id") is not None:
                    successful_docs += 1
                    doc_id = result.get("document_id")
                    # Add to processed files list
                    processed_files.append(str(doc_path))
                    
                    # Save rich metadata (for statistics even if not used in session)
                    file_path_str = str(doc_path)
                    file_metadata[file_path_str] = {
                        "size": doc_path.stat().st_size,
                        "chunks": result.get("chunk_count", 0),
                        "processing_time": result.get("processing_time", 0)
                    }
                    
                    logger.info(f"({processed_count}/{total_files}) Document processed OK: {doc_path} -> ID: {doc_id}")
                else:
                    failed_docs += 1
                    logger.error(f"({processed_count}/{total_files}) Error processing: {doc_path}")
        else:
            logger.error(f"{C_ERROR}No results received from parallel processing.")
            failed_docs = total_files  # Mark all as failed if no results
    else:
        # Sequential processing if no ConcurrencyManager or is single file
        logger.info("Processing sequentially.")
        for i, doc_path in enumerate(md_files):
            logger.info(f"({i+1}/{total_files}) Processing: {doc_path}")
            result = process_single_document(str(doc_path), markdown_processor, chunker, db, resource_manager)
            if isinstance(result, dict) and result.get("document_id") is not None:
                successful_docs += 1
                processed_files.append(str(doc_path))
                
                # Save rich metadata (for statistics)
                file_path_str = str(doc_path)
                file_metadata[file_path_str] = {
                    "size": doc_path.stat().st_size,
                    "chunks": result.get("chunk_count", 0),
                    "processing_time": result.get("processing_time", 0)
                }
            else:
                failed_docs += 1

    # Calculate elapsed time for processing
    elapsed_time = time.time() - start_time
    
    # Optimize database after inserting all documents
    logger.info("Optimizing database...")
    db.optimize_database()
    
    # If processing was successful, create unified session
    if successful_docs > 0:
        try:
            # Calculate statistics for session
            total_chunks_processed = sum(metadata.get("chunks", 0) for metadata in file_metadata.values())
            total_processing_time = sum(metadata.get("processing_time", 0) for metadata in file_metadata.values())
            
            # Update database metadata with statistics
            db_metadata.update({
                "total_chunks": total_chunks_processed,
                "total_files": successful_docs,
                "processing_time": total_processing_time,
                "processing_complete": True,
                "processing_date": datetime.now().isoformat()
            })
            
            # Create or get session through new unified method
            from modulos.session_manager.session_manager import SessionManager
            session_manager = SessionManager()
            
            # If session name was provided, use it
            if session_name:
                db_metadata["name"] = session_name
                db_metadata["id"] = session_name
            
            # Create unified session with all metadata
            unified_session_id = session_manager.create_unified_session(
                database_metadata=db_metadata,
                files_list=processed_files
            )
            
            logger.info(f"Unified session created successfully: {unified_session_id}")
        except Exception as e:
            logger.error(f"{C_ERROR}Error creating unified session: {e}")
    else:
        logger.warning(f"{C_WARNING}No session created because no documents processed successfully.")

    # Show processing summary
    logger.info(f"Processing completed in {C_VALUE}{elapsed_time:.2f}{C_RESET} seconds")
    logger.info(f"Documents processed correctly: {C_VALUE}{successful_docs}{C_RESET}")
    
    if failed_docs:
        logger.warning(f"Documents with errors: {C_VALUE}{failed_docs}{C_RESET}")
    
    # Release resources
    if resource_manager and resource_manager.memory_manager:
        logger.info("Performing final memory cleanup...")
        resource_manager.memory_manager.cleanup(reason="processing_completed")
    
    logger.info("Ingestion process completed")

def process_single_document_wrapper(args_tuple):
    """
    Wrapper function for use with concurrent processing.
    
    Unpacks arguments tuple and calls process_single_document.
    Handles exceptions at worker level to avoid worker crashes.
    
    Args:
        args_tuple: Tuple containing (file_path, markdown_processor, chunker, db, resource_manager)
        
    Returns:
        Result from process_single_document or None if error occurs
    """
    # Unpack arguments
    file_path, markdown_processor, chunker, db, resource_manager = args_tuple
    
    # Call original function with exception handling
    try:
        return process_single_document(file_path, markdown_processor, chunker, db, resource_manager)
    except Exception as e:
        # Print to stderr since worker logging might not be properly configured
        print(f"[Worker Error] Error processing {os.path.basename(file_path)}: {e}", file=sys.stderr)
        return None

def process_single_document(file_path: str, 
                           markdown_processor: Any, 
                           chunker: Any, 
                           db: Any,
                           resource_manager: Any = None) -> Optional[int]:
    """
    Processes a single Markdown document and inserts it into the database.
    Uses a streaming approach to process large documents without
    loading all chunks into memory simultaneously, and ensures
    that all processing is done in a single transaction.
    
    Args:
        file_path: Path to Markdown file
        markdown_processor: Markdown processor instance
        chunker: Chunker instance
        db: Database instance
        resource_manager: ResourceManager instance
        
    Returns:
        Inserted document ID or None if failed
    """
    if resource_manager is None:
        # Only if we didn't receive resource_manager as parameter
        try:
            from modulos.resource_management.resource_manager import ResourceManager
            resource_manager = ResourceManager()
        except Exception as e:
            logger.error(f"Error initializing ResourceManager: {e}")
            resource_manager = None
    
    memory_manager = None
    if resource_manager:
        memory_manager = resource_manager.memory_manager

    try:
        # Get memory optimization configuration
        from config import config
        chunks_config = config.get_chunks_config()
        memory_config = chunks_config.get("memory_optimization", {})
        
        # Configure interval for memory checks
        memory_check_interval = memory_config.get("memory_check_interval", 15.0)  # Check memory every 15 seconds
        
        # Process document to get metadata and content
        start_processing_time = time.time()
        metadata, content = markdown_processor.process_document(file_path)
        
        # Determine if we need GC before continuing with main processing
        if memory_manager and (time.time() - start_processing_time > 2.0):  # If processing took more than 2 seconds
            # Document could be large, do preventive cleanup
            logger.debug("Performing preventive cleanup before processing large document")
            memory_manager.cleanup(aggressive=False, reason="pre_doc_processing")
        
        # Get document title from metadata
        doc_title = metadata.get('title', 'Documento')
        
        # Determine document size for possible suspension of checks
        document_size_kb = len(content) / 1024  # Approximate size in KB
        
        # Determine if document is large (to adjust strategies)
        is_large_document = document_size_kb > 500  # Documents > 500KB are considered large
        
        # Adjust processing strategy based on size
        if is_large_document:
            logger.info(f"Large document detected ({document_size_kb:.1f}KB). Adjusting processing parameters.")
            
            # For large documents, consider suspension of checks
            if resource_manager:
                if document_size_kb > 5000:  # Extremely large document (>5MB)
                    # Suspend checks for longer time for very large documents
                    resource_manager.auto_suspend_if_needed(document_size_kb=document_size_kb, duration_seconds=600)
                    logger.info("Checks suspended for optimizing processing of very large document")
                else:
                    resource_manager.auto_suspend_if_needed(document_size_kb=document_size_kb, duration_seconds=300)
                    logger.info("Checks suspended for large document")
        else:
            logger.info(f"Standard size document ({document_size_kb:.1f}KB). Keeping normal configuration.")
            
            # For small documents, consider temporary brief suspension
            if resource_manager and document_size_kb < 50:  # Very small documents
                resource_manager.auto_suspend_if_needed(document_size_kb=document_size_kb, duration_seconds=30)
        
        # Insert document metadata first
        document_id = db.insert_document_metadata(metadata)
        
        if not document_id:
            logger.error(f"{C_ERROR}Error inserting document metadata {file_path}")
            return None
        
        # Start a single transaction for entire document processing
        db.begin_transaction()
        
        # Variables for statistics and control
        processed_chunks = 0
        start_time = time.time()
        last_memory_check_time = time.time()
        embedding_dim = chunker.model.get_dimensions()
        
        try:
            # Generate, process, and insert chunks one by one using streaming
            # Without accumulating them in a buffer to minimize memory usage
            for chunk in chunker.process_content_stream(content, doc_title=doc_title):
                logger.debug(f"Processing chunk with header: {chunk.get('header', 'No header')} and text length: {len(chunk.get('text', ''))} characters")
                
                # Verify if model is initialized
                if chunker.model is None:
                    logger.error(f"{C_ERROR}Error: Embedding model in chunker is None. Cannot generate embedding.")
                    db.rollback_transaction()
                    return None
                
                try:
                    # Extract data from chunk
                    header = chunk.get('header', '')
                    text = chunk.get('text', '')
                    page = chunk.get('page', '')
                    
                    # Verify data
                    if not text:
                        logger.warning("Empty text chunk, skipping")
                        continue
                    
                    # Generate embedding for this individual chunk
                    try:
                        embedding = chunker.model.get_document_embedding(header, text)
                    except Exception as e:
                        logger.error(f"Error generating embedding: {e}")
                        # Create empty embedding as fallback
                        embedding = [0.0] * embedding_dim
                    
                    # Prepare final chunk with its embedding
                    prepared_chunk = {
                        'text': text,
                        'header': header,
                        'page': page,
                        'embedding': embedding,
                        'embedding_dim': embedding_dim
                    }
                    
                    # Insert directly this chunk into the database (within the same transaction)
                    db.insert_single_chunk(document_id, prepared_chunk)
                    processed_chunks += 1
                    
                    # Show progress periodically
                    if processed_chunks % 10 == 0:  # Show every 10 processed chunks
                        elapsed = time.time() - start_time
                        rate = processed_chunks / elapsed if elapsed > 0 else 0
                        logger.info(f"Processed {C_VALUE}{processed_chunks}{C_RESET} chunks ({C_VALUE}{rate:.2f}{C_RESET} chunks/sec)")
                    
                    # Verify memory usage periodically
                    current_time = time.time()
                    if memory_manager and (current_time - last_memory_check_time) >= memory_check_interval:
                        last_memory_check_time = current_time
                        memory_manager.check_memory_usage()
                    
                    # Explicitly release references to help GC
                    del embedding
                    del prepared_chunk
                    
                except Exception as chunk_e:
                    logger.error(f"{C_ERROR}Error processing/inserting individual chunk: {chunk_e}", exc_info=True)
            
            # Confirm transaction - entire document is confirmed at once
            db.commit_transaction()
            
            # Final cleanup after complete processing
            if memory_manager:
                # Request cleanup but adjust aggressiveness based on size of processed document
                is_large_processing = processed_chunks > 100 or document_size_kb > 1000
                memory_manager.cleanup(
                    aggressive=is_large_processing,  
                    reason=f"post_doc_processing_{processed_chunks}_chunks"
                )
            else:
                # If no memory_manager, use traditional GC
                gc.collect()
            
            # Evaluate at the end of processing if we should reconsider suspension
            # based on total chunks processed
            if resource_manager and is_large_document and processed_chunks < 10:
                logger.info(f"Large document ({document_size_kb:.1f}KB) but generated few chunks ({processed_chunks}). Reconsidering suspension.")
                resource_manager.auto_suspend_if_needed(
                    document_size_kb=document_size_kb, 
                    chunk_count=processed_chunks,
                    duration_seconds=60  # Shorter suspension for documents that turned out to be simple
                )
                
            # Calculate final statistics
            total_time = time.time() - start_time
            logger.info(f"Completed: {file_path} -> {C_VALUE}{processed_chunks}{C_RESET} chunks processed in {C_VALUE}{total_time:.2f}{C_RESET}s")
            if processed_chunks > 0:
                logger.info(f"Performance: {C_VALUE}{processed_chunks / total_time:.2f}{C_RESET} chunks/sec, {C_VALUE}{document_size_kb / total_time:.2f}{C_RESET} KB/sec")
            
            return {
                "document_id": document_id, 
                "chunk_count": processed_chunks,
                "processing_time": total_time,
                "document_size_kb": document_size_kb,
                "chunks_per_second": processed_chunks / total_time if total_time > 0 else 0,
                "kb_per_second": document_size_kb / total_time if total_time > 0 else 0
            }
            
        except Exception as e:
            # Rollback in case of error - entire transaction is rolled back
            logger.error(f"{C_ERROR}Error processing chunks of {file_path}: {e}", exc_info=True)
            try:
                db.rollback_transaction()
                logger.info("Transaction rolled back.")
            except Exception as rollback_e:
                logger.error(f"Additional error when trying to rollback: {rollback_e}")
            return None
    except Exception as e:
        logger.error(f"{C_ERROR}Error processing document {file_path}: {e}", exc_info=True)
        return None

def process_query(query: str, n_chunks: int = 5, model: Optional[str] = None, 
                 session_id: Optional[str] = None, db_index: Optional[int] = None) -> str:
    """
    Processes a query using the RAG system.
    
    This function:
    1. Retrieves relevant chunks from the database based on query embedding
    2. Sends these chunks as context to an AI model
    3. Returns the generated response
    
    Args:
        query: Query text
        n_chunks: Number of chunks to retrieve
        model: AI model to use (optional)
        session_id: Specific session ID to use (optional)
        db_index: Index of database to use (optional)
        
    Returns:
        Generated response text
    """
    # Imports on demand
    from config import config  
    from modulos.session_manager.session_manager import SessionManager
    from modulos.embeddings.embeddings_factory import EmbeddingFactory
    from modulos.clientes.FactoryClient import ClientFactory
    
    try:
        # Use session_manager to get correct configuration
        session_manager = SessionManager()
    
        # Get database and configuration
        if db_index is not None:
            # If specific index requested, use that index
            db, session = session_manager.get_database_by_index(db_index, session_id=session_id)
        elif session_id:
            # If session ID provided but no index, use most recent database for that session
            session_dbs = session_manager.get_session_databases(session_id)
            if session_dbs:
                db, session = session_manager.get_database_by_index(0, session_id=session_id)
            else:
                db, session = session_manager.get_database_by_index(0)
        else:
            # Default: use most recent database
            db, session = session_manager.get_database_by_index(0)
        
        # Initialize embedding model that matches the database
        embedding_model = session.get("embedding_model", "modernbert")
        embedding_manager = EmbeddingFactory.get_embedding_manager(embedding_model)
        
        try:
            embedding_manager.load_model()
        except Exception as e:
            logger.error(f"Error loading embeddings model: {e}")
            return "Could not load embeddings model. Please check configuration or try with another database."
        
        # Generate embedding for the query
        query_embedding = embedding_manager.get_query_embedding(query)
        
        # Search for most relevant chunks
        try:
            search_results = db.vector_search(query_embedding, n_results=n_chunks)
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return "There was a problem finding relevant information. The selected database may not be compatible with the current query."
        
        if not search_results:
            return "No relevant information found to answer this query."
        
        # Determine which AI model to use
        if model is None:
            ai_config = config.get_ai_client_config()
            model = ai_config.get("type", "openai")
            
        # Initialize AI client
        try:
            ai_client = ClientFactory.get_client(client_type=model)
        except Exception as e:
            logger.error(f"Error creating AI client: {e}")
            return "Could not initialize AI model. Please check your configuration and API keys."
        
        # Prepare context chunks for the response
        context_chunks = []
        for chunk in search_results:
            context_chunks.append({
                "text": chunk["text"],
                "header": chunk.get("header", ""),
                "similarity": chunk.get("similarity", 0.0),
                "page": chunk.get("page", "N/A")
            })
        
        # Generate response
        try:
            response = ai_client.generate_response(query, context=context_chunks, show_context=True)
            
            # Handle streaming responses
            if hasattr(response, '__iter__') and not isinstance(response, str):
                try:
                    # Collect all chunks from the generator
                    chunks = list(response)
                    full_response = "".join(chunks)
                    
                    if full_response:
                        return full_response
                    elif hasattr(ai_client, 'last_response_text') and ai_client.last_response_text:
                        # Fallback to stored response text
                        return ai_client._format_response_with_context(ai_client.last_response_text, query)
                    else:
                        return "Response generated successfully but could not be displayed. Please try again."
                except Exception as e:
                    # If iteration fails, try using response as string
                    if response:
                        return str(response)
                    else:
                        return "Error processing response. Please try again."
            
            # Handle null response
            if response is None:
                return "No response received from model. This may be a connection or API problem."
                
            # Return string response
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "There was a problem generating response. Please check your configuration and internet connection."
        
    except Exception as e:
        # Update database last use timestamp if possible
        try:
            if 'session' in locals() and 'db_name' in session:
                db_name = session.get("id", "")
                # Update timestamp
                session_manager.register_database(db_name, {
                    "last_used": time.time(),
                    **session
                })
        except Exception as update_err:
            logger.debug(f"Error updating metadata: {update_err}")
    
        # Format error message
        try:
            error_msg = f"{e.__class__.__name__}: {str(e)}"
        except Exception:
            error_msg = "Unknown error during processing"
            
        logger.error(f"Error processing query: {error_msg}")
        
        # Return user-friendly error message
        if "database" in error_msg.lower():
            return "Error accessing database. Please check that the selected database exists and is accessible."
        elif "embedding" in error_msg.lower() or "model" in error_msg.lower():
            return "Embedding model error. Please check configuration or try with another model."
        else:
            return f"An error occurred processing your query: {error_msg}. Please try again or select another database."

def verify_database_file(db_path: str) -> bool:
    """
    Verifies that the physical database file exists.
    
    Args:
        db_path: Path to database file
        
    Returns:
        bool: True if file exists, False otherwise
    """
    if db_path == ":memory:":
        return True
    
    file_path = Path(db_path)
    exists = file_path.exists()
    
    if exists:
        size_kb = file_path.stat().st_size / 1024
        logger.info(f"{C_SUCCESS}✓ Database verified: {C_VALUE}{db_path} ({C_SUCCESS}{size_kb:.2f} KB{C_INFO})")
    else:
        logger.error(f"{C_ERROR}✗ Database file not found: {C_VALUE}{db_path}")
        
    return exists
