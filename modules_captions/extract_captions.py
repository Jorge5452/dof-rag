#!/usr/bin/env python3
"""Extract Captions - Enhanced Image Description Extraction System

This script provides an improved version of the caption extraction system
with SQLite database storage, better error handling, and enhanced processing
capabilities. It supports multiple AI providers through direct command-line flags.

Usage:
    python extract_captions.py --root-dir /path/to/images --db-path captions.db
    python extract_captions.py --root-dir /path/to/images --openai --batch-size 20
    python extract_captions.py --root-dir /path/to/images --gemini
    python extract_captions.py --root-dir /path/to/images --claude
"""

import argparse
import json
import os
import sys
import signal
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Import colorama for cross-platform colored output
try:
    import colorama
    from colorama import Fore, Back, Style, init
    # Initialize colorama for Windows compatibility
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    # Fallback if colorama is not available
    class MockColor:
        def __getattr__(self, name):
            return ''
    
    Fore = Back = Style = MockColor()
    COLORAMA_AVAILABLE = False

try:
    from .db.manager import DatabaseManager
    from .clients import create_client
    from .utils.file_processor import FileProcessor
    from .utils.error_handler import ErrorHandler
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(Path(__file__).parent))
    from db.manager import DatabaseManager
    from clients import create_client
    from utils.file_processor import FileProcessor
    from utils.error_handler import ErrorHandler

def print_header(text: str, color: str = Fore.CYAN):
    """Print a formatted header with colors."""
    separator = "=" * len(text)
    print(f"\n{color}{separator}")
    print(f"{text}")
    print(f"{separator}{Style.RESET_ALL}")

def print_info(label: str, value: str, color: str = Fore.GREEN):
    """Print formatted information with colors."""
    print(f"{color}📋 {label}:{Style.RESET_ALL} {value}")

def print_success(message: str):
    """Print success message with green color."""
    print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")

def print_warning(message: str):
    """Print warning message with yellow color."""
    print(f"{Fore.YELLOW}⚠️  {message}{Style.RESET_ALL}")

def print_error(message: str):
    """Print error message with red color."""
    print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")

def print_stats(stats: Dict[str, Any]):
    """Print essential processing statistics with colors and formatting."""
    print_header("📊 Processing Summary", Fore.CYAN)
    
    # Essential statistics only
    processed = stats.get('total_processed', 0)
    total = stats.get('total_images', 0)
    errors = stats.get('total_errors', 0)
    
    print_info("Processed", f"{processed}/{total} images", Fore.GREEN if processed > 0 else Fore.YELLOW)
    
    if errors > 0:
        print_info("Errors", str(errors), Fore.RED)
    
    # Success rate with color coding
    success_rate = stats.get('success_rate', 0)
    if success_rate >= 90:
        color = Fore.GREEN
    elif success_rate >= 70:
        color = Fore.YELLOW
    else:
        color = Fore.RED
    print_info("Success rate", f"{success_rate:.1f}%", color)
    
    # Time information (simplified)
    total_time = stats.get('total_time_seconds', 0)
    if total_time > 0:
        print_info("Total time", f"{total_time:.1f}s", Fore.MAGENTA)
    
    # Database summary (simplified)
    db_stats = stats.get('database_stats', {})
    if db_stats and db_stats.get('total_descriptions', 0) > 0:
        print_info("Total descriptions in DB", str(db_stats.get('total_descriptions', 0)), Fore.CYAN)

class CaptionExtractor:
    """
    Main caption extraction orchestrator.
    
    This class coordinates the entire caption extraction process,
    managing the database, AI client, file processor, and error handling.
    """
    
    def __init__(self, config: Dict[str, Any], status_only: bool = False):
        """
        Initialize the caption extractor.
        
        Args:
            config: Configuration dictionary with all necessary parameters.
            status_only: If True, skip API key validation for status-only operations.
        """
        self.config = config
        self.interrupted = False
        self.status_only = status_only
        
        # Initialize error handler first
        self.error_handler = ErrorHandler(
            log_dir=config.get('log_dir', 'logs'),
            log_level=config.get('log_level', 20),  # INFO level
            debug_mode=config.get('debug_mode', False)
        )
        
        # Initialize database manager
        self.db_manager = DatabaseManager(config['db_path'])
        
        # Initialize AI client
        self.ai_client = self._create_ai_client()
        
        # Pass debug mode to AI client if supported
        if hasattr(self.ai_client, 'debug_mode'):
            self.ai_client.debug_mode = config.get('debug_mode', False)
        
        # Initialize file processor only if not status-only mode
        if not status_only:
            self.file_processor = FileProcessor(
                root_directory=config['root_directory'],
                db_manager=self.db_manager,
                ai_client=self.ai_client,
                log_dir=config.get('log_directory', 'logs'),
                commit_interval=config.get('commit_interval', 10),
                cooldown_seconds=config.get('cooldown_seconds', 0),
                debug_mode=config.get('debug_mode', False)
            )
            
            # Set file processor reference in AI client for interrupt handling
            if hasattr(self.ai_client, 'set_file_processor'):
                self.ai_client.set_file_processor(self.file_processor)
        else:
            self.file_processor = None
        
        # Setup signal handlers for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _create_ai_client(self):
        """
        Create and configure the AI client.
        
        Returns:
            Configured AI client instance.
        """
        try:
            # Get provider configuration
            provider = self.config.get('provider', 'openai')
            client_config = self.config.get('client_config', {})
            
            # Create client
            client = create_client(provider, **client_config)
            
            # Set API key
            api_key = self.config.get('api_key') or os.getenv(f"{provider.upper()}_API_KEY")
            if not api_key:
                error_msg = (f"API key not found for provider {provider}. "
                           f"Set {provider.upper()}_API_KEY environment variable or provide in config.")
                if self.status_only:
                    self.error_handler.logger.warning(error_msg + " (Status-only mode, continuing without API key)")
                    # Don't set API key, client will be in limited mode
                else:
                    self.error_handler.logger.error(error_msg)
                    raise ValueError(error_msg)
            else:
                client.set_api_key(api_key)
            
            # Set custom prompt if provided
            if 'prompt' in self.config:
                client.set_prompt(self.config['prompt'])
            
            # Set error handler for the client
            client.set_error_handler(self.error_handler)
            
            # Configure rate limiting if available
            rate_limits = self.config.get('rate_limits', {})
            requests_per_minute = rate_limits.get('requests_per_minute')
            if requests_per_minute and hasattr(client, 'set_rate_limits'):
                client.set_rate_limits(requests_per_minute)
            
            # Print additional client configuration
            print(client.get_model_info())
            return client
            
        except Exception as e:
            self.error_handler.handle_error(e, {'provider': provider, 'config': client_config}, 'api')
            raise
    
    def _signal_handler(self, signum, frame):
        """
        Handle interrupt signals for graceful shutdown.
        
        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        self.error_handler.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.interrupted = True
        self.file_processor.interrupt()
    
    def extract_captions(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Main method to extract captions from images.
        
        Args:
            force_reprocess: If True, reprocess images even if descriptions exist.
            
        Returns:
            Dict with processing results and statistics.
        """
        try:
            # Validate API key configuration before processing
            if not hasattr(self.ai_client, '_client') or self.ai_client._client is None:
                error_msg = "AI client not properly configured. API key validation failed."
                self.error_handler.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Log processing start
            self.error_handler.log_processing_start(self.config)
            
            # Find images to process
            self.error_handler.logger.info("Discovering images to process...")
            images = self.file_processor.find_images(skip_existing=not force_reprocess)
            
            if not images:
                self.error_handler.logger.info("No images found to process")
                return {'status': 'completed', 'message': 'No images to process'}
            
            # Process images
            self.error_handler.logger.info(f"Starting processing of {len(images)} images")
            results = self.file_processor.process_images(images, force_reprocess)
            
            # Add database statistics
            db_stats = self.db_manager.get_statistics()
            results['database_stats'] = db_stats
            
            # Add error summary
            error_summary = self.error_handler.get_error_summary()
            results['error_summary'] = error_summary
            
            # Log processing end
            self.error_handler.log_processing_end(results)
            
            return {
                'status': 'completed' if not results.get('interrupted', False) else 'interrupted',
                'results': results
            }
            
        except Exception as e:
            self.error_handler.handle_error(e, {'operation': 'extract_captions'}, 'unknown')
            return {
                'status': 'error',
                'error': str(e),
                'error_summary': self.error_handler.get_error_summary()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status and statistics.
        
        Returns:
            Dict with system status information.
        """
        try:
            db_stats = self.db_manager.get_statistics()
            error_summary = self.error_handler.get_error_summary()
            
            # Try to get model information, but don't fail if not possible
            try:
                model_info = self.ai_client.get_model_info()
            except Exception as model_error:
                self.error_handler.logger.warning(f"Could not get model information: {model_error}")
                model_info = {"status": "unavailable", "error": str(model_error)}
            
            # Filter sensitive information from configuration
            safe_config = self.config.copy()
            if 'api_key' in safe_config:
                safe_config['api_key'] = '***REDACTED***'
            
            return {
                'database_stats': db_stats,
                'error_summary': error_summary,
                'model_info': model_info,
                'config': safe_config
            }
            
        except Exception as e:
            self.error_handler.handle_error(e, {'operation': 'get_status'}, 'unknown')
            return {'error': str(e)}



def load_provider_config(provider: str) -> Dict[str, Any]:
    """
    Load configuration for the specified provider from config.json.
    
    Args:
        provider: Provider name (openai, gemini, claude, ollama, azure)
        
    Returns:
        Complete configuration for the specified provider
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            unified_config = json.load(f)
    except Exception as e:
        print_error(f"Error loading config.json: {e}")
        print_warning("Using default configuration...")
        return create_default_config()
    
    # Get base configuration (excluding providers section)
    config = {k: v for k, v in unified_config.items() if k not in ["providers", "_comment", "_instructions", "_configuration_notes", "_environment_variables", "_usage_examples"]}
    
    # Check if provider exists
    if provider not in unified_config.get("providers", {}):
        print_error(f"Provider '{provider}' not found in configuration. Available providers:")
        for available_provider in unified_config.get('providers', {}).keys():
            print(f"  {Fore.CYAN}- {available_provider}{Style.RESET_ALL}")
        print_warning("Using default configuration...")
        return config
    
    # Update with provider-specific configuration
    provider_config = unified_config["providers"][provider]
    
    # Set the provider
    config["provider"] = provider
    
    # Update client_config
    if "client_config" in provider_config:
        config["client_config"] = provider_config["client_config"]
    
    # Update rate_limits
    if "rate_limits" in provider_config:
        config["rate_limits"] = provider_config["rate_limits"]
    
    # Convert relative paths to absolute paths within modules_captions
    module_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Handle log_directory
    if "log_directory" in config:
        if not os.path.isabs(config["log_directory"]):
            config["log_directory"] = os.path.join(module_dir, config["log_directory"])
            config["log_dir"] = config["log_directory"]  # Alias for compatibility
    
    # Handle checkpoint_directory
    if "checkpoint_directory" in config:
        if not os.path.isabs(config["checkpoint_directory"]):
            config["checkpoint_directory"] = os.path.join(module_dir, config["checkpoint_directory"])
            config["checkpoint_dir"] = config["checkpoint_directory"]  # Alias for compatibility
    
    # Handle db_path - place it in modules_captions/db/
    if "db_path" in config:
        if not os.path.isabs(config["db_path"]):
            # Extract just the filename from the path
            db_filename = os.path.basename(config["db_path"])
            config["db_path"] = os.path.join(module_dir, "db", db_filename)
    
    # Create directories if they don't exist
    if "log_directory" in config:
        os.makedirs(config["log_directory"], exist_ok=True)
    if "checkpoint_directory" in config:
        os.makedirs(config["checkpoint_directory"], exist_ok=True)
    if "db_path" in config:
        db_dir = os.path.dirname(config["db_path"])
        os.makedirs(db_dir, exist_ok=True)
    
    # Show rate limit information
    if "rate_limits" in provider_config:
        rate_limits = provider_config["rate_limits"]
        print_header(f"⚡ Rate Limits for {provider}", Fore.YELLOW)
        print_info("Requests per minute", str(rate_limits.get('requests_per_minute', 'Not specified')), Fore.CYAN)
        print_info("Tokens per minute", str(rate_limits.get('tokens_per_minute', 'Not specified')), Fore.CYAN)
        print_info("Requests per day", str(rate_limits.get('requests_per_day', 'Not specified')), Fore.CYAN)
    
    # Check environment variable for API key
    if "env_var" in provider_config and provider_config["env_var"]:
        env_var = provider_config["env_var"]
        if os.getenv(env_var):
            print_success(f"Using API key from environment variable {env_var}")
            # Import OpenAIClient to use clean_api_key method
            from clients.openai import OpenAIClient
            config["api_key"] = OpenAIClient.clean_api_key(os.getenv(env_var))
        else:
            print_warning(f"Environment variable {env_var} not found")
            print_warning(f"Set {env_var} or provide an API key with --api-key")
    
    return config

def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration dictionary.
    
    Returns:
        Default configuration.
    """
    # Path to modules_captions directory for logs and checkpoints
    module_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(module_dir, 'logs')
    
    # Create logs directory if it doesn't exist
    os.makedirs(logs_dir, exist_ok=True)
    
    return {
        'provider': 'openai',

        'log_dir': logs_dir,
        'log_level': 20,  # INFO
        'client_config': {
            'model': 'gpt-4o',
            'max_tokens': 256,
            'temperature': 0.6,
            'top_p': 0.6
        },
        'prompt': """Resume brevemente la imagen en español (máximo 3-4 oraciones por categoría):
- **Texto:** Menciona solo el título y 2-3 puntos clave si hay texto.
- **Mapas:** Identifica la región principal y máximo 2-3 ubicaciones relevantes.
- **Diagramas:** Resume el concepto central en 1-2 oraciones.
- **Logos:** Identifica la entidad y sus características distintivas.
- **Datos visuales:** Menciona solo los 2-3 valores o tendencias más importantes.
Prioriza la información esencial sobre los detalles, manteniendo la descripción breve y directa."""
    }

# Load environment variables from .env file
try:
    load_dotenv()
    print("Environment variables loaded from .env")
except Exception as e:
    print(f"Error loading environment variables: {e}")

def main():
    """
    Main entry point for the caption extraction script.
    """
    parser = argparse.ArgumentParser(
        description="Enhanced Image Caption Extraction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_captions.py --root-dir ./images --db-path captions.db
  python extract_captions.py --force-reprocess
  python extract_captions.py --root-dir ./images --openai
  python extract_captions.py --root-dir ./images --gemini
  python extract_captions.py --root-dir ./images --claude
        """
    )
    
    # Configuration options
    parser.add_argument('--root-dir', type=str, help='Root directory containing images')
    parser.add_argument('--db-path', type=str, default='captions.db', help='Path to SQLite database')
    
    # Provider selection (mutually exclusive)
    provider_group = parser.add_mutually_exclusive_group()
    provider_group.add_argument('--openai', action='store_true', help='Use OpenAI provider')
    provider_group.add_argument('--gemini', action='store_true', help='Use Google Gemini provider')
    provider_group.add_argument('--claude', action='store_true', help='Use Anthropic Claude provider')
    provider_group.add_argument('--ollama', action='store_true', help='Use Ollama provider')
    provider_group.add_argument('--azure', action='store_true', help='Use Azure OpenAI provider')
    
    parser.add_argument('--api-key', type=str, help='API key for the AI provider')

    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Reprocess images even if descriptions exist')
    parser.add_argument('--status', action='store_true', help='Show system status and exit')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode with verbose logging and detailed API information')
    
    args = parser.parse_args()
    
    try:
        # Determine provider from command line arguments
        provider = 'openai'  # default
        if args.gemini:
            provider = 'gemini'
        elif args.claude:
            provider = 'claude'
        elif args.ollama:
            provider = 'ollama'
        elif args.azure:
            provider = 'azure'
        elif args.openai:
            provider = 'openai'
        
        # Load provider-specific configuration from config.json
        config = load_provider_config(provider)
        
        # Set default root directory to dof_markdown if not specified
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_root_dir = os.path.join(project_root, 'dof_markdown')
        
        # Override config with command line arguments
        if args.root_dir:
            config['root_directory'] = args.root_dir
        else:
            # Use dof_markdown as default root directory
            config['root_directory'] = default_root_dir
            
        if args.db_path and args.db_path != 'captions.db':  # Only override if explicitly set by user
            # Apply the same path conversion logic as in load_config
            if not os.path.isabs(args.db_path):
                module_dir = os.path.dirname(os.path.abspath(__file__))
                db_filename = os.path.basename(args.db_path)
                config['db_path'] = os.path.join(module_dir, "db", db_filename)
                # Create directory if it doesn't exist
                db_dir = os.path.dirname(config['db_path'])
                os.makedirs(db_dir, exist_ok=True)
            else:
                config['db_path'] = args.db_path
        if args.api_key:
            # Import OpenAIClient to use clean_api_key method
            from clients.openai import OpenAIClient
            config['api_key'] = OpenAIClient.clean_api_key(args.api_key)

        if args.log_level:
            log_levels = {'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40}
            config['log_level'] = log_levels[args.log_level]
        
        # Enable debug mode if requested
        if args.debug:
            config['log_level'] = 10  # DEBUG level
            config['debug_mode'] = True
            print("🔍 Debug mode enabled - verbose logging activated")
        else:
            config['debug_mode'] = False
        
        # Validate required parameters
        if 'root_directory' not in config:
            parser.error("Root directory is required. Use --root-dir or provide in config file.")
        
        if not os.path.exists(config['root_directory']):
            parser.error(f"Root directory does not exist: {config['root_directory']}")
        
        # Handle status request specially to avoid API key requirement
        if args.status:
            try:
                # Initialize extractor in status-only mode
                extractor = CaptionExtractor(config, status_only=True)
                status = extractor.get_status()
                
                # Display status with colored formatting instead of JSON
                print_header("📊 System Status", Fore.BLUE)
                print_info("Status", status.get('status', 'Unknown'), Fore.GREEN)
                print_info("Images found", str(status.get('total_images', 0)), Fore.CYAN)
                
                # Database statistics
                db_stats = status.get('database_stats', {})
                if db_stats:
                    print_header("💾 Database Statistics", Fore.BLUE)
                    print_info("Total descriptions", str(db_stats.get('total_descriptions', 0)), Fore.CYAN)
                    print_info("Unique documents", str(db_stats.get('unique_documents', 0)), Fore.CYAN)
                    print_info("Recent descriptions", str(db_stats.get('recent_descriptions', 0)), Fore.CYAN)
                
                return
            except Exception as e:
                print_error(f"Error getting status: {e}")
                sys.exit(1)
        
        # Initialize extractor for normal operation
        print_header(f"🚀 Starting Caption Extraction", Fore.GREEN)
        print_info("Provider", provider, Fore.CYAN)
        print_info("Model", config.get('client_config', {}).get('model', 'Not specified'), Fore.CYAN)
        print_info("Base URL", config.get('client_config', {}).get('base_url', 'Default'), Fore.CYAN)
        
        if config.get('debug_mode', False):
            print_warning("🔍 Debug mode: ENABLED")
            print_info("Root directory", config['root_directory'], Fore.YELLOW)
            print_info("Database path", config.get('db_path', 'captions.db'), Fore.YELLOW)
        
        extractor = CaptionExtractor(config)
        
        # Extract captions
        print_info("Starting caption extraction", "Processing images...", Fore.GREEN)
        results = extractor.extract_captions(force_reprocess=args.force_reprocess)
        
        # Print results with colored formatting
        if results['status'] in ['completed', 'interrupted']:
            # Only print stats if we have results data
            if 'results' in results:
                print_stats(results['results'])
            else:
                print_info("Processing completed", results.get('message', 'No additional information'), Fore.GREEN)
            if results['status'] == 'interrupted':
                print_warning("Processing was interrupted")
        else:
            print_error(f"Error during processing: {results.get('error', 'Unknown error')}")
            # Show simplified error summary instead of full report
            if 'error_summary' in results:
                error_summary = results['error_summary']
                if error_summary.get('total_errors', 0) > 0:
                    print_error(f"Total errors encountered: {error_summary['total_errors']}")
                    print_info("Error log", error_summary.get('error_log_file', 'Not available'), Fore.YELLOW)
        
    except KeyboardInterrupt:
        print_warning("Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()