import logging
import os
import json
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import sys

class ErrorHandler:
    """
    Centralized error handling and logging for the caption extraction system.
    
    This class provides comprehensive error tracking, logging, and recovery
    mechanisms for the image processing pipeline. It maintains error logs
    and provides utilities for debugging and system monitoring.
    """
    
    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO, debug_mode: bool = False):
        """
        Initialize the error handler.
        
        Args:
            log_dir: Directory for storing log files.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            debug_mode: Enable debug mode with enhanced logging.
        """
        self.log_dir = Path(log_dir)
        self.debug_mode = debug_mode
        
        # Crear directorio de logs y subdirectorios si no existen
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Error tracking
        self.error_counts = {
            'api_errors': 0,
            'file_errors': 0,
            'database_errors': 0,
            'validation_errors': 0,
            'unknown_errors': 0
        }
        
        self.error_details = []
        
        # Setup logging
        self._setup_logging(log_level)
        
        # Error log file
        self.error_log_file = self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Debug mode logging
        if self.debug_mode:
            self.logger.debug("🔍 Debug mode activated in ErrorHandler")
            self.logger.debug(f"Log directory: {self.log_dir}")
            self.logger.debug(f"Log level: {log_level}")
            self.logger.debug(f"Error log file: {self.error_log_file}")
    
    def _setup_logging(self, log_level: int) -> None:
        """
        Setup logging configuration.
        
        Args:
            log_level: Logging level to use.
        """
        # Create logger
        self.logger = logging.getLogger('caption_extractor')
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Enhanced formatter for debug mode
        if self.debug_mode:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"caption_extractor_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def handle_error(self, 
                    error: Exception, 
                    context: Dict[str, Any], 
                    error_type: str = "unknown") -> None:
        """
        Handle and log an error with context information.
        
        Args:
            error: The exception that occurred.
            context: Context information about when/where the error occurred.
            error_type: Type of error (api, file, database, validation, unknown).
        """
        # Increment error count
        error_key = f"{error_type}_errors"
        if error_key in self.error_counts:
            self.error_counts[error_key] += 1
        else:
            self.error_counts['unknown_errors'] += 1
        
        # Create error record
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_class': error.__class__.__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context
        }
        
        # Add to error details
        self.error_details.append(error_record)
        
        # Log the error with enhanced debug information
        if self.debug_mode:
            self.logger.error(
                f"[{error_type.upper()}] {error.__class__.__name__}: {str(error)}"
            )
            self.logger.debug(f"Error context: {json.dumps(context, indent=2, default=str)}")
            self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        else:
            self.logger.error(
                f"[{error_type.upper()}] {error.__class__.__name__}: {str(error)} | Context: {context}"
            )
        
        # Save to error log file
        self._save_error_log(error_record)
    
    def handle_api_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """
        Handle API-related errors with improved categorization and logging.
        
        Args:
            error: The API exception.
            context: Additional context information about the error.
        """
        if context is None:
            context = {}
            
        error_message = str(error)
        error_category = context.get('error_category', 'api_communication')
        http_status = context.get('http_status')
        
        # Enhanced error detection and categorization
        is_server_error = (
            context.get('is_server_error', False) or
            error_category == 'api_server_error' or
            (http_status and 500 <= http_status < 600) or
            any(term in error_message.lower() for term in ['503', '500', 'unavailable', 'overloaded'])
        )
        
        is_rate_limit = (
            context.get('is_rate_limit', False) or
            error_category == 'api_rate_limit' or
            'rate limit' in error_message.lower() or
            (http_status == 429)
        )
        
        is_auth_error = (
            context.get('is_auth_error', False) or
            error_category == 'api_authentication' or
            'authentication' in error_message.lower() or
            'unauthorized' in error_message.lower() or
            (http_status == 401)
        )
        
        is_network_error = (
            context.get('is_network_error', False) or
            error_category == 'network_error' or
            'connection' in error_message.lower() or
            'network' in error_message.lower()
        )
        
        is_timeout_error = (
            context.get('is_timeout_error', False) or
            error_category == 'timeout_error' or
            'timeout' in error_message.lower()
        )
        
        # Update context with enhanced categorization
        context.update({
            'error_category': error_category,
            'is_server_error': is_server_error,
            'is_rate_limit': is_rate_limit,
            'is_auth_error': is_auth_error,
            'is_network_error': is_network_error,
            'is_timeout_error': is_timeout_error,
            'http_status': http_status
        })
        
        # Enhanced logging based on error type
        if is_server_error:
            self.logger.critical(f"Server error detected (HTTP {http_status}): {error_message}. This will stop processing.")
            self.consecutive_api_errors += 3  # Force processing stop
        elif is_rate_limit:
            self.logger.warning(f"Rate limit error detected (HTTP {http_status}): {error_message}. Consider reducing request frequency.")
            self.consecutive_api_errors += 1
        elif is_auth_error:
            self.logger.error(f"Authentication error detected (HTTP {http_status}): {error_message}. Check API credentials.")
            self.consecutive_api_errors += 2
        elif is_network_error:
            self.logger.warning(f"Network error detected: {error_message}. Check internet connection.")
            self.consecutive_api_errors += 1
        elif is_timeout_error:
            self.logger.warning(f"Timeout error detected: {error_message}. API response took too long.")
            self.consecutive_api_errors += 1
        else:
            self.logger.error(f"API communication error (HTTP {http_status}): {error_message}")
            self.consecutive_api_errors += 1
            
        self.handle_error(error, context, 'api')
    
    def handle_file_error(self, error: Exception, file_path: str, operation: str) -> None:
        """
        Handle file-related errors.
        
        Args:
            error: The file exception.
            file_path: Path to the file that caused the error.
            operation: The file operation being performed (read, write, delete, etc.).
        """
        context = {
            'file_path': file_path,
            'operation': operation,
            'file_exists': os.path.exists(file_path),
            'error_category': 'file_operation'
        }
        self.handle_error(error, context, 'file')
    
    def handle_database_error(self, error: Exception, operation: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle database-related errors.
        
        Args:
            error: The database exception.
            operation: The database operation being performed.
            data: Data involved in the operation (optional).
        """
        context = {
            'operation': operation,
            'data': data,
            'error_category': 'database_operation'
        }
        self.handle_error(error, context, 'database')
    
    def handle_validation_error(self, error: Exception, validation_type: str, data: Any) -> None:
        """
        Handle validation-related errors.
        
        Args:
            error: The validation exception.
            validation_type: Type of validation that failed.
            data: Data that failed validation.
        """
        context = {
            'validation_type': validation_type,
            'data': str(data)[:500],  # Limit data size in logs
            'error_category': 'data_validation'
        }
        self.handle_error(error, context, 'validation')
    
    def _save_error_log(self, error_record: Dict[str, Any]) -> None:
        """
        Save error record to the error log file.
        
        Args:
            error_record: Error record to save.
        """
        try:
            # Load existing errors if file exists
            if self.error_log_file.exists():
                with open(self.error_log_file, 'r', encoding='utf-8') as f:
                    errors = json.load(f)
            else:
                errors = []
            
            # Add new error
            errors.append(error_record)
            
            # Save back to file
            with open(self.error_log_file, 'w', encoding='utf-8') as f:
                json.dump(errors, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            # If we can't save the error log, at least log it to console
            self.logger.critical(f"Failed to save error log: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all errors encountered.
        
        Returns:
            Dict with error statistics and recent errors.
        """
        return {
            'error_counts': self.error_counts.copy(),
            'total_errors': sum(self.error_counts.values()),
            'recent_errors': self.error_details[-10:],  # Last 10 errors
            'error_log_file': str(self.error_log_file)
        }
    
    def get_error_report(self) -> str:
        """
        Generate a human-readable error report.
        
        Returns:
            Formatted string with error summary.
        """
        summary = self.get_error_summary()
        
        # Count specific error categories
        rate_limit_errors = sum(1 for e in self.error_details if e.get('context', {}).get('is_rate_limit', False))
        auth_errors = sum(1 for e in self.error_details if e.get('context', {}).get('is_auth_error', False))
        server_errors = sum(1 for e in self.error_details if e.get('context', {}).get('is_server_error', False))
        network_errors = sum(1 for e in self.error_details if e.get('context', {}).get('is_network_error', False))
        timeout_errors = sum(1 for e in self.error_details if e.get('context', {}).get('is_timeout_error', False))
        
        report = f"""
=== Error Report ===
Total Errors: {summary['total_errors']}
API Errors: {self.error_counts['api_errors']}
  - Rate Limit Errors: {rate_limit_errors}
  - Authentication Errors: {auth_errors}
  - Server Errors (5xx): {server_errors}
  - Network Errors: {network_errors}
  - Timeout Errors: {timeout_errors}
File Errors: {self.error_counts['file_errors']}
Database Errors: {self.error_counts['database_errors']}
Validation Errors: {self.error_counts['validation_errors']}
Unknown Errors: {self.error_counts['unknown_errors']}

Recent Errors:"""
        
        for i, error in enumerate(summary['recent_errors'][-5:], 1):
            report += f"""
{i}. [{error['error_type'].upper()}] {error['error_class']}: {error['error_message'][:100]}...
   Time: {error['timestamp']}
   Context: {error['context']}"""
        
        report += f"""

Detailed error log: {summary['error_log_file']}
=================="""
        
        return report
    
    def clear_errors(self) -> None:
        """
        Clear the current error tracking (not the log files).
        """
        self.error_counts = {
            'api_errors': 0,
            'file_errors': 0,
            'database_errors': 0,
            'validation_errors': 0,
            'unknown_errors': 0
        }
        self.error_details.clear()
        self.logger.info("Error tracking cleared - All error counters and details have been reset for new processing session")
    
    def should_continue_processing(self, max_consecutive_errors: int = 5) -> bool:
        """
        Determine if processing should continue based on error patterns.
        
        Args:
            max_consecutive_errors: Maximum consecutive errors before stopping.
            
        Returns:
            bool: True if processing should continue, False otherwise.
        """
        # Check for server errors (500, 503) that indicate the API is unavailable
        if self.error_details:  # If there are any errors
            latest_error = self.error_details[-1]  # Get the most recent error
            error_message = latest_error.get('error_message', '')
            error_context = latest_error.get('context', {})
            
            # Verificar si es un error de servidor en el contexto o en el mensaje
            is_server_error = error_context.get('is_server_error', False) or \
                             error_context.get('error_category') == 'api_server_error' or \
                             '503 UNAVAILABLE' in error_message or \
                             '500 ' in error_message or \
                             'unavailable' in error_message.lower() or \
                             'overloaded' in error_message.lower()
            
            if is_server_error:
                self.logger.critical(f"Detected server error: {error_message}. Stopping processing immediately.")
                return False
        
        # Check recent errors for consecutive failures
        if len(self.error_details) >= max_consecutive_errors:
            recent_errors = self.error_details[-max_consecutive_errors:]
            
            # If all recent errors are API errors, might be a temporary issue
            api_errors = sum(1 for e in recent_errors if e['error_type'] == 'api')
            if api_errors == max_consecutive_errors:
                self.logger.warning(f"Detected {max_consecutive_errors} consecutive API errors")
                return False
        
        # Check total error rate
        total_errors = sum(self.error_counts.values())
        if total_errors > 50:  # Arbitrary threshold
            self.logger.warning(f"High error count detected: {total_errors}")
            return False
        
        return True
    
    def log_processing_start(self, config: Dict[str, Any]) -> None:
        """
        Log the start of a processing session.
        
        Args:
            config: Processing configuration parameters.
        """
        self.logger.info("=== Starting Caption Extraction Session ===")
        self.clear_errors()
    
    def log_processing_end(self, stats: Dict[str, Any]) -> None:
        """
        Log the end of a processing session.
        
        Args:
            stats: Processing statistics.
        """
        self.logger.info("=== Caption Extraction Session Complete ===")
        
        if sum(self.error_counts.values()) > 0:
            self.logger.warning(f"Errors encountered: {sum(self.error_counts.values())} total")
        else:
            self.logger.info("No errors encountered during processing")
    
    def close(self) -> None:
        """
        Close all logging handlers to release file handles.
        This should be called when the ErrorHandler is no longer needed.
        """
        for handler in self.logger.handlers[:]:  # Create a copy of the list to avoid modification during iteration
            handler.close()
            self.logger.removeHandler(handler)
        self.logger.info("Logging handlers closed")
        logging.shutdown()