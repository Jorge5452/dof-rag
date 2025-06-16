import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback if tqdm is not available
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None, **kwargs):
            self.iterable = iterable or []
            self.total = total or len(self.iterable) if hasattr(self.iterable, '__len__') else 0
            self.desc = desc or ''
            self.n = 0
            
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)
                
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
            
        def update(self, n=1):
            self.n += n
            
        def set_description(self, desc):
            self.desc = desc
            
        def close(self):
            pass

# Import colorama for better visual output
try:
    import colorama
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    class MockColor:
        def __getattr__(self, name):
            return ''
    Fore = Style = MockColor()
    COLORAMA_AVAILABLE = False

try:
    from ..db.manager import DatabaseManager
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from db.manager import DatabaseManager

class FileProcessor:
    """
    File processor for batch image description generation with database storage.
    
    This class handles the discovery, processing, and management of image files
    for caption extraction. It replaces the file-based storage approach with
    SQLite database integration and provides improved error handling and
    checkpoint management.
    """
    
    def __init__(self, 
                 root_directory: str,
                 db_manager: DatabaseManager,
                 ai_client,
                 log_dir: str = "logs",
                 commit_interval: int = 10,
                 cooldown_seconds: int = 0,
                 debug_mode: bool = False):
        """
        Initialize the file processor.
        
        Args:
            root_directory: Root directory containing image files to process.
            db_manager: Database manager for storing descriptions.
            ai_client: AI client for generating descriptions.
            log_dir: Directory for storing log files.
            commit_interval: Number of images to process before committing to database.
            cooldown_seconds: Seconds to wait between processing batches.
            debug_mode: Enable debug mode with enhanced logging.
        """
        self.root_directory = Path(root_directory)
        self.db_manager = db_manager
        self.ai_client = ai_client
        self.log_dir = Path(log_dir)
        self.commit_interval = commit_interval
        self.cooldown_seconds = cooldown_seconds
        self.debug_mode = debug_mode
        
        # Create necessary directories
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Checkpoint files in logs directory
        self.completed_checkpoints_file = self.log_dir / "completed_directories.json"
        self.pending_checkpoint_file = self.log_dir / "pending_directory.json"
        self.error_images_file = self.log_dir / "error_images.json"
        
        # Processing statistics
        self.stats = {
            'total_found': 0,
            'total_processed': 0,
            'total_skipped': 0,
            'total_errors': 0,
            'commits_made': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Interruption flag
        self.interrupted = False
        
        # Logger - use the same logger as the main application
        self.logger = logging.getLogger('caption_extractor')
        
        # Supported image extensions
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
        
        # Debug mode initialization
        if self.debug_mode:
            self.logger.debug("🔍 Debug mode activated in FileProcessor")
            self.logger.debug(f"Root directory: {self.root_directory}")
            self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")
            self.logger.debug(f"Commit interval: {self.commit_interval} images")
            self.logger.debug(f"Supported extensions: {self.image_extensions}")
    
    def find_images(self, skip_existing: bool = True) -> List[Tuple[str, str, int, str]]:
        """
        Find all image files in the root directory that need processing.
        
        Args:
            skip_existing: If True, skip images that already have descriptions in the database.
            
        Returns:
            List of tuples (document_name, image_path, page_number, image_filename).
        """
        images_to_process = []
        completed_dirs = self._load_completed_directories()
        error_images = self._load_error_images()
        
        self.logger.info(f"Scanning for images in {self.root_directory}")
        
        # Group images by directory
        directories_with_images = {}
        
        for image_path in self.root_directory.rglob('*'):
            if image_path.suffix.lower() in self.image_extensions:
                parent_dir = str(image_path.parent.relative_to(self.root_directory))
                
                # Skip if directory is already completed
                if parent_dir in completed_dirs:
                    self.stats['total_skipped'] += 1
                    continue
                
                # Skip if image had errors
                image_key = f"{parent_dir}/{image_path.name}"
                if image_key in error_images:
                    self.stats['total_skipped'] += 1
                    continue
                
                # Extract document and page information from path
                document_name, page_number = self._extract_document_info(image_path)
                image_filename = image_path.name
                
                # Check if description already exists
                if skip_existing and self.db_manager.description_exists(
                    document_name, page_number, image_filename
                ):
                    self.stats['total_skipped'] += 1
                    continue
                
                if parent_dir not in directories_with_images:
                    directories_with_images[parent_dir] = []
                
                directories_with_images[parent_dir].append((
                    document_name, 
                    str(image_path), 
                    page_number, 
                    image_filename,
                    parent_dir
                ))
        
        # Flatten the dictionary to a list, prioritizing pending directory
        pending_dir = self._load_pending_directory()
        if pending_dir and pending_dir in directories_with_images:
            # Process pending directory first
            images_to_process.extend(directories_with_images[pending_dir])
            del directories_with_images[pending_dir]
        
        # Add remaining directories
        for dir_images in directories_with_images.values():
            images_to_process.extend(dir_images)
        
        self.stats['total_found'] = len(images_to_process)
        self.logger.info(f"Found {len(images_to_process)} images to process across {len(directories_with_images) + (1 if pending_dir else 0)} directories")
        
        if self.debug_mode:
            self.logger.debug(f"🔍 Total files scanned: {self.stats['total_found'] + self.stats['total_skipped']}")
            self.logger.debug(f"🔍 Directories with images: {list(directories_with_images.keys())}")
            self.logger.debug(f"🔍 Completed directories: {completed_dirs}")
            self.logger.debug(f"🔍 Error images: {len(error_images)}")
        
        return images_to_process
    
    def process_images(self, 
                      images: Optional[List[Tuple[str, str, int, str, str]]] = None,
                      force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process a list of images to generate descriptions.
        
        Args:
            images: List of image tuples to process. If None, will find images automatically.
            force_reprocess: If True, reprocess images even if descriptions exist.
            
        Returns:
            Dict with processing results and statistics.
        """
        if images is None:
            images = self.find_images(skip_existing=not force_reprocess)
        else:
            # Update total_found when images are provided directly
            self.stats['total_found'] = len(images)
        
        if not images:
            self.logger.info("No images to process")
            return self._get_final_stats()
        
        self.stats['start_time'] = time.time()
        self.logger.info(f"🚀 Starting directory-based processing of {len(images)} images (commit every {self.commit_interval} images)")
        
        if self.debug_mode:
            self.logger.debug(f"Processing configuration:")
            self.logger.debug(f"  - Total images: {len(images)}")
            self.logger.debug(f"  - Commit interval: {self.commit_interval}")
            self.logger.debug(f"  - AI client: {type(self.ai_client).__name__}")
            self.logger.debug(f"  - Database path: {self.db_manager.db_path if hasattr(self.db_manager, 'db_path') else 'Unknown'}")
            
            # Log first few images for debugging
            sample_images = images[:3] if len(images) > 3 else images
            self.logger.debug(f"Sample images to process:")
            for i, (doc, path, page, filename, parent_dir) in enumerate(sample_images):
                self.logger.debug(f"  {i+1}. {filename} (doc: {doc}, page: {page}, dir: {parent_dir})")
            if len(images) > 3:
                self.logger.debug(f"  ... and {len(images) - 3} more images")
        
        # Group images by directory for processing
        images_by_directory = {}
        for image_tuple in images:
            parent_dir = image_tuple[4]  # parent_dir is the 5th element
            if parent_dir not in images_by_directory:
                images_by_directory[parent_dir] = []
            images_by_directory[parent_dir].append(image_tuple)
        
        # Process directories one by one
        for current_dir, dir_images in images_by_directory.items():
            if self.interrupted:
                break
                
            # Enhanced directory processing logs
            dir_start_msg = f"{Fore.BLUE}📂 Starting directory: {current_dir}{Style.RESET_ALL}"
            dir_info_msg = f"{Fore.CYAN}   └─ Found {len(dir_images)} images to process{Style.RESET_ALL}"
            
            self.logger.info(f"\n{dir_start_msg}")
            self.logger.info(dir_info_msg)
            self._save_pending_directory(current_dir)
            
            success = self._process_directory_images(dir_images)
            
            if success and not self.interrupted:
                # Mark directory as completed
                self._mark_directory_completed(current_dir)
                self._clear_pending_directory()
                
                # Enhanced completion message
                completion_msg = f"{Fore.GREEN}✅ Directory completed: {current_dir}{Style.RESET_ALL}"
                stats_msg = f"{Fore.CYAN}   └─ Processed: {len(dir_images)} images | Errors: {self.stats.get('directory_errors', 0)}{Style.RESET_ALL}"
                
                self.logger.info(f"\n{completion_msg}")
                self.logger.info(stats_msg)
            elif self.interrupted:
                interrupt_msg = f"{Fore.YELLOW}⚠️ Processing interrupted in directory: {current_dir}{Style.RESET_ALL}"
                self.logger.warning(f"\n{interrupt_msg}")
                break
        
        self.stats['end_time'] = time.time()
        
        # Clean up checkpoint if processing completed successfully
        if not self.interrupted:
            self._cleanup_checkpoint()
        
        return self._get_final_stats()
    
    def _process_directory_images(self, dir_images: List[Tuple[str, str, int, str, str]]) -> bool:
        """
        Process all images in a specific directory.
        
        Args:
            dir_images: List of image tuples for the directory.
            
        Returns:
            True if all images were processed successfully, False otherwise.
        """
        pending_descriptions = []
        directory_success = True
        
        # Configure tqdm for better visual output
        tqdm_config = {
            'total': len(dir_images),
            'desc': f"{Fore.CYAN}🔄 Processing {dir_images[0][4] if dir_images else 'directory'}{Style.RESET_ALL}",
            'unit': "img",
            'ncols': 100,
            'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            'leave': True,
            'position': 0
        }
        
        if TQDM_AVAILABLE:
            # Use tqdm with proper configuration to avoid log interference
            tqdm_config['file'] = None  # Use stdout
            tqdm_config['dynamic_ncols'] = True
        
        with tqdm(**tqdm_config) as pbar:
            for i, (document_name, image_path, page_number, image_filename, parent_dir) in enumerate(dir_images):
                if self.interrupted:
                    directory_success = False
                    break
                
                try:
                    # Update progress bar description with current file
                    pbar.set_description(f"{Fore.CYAN}🔄 Processing {parent_dir}{Style.RESET_ALL} - {image_filename[:20]}...")
                    
                    # Generate description
                    description = self.ai_client.describe(image_path)
                    
                    if description:
                        pending_descriptions.append((document_name, page_number, image_filename, description))
                        self.stats['total_processed'] += 1
                        self.logger.debug(f"Successfully processed {image_filename}")
                    else:
                        raise ValueError("Empty description returned")
                        
                except Exception as e:
                    self.stats['total_errors'] += 1
                    error_msg = str(e)
                    
                    # Temporarily disable tqdm to show error clearly
                    pbar.clear()
                    self.logger.error(f"{Fore.RED}❌ Error processing {image_filename}: {error_msg}{Style.RESET_ALL}")
                    pbar.refresh()
                    
                    # Save error image
                    self._save_error_image(parent_dir, image_filename, error_msg)
                    
                    # Check for critical server errors (400, 500, 503)
                    if any(code in error_msg for code in ['400', '500', '503', 'UNAVAILABLE', 'overloaded']):
                        pbar.clear()
                        self.logger.critical(f"{Fore.RED}🚨 Detected server error: {error_msg}. Stopping directory processing.{Style.RESET_ALL}")
                        directory_success = False
                        self.interrupted = True
                        break
                
                pbar.update(1)
                
                # Commit batch if needed
                if len(pending_descriptions) >= self.commit_interval:
                    # Temporarily clear progress bar for clean commit message
                    pbar.clear()
                    self._commit_descriptions(pending_descriptions)
                    pbar.refresh()
                    pending_descriptions = []
        
        # Commit remaining descriptions
        if pending_descriptions:
            self._commit_descriptions(pending_descriptions)
        
        return directory_success and not self.interrupted
    
    # Checkpoint management methods for directory-based processing
    def _load_completed_directories(self) -> set:
        """Load the set of completed directories."""
        if not self.completed_checkpoints_file.exists():
            return set()
        
        try:
            with open(self.completed_checkpoints_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('completed_directories', []))
        except Exception as e:
            self.logger.error(f"Error loading completed directories: {e}")
            return set()
    
    def _mark_directory_completed(self, directory: str) -> None:
        """Mark a directory as completed."""
        completed_dirs = self._load_completed_directories()
        completed_dirs.add(directory)
        
        data = {
            'completed_directories': list(completed_dirs),
            'last_updated': time.time()
        }
        
        try:
            with open(self.completed_checkpoints_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Marked directory as completed: {directory}")
        except Exception as e:
            self.logger.error(f"Error marking directory as completed: {e}")
    
    def _load_pending_directory(self) -> Optional[str]:
        """Load the currently pending directory."""
        if not self.pending_checkpoint_file.exists():
            return None
        
        try:
            with open(self.pending_checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('pending_directory')
        except Exception as e:
            self.logger.error(f"Error loading pending directory: {e}")
            return None
    
    def _save_pending_directory(self, directory: str) -> None:
        """Save the currently processing directory as pending."""
        data = {
            'pending_directory': directory,
            'timestamp': time.time()
        }
        
        try:
            with open(self.pending_checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved pending directory: {directory}")
        except Exception as e:
            self.logger.error(f"Error saving pending directory: {e}")
    
    def _clear_pending_directory(self) -> None:
        """Clear the pending directory file."""
        try:
            if self.pending_checkpoint_file.exists():
                self.pending_checkpoint_file.unlink()
                self.logger.debug("Cleared pending directory")
        except Exception as e:
            self.logger.error(f"Error clearing pending directory: {e}")
    
    def _load_error_images(self) -> dict:
        """Load the dictionary of images that had errors."""
        if not self.error_images_file.exists():
            return {}
        
        try:
            with open(self.error_images_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading error images: {e}")
            return {}
    
    def _save_error_image(self, directory: str, filename: str, error_msg: str) -> None:
        """Save an image that had an error."""
        error_images = self._load_error_images()
        image_key = f"{directory}/{filename}"
        
        error_images[image_key] = {
            'error_message': error_msg,
            'timestamp': time.time(),
            'directory': directory,
            'filename': filename
        }
        
        try:
            with open(self.error_images_file, 'w', encoding='utf-8') as f:
                json.dump(error_images, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved error image: {image_key}")
        except Exception as e:
            self.logger.error(f"Error saving error image: {e}")
    
    def _commit_descriptions(self, descriptions: List[Tuple[str, int, str, str]]) -> None:
        """
        Commit a batch of descriptions to the database.
        
        Args:
            descriptions: List of description tuples to commit.
        """
        if not descriptions:
            return
            
        try:
            inserted_count = self.db_manager.batch_insert_descriptions(descriptions)
            if inserted_count != len(descriptions):
                self.logger.warning(f"Expected to insert {len(descriptions)} descriptions, but only {inserted_count} were inserted")
                self.stats['total_errors'] += len(descriptions) - inserted_count
            
            self.stats['commits_made'] += 1
            
            # Enhanced commit message with better formatting
            commit_msg = f"{Fore.GREEN}💾 Committed {len(descriptions)} descriptions to database{Style.RESET_ALL}"
            commit_details = f"{Fore.CYAN}   └─ Commit #{self.stats['commits_made']} | Total processed: {self.stats['total_processed']}{Style.RESET_ALL}"
            
            self.logger.info(f"\n{commit_msg}")
            self.logger.info(commit_details)
            
        except Exception as e:
            error_msg = f"{Fore.RED}❌ Failed to commit {len(descriptions)} descriptions: {str(e)}{Style.RESET_ALL}"
            self.logger.error(error_msg)
            self.stats['total_errors'] += len(descriptions)
    
    def _process_batch(self, batch: List[Tuple[str, str, int, str]]) -> None:
        """
        Process a single batch of images (deprecated method, kept for compatibility).
        
        Args:
            batch: List of image tuples to process.
        """
        self.logger.warning("_process_batch method is deprecated. Using sequential processing instead.")
        
        # Process each image in the batch sequentially
        pending_descriptions = []
        
        for document_name, image_path, page_number, image_filename in batch:
            if self.interrupted:
                break
            
            try:
                self.logger.debug(f"Processing {image_filename}")
                
                # Generate description
                description = self.ai_client.describe(image_path)
                
                # Add to pending descriptions
                pending_descriptions.append((document_name, page_number, image_filename, description))
                
                self.stats['total_processed'] += 1
                self.logger.debug(f"Successfully processed {image_filename}")
                
            except Exception as e:
                self.stats['total_errors'] += 1
                self.logger.error(f"Error processing {image_filename}: {str(e)}")
                
                # Check for critical server errors
                error_handler = getattr(self.ai_client, 'error_handler', None)
                if error_handler and ('UNAVAILABLE' in str(e) or '503' in str(e) or '500' in str(e) or 'overloaded' in str(e).lower()):
                    self.logger.critical(f"Detected server error: {str(e)}. Stopping processing.")
                    self.interrupted = True
                    break
        
        # Commit all descriptions from this batch
        if pending_descriptions:
            self._commit_descriptions(pending_descriptions)
    
    def _extract_document_info(self, image_path: Path) -> Tuple[str, int]:
        """
        Extract document name and page number from image path.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Tuple of (document_name, page_number).
        """
        # Try to extract page number from filename
        filename = image_path.stem
        page_match = re.search(r'page[_-]?(\d+)', filename, re.IGNORECASE)
        page_number = int(page_match.group(1)) if page_match else 0
        
        # Use parent directory as document name, or filename if no clear structure
        if image_path.parent != self.root_directory:
            document_name = image_path.parent.name
        else:
            # Extract document name from filename (remove page info)
            document_name = re.sub(r'[_-]?page[_-]?\d+.*$', '', filename, flags=re.IGNORECASE)
            if not document_name:
                document_name = filename
        
        return document_name, page_number
    
    def _cooldown(self) -> None:
        """
        Wait for the cooldown period between batches.
        """
        if self.cooldown_seconds <= 0:
            return
        
        self.logger.info(f"Cooling down for {self.cooldown_seconds} seconds...")
        
        # Interruptible sleep
        for _ in range(self.cooldown_seconds):
            if self.interrupted:
                break
            time.sleep(1)
    
    def _save_checkpoint(self, processed_count: int, total_count: int) -> None:
        """
        Save processing checkpoint.
        
        Args:
            processed_count: Number of images processed so far.
            total_count: Total number of images to process.
        """
        checkpoint_data = {
            'processed_count': processed_count,
            'total_count': total_count,
            'stats': self.stats.copy(),
            'timestamp': time.time()
        }
        
        checkpoint_file = self.checkpoint_dir / 'processing_checkpoint.json'
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Checkpoint saved: {processed_count}/{total_count} processed")
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the last processing checkpoint.
        
        Returns:
            Dict with checkpoint data if available, None otherwise.
        """
        checkpoint_file = self.checkpoint_dir / 'processing_checkpoint.json'
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def interrupt(self) -> None:
        """
        Signal the processor to stop at the next safe point.
        """
        self.interrupted = True
        self.logger.info("Interrupt signal received")
    
    def _cleanup_checkpoint(self) -> None:
        """
        Remove checkpoint file after successful completion.
        """
        checkpoint_file = self.checkpoint_dir / 'processing_checkpoint.json'
        
        try:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                self.logger.debug("Checkpoint file cleaned up")
        except Exception as e:
            self.logger.warning(f"Could not clean up checkpoint file: {e}")
    
    def _get_final_stats(self) -> Dict[str, Any]:
        """
        Calculate and return final processing statistics.
        
        Returns:
            Dictionary containing processing statistics.
        """
        if self.stats['start_time'] and self.stats['end_time']:
            total_time = self.stats['end_time'] - self.stats['start_time']
        else:
            total_time = 0
        
        return {
            'total_images': self.stats['total_found'],
            'total_processed': self.stats['total_processed'],
            'total_skipped': self.stats['total_skipped'],
            'total_errors': self.stats['total_errors'],
            'commits_made': self.stats.get('commits_made', 0),
            'total_time_seconds': round(total_time, 2),
            'average_time_per_image': round(total_time / max(self.stats['total_processed'], 1), 2),
            'success_rate': round((self.stats['total_processed'] / max(self.stats['total_found'], 1)) * 100, 2),
            'interrupted': self.interrupted
        }
    
    def get_processing_summary(self) -> str:
        """
        Get a human-readable summary of processing results.
        
        Returns:
            Formatted string with processing summary.
        """
        stats = self._get_final_stats()
        
        summary = f"""
=== Processing Summary ===
Images found: {stats['total_images']}
Images processed: {stats['total_processed']}
Images skipped: {stats['total_skipped']}
Errors: {stats['total_errors']}
Commits made: {stats['commits_made']}
Total time: {stats['total_time_seconds']:.2f} seconds
Average time per image: {stats['average_time_per_image']:.2f} seconds
Success rate: {stats['success_rate']:.1f}%
Interrupted: {stats['interrupted']}
========================"""
        
        return summary