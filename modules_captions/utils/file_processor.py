import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re

try:
    from tqdm import tqdm
except ImportError:
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

                 checkpoint_dir: str = "checkpoints",
                 debug_mode: bool = False):
        """
        Initialize the file processor.
        
        Args:
            root_directory: Root directory containing image files to process.
            db_manager: Database manager for storing descriptions.
            ai_client: AI client for generating descriptions.
            checkpoint_dir: Directory for storing checkpoint files.
            debug_mode: Enable debug mode with enhanced logging.
        """
        self.root_directory = Path(root_directory)
        self.db_manager = db_manager
        self.ai_client = ai_client
        self.checkpoint_dir = Path(checkpoint_dir)
        self.commit_interval = 15  # Commit every 15 images
        self.debug_mode = debug_mode
        
        # Create checkpoint directory and parent directories if they don't exist
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
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
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
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
        
        self.logger.info(f"Scanning for images in {self.root_directory}")
        
        for image_path in self.root_directory.rglob('*'):
            if image_path.suffix.lower() in self.image_extensions:
                # Extract document and page information from path
                document_name, page_number = self._extract_document_info(image_path)
                image_filename = image_path.name
                
                # Check if description already exists
                if skip_existing and self.db_manager.description_exists(
                    document_name, page_number, image_filename
                ):
                    self.stats['total_skipped'] += 1
                    continue
                
                images_to_process.append((
                    document_name, 
                    str(image_path), 
                    page_number, 
                    image_filename
                ))
        
        self.stats['total_found'] = len(images_to_process)
        self.logger.info(f"Found {len(images_to_process)} images to process")
        
        return images_to_process
    
    def process_images(self, 
                      images: Optional[List[Tuple[str, str, int, str]]] = None,
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
        self.logger.info(f"🚀 Starting sequential processing of {len(images)} images (commit every {self.commit_interval} images)")
        
        if self.debug_mode:
            self.logger.debug(f"Processing configuration:")
            self.logger.debug(f"  - Total images: {len(images)}")
            self.logger.debug(f"  - Commit interval: {self.commit_interval}")
            self.logger.debug(f"  - AI client: {type(self.ai_client).__name__}")
            self.logger.debug(f"  - Database path: {self.db_manager.db_path if hasattr(self.db_manager, 'db_path') else 'Unknown'}")
            
            # Log first few images for debugging
            sample_images = images[:3] if len(images) > 3 else images
            self.logger.debug(f"Sample images to process:")
            for i, (doc, path, page, filename) in enumerate(sample_images):
                self.logger.debug(f"  {i+1}. {filename} (doc: {doc}, page: {page})")
            if len(images) > 3:
                self.logger.debug(f"  ... and {len(images) - 3} more images")
        
        # Check for existing checkpoint
        checkpoint = self.load_checkpoint()
        start_index = 0
        
        if checkpoint and not force_reprocess:
            processed_count = checkpoint.get('processed_count', 0)
            total_count = checkpoint.get('total_count', 0)
            
            # Validate checkpoint against current image list
            if total_count == len(images) and processed_count < len(images):
                start_index = processed_count
                self.logger.info(f"📂 Resuming from checkpoint: {processed_count}/{total_count} images already processed")
                
                # Restore stats from checkpoint if available
                if 'stats' in checkpoint:
                    saved_stats = checkpoint['stats']
                    self.stats['total_processed'] = saved_stats.get('total_processed', 0)
                    self.stats['total_errors'] = saved_stats.get('total_errors', 0)
            else:
                self.logger.info("🔄 Checkpoint found but invalid (different image count), starting fresh")
                self._cleanup_checkpoint()
        
        # Process images sequentially with progress bar
        pending_descriptions = []
        
        # Initialize progress bar with current progress
        with tqdm(total=len(images), desc="🖼️  Processing images", unit="img", 
                  initial=start_index,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}") as pbar:
            
            for i, (document_name, image_path, page_number, image_filename) in enumerate(images[start_index:], start=start_index):
                if self.interrupted:
                    pbar.set_description("⚠️  Processing interrupted")
                    self.logger.info("Processing interrupted by user")
                    break
                
                try:
                    # Update progress bar description with current file
                    pbar.set_description(f"🖼️  Processing {image_filename[:30]}...")
                    
                    # Add rate limiting info to progress bar
                    rate_info = ""
                    if hasattr(self.ai_client, 'rate_limit_enabled') and self.ai_client.rate_limit_enabled:
                        current_requests = len(getattr(self.ai_client, 'request_timestamps', []))
                        max_requests = getattr(self.ai_client, 'requests_per_minute', 0)
                        rate_info = f"Rate: {current_requests}/{max_requests}/min"
                        pbar.set_postfix_str(rate_info)
                    
                    # Record start time for this image
                    image_start_time = time.time()
                    
                    # Generate description
                    description = self.ai_client.describe(image_path)
                    
                    # Calculate processing time for this image
                    image_processing_time = time.time() - image_start_time
                    
                    if description:
                        # Add to pending descriptions
                        pending_descriptions.append((document_name, page_number, image_filename, description))
                        
                        self.stats['total_processed'] += 1
                        pbar.set_postfix_str(f"{rate_info} ✅ Success")
                        
                        # Enhanced success logging with timing
                        self.logger.info(f"✓ Completed {image_filename} in {image_processing_time:.2f}s")
                        self.logger.debug(f"Description length: {len(description)} characters")
                    else:
                        self.stats['total_errors'] += 1
                        pbar.set_postfix_str(f"{rate_info} ⚠️ Failed")
                        self.logger.warning(f"⚠️  No description generated for {image_filename}")
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Commit every commit_interval images or at the end
                    if (i + 1) % self.commit_interval == 0 or (i + 1) == len(images):
                        self._commit_descriptions(pending_descriptions)
                        pending_descriptions = []
                        pbar.set_postfix_str(f"{rate_info} 💾 Saved batch")
                        self.logger.info(f"📊 Committed descriptions for {i + 1} images")
                        
                        # Save checkpoint
                        self._save_checkpoint(i + 1, len(images))
                    
                except Exception as e:
                    self.stats['total_errors'] += 1
                    
                    # Update progress bar with error
                    pbar.set_postfix_str(f"❌ Error: {str(e)[:30]}...")
                    pbar.update(1)
                    
                    # Enhanced error logging with progress context
                    self.logger.error(f"❌ Error processing {image_filename} ({i+1}/{len(images)}): {str(e)}")
                    self.logger.debug(f"Error details - Document: {document_name}, Page: {page_number}")
                    self.logger.debug(f"Error details - Full path: {image_path}")
                    
                    # Check for critical server errors
                    error_handler = getattr(self.ai_client, 'error_handler', None)
                    if error_handler and ('UNAVAILABLE' in str(e) or '503' in str(e) or '500' in str(e) or 'overloaded' in str(e).lower()):
                        self.logger.critical(f"🚨 Detected server error: {str(e)}. Stopping processing.")
                        pbar.set_description("🚨 Server error - stopping")
                        # Commit any pending descriptions before stopping
                        if pending_descriptions:
                            self._commit_descriptions(pending_descriptions)
                        self.interrupted = True
                        break
            
            # Final update
            if not self.interrupted:
                pbar.set_description("🎉 Processing completed")
                pbar.set_postfix_str(f"✅ {self.stats['total_processed']} processed")
        
        # Final commit for any remaining descriptions
        if pending_descriptions and not self.interrupted:
            self._commit_descriptions(pending_descriptions)
        
        # Final statistics
        self.stats['end_time'] = time.time()
        self.stats['total_time'] = self.stats['end_time'] - self.stats['start_time']
        
        # Clean up checkpoint file if processing completed successfully
        if not self.interrupted:
            self._cleanup_checkpoint()
            
        # Enhanced completion logging
        success_rate = (self.stats['total_processed'] / len(images) * 100) if len(images) > 0 else 0
        avg_time_per_image = self.stats['total_time'] / self.stats['total_processed'] if self.stats['total_processed'] > 0 else 0
        
        self.logger.info(f"🎉 Processing completed!")
        self.logger.info(f"📊 Summary: {self.stats['total_processed']}/{len(images)} images processed ({success_rate:.1f}% success rate)")
        self.logger.info(f"⏱️  Total time: {self.stats['total_time']:.2f}s (avg: {avg_time_per_image:.2f}s per image)")
        
        if self.stats['total_errors'] > 0:
            self.logger.warning(f"⚠️  Errors encountered: {self.stats['total_errors']}")
        
        if self.debug_mode:
            self.logger.debug(f"🔍 Debug summary:")
            self.logger.debug(f"  - Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.stats['start_time']))}")
            self.logger.debug(f"  - End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.stats['end_time']))}")
            self.logger.debug(f"  - Commit interval: {self.commit_interval} images")
            self.logger.debug(f"  - Interrupted: {self.interrupted}")
            if hasattr(self.ai_client, 'model'):
                self.logger.debug(f"  - AI model used: {self.ai_client.model}")
            
            # Log processing rate statistics
            if self.stats['total_time'] > 0:
                images_per_minute = (self.stats['total_processed'] / self.stats['total_time']) * 60
                self.logger.debug(f"  - Processing rate: {images_per_minute:.1f} images/minute")
                
            # Log memory and performance info if available
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.logger.debug(f"  - Memory usage: {memory_mb:.1f} MB")
            except ImportError:
                pass
        
        return self._get_final_stats()
    
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
            self.logger.info(f"Committed {len(descriptions)} descriptions to database (commit #{self.stats['commits_made']})")
            
        except Exception as e:
            self.logger.error(f"Failed to commit {len(descriptions)} descriptions: {str(e)}")
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