"""Chunk embeddings visualizer using t-SNE.

This module provides tools to visualize chunk embeddings generated from
Markdown document processing using t-SNE (t-Distributed Stochastic Neighbor
Embedding), a dimensionality reduction algorithm. It generates 2D and 3D plots
that allow visualization of how embeddings are distributed in space and helps
identify semantic patterns and groups.
"""

import os
import logging
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from ..session_manager.session_manager import SessionManager

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

# Configure matplotlib for server environments without display
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI dependencies
plt.ioff()  # Disable interactive plotting to prevent blocking operations

# Limit parallel processing to prevent resource exhaustion
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Restrict joblib to 4 cores to avoid memory issues

# Configure logging
logger = logging.getLogger(__name__)

# Constants for optimization
CONSTANTS = {
    'MIN_CHUNKS_NEEDED': 2,
    'DEFAULT_LIMIT': 1000,
    'DEFAULT_DPI': 150,
    'RANDOM_STATE': 42,
    'FIGURE_SIZE_2D': (12, 10),
    'FIGURE_SIZE_3D': (14, 12),
    'SMALL_CHUNK_THRESHOLD': 5,
    'MEDIUM_CHUNK_THRESHOLD': 10,
    'LARGE_CHUNK_THRESHOLD': 20,
    'DERIVED_3D_THRESHOLD': 15
}

class TSNEVisualizer:
    """
    Class for visualizing chunk embeddings using t-SNE.
    
    Generates 2D and 3D visualizations of chunk embeddings
    to facilitate analysis and pattern identification.
    """
    
    def __init__(self, db_instance):
        """
        Initializes the visualizer with a database instance.
        
        Args:
            db_instance: Vector database instance
        """
        self.db = db_instance
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def generate_tsne_embeddings(self, document_id: int, perplexity: float = 30.0) -> Optional[Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray, np.ndarray]]:
        """
        Generates 2D and 3D t-SNE embeddings for a specific document.
        
        Args:
            document_id: Document ID in the database
            perplexity: Perplexity parameter for t-SNE (recommended value: 5-50)
            
        Returns:
            Tuple with (original embeddings, chunk list, 2D embeddings, 3D embeddings)
            Returns None if there are not enough chunks with embeddings
        """
        try:
            # Retrieve all text chunks for the specified document from database
            chunks = self.db.get_chunks_by_document(document_id, 0, CONSTANTS['DEFAULT_LIMIT'])
            
            # Filter out chunks without embedding vectors (required for t-SNE)
            chunks_with_embeddings = [
                chunk for chunk in chunks 
                if chunk.get("embedding") is not None
            ]
            
            # Verify there are enough chunks for t-SNE
            n_chunks = int(len(chunks_with_embeddings))
            if n_chunks < CONSTANTS['MIN_CHUNKS_NEEDED']:
                self.logger.warning(
                    f"Not enough chunks with embeddings for document {document_id}. "
                    f"At least {CONSTANTS['MIN_CHUNKS_NEEDED']} chunks are needed."
                )
                return None
            
            # Convert embedding lists to numpy array and apply L2 normalization
            embeddings = np.array([chunk["embedding"] for chunk in chunks_with_embeddings])
            normalized_embeddings = normalize(embeddings, axis=1, norm='l2')
            
            # Calculate adaptive t-SNE parameters based on dataset size
            tsne_params = self._get_tsne_parameters(n_chunks, perplexity)
            
            # Generate 2D t-SNE embeddings
            tsne_2d = TSNE(
                n_components=2,
                perplexity=tsne_params['perplexity'],
                max_iter=tsne_params['n_iter'],
                early_exaggeration=tsne_params['early_exaggeration'],
                random_state=CONSTANTS['RANDOM_STATE'],
                n_jobs=1  # Explicit single-threaded for consistency
            )
            tsne_embeddings_2d = tsne_2d.fit_transform(normalized_embeddings)
            
            # Generate 3D embeddings using hybrid approach (derived or computed)
            tsne_embeddings_3d = self._generate_3d_embeddings(
                normalized_embeddings, tsne_embeddings_2d, n_chunks, tsne_params
            )
            
            self.logger.info(
                f"Generated t-SNE embeddings for {n_chunks} chunks "
                f"with perplexity={tsne_params['perplexity']:.2f}"
            )
            return normalized_embeddings, chunks_with_embeddings, tsne_embeddings_2d, tsne_embeddings_3d
            
        except Exception as e:
            self.logger.error(f"Error generating t-SNE embeddings: {e}")
            return None
    
    def _get_tsne_parameters(self, n_chunks: int, perplexity: float) -> Dict[str, float]:
        """
        Get optimized t-SNE parameters based on the number of chunks.
        
        Args:
            n_chunks: Number of chunks
            perplexity: Base perplexity value
            
        Returns:
            Dictionary with t-SNE parameters
        """
        # t-SNE requires perplexity < n_samples, adjust parameters by dataset size
        n_chunks = int(n_chunks)  # Convert to integer for threshold comparisons
        if n_chunks <= CONSTANTS['SMALL_CHUNK_THRESHOLD']:
            return {
                'perplexity': max(1.0, n_chunks / 3),
                'n_iter': 250,
                'early_exaggeration': 6.0
            }
        elif n_chunks <= CONSTANTS['MEDIUM_CHUNK_THRESHOLD']:
            return {
                'perplexity': max(2.0, n_chunks / 3),
                'n_iter': 300,
                'early_exaggeration': 8.0
            }
        elif n_chunks <= CONSTANTS['LARGE_CHUNK_THRESHOLD']:
            return {
                'perplexity': min(perplexity, n_chunks - 1),
                'n_iter': 400,
                'early_exaggeration': 10.0
            }
        else:
            return {
                'perplexity': min(perplexity, n_chunks - 1),
                'n_iter': 500,
                'early_exaggeration': 12.0
            }
    
    def _generate_3d_embeddings(self, normalized_embeddings: np.ndarray, 
                               tsne_embeddings_2d: np.ndarray, n_chunks: int, 
                               tsne_params: Dict[str, float]) -> np.ndarray:
        """
        Generate 3D t-SNE embeddings using optimized approach.
        
        Args:
            normalized_embeddings: Normalized input embeddings
            tsne_embeddings_2d: 2D t-SNE embeddings
            n_chunks: Number of chunks
            tsne_params: t-SNE parameters
            
        Returns:
            3D t-SNE embeddings
        """
        np.random.seed(CONSTANTS['RANDOM_STATE'])  # Ensure reproducibility
        n_chunks = int(n_chunks)  # Ensure n_chunks is integer
        
        if n_chunks >= 3 and n_chunks <= CONSTANTS['DERIVED_3D_THRESHOLD']:
            # For datasets ≤15 chunks: create pseudo-3D by adding structured Z-axis
            z_component = np.random.normal(0, 0.1, size=n_chunks)
            # Correlate Z-axis with 2D position to maintain some spatial relationship
            z_component += 0.05 * (tsne_embeddings_2d[:, 0] + tsne_embeddings_2d[:, 1])
            return np.column_stack((tsne_embeddings_2d, z_component))
        
        elif n_chunks > CONSTANTS['DERIVED_3D_THRESHOLD']:
            # For datasets >15 chunks: compute true 3D t-SNE transformation
            tsne_3d = TSNE(
                n_components=3,
                perplexity=tsne_params['perplexity'],
                max_iter=tsne_params['n_iter'],
                early_exaggeration=tsne_params['early_exaggeration'],
                random_state=CONSTANTS['RANDOM_STATE'],
                n_jobs=1
            )
            return tsne_3d.fit_transform(normalized_embeddings)
        
        else:
            # For datasets with 1-2 chunks: add minimal Z variation for 3D display
            z_component = np.random.normal(0, 0.01, size=n_chunks)
            return np.column_stack((tsne_embeddings_2d, z_component))
    
    def create_2d_plot(self, tsne_embeddings_2d: np.ndarray, chunks: List[Dict[str, Any]], 
                      output_path: Union[str, Path], title: Optional[str] = None) -> bool:
        """
        Creates and saves a 2D plot of t-SNE embeddings.
        
        Args:
            tsne_embeddings_2d: 2D t-SNE embeddings
            chunks: List of chunks with related information
            output_path: Path where to save the plot
            title: Optional title for the plot
            
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            self._remove_existing_file(output_path)
            
            # Initialize matplotlib figure with predefined dimensions
            fig, ax = plt.subplots(figsize=CONSTANTS['FIGURE_SIZE_2D'])
            
            # Extract page-based colors and size-adaptive plot parameters
            colors = self._extract_colors(chunks)
            plot_params = self._get_plot_parameters(len(chunks))
            
            # Create scatter plot
            scatter = ax.scatter(
                tsne_embeddings_2d[:, 0], 
                tsne_embeddings_2d[:, 1],
                c=colors, 
                cmap='viridis',
                **plot_params
            )
            
            # Add text labels only for datasets with ≤10 chunks to avoid clutter
            self._add_chunk_labels(ax, tsne_embeddings_2d, chunks)
            
            # Configure the plot
            n_chunks = len(chunks)
            ax.set_title(title or f"2D t-SNE Visualization of {n_chunks} Chunk Embeddings", fontsize=16)
            ax.set_xlabel("t-SNE Component 1", fontsize=12)
            ax.set_ylabel("t-SNE Component 2", fontsize=12)
            ax.grid(alpha=0.3)
            
            # Display colorbar only when chunks span multiple pages
            if len(set(colors)) > 1:
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label("Page Number", fontsize=12)
            
            # Save and cleanup
            self._save_and_cleanup(fig, output_path)
            
            self.logger.info(f"2D t-SNE plot saved at: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating 2D t-SNE plot: {e}")
            return False
    
    def create_3d_plot(self, tsne_embeddings_3d: np.ndarray, chunks: List[Dict[str, Any]], 
                      output_path: Union[str, Path], title: Optional[str] = None) -> bool:
        """
        Creates and saves a 3D plot of t-SNE embeddings.
        
        Args:
            tsne_embeddings_3d: 3D t-SNE embeddings
            chunks: List of chunks with related information
            output_path: Path where to save the plot
            title: Optional title for the plot
            
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            self._remove_existing_file(output_path)
            self._cleanup_old_view_files(output_path)
            
            # Initialize 3D matplotlib figure with larger dimensions
            fig = plt.figure(figsize=CONSTANTS['FIGURE_SIZE_3D'])
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract page-based colors and size-adaptive plot parameters
            colors = self._extract_colors(chunks)
            plot_params = self._get_plot_parameters(len(chunks))
            plot_params['depthshade'] = True  # Enable depth shading for 3D visual effect
            
            # Create 3D scatter plot
            scatter = ax.scatter(
                tsne_embeddings_3d[:, 0],
                tsne_embeddings_3d[:, 1], 
                tsne_embeddings_3d[:, 2],
                c=colors,
                cmap='viridis',
                **plot_params
            )
            
            # Add 3D text labels only for datasets with ≤10 chunks
            self._add_3d_chunk_labels(ax, tsne_embeddings_3d, chunks)
            
            # Configure the plot
            n_chunks = len(chunks)
            ax.set_title(title or f"3D t-SNE Visualization of {n_chunks} Chunk Embeddings", fontsize=16)
            ax.set_xlabel("t-SNE Component 1", fontsize=12)
            ax.set_ylabel("t-SNE Component 2", fontsize=12)
            ax.set_zlabel("t-SNE Component 3", fontsize=12)
            
            # Add colorbar if there is color variety
            if len(set(colors)) > 1:
                cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
                cbar.set_label("Page Number", fontsize=12)
            
            # Set consistent 3D viewing angle (30° elevation, 45° azimuth)
            ax.view_init(elev=30, azim=45)
            
            # Save and cleanup
            self._save_and_cleanup(fig, output_path)
            
            self.logger.info(f"3D t-SNE plot saved at: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating 3D t-SNE plot: {e}")
            return False
    
    def _remove_existing_file(self, file_path: Path) -> None:
        """Delete existing plot file to prevent overwrite conflicts."""
        if file_path.exists():
            file_path.unlink()
            self.logger.info(f"Replacing existing file: {file_path}")
    
    def _cleanup_old_view_files(self, output_path: Path) -> None:
        """Remove any multi-angle view files from previous 3D plot generations."""
        views = [(30, 45), (0, 0), (0, 90), (90, 0)]
        for i in range(len(views)):
            view_path = output_path.with_name(f"{output_path.stem}_view{i}.png")
            if view_path.exists():
                view_path.unlink()
                self.logger.debug(f"Removing old view file: {view_path}")
    
    def _extract_colors(self, chunks: List[Dict[str, Any]]) -> List[int]:
        """Convert chunk page numbers to integer color values for plot coloring."""
        colors = []
        for chunk in chunks:
            page = chunk.get("page")
            if page:
                if isinstance(page, (int, float)):
                    colors.append(int(page))
                elif isinstance(page, str) and page.isdigit():
                    colors.append(int(page))
                else:
                    colors.append(0)
            else:
                colors.append(0)
        return colors
    
    def _get_plot_parameters(self, n_chunks: int) -> Dict[str, Any]:
        """Return scatter plot styling parameters adapted to dataset size."""
        if n_chunks <= CONSTANTS['SMALL_CHUNK_THRESHOLD']:
            return {
                'alpha': 0.9,      # Higher opacity for small datasets
                's': 200,          # Larger point size for visibility
                'edgecolors': 'w', # White borders for definition
                'linewidths': 1.0  # Thicker borders
            }
        else:
            return {
                'alpha': 0.7,      # Lower opacity to reduce visual clutter
                's': 100,          # Smaller points for dense plots
                'edgecolors': 'w', # White borders for definition
                'linewidths': 0.5  # Thinner borders
            }
    
    def _add_chunk_labels(self, ax, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        """Add labels to chunks for small datasets (2D)."""
        if len(chunks) <= CONSTANTS['MEDIUM_CHUNK_THRESHOLD']:
            for i, chunk in enumerate(chunks):
                chunk_id = chunk.get('id', i)
                ax.annotate(
                    f'Chunk {chunk_id}', 
                    (embeddings[i, 0], embeddings[i, 1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9
                )
    
    def _add_3d_chunk_labels(self, ax, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        """Add labels to chunks for small datasets (3D)."""
        if len(chunks) <= CONSTANTS['MEDIUM_CHUNK_THRESHOLD']:
            for i, chunk in enumerate(chunks):
                chunk_id = chunk.get('id', i)
                ax.text(
                    embeddings[i, 0],
                    embeddings[i, 1], 
                    embeddings[i, 2],
                    f'Chunk {chunk_id}',
                    size=9
                )
    
    def _save_and_cleanup(self, fig, output_path: Path) -> None:
        """Save plot to file and release matplotlib memory resources."""
        plt.tight_layout()
        fig.savefig(output_path, dpi=CONSTANTS['DEFAULT_DPI'], bbox_inches="tight")
        plt.close(fig)
        gc.collect()  # Force garbage collection to prevent memory leaks
    
    def visualize_document_embeddings(self, document_path: Union[str, Path]) -> bool:
        """
        Visualizes document embeddings using t-SNE.
        
        Args:
            document_path: Path to the Markdown document
            
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        try:
            # Convert to absolute path for consistent database matching
            document_path = Path(document_path)
            normalized_path = str(document_path.resolve())
            
            # Query database for document record using file path
            document = self.find_document_by_path(normalized_path)
            if not document:
                self.logger.warning(f"No document found for: {document_path}")
                return False
            
            # Generate t-SNE embeddings
            embeddings_result = self.generate_tsne_embeddings(document["id"], perplexity=30.0)
            if embeddings_result is None:
                self.logger.warning(f"Could not generate t-SNE embeddings for: {document_path}")
                return False
            
            _, chunks, tsne_2d, tsne_3d = embeddings_result
            title = document.get("title", document_path.name)
            
            # Construct output file paths in same directory as source document
            output_dir = document_path.parent
            base_name = document_path.stem
            
            # Create plots
            success_2d = self.create_2d_plot(
                tsne_2d, chunks, 
                output_dir / f"{base_name}_tsne_2d.png", 
                f"t-SNE 2D: {title}"
            )
            
            success_3d = self.create_3d_plot(
                tsne_3d, chunks, 
                output_dir / f"{base_name}_tsne_3d.png", 
                f"t-SNE 3D: {title}"
            )
            
            if success_2d and success_3d:
                self.logger.info(f"t-SNE visualizations generated for: {document_path}")
                return True
            else:
                self.logger.warning(f"Some visualizations failed for: {document_path}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error visualizing document embeddings: {e}")
            return False
    
    def find_document_by_path(self, document_path: str) -> Optional[Dict[str, Any]]:
        """
        Searches for a document in the database by its file path.
        
        Args:
            document_path: Path to the file to search for
            
        Returns:
            Dict with document information or None if not found
        """
        try:
            self.logger.debug(f"Searching for document with path: {document_path}")
            
            # Access database cursor for SQL query execution
            cursor = self.db._cursor
            
            # Query documents table using exact file path match
            cursor.execute(
                "SELECT id, title, url, file_path, created_at FROM documents WHERE file_path = ?", 
                (document_path,)
            )
            doc = cursor.fetchone()
            
            if doc:
                # Convert SQLite row to dictionary format
                if hasattr(doc, 'keys'):
                    return dict(doc)
                else:
                    # Manual conversion for tuple-based results
                    return {
                        'id': doc[0],
                        'title': doc[1],
                        'url': doc[2],
                        'file_path': doc[3],
                        'created_at': doc[4]
                    }
            else:
                # Fallback: search using filename pattern matching
                cursor.execute(
                    "SELECT id, title, url, file_path, created_at FROM documents WHERE file_path LIKE ?", 
                    (f"%{os.path.basename(document_path)}%",)
                )
                doc = cursor.fetchone()
                
                if doc:
                    if hasattr(doc, 'keys'):
                        return dict(doc)
                    else:
                        return {
                            'id': doc[0],
                            'title': doc[1],
                            'url': doc[2],
                            'file_path': doc[3],
                            'created_at': doc[4]
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error searching for document: {e}")
            return None


def visualize_tsne_for_files(target_path: str, db_index: int = 0) -> bool:
    """
    Generates t-SNE visualizations for documents in the specified path.
    
    Args:
        target_path: Path to file or directory to visualize
        db_index: Database index (default: 0)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize database connection through session management
        session_manager = SessionManager()
        db, db_metadata = session_manager.get_database_by_index(db_index)
        
        if not db:
            print(f"Database with index {db_index} not found")
            return False
        
        # Initialize visualizer
        visualizer = TSNEVisualizer(db)
        
        # Convert to absolute path for consistent file system operations
        target_path = os.path.abspath(target_path)
        
        if os.path.isfile(target_path):
            # Generate t-SNE plots for single markdown file
            document = visualizer.find_document_by_path(target_path)
            if document:
                return visualizer.visualize_document_embeddings(target_path)
            else:
                print(f"Document not found for path: {target_path}")
                return False
        elif os.path.isdir(target_path):
            # Recursively process all markdown files in directory tree
            success = True
            for root, dirs, files in os.walk(target_path):
                for file in files:
                    if file.endswith('.md'):
                        file_path = os.path.join(root, file)
                        document = visualizer.find_document_by_path(file_path)
                        if document:
                            result = visualizer.visualize_document_embeddings(file_path)
                            if not result:
                                success = False
            return success
        else:
            print(f"Path not found: {target_path}")
            return False
            
    except Exception as e:
        print(f"Error generating t-SNE visualization: {e}")
        return False