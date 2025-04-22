"""
Visualizador de embeddings de chunks utilizando t-SNE.

Este módulo proporciona herramientas para visualizar los embeddings de chunks 
generados del procesamiento de documentos Markdown utilizando t-SNE (t-Distributed 
Stochastic Neighbor Embedding), un algoritmo de reducción de dimensionalidad.
Genera gráficos 2D y 3D que permiten visualizar cómo se distribuyen los embeddings
en el espacio y ayuda a identificar patrones y grupos semánticos.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

# Silenciar warning de joblib configurando el número de cores
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Un valor razonable para la mayoría de sistemas

# Configurar logging
logger = logging.getLogger(__name__)

class TSNEVisualizer:
    """
    Clase para visualizar embeddings de chunks utilizando t-SNE.
    
    Genera visualizaciones 2D y 3D de los embeddings de chunks
    para facilitar el análisis y la identificación de patrones.
    """
    
    def __init__(self, db_instance):
        """
        Inicializa el visualizador con una instancia de base de datos.
        
        Args:
            db_instance: Instancia de base de datos vectorial
        """
        self.db = db_instance
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def generate_tsne_embeddings(self, document_id: int, perplexity: float = 30.0) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """
        Genera embeddings t-SNE en 2D y 3D para un documento específico.
        
        Args:
            document_id: ID del documento en la base de datos
            perplexity: Parámetro de perplexity para t-SNE (valor recomendado: 5-50)
            
        Returns:
            Tupla con (embeddings originales, lista de chunks, embeddings 2D, embeddings 3D)
            Retorna None si no hay suficientes chunks con embeddings
        """
        try:
            # Obtener todos los chunks del documento con sus embeddings
            offset = 0
            limit = 1000  # Un límite grande para obtener todos los chunks
            chunks = self.db.get_chunks_by_document(document_id, offset, limit)
            
            # Filtrar chunks que tienen embeddings
            chunks_with_embeddings = [chunk for chunk in chunks if "embedding" in chunk and chunk["embedding"] is not None]
            
            # Verificar que hay suficientes chunks para t-SNE
            min_chunks_needed = 2  # Mínimo absoluto para visualizar puntos
            
            if len(chunks_with_embeddings) < min_chunks_needed:
                self.logger.warning(f"No hay suficientes chunks con embeddings para el documento {document_id}. Se necesitan al menos {min_chunks_needed} chunks.")
                return None, None, None, None
            
            # Extraer embeddings
            embeddings = np.array([chunk["embedding"] for chunk in chunks_with_embeddings])
            
            # Normalizar embeddings para consistencia
            normalized_embeddings = normalize(embeddings, axis=1, norm='l2')
            
            # Adaptar parámetros de t-SNE según el número de chunks
            n_chunks = len(chunks_with_embeddings)
            
            # Para t-SNE la perplexity debe ser menor que n_samples - 1
            # Se recomienda entre 5 y 50, pero adaptamos para conjuntos pequeños
            if n_chunks <= 5:
                # Para conjuntos muy pequeños, usar perplexity baja
                adjusted_perplexity = max(1.0, n_chunks / 3)
                n_iter = 500  # Más iteraciones para conjuntos pequeños
                early_exaggeration = 6.0  # Valor equilibrado
            elif n_chunks <= 10:
                adjusted_perplexity = max(2.0, n_chunks / 3)
                n_iter = 1000
                early_exaggeration = 8.0
            else:
                # Para conjuntos normales, usar la perplexity recomendada
                adjusted_perplexity = min(perplexity, n_chunks - 1)
                n_iter = 1000
                early_exaggeration = 12.0
            
            # Generar embeddings t-SNE en 2D
            tsne_2d = TSNE(
                n_components=2, 
                perplexity=adjusted_perplexity,
                n_iter=n_iter,
                early_exaggeration=early_exaggeration,
                random_state=42
            )
            tsne_embeddings_2d = tsne_2d.fit_transform(normalized_embeddings)
            
            # Generar embeddings t-SNE en 3D (solo si hay suficientes chunks)
            if n_chunks >= 3:  # Para 3D necesitamos al menos 3 puntos
                tsne_3d = TSNE(
                    n_components=3, 
                    perplexity=adjusted_perplexity,
                    n_iter=n_iter,
                    early_exaggeration=early_exaggeration,
                    random_state=42
                )
                tsne_embeddings_3d = tsne_3d.fit_transform(normalized_embeddings)
            else:
                # Para 1 o 2 chunks, generar una visualización 3D simple
                # Añadimos una dimensión con valores aleatorios pequeños para visualizarlos
                tsne_embeddings_3d = np.column_stack((
                    tsne_embeddings_2d, 
                    np.random.normal(0, 0.01, size=n_chunks)
                ))
            
            self.logger.info(f"Generados embeddings t-SNE para {n_chunks} chunks con perplexity={adjusted_perplexity:.2f}")
            return normalized_embeddings, chunks_with_embeddings, tsne_embeddings_2d, tsne_embeddings_3d
            
        except Exception as e:
            self.logger.error(f"Error al generar embeddings t-SNE: {e}")
            return None, None, None, None
    
    def create_2d_plot(self, tsne_embeddings_2d: np.ndarray, chunks: List[Dict[str, Any]], 
                      output_path: str, title: Optional[str] = None) -> bool:
        """
        Crea y guarda un gráfico 2D de los embeddings t-SNE.
        
        Args:
            tsne_embeddings_2d: Embeddings t-SNE en 2D
            chunks: Lista de chunks con información relacionada
            output_path: Ruta donde guardar el gráfico
            title: Título opcional para el gráfico
            
        Returns:
            bool: True si la operación fue exitosa, False en caso contrario
        """
        try:
            # Eliminar archivos existentes si existen
            if os.path.exists(output_path):
                os.remove(output_path)
                self.logger.info(f"Reemplazando archivo existente: {output_path}")
            
            plt.figure(figsize=(12, 10))
            
            # Obtener información para colorear puntos (usando páginas o encabezados)
            colors = []
            unique_headers = set()
            
            for chunk in chunks:
                # Usar página como color si está disponible
                if "page" in chunk and chunk["page"]:
                    colors.append(int(chunk["page"]) if isinstance(chunk["page"], (int, float)) or 
                                 (isinstance(chunk["page"], str) and chunk["page"].isdigit()) else 0)
                else:
                    # Si no hay página, usar un valor predeterminado
                    colors.append(0)
                
                # Recopilar encabezados únicos para la leyenda
                if "header" in chunk and chunk["header"]:
                    unique_headers.add(chunk["header"])
            
            # Ajustar visualización según cantidad de puntos
            n_chunks = len(chunks)
            
            if n_chunks <= 5:
                s = 200  # Puntos más grandes para pocos chunks
                alpha = 0.9
                linewidths = 1.0
            else:
                s = 100
                alpha = 0.7
                linewidths = 0.5
            
            # Crear scatter plot
            scatter = plt.scatter(
                tsne_embeddings_2d[:, 0], 
                tsne_embeddings_2d[:, 1],
                c=colors, 
                cmap='viridis',
                alpha=alpha,
                s=s,
                edgecolors='w',
                linewidths=linewidths
            )
            
            # Añadir etiquetas para pocos puntos
            if n_chunks <= 10:
                for i, chunk in enumerate(chunks):
                    chunk_id = chunk.get('id', i)
                    plt.annotate(
                        f'Chunk {chunk_id}', 
                        (tsne_embeddings_2d[i, 0], tsne_embeddings_2d[i, 1]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9
                    )
            
            # Configurar el gráfico
            plt.title(title or f"Visualización t-SNE 2D de {n_chunks} Embeddings de Chunks", fontsize=16)
            plt.xlabel("t-SNE Componente 1", fontsize=12)
            plt.ylabel("t-SNE Componente 2", fontsize=12)
            plt.grid(alpha=0.3)
            
            # Añadir colorbar si hay variedad de colores
            if len(set(colors)) > 1:
                cbar = plt.colorbar(scatter)
                cbar.set_label("Número de Página", fontsize=12)
            
            # Guardar el gráfico
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            self.logger.info(f"Gráfico t-SNE 2D guardado en: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al crear gráfico t-SNE 2D: {e}")
            return False
    
    def create_3d_plot(self, tsne_embeddings_3d: np.ndarray, chunks: List[Dict[str, Any]], 
                      output_path: str, title: Optional[str] = None) -> bool:
        """
        Crea y guarda un gráfico 3D de los embeddings t-SNE.
        
        Args:
            tsne_embeddings_3d: Embeddings t-SNE en 3D
            chunks: Lista de chunks con información relacionada
            output_path: Ruta donde guardar el gráfico
            title: Título opcional para el gráfico
            
        Returns:
            bool: True si la operación fue exitosa, False en caso contrario
        """
        try:
            # Eliminar archivos existentes si existen
            if os.path.exists(output_path):
                os.remove(output_path)
                self.logger.info(f"Reemplazando archivo existente: {output_path}")
            
            # Crear figura 3D
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Obtener información para colorear puntos (usando páginas o encabezados)
            colors = []
            for chunk in chunks:
                # Usar página como color si está disponible
                if "page" in chunk and chunk["page"]:
                    colors.append(int(chunk["page"]) if isinstance(chunk["page"], (int, float)) or 
                                 (isinstance(chunk["page"], str) and chunk["page"].isdigit()) else 0)
                else:
                    # Si no hay página, usar un valor predeterminado
                    colors.append(0)
            
            # Ajustar visualización según cantidad de puntos
            n_chunks = len(chunks)
            
            if n_chunks <= 5:
                s = 200  # Puntos más grandes para pocos chunks
                alpha = 0.9
                linewidths = 1.0
            else:
                s = 100
                alpha = 0.7
                linewidths = 0.5
            
            # Crear scatter plot 3D
            scatter = ax.scatter(
                tsne_embeddings_3d[:, 0],
                tsne_embeddings_3d[:, 1], 
                tsne_embeddings_3d[:, 2],
                c=colors,
                cmap='viridis',
                alpha=alpha,
                s=s,
                edgecolors='w',
                linewidths=linewidths,
                depthshade=True
            )
            
            # Añadir etiquetas para pocos puntos
            if n_chunks <= 10:
                for i, chunk in enumerate(chunks):
                    chunk_id = chunk.get('id', i)
                    ax.text(
                        tsne_embeddings_3d[i, 0],
                        tsne_embeddings_3d[i, 1], 
                        tsne_embeddings_3d[i, 2],
                        f'Chunk {chunk_id}',
                        size=9
                    )
            
            # Configurar el gráfico
            ax.set_title(title or f"Visualización t-SNE 3D de {n_chunks} Embeddings de Chunks", fontsize=16)
            ax.set_xlabel("t-SNE Componente 1", fontsize=12)
            ax.set_ylabel("t-SNE Componente 2", fontsize=12)
            ax.set_zlabel("t-SNE Componente 3", fontsize=12)
            
            # Añadir colorbar si hay variedad de colores
            if len(set(colors)) > 1:
                cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
                cbar.set_label("Número de Página", fontsize=12)
            
            # Configurar varias vistas para mejorar interpretación
            views = [(30, 45), (0, 0), (0, 90), (90, 0)]
            if n_chunks <= 5:
                # Para pocos puntos, guardar varias vistas
                for i, (elev, azim) in enumerate(views):
                    view_path = output_path.replace('.png', f'_view{i}.png')
                    
                    # Eliminar archivos de vistas existentes si existen
                    if os.path.exists(view_path):
                        os.remove(view_path)
                        self.logger.debug(f"Reemplazando archivo existente: {view_path}")
                    
                    ax.view_init(elev=elev, azim=azim)
                    plt.savefig(view_path, dpi=300, bbox_inches="tight")
                
                # Volver a la vista principal
                ax.view_init(elev=30, azim=45)
            else:
                # Para muchos puntos, solo la vista principal
                ax.view_init(elev=30, azim=45)
                
                # Eliminar archivos de vistas si existen de ejecuciones anteriores
                # (por si antes hubo pocos chunks y ahora hay muchos)
                for i in range(len(views)):
                    view_path = output_path.replace('.png', f'_view{i}.png')
                    if os.path.exists(view_path):
                        os.remove(view_path)
                        self.logger.debug(f"Eliminando vista no necesaria: {view_path}")
            
            # Guardar el gráfico principal
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            self.logger.info(f"Gráfico t-SNE 3D guardado en: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al crear gráfico t-SNE 3D: {e}")
            return False
    
    def visualize_document_embeddings(self, document_path: str) -> bool:
        """
        Visualiza los embeddings de un documento usando t-SNE.
        
        Args:
            document_path: Ruta del documento Markdown
            
        Returns:
            bool: True si la operación fue exitosa, False en caso contrario
        """
        try:
            # Normalizar la ruta para búsqueda en la base de datos
            normalized_path = os.path.normpath(document_path)
            
            # Buscar documento en la base de datos
            document = self.find_document_by_path(normalized_path)
            if not document:
                self.logger.warning(f"No se encontró documento para: {document_path}")
                return False
            
            # Obtener ID del documento
            document_id = document["id"]
            title = document.get("title", os.path.basename(document_path))
            
            # Generar embeddings t-SNE
            _, chunks, tsne_2d, tsne_3d = self.generate_tsne_embeddings(document_id, perplexity=30.0)
            
            if tsne_2d is None or tsne_3d is None:
                self.logger.warning(f"No se pudieron generar embeddings t-SNE para: {document_path}")
                return False
            
            # Definir rutas de salida para los gráficos
            output_dir = os.path.dirname(os.path.abspath(document_path))
            base_name = os.path.splitext(os.path.basename(document_path))[0]
            
            # Crear gráfico 2D
            output_path_2d = os.path.join(output_dir, f"{base_name}_tsne_2d.png")
            title_2d = f"t-SNE 2D: {title}"
            self.create_2d_plot(tsne_2d, chunks, output_path_2d, title_2d)
            
            # Crear gráfico 3D
            output_path_3d = os.path.join(output_dir, f"{base_name}_tsne_3d.png")
            title_3d = f"t-SNE 3D: {title}"
            self.create_3d_plot(tsne_3d, chunks, output_path_3d, title_3d)
            
            self.logger.info(f"Visualizaciones t-SNE generadas para: {document_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al visualizar embeddings del documento: {e}")
            return False
    
    def find_document_by_path(self, document_path: str) -> Optional[Dict[str, Any]]:
        """
        Busca un documento en la base de datos por su ruta de archivo.
        
        Args:
            document_path: Ruta del archivo a buscar
            
        Returns:
            Dict con información del documento o None si no se encuentra
        """
        try:
            self.logger.debug(f"Buscando documento con ruta: {document_path}")
            
            # Ejecutar consulta en la base de datos
            cursor = self.db._cursor
            
            # Buscar el documento por ruta exacta
            cursor.execute(
                "SELECT id, title, url, file_path, created_at FROM documents WHERE file_path = ?", 
                (document_path,)
            )
            doc = cursor.fetchone()
            
            if doc:
                # Convertir a diccionario si es una fila de SQLite
                if hasattr(doc, 'keys'):
                    return dict(doc)
                else:
                    # Crear diccionario manualmente
                    return {
                        'id': doc[0],
                        'title': doc[1],
                        'url': doc[2],
                        'file_path': doc[3],
                        'created_at': doc[4]
                    }
            else:
                # Intentar búsqueda alternativa - con coincidencia parcial de ruta
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
            self.logger.error(f"Error al buscar documento: {e}")
            return None


def visualize_tsne_for_files(file_paths: str, db_instance) -> Dict[str, bool]:
    """
    Genera visualizaciones t-SNE para los embeddings de chunks de los archivos especificados.
    
    Args:
        file_paths: Ruta a un directorio o archivo individual
        db_instance: Instancia de base de datos vectorial
        
    Returns:
        Diccionario con las rutas procesadas y si su visualización fue exitosa
    """
    visualizer = TSNEVisualizer(db_instance)
    results = {}
    
    try:
        if os.path.isdir(file_paths):
            # Recorrer recursivamente el directorio
            logger.info(f"Generando visualizaciones t-SNE para todos los Markdown en: {file_paths}")
            
            for root, _, files in os.walk(file_paths):
                for file in files:
                    if file.lower().endswith('.md'):
                        md_path = os.path.join(root, file)
                        result = visualizer.visualize_document_embeddings(md_path)
                        results[md_path] = result
                        
        elif os.path.isfile(file_paths) and file_paths.lower().endswith('.md'):
            # Visualizar un solo archivo
            logger.info(f"Generando visualizaciones t-SNE para archivo Markdown: {file_paths}")
            result = visualizer.visualize_document_embeddings(file_paths)
            results[file_paths] = result
        else:
            logger.warning(f"La ruta proporcionada no es un archivo Markdown o directorio válido: {file_paths}")
    
    except Exception as e:
        logger.error(f"Error al generar visualizaciones t-SNE: {e}")
    
    # Liberar recursos
    del visualizer
    
    return results 