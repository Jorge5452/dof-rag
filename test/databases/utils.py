"""
Utilidades para las pruebas de bases de datos vectoriales.
"""

import os
import numpy as np
import logging
import tempfile
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

def generate_random_embedding(dim: int = 384) -> List[float]:
    """
    Genera un vector de embedding aleatorio para pruebas.
    
    Args:
        dim: Dimensión del vector de embedding
        
    Returns:
        Lista de valores float que representan el vector normalizado
    """
    # Generar un vector aleatorio con valores entre 0 y 1
    vector = np.random.rand(dim).astype(np.float32)
    
    # Normalizar el vector (como suelen estar los embeddings en la práctica)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    # Convertir a lista de Python con redondeo para limitar posibles problemas de precisión
    return [round(float(x), 6) for x in vector.tolist()]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcula la similitud de coseno entre dos vectores.
    
    Args:
        vec1: Primer vector
        vec2: Segundo vector
        
    Returns:
        Valor de similitud de coseno entre 0 y 1
    """
    # Convertir a arrays de numpy
    a = np.array(vec1)
    b = np.array(vec2)
    
    # Calcular similitud de coseno: cos(θ) = (a·b)/(|a|·|b|)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Evitar división por cero
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))

def create_test_document(doc_id=1, with_embeddings=True, embedding_dim=10):
    """
    Crea un documento de prueba con metadatos estándar.
    
    Args:
        doc_id: ID numérico para el documento
        with_embeddings: Si se deben generar embeddings para los chunks
        embedding_dim: Dimensión de los embeddings
        
    Returns:
        Diccionario con información del documento
    """
    return {
        "id": doc_id,
        "title": f"Documento de prueba {doc_id}",
        "url": f"http://example.com/doc_{doc_id}",
        "file_path": f"/path/to/test_{doc_id}.md",
    }

def create_test_chunks(num_chunks=2, embedding_dim=10):
    """
    Crea una lista de chunks de prueba con embeddings.
    
    Args:
        num_chunks: Número de chunks a crear
        embedding_dim: Dimensión de los embeddings
        
    Returns:
        Lista de diccionarios con información de los chunks
    """
    chunks = []
    for i in range(num_chunks):
        chunks.append({
            "text": f"Este es un texto de prueba para el chunk {i+1}.",
            "header": f"Encabezado de prueba {i//2 + 1}",
            "page": f"{i//2 + 1}",
            "embedding": generate_random_embedding(embedding_dim)
        })
    return chunks

class TestDatabaseContext:
    """
    Contexto para pruebas de bases de datos que gestiona la creación y limpieza de recursos.
    Útil para tests que no heredan de las clases base.
    """
    
    def __init__(self, db_class, embedding_dim=10):
        """
        Inicializa el contexto de prueba.
        
        Args:
            db_class: Clase de la base de datos vectorial a probar
            embedding_dim: Dimensión de los embeddings para pruebas
        """
        self.db_class = db_class
        self.embedding_dim = embedding_dim
        self.temp_dir = None
        self.db_path = None
        self.db_instance = None
    
    def __enter__(self):
        """Configure el contexto de prueba al entrar."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_db.db")
        
        # Crear instancia de base de datos
        self.db_instance = self.db_class(embedding_dim=self.embedding_dim)
        self.db_instance.connect(self.db_path)
        self.db_instance.create_schema()
        
        logger.info(f"Contexto de prueba creado: {self.db_path}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Limpiar recursos al salir del contexto."""
        if self.db_instance:
            try:
                self.db_instance.close_connection()
                logger.info("Base de datos cerrada correctamente")
            except Exception as e:
                logger.error(f"Error al cerrar base de datos: {e}")
        
        # Eliminar directorio temporal
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"Directorio temporal eliminado: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"No se pudo eliminar el directorio temporal: {e}")
