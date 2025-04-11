#!/usr/bin/env python
"""
Herramienta para la creación de bases de datos de prueba con datos sintéticos.

Este script permite generar bases de datos SQLite o DuckDB pobladas con
documentos y chunks aleatorios para pruebas de rendimiento, desarrollo
y depuración.

Ejemplos de uso:
---------------
1. Crear una base de datos SQLite con 10 documentos y 5 chunks por documento:
   python create_test_data.py --type sqlite --path ./test_db.sqlite --docs 10 --chunks 5

2. Crear una base de datos DuckDB con 50 documentos y dimensión de embedding 768:
   python create_test_data.py --type duckdb --path ./test_db.duckdb --docs 50 --dim 768
"""

import os
import sys
import logging
import numpy as np
import argparse
from pathlib import Path

# Añadir el directorio raíz al path para permitir importaciones relativas
sys.path.insert(0, str(Path(__file__).parents[2]))

from test.databases.utils import generate_random_embedding
# Importar la implementación mock para SQLite para evitar el error de método abstracto
from test.databases.mocks import MockSQLiteVectorialDatabase
try:
    from modulos.databases.implementaciones.duckdb import DuckDBVectorialDatabase
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_document(doc_id: int, num_chunks: int = 10, embedding_dim: int = 384) -> tuple:
    """
    Genera un documento de prueba con sus chunks.
    
    Args:
        doc_id: Identificador del documento
        num_chunks: Número de chunks a generar
        embedding_dim: Dimensión de los vectores de embedding
        
    Returns:
        Tupla (documento, lista de chunks)
    """
    # Generar documento
    document = {
        "title": f"Documento de prueba {doc_id}",
        "url": f"http://example.com/doc_{doc_id}",
        "file_path": f"/path/to/doc_{doc_id}.md"
    }
    
    # Generar chunks
    chunks = []
    for i in range(num_chunks):
        chunk = {
            "text": f"Este es el contenido del chunk {i} del documento {doc_id}. " 
                   f"Contiene información de prueba para realizar búsquedas vectoriales.",
            "header": f"Sección {i // 3 + 1}",
            "page": str(i // 2 + 1),
            "embedding": generate_random_embedding(embedding_dim)
        }
        chunks.append(chunk)
    
    return document, chunks

def create_test_database(db_type: str, db_path: str, num_docs: int = 5, 
                        chunks_per_doc: int = 10, embedding_dim: int = 384) -> None:
    """
    Crea una base de datos de prueba con documentos y chunks aleatorios.
    
    Args:
        db_type: Tipo de base de datos ('sqlite' o 'duckdb')
        db_path: Ruta donde crear la base de datos
        num_docs: Número de documentos a generar
        chunks_per_doc: Número de chunks por documento
        embedding_dim: Dimensión de los embeddings
    """
    # Verificar tipo de base de datos
    if db_type.lower() == 'sqlite':
        db = MockSQLiteVectorialDatabase(embedding_dim=embedding_dim)
    elif db_type.lower() == 'duckdb':
        if not DUCKDB_AVAILABLE:
            logger.error("DuckDB no está disponible. Instale el paquete duckdb primero.")
            return
        db = DuckDBVectorialDatabase(embedding_dim=embedding_dim)
    else:
        logger.error(f"Tipo de base de datos no soportado: {db_type}")
        return
    
    # Crear directorio para la BD si no existe
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    
    try:
        # Conectar y crear esquema
        logger.info(f"Conectando a la base de datos {db_type} en {db_path}")
        db.connect(db_path)
        db.create_schema()
        
        # Generar e insertar documentos y chunks
        logger.info(f"Generando {num_docs} documentos con {chunks_per_doc} chunks cada uno...")
        
        for doc_id in range(1, num_docs + 1):
            document, chunks = generate_test_document(
                doc_id, chunks_per_doc, embedding_dim
            )
            
            # Insertar documento y chunks
            inserted_id = db.insert_document(document, chunks)
            logger.info(f"Documento {inserted_id} insertado con {len(chunks)} chunks")
            
            # Cada 20 documentos, mostrar progreso
            if doc_id % 20 == 0 or doc_id == num_docs:
                logger.info(f"Progreso: {doc_id}/{num_docs} documentos insertados")
        
        # Crear índices vectoriales si es posible
        logger.info("Creando índices vectoriales...")
        db.create_vector_index()
        
        logger.info(f"Base de datos de prueba creada exitosamente con {num_docs} documentos "
                    f"y {num_docs * chunks_per_doc} chunks en total")
        
    except Exception as e:
        logger.error(f"Error al crear la base de datos de prueba: {e}")
    finally:
        # Cerrar la conexión
        db.close_connection()
        logger.info("Conexión cerrada")

def main():
    """Función principal que procesa los argumentos y ejecuta la creación de la BD."""
    parser = argparse.ArgumentParser(
        description="Crear bases de datos vectoriales de prueba con datos sintéticos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--type", choices=["sqlite", "duckdb"], default="sqlite",
                        help="Tipo de base de datos a generar")
    parser.add_argument("--path", type=str, default="./test_vector_db",
                        help="Ruta donde guardar la base de datos")
    parser.add_argument("--docs", type=int, default=5,
                        help="Número de documentos a generar")
    parser.add_argument("--chunks", type=int, default=10,
                        help="Número de chunks por documento")
    parser.add_argument("--dim", type=int, default=384,
                        help="Dimensión de los embeddings")
    
    args = parser.parse_args()
    
    # Asegurar que la extensión es adecuada
    if args.type == "sqlite" and not args.path.endswith(".db") and not args.path.endswith(".sqlite"):
        args.path += ".db"
    elif args.type == "duckdb" and not args.path.endswith(".db") and not args.path.endswith(".duckdb"):
        args.path += ".db"
    
    create_test_database(
        args.type, args.path, args.docs, args.chunks, args.dim
    )

if __name__ == "__main__":
    main()
