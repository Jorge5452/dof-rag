import os
import unittest
import tempfile
import shutil
import numpy as np
import sys
import logging
from pathlib import Path

# Añadir el directorio raíz al path para permitir importaciones relativas
sys.path.insert(0, str(Path(__file__).parents[2]))

from test.databases.mocks import MockSQLiteVectorialDatabase
try:
    from modulos.databases.implementaciones.duckdb import DuckDBVectorialDatabase
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# Configurar logging para el test
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChunkInsertionTest(unittest.TestCase):
    """Pruebas para la inserción de chunks en diferentes bases de datos."""
    
    def setUp(self):
        """Configuración inicial para las pruebas."""
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Directorio temporal para pruebas: {self.temp_dir}")
        
        # Dimensión de embedding para pruebas - usar valor pequeño para pruebas
        self.test_embedding_dim = 10
        
        # Creamos bases de datos para las pruebas
        self.sqlite_db_path = os.path.join(self.temp_dir, "test_chunks_sqlite.db")
        self.sqlite_db = MockSQLiteVectorialDatabase(embedding_dim=self.test_embedding_dim)
        self.sqlite_db.connect(self.sqlite_db_path)
        self.sqlite_db.create_schema()
        
        # Creamos una base de datos DuckDB para las pruebas si está disponible
        try:
            if DUCKDB_AVAILABLE:
                self.duckdb_db_path = os.path.join(self.temp_dir, "test_chunks_duckdb.db")
                self.duckdb_db = DuckDBVectorialDatabase(embedding_dim=self.test_embedding_dim)
                self.duckdb_db.connect(self.duckdb_db_path)
                self.duckdb_db.create_schema()
            else:
                self.duckdb_db = None
                self.duckdb_available = False
        except Exception as e:
            logger.error(f"Error inicializando DuckDB: {e}")
            self.duckdb_db = None
            self.duckdb_available = False
    
    def tearDown(self):
        """Limpieza después de las pruebas."""
        # Cerramos todas las conexiones
        if hasattr(self, 'sqlite_db') and self.sqlite_db:
            self.sqlite_db.close_connection()
        
        if hasattr(self, 'duckdb_db') and self.duckdb_db:
            self.duckdb_db.close_connection()
        
        # Eliminamos el directorio temporal
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Directorio temporal eliminado: {self.temp_dir}")
    
    def test_sqlite_insert_single_chunk(self):
        """Prueba la inserción de un único chunk en SQLite."""
        # Creamos un embedding de prueba (aseguramos que sea una lista de floats)
        embedding = list(np.random.rand(self.test_embedding_dim).astype(float))
        
        # Creamos un documento de prueba primero (para tener document_id)
        test_doc = {
            "title": "Documento de prueba",
            "url": "http://example.com/test",
            "file_path": "/path/to/test.md"
        }
        
        # Insertamos el documento
        doc_id = self.sqlite_db.insert_document(test_doc, [])
        
        # Ahora insertamos un chunk con el document_id
        metadata = {"document_id": doc_id, "page": "1", "header": "Cabecera de prueba"}
        chunk_id = self.sqlite_db.insert_chunk(
            text="Este es un texto de prueba",
            embedding=embedding,
            metadata=metadata
        )
        
        # Verificamos que se haya insertado correctamente
        self.assertIsNotNone(chunk_id)
        
        # Verificamos que el chunk exista en la base de datos
        cursor = self.sqlite_db._conn.cursor()
        cursor.execute("SELECT id, text, document_id FROM chunks WHERE id = ?", (chunk_id,))
        result = cursor.fetchone()
        
        self.assertIsNotNone(result)
        self.assertEqual(result[1], "Este es un texto de prueba")
        self.assertEqual(result[2], doc_id)
    
    def test_sqlite_insert_multiple_chunks(self):
        """Prueba la inserción de múltiples chunks en SQLite."""
        # Creamos un documento de prueba primero
        test_doc = {
            "title": "Documento para múltiples chunks",
            "url": "http://example.com/multiple",
            "file_path": "/path/to/multiple.md"
        }
        
        # Insertamos el documento
        doc_id = self.sqlite_db.insert_document(test_doc, [])
        
        # Creamos varios chunks de prueba
        chunks = []
        for i in range(5):
            chunks.append({
                'text': f"Texto de prueba {i}",
                'embedding': list(np.random.rand(self.test_embedding_dim).astype(float)),
                'document_id': doc_id,
                'header': f"Cabecera {i}",
                'page': str(i)
            })
        
        # Insertamos los chunks
        result = self.sqlite_db.insert_chunks(chunks)
        self.assertTrue(result)
        
        # Verificamos que todos los chunks se hayan insertado
        cursor = self.sqlite_db._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,))
        count = cursor.fetchone()[0]
        self.assertEqual(count, 5)
    
    def test_duckdb_insert_chunks(self):
        """Prueba la inserción de chunks en DuckDB."""
        if not hasattr(self, 'duckdb_db') or self.duckdb_db is None:
            self.skipTest("DuckDB no está disponible")
        
        # Creamos un documento de prueba primero
        test_doc = {
            "title": "Documento para DuckDB",
            "url": "http://example.com/duckdb",
            "file_path": "/path/to/duckdb.md"
        }
        
        # Insertamos el documento
        doc_id = self.duckdb_db.insert_document(test_doc, [])
        
        # Creamos varios chunks de prueba - con la dimensión correcta
        chunks = []
        for i in range(3):
            chunks.append({
                'text': f"Texto de prueba DuckDB {i}",
                'embedding': list(np.random.rand(self.test_embedding_dim).astype(float)),
                'document_id': doc_id,
                'header': f"Cabecera DuckDB {i}",
                'page': str(i)
            })
        
        # Insertamos los chunks
        result = self.duckdb_db.insert_chunks(chunks)
        self.assertTrue(result)
        
        # Verificamos que se hayan insertado correctamente
        count = self.duckdb_db._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ?", 
            [doc_id]
        ).fetchone()[0]
        self.assertEqual(count, 3)

if __name__ == "__main__":
    unittest.main()
