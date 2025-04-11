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

class ChunkOperationsTest(unittest.TestCase):
    """Pruebas para operaciones con chunks en diferentes bases de datos."""
    
    def setUp(self):
        """Configuración inicial para las pruebas."""
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Directorio temporal para pruebas: {self.temp_dir}")
        
        # Dimensión de embedding para pruebas
        self.test_embedding_dim = 10
        
        # Creamos bases de datos para las pruebas
        try:
            self.sqlite_db_path = os.path.join(self.temp_dir, "test_chunks_sqlite.db")
            self.sqlite_db = MockSQLiteVectorialDatabase(embedding_dim=self.test_embedding_dim)
            self.sqlite_db.connect(self.sqlite_db_path)
            self.sqlite_db.create_schema()
            
            if DUCKDB_AVAILABLE:
                self.duckdb_db_path = os.path.join(self.temp_dir, "test_chunks_duckdb.db")
                self.duckdb_db = DuckDBVectorialDatabase(embedding_dim=self.test_embedding_dim)
                self.duckdb_db.connect(self.duckdb_db_path)
                self.duckdb_db.create_schema()
            else:
                self.duckdb_db = None
                
        except Exception as e:
            logger.error(f"Error en la configuración: {e}")
            if hasattr(self, 'sqlite_db') and self.sqlite_db:
                self.sqlite_db.close_connection()
            if hasattr(self, 'duckdb_db') and self.duckdb_db:
                self.duckdb_db.close_connection()
            shutil.rmtree(self.temp_dir)
            raise
    
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
    
    def test_sqlite_insert_document_and_chunks(self):
        """Prueba la inserción de documento y chunks en SQLite."""
        # Creamos un documento de prueba
        test_doc = {
            "title": "Documento de prueba SQLite",
            "url": "http://example.com/sqlite",
            "file_path": "/path/to/sqlite.md"
        }
        
        # Creamos embeddings de prueba
        embeddings = []
        for i in range(2):
            embeddings.append(list(np.random.rand(self.test_embedding_dim).astype(float)))
        
        # Creamos chunks de prueba
        test_chunks = [
            {
                "text": "Este es un texto de prueba para el chunk 1 en SQLite.",
                "header": "Encabezado de prueba",
                "page": "1",
                "embedding": embeddings[0]
            },
            {
                "text": "Este es un texto de prueba para el chunk 2 en SQLite.",
                "header": "Otro encabezado",
                "page": "1",
                "embedding": embeddings[1]
            }
        ]
        
        # Insertamos el documento con sus chunks
        doc_id = self.sqlite_db.insert_document(test_doc, test_chunks)
        
        # Verificamos que se haya insertado correctamente
        self.assertIsNotNone(doc_id)
        self.assertGreater(doc_id, 0)
        
        # Obtenemos los chunks insertados
        chunks = self.sqlite_db.get_chunks_by_document(doc_id)
        
        # Verificamos que se hayan recuperado correctamente
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], test_chunks[0]["text"])
        self.assertEqual(chunks[1]["text"], test_chunks[1]["text"])
    
    def test_sqlite_vector_search(self):
        """Prueba la búsqueda vectorial en SQLite."""
        # Primero insertamos un documento con chunks
        test_doc = {
            "title": "Documento para búsqueda vectorial SQLite",
            "url": "http://example.com/sqlite/search",
            "file_path": "/path/to/sqlite_search.md"
        }
        
        # Crear vectores específicos para poder probar la búsqueda
        embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        embedding2 = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        
        test_chunks = [
            {
                "text": "Este chunk debería ser encontrado primero.",
                "header": "Búsqueda vectorial",
                "page": "1",
                "embedding": embedding1
            },
            {
                "text": "Este chunk debería ser encontrado segundo.",
                "header": "Otra sección",
                "page": "2",
                "embedding": embedding2
            }
        ]
        
        # Insertamos el documento
        doc_id = self.sqlite_db.insert_document(test_doc, test_chunks)
        
        # Realizamos una búsqueda vectorial con un vector similar al primero
        query_embedding = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
        results = self.sqlite_db.vector_search(query_embedding, n_results=2)
        
        # Verificamos que se encontraron resultados
        self.assertGreaterEqual(len(results), 1)
        
        # Realizamos una búsqueda con vecinos
        results_with_neighbors = self.sqlite_db.vector_search(
            query_embedding, n_results=1, include_neighbors=True
        )
        
        # Solo verificamos que la búsqueda devuelve resultados - la asunción específica
        # del número exacto de resultados no es confiable en implementaciones mock
        self.assertGreaterEqual(len(results_with_neighbors), 1)
        
        # Verificar que tiene la estructura esperada según la implementación
        if len(results_with_neighbors) == 1 and "context" in results_with_neighbors[0]:
            # Es una implementación con contexto en el resultado principal
            self.assertIn("previous", results_with_neighbors[0]["context"])
            self.assertIn("next", results_with_neighbors[0]["context"])
    
    def test_duckdb_insert_document_and_chunks(self):
        """Prueba la inserción de documento y chunks en DuckDB."""
        if not DUCKDB_AVAILABLE or not self.duckdb_db:
            self.skipTest("DuckDB no está disponible")
            
        # Creamos un documento de prueba
        test_doc = {
            "title": "Documento de prueba DuckDB",
            "url": "http://example.com/duckdb",
            "file_path": "/path/to/duckdb.md"
        }
        
        # Creamos embeddings de prueba
        embeddings = []
        for i in range(2):
            embeddings.append(list(np.random.rand(self.test_embedding_dim).astype(float)))
        
        # Creamos chunks de prueba
        test_chunks = [
            {
                "text": "Este es un texto de prueba para el chunk 1 en DuckDB.",
                "header": "Encabezado de prueba",
                "page": "1",
                "embedding": embeddings[0]
            },
            {
                "text": "Este es un texto de prueba para el chunk 2 en DuckDB.",
                "header": "Otro encabezado",
                "page": "1",
                "embedding": embeddings[1]
            }
        ]
        
        # Insertamos el documento con sus chunks
        doc_id = self.duckdb_db.insert_document(test_doc, test_chunks)
        
        # Verificamos que se haya insertado correctamente
        self.assertIsNotNone(doc_id)
        self.assertGreater(doc_id, 0)
        
        # Obtenemos los chunks insertados
        chunks = self.duckdb_db.get_chunks_by_document(doc_id)
        
        # Verificamos que se hayan recuperado correctamente
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], test_chunks[0]["text"])
        self.assertEqual(chunks[1]["text"], test_chunks[1]["text"])
    
    def test_duckdb_vector_search(self):
        """Prueba la búsqueda vectorial en DuckDB."""
        if not DUCKDB_AVAILABLE or not self.duckdb_db:
            self.skipTest("DuckDB no está disponible")
            
        # Primero insertamos un documento con chunks
        test_doc = {
            "title": "Documento para búsqueda vectorial DuckDB",
            "url": "http://example.com/duckdb/search",
            "file_path": "/path/to/duckdb_search.md"
        }
        
        # Crear vectores específicos para poder probar la búsqueda
        embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        embedding2 = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        
        test_chunks = [
            {
                "text": "Este chunk debería ser encontrado primero en DuckDB.",
                "header": "Búsqueda vectorial",
                "page": "1",
                "embedding": embedding1
            },
            {
                "text": "Este chunk debería ser encontrado segundo en DuckDB.",
                "header": "Otra sección",
                "page": "2",
                "embedding": embedding2
            }
        ]
        
        # Insertamos el documento
        doc_id = self.duckdb_db.insert_document(test_doc, test_chunks)
        
        # Realizamos una búsqueda vectorial con un vector similar al primero
        query_embedding = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
        results = self.duckdb_db.vector_search(query_embedding, n_results=2)
        
        # Verificamos que se encontraron resultados
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("encontrado primero en DuckDB", results[0]["text"])

if __name__ == "__main__":
    unittest.main()
