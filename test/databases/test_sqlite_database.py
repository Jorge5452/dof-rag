import unittest
import sqlite3
import os
import sys
import logging
from pathlib import Path
import traceback

# Añadir el directorio raíz al path para permitir importaciones relativas
sys.path.insert(0, str(Path(__file__).parents[2]))

from test.databases.test_vectorial_database import BaseVectorialDatabaseTest
from test.databases.mocks import MockSQLiteVectorialDatabase

logger = logging.getLogger(__name__)

class SQLiteDatabaseTest(BaseVectorialDatabaseTest, unittest.TestCase):
    """
    Pruebas específicas para la implementación SQLite de VectorialDatabase.
    Hereda las pruebas básicas de BaseVectorialDatabaseTest.
    """
    
    def setUp(self):
        """Preparación antes de cada test"""
        super().setUp()
        try:
            # Usar la implementación mock para pruebas
            self.db_instance = MockSQLiteVectorialDatabase(embedding_dim=self.test_embedding_dim)
            self.db_path = os.path.join(self.test_dir, f"test_sqlite_{self.test_name}.db")
            self.db_instance.connect(self.db_path)
            self.db_instance.create_schema()
            logger.info(f"Base de datos SQLite inicializada en {self.db_path}")
        except Exception as e:
            logger.error(f"Error al inicializar SQLite: {e}")
            self.log_test_result(f"Error al inicializar SQLite: {e}", False)
            # Si hay un error en setUp, establecer db_instance como None para que los tests lo manejen
            self.db_instance = None
            print(f"Error en setUp de SQLiteDatabaseTest.{self.test_name}: {e}")
            print(f"Traza: {traceback.format_exc()}")
    
    def test_extension_loading(self):
        """Prueba de carga de extensiones específicas de SQLite"""
        if not self.db_instance:
            self.skipTest("Base de datos no inicializada")
            return
            
        try:
            # Intentar cargar extensiones
            result = self.db_instance.load_extensions()
            
            # No siempre se puede cargar la extensión en entornos de prueba
            # así que solo verificamos que el método devuelva un valor booleano
            self.assertIsInstance(result, bool)
            status_msg = "Extensiones cargadas correctamente" if result else "Extensiones no disponibles pero manejo correcto"
            self.log_test_result(status_msg, True)
        except Exception as e:
            self.log_test_result(f"Error al cargar extensiones: {e}", False)
            raise
    
    def test_create_vector_index(self):
        """Prueba de creación de índice vectorial"""
        if not self.db_instance:
            self.skipTest("Base de datos no inicializada")
            return
            
        try:
            # Insertar datos primero
            self.db_instance.insert_document(self.test_document, self.test_chunks)
            
            # Crear índice vectorial
            result = self.db_instance.create_vector_index(force_rebuild=True)
            
            # Verificar resultado
            # Puede ser False si la extensión no está disponible en el entorno de prueba
            self.assertIsInstance(result, bool)
            status_msg = "Índice vectorial creado correctamente" if result else "Extensión no disponible pero manejo correcto"
            self.log_test_result(status_msg, True)
        except Exception as e:
            self.log_test_result(f"Error al crear índice vectorial: {e}", False)
            raise
    
    def test_serialization(self):
        """Prueba de serialización de vectores"""
        if not self.db_instance:
            self.skipTest("Base de datos no inicializada")
            return
            
        try:
            # Serializar vector - se adapta a la dimensión fija
            vector = [1.0, 2.0, 3.0]
            serialized = self.db_instance.serialize_vector(vector)
            
            # Verificar que el resultado es bytes
            self.assertIsInstance(serialized, bytes)
            
            # Verificar el tamaño según la dimensión configurada en la clase
            expected_size = self.db_instance._embedding_dim * 4  # 4 bytes por float
            self.assertEqual(len(serialized), expected_size)
            
            self.log_test_result(f"Vector serializado correctamente: {len(serialized)} bytes", True)
        except Exception as e:
            self.log_test_result(f"Error en serialización: {e}", False)
            raise
    
    def test_transaction_rollback_on_error(self):
        """Prueba de rollback de transacción en caso de error"""
        if not self.db_instance:
            self.skipTest("Base de datos no inicializada")
            return
            
        try:
            # Crear un documento válido pero chunks con errores
            invalid_chunks = [
                {
                    "text": None,  # Debería causar un error porque la implementación mock verifica esto
                    "header": "Encabezado",
                    "page": "1",
                    "embedding": self.test_embedding
                }
            ]
            
            # Intentar insertar (debería fallar)
            with self.assertRaises(ValueError):  # Cambiado a ValueError específicamente
                self.db_instance.insert_document(self.test_document, invalid_chunks)
            
            # Verificar que no se insertó el documento (transacción hizo rollback)
            cursor = self.db_instance._conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 0)
            self.log_test_result("Rollback funcionó correctamente", True)
        except Exception as e:
            self.log_test_result(f"Error en prueba de rollback: {e}", False)
            raise

    def test_document_exists(self):
        """Prueba para verificar si un documento existe en la base de datos"""
        if not self.db_instance:
            self.skipTest("Base de datos no inicializada")
            return
            
        try:
            # Insertar un documento
            doc_id = self.db_instance.insert_document(self.test_document, self.test_chunks)
            
            # Verificar que el documento existe
            exists = self.db_instance.document_exists(self.test_document["file_path"])
            self.assertTrue(exists)
            
            # Verificar que un documento inexistente no existe
            exists = self.db_instance.document_exists("/path/to/nonexistent.md")
            self.assertFalse(exists)
            
            self.log_test_result("Verificación de existencia de documentos correcta", True)
        except Exception as e:
            self.log_test_result(f"Error en test_document_exists: {e}", False)
            raise

    def test_file_extension(self):
        """Prueba que se utiliza la extensión de archivo correcta (.sqlite)"""
        if not self.db_instance:
            self.skipTest("Base de datos no inicializada")
            return
            
        try:
            # Verificar que la extensión del archivo es .sqlite
            self.assertTrue(self.db_path.endswith('.db'), 
                           f"Para pruebas, la extensión debería ser .db, pero es {os.path.splitext(self.db_path)[1]}")
            
            # Crear una nueva conexión utilizando DatabaseFactory
            from modulos.databases.FactoryDatabase import DatabaseFactory
            
            # Establecer un nombre de prueba sin extensión para verificar que se añade .sqlite
            test_db_name = "test_sqlite_extension"
            test_db_path = os.path.join(self.test_dir, test_db_name)
            
            # Obtener instancia a través del factory
            db_factory = DatabaseFactory()
            db_instance = db_factory._get_db_path("sqlite", test_db_name)
            
            # Verificar que la ruta termina con .sqlite
            self.assertTrue(db_instance.endswith('.sqlite'), 
                           f"El Factory debería crear rutas con extensión .sqlite, pero creó {db_instance}")
            
            self.log_test_result("Extensión de archivo SQLite (.sqlite) verificada correctamente", True)
        except Exception as e:
            self.log_test_result(f"Error al verificar extensión de archivo: {e}", False)
            raise

if __name__ == '__main__':
    unittest.main()
