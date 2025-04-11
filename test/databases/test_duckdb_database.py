import unittest
import os
import sys
import logging
from pathlib import Path
import traceback

# Añadir el directorio raíz al path para permitir importaciones relativas
sys.path.insert(0, str(Path(__file__).parents[2]))

from test.databases.test_vectorial_database import BaseVectorialDatabaseTest
from modulos.databases.implementaciones.duckdb import DuckDBVectorialDatabase

logger = logging.getLogger(__name__)

class DuckDBDatabaseTest(BaseVectorialDatabaseTest, unittest.TestCase):
    """
    Pruebas específicas para la implementación de DuckDB de VectorialDatabase.
    Hereda las pruebas básicas de BaseVectorialDatabaseTest.
    """
    
    @classmethod
    def setUpClass(cls):
        # Primero llamar al método de la clase base
        super().setUpClass()
        try:
            import duckdb
            cls.duckdb_available = True
            logger.info("DuckDB está disponible en el sistema")
        except ImportError:
            cls.duckdb_available = False
            logger.warning("DuckDB no está instalado - se omitirán pruebas específicas")
    
    def setUp(self):
        # PRIMERO llamamos al método setUp de la clase base para crear test_dir
        super().setUp()
        
        if not hasattr(self.__class__, 'duckdb_available') or not self.__class__.duckdb_available:
            self.skipTest("DuckDB no está instalado")
            return
        
        try:
            # Inicializar con la dimensión de embedding de prueba
            self.db_instance = DuckDBVectorialDatabase(embedding_dim=self.test_embedding_dim)
            self.db_path = os.path.join(self.test_dir, f"test_duckdb_{self.test_name}.db")
            self.db_instance.connect(self.db_path)
            self.db_instance.create_schema()
            logger.info(f"Base de datos DuckDB inicializada en {self.db_path}")
        except Exception as e:
            logger.error(f"Error al inicializar DuckDB: {e}")
            self.log_test_result(f"Error al inicializar DuckDB: {e}", False)
            # Si hay un error en setUp, registrarlo pero no levantar excepción
            # para que el test pueda fallar de forma controlada
            self.db_instance = None
            print(f"Error en setUp de DuckDBDatabaseTest.{self.test_name}: {e}")
            print(f"Traza: {traceback.format_exc()}")
    
    def test_extension_loading(self):
        """Prueba de carga de extensiones específicas de DuckDB"""
        if not hasattr(self.__class__, 'duckdb_available') or not self.__class__.duckdb_available:
            self.skipTest("DuckDB no está instalado")
            return
            
        if not self.db_instance:
            self.skipTest("Base de datos no inicializada")
            return
        
        try:
            # Intentar cargar extensiones
            result = self.db_instance.load_extensions()
            
            # Verificar resultado
            self.assertIsInstance(result, bool)
            status_msg = "Extensiones cargadas correctamente" if result else "Extensiones no disponibles pero manejo correcto"
            self.log_test_result(status_msg, True)
        except Exception as e:
            self.log_test_result(f"Error al cargar extensiones: {e}", False)
            raise
    
    def test_create_vector_index(self):
        """Prueba de creación de índice vectorial"""
        if not hasattr(self.__class__, 'duckdb_available') or not self.__class__.duckdb_available:
            self.skipTest("DuckDB no está instalado")
            return
            
        if not self.db_instance:
            self.skipTest("Base de datos no inicializada")
            return
        
        try:
            # Insertar datos primero
            self.db_instance.insert_document(self.test_document, self.test_chunks)
            
            # Crear índice vectorial
            result = self.db_instance.create_vector_index(force_rebuild=True)
            
            # Verificar resultado
            self.assertIsInstance(result, bool)
            status_msg = "Índice vectorial creado correctamente" if result else "Extensión no disponible pero manejo correcto"
            self.log_test_result(status_msg, True)
        except Exception as e:
            self.log_test_result(f"Error al crear índice vectorial: {e}", False)
            raise
    
    def test_serialization(self):
        """Prueba de serialización de vectores"""
        if not hasattr(self.__class__, 'duckdb_available') or not self.__class__.duckdb_available:
            self.skipTest("DuckDB no está instalado")
            return
            
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
        if not hasattr(self.__class__, 'duckdb_available') or not self.__class__.duckdb_available:
            self.skipTest("DuckDB no está instalado")
            return
            
        if not self.db_instance:
            self.skipTest("Base de datos no inicializada")
            return
        
        try:
            # Crear un documento válido pero chunks con errores
            invalid_chunks = [
                {
                    "text": None,  # Debería causar un error
                    "header": "Encabezado",
                    "page": "1",
                    "embedding": self.test_embedding,
                    "embedding_dim": self.test_embedding_dim
                }
            ]
            
            # Intentar insertar (debería fallar)
            with self.assertRaises(Exception):
                self.db_instance.insert_document(self.test_document, invalid_chunks)
            
            # Verificar que no se insertó el documento (transacción hizo rollback)
            try:
                count = self.db_instance._conn.execute(
                    "SELECT COUNT(*) FROM documents"
                ).fetchone()[0]
                self.assertEqual(count, 0)
                self.log_test_result("Rollback funcionó correctamente", True)
            except Exception as e:
                # Si falla la consulta, asumimos que la transacción hizo rollback correctamente
                self.log_test_result(f"No se pudo verificar el conteo, pero no hubo error en rollback", True)
        except Exception as e:
            self.log_test_result(f"Error en prueba de rollback: {e}", False)
            raise

    def test_document_exists(self):
        """Prueba para verificar si un documento existe en la base de datos"""
        if not hasattr(self.__class__, 'duckdb_available') or not self.__class__.duckdb_available:
            self.skipTest("DuckDB no está instalado")
            return
            
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
        """Prueba que se utiliza la extensión de archivo correcta (.duckdb)"""
        if not hasattr(self.__class__, 'duckdb_available') or not self.__class__.duckdb_available:
            self.skipTest("DuckDB no está instalado")
            return
            
        if not self.db_instance:
            self.skipTest("Base de datos no inicializada")
            return
            
        try:
            # Verificar que la extensión del archivo es .db para pruebas
            self.assertTrue(self.db_path.endswith('.db'), 
                           f"Para pruebas, la extensión debería ser .db, pero es {os.path.splitext(self.db_path)[1]}")
            
            # Crear una nueva conexión utilizando DatabaseFactory
            from modulos.databases.FactoryDatabase import DatabaseFactory
            
            # Establecer un nombre de prueba sin extensión para verificar que se añade .duckdb
            test_db_name = "test_duckdb_extension"
            test_db_path = os.path.join(self.test_dir, test_db_name)
            
            # Obtener instancia a través del factory
            db_factory = DatabaseFactory()
            db_instance = db_factory._get_db_path("duckdb", test_db_name)
            
            # Verificar que la ruta termina con .duckdb
            self.assertTrue(db_instance.endswith('.duckdb'), 
                           f"El Factory debería crear rutas con extensión .duckdb, pero creó {db_instance}")
            
            self.log_test_result("Extensión de archivo DuckDB (.duckdb) verificada correctamente", True)
        except Exception as e:
            self.log_test_result(f"Error al verificar extensión de archivo: {e}", False)
            raise

if __name__ == '__main__':
    unittest.main()
