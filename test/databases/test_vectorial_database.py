import os
import sys
import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import logging
import traceback
import json

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parents[2]))

# Importar las constantes y utilidades
from test.utils.constants import DATABASE_RESULTS_DIR
from test.utils.environment import ensure_dir_exists, get_test_result_path

from modulos.databases.VectorialDatabase import VectorialDatabase
from test.databases.utils import generate_random_embedding

class BaseVectorialDatabaseTest:
    """
    Clase base para pruebas de bases de datos vectoriales.
    
    Esta clase contiene pruebas comunes que todas las implementaciones
    de bases de datos vectoriales deben pasar.
    """
    
    @classmethod
    def setUpClass(cls):
        """Configuración antes de todas las pruebas"""
        # Generar directorio para resultados usando la ruta estándar
        cls.results_dir = ensure_dir_exists(DATABASE_RESULTS_DIR)
        
        # Dimensión de embedding para pruebas (usar un valor pequeño por eficiencia)
        cls.test_embedding_dim = 10
        
    def setUp(self):
        """Preparación antes de cada prueba"""
        # Crear directorio temporal para tests - DEBE SER LO PRIMERO
        self.test_dir = tempfile.mkdtemp()
        
        # Obtener nombre del test actual
        self.test_name = self.id().split('.')[-1]
        
        # Ahora podemos usar self.test_dir con seguridad en los logs
        logger.info(f"[{self.__class__.__name__}] Pruebas de base de datos en: {self.test_dir}")
        logger.info(f"[{self.__class__.__name__}] Resultados guardados en: {self.results_dir}")
        logger.info(f"[{self.__class__.__name__}] Iniciando test: {self.test_name}")
        
        # Crear documento de prueba
        self.test_document = {
            "title": "Documento de prueba",
            "url": "http://example.com/test",
            "file_path": "/path/to/test.md"
        }
        
        # Crear embedding de prueba fijo para consistencia en tests
        self.test_embedding = generate_random_embedding(self.test_embedding_dim)
        
        # Crear chunks de prueba con el embedding fijo
        self.test_chunks = [
            {
                "text": "Este es el primer chunk de prueba para pruebas de búsqueda vectorial.",
                "header": "Sección 1",
                "page": "1",
                "embedding": self.test_embedding
            },
            {
                "text": "Este es el segundo chunk que debería ser menos relevante para la búsqueda.",
                "header": "Sección 2",
                "page": "1",
                "embedding": generate_random_embedding(self.test_embedding_dim)
            }
        ]
        
        # La implementación específica debe establecer self.db_instance
        self.db_instance = None
    
    def tearDown(self):
        """Limpieza después de cada prueba"""
        # Limpiar después de cada prueba
        if hasattr(self, 'db_instance') and self.db_instance:
            try:
                self.db_instance.close_connection()
                logger.info(f"[{self.__class__.__name__}] Conexión cerrada correctamente para: {self.test_name}")
            except Exception as e:
                logger.error(f"Error al cerrar conexión: {e}")
        
        # Limpiar directorio temporal si existe
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
                logger.info(f"[{self.__class__.__name__}] Directorio temporal eliminado: {self.test_dir}")
            except Exception as e:
                logger.error(f"Error al eliminar directorio temporal: {e}")
    
    def log_test_result(self, message, success=True):
        """
        Registra el resultado de la prueba en un archivo de log
        
        Args:
            message (str): Mensaje descriptivo
            success (bool): Si la prueba fue exitosa
        """
        try:
            # Usar la ruta estandarizada para los archivos de resultados
            log_file = self.results_dir / f"{self.db_instance.__class__.__name__}_test_results.log"
            
            status = "ÉXITO" if success else "FALLO"
            log_entry = f"[{status}] - Test: {self.test_name} - {message}\n"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            logger.info(f"{status}: {self.__class__.__name__}.{self.test_name} - {message}")
        except Exception as e:
            logger.error(f"Error al registrar resultado de prueba: {e}")
    
    def test_connect_and_close(self):
        """Prueba la conexión y cierre de la base de datos"""
        if not self.db_instance:
            self.skipTest("Base de datos no inicializada")
            return
            
        try:
            # Verificar que la conexión está activa
            self.assertTrue(hasattr(self.db_instance, '_conn'))
            self.assertIsNotNone(self.db_instance._conn)
            
            # Cerrar conexión
            self.db_instance.close_connection()
            
            # Verificar que se cerró correctamente - esto depende de la implementación
            # Solo verificamos que el método no genere errores
            
            # Reconectar para otros tests
            db_path = self.db_path if hasattr(self, 'db_path') else ":memory:"
            self.db_instance.connect(db_path)
            
            self.log_test_result("Conexión y cierre exitosos", True)
        except Exception as e:
            self.log_test_result(f"Error en conexión o cierre: {e}", False)
            raise
    
    def test_schema_creation(self):
        """Prueba la creación del esquema de la base de datos"""
        if not self.db_instance:
            self.skipTest("Base de datos no inicializada")
            return
            
        try:
            # Crear esquema (este método debería ser idempotente)
            self.db_instance.create_schema()
            
            # El esquema ya se creó durante setUp, así que solo verificamos que no falle
            self.log_test_result("Creación del esquema exitoso", True)
        except Exception as e:
            self.log_test_result(f"Error en creación de esquema: {e}", False)
            raise
    
    def test_document_insertion(self):
        """Prueba la inserción de un documento con sus chunks"""
        if not self.db_instance:
            self.skipTest("Base de datos no inicializada")
            return
            
        try:
            # Insertar documento con sus chunks
            document_id = self.db_instance.insert_document(self.test_document, self.test_chunks)
            
            # Verificar que se insertó correctamente
            self.assertIsNotNone(document_id)
            self.assertGreater(document_id, 0)
            
            self.log_test_result(f"Documento insertado con ID: {document_id}", True)
        except Exception as e:
            self.log_test_result(f"Error en inserción de documento: {e}", False)
            raise
    
    def test_get_chunks_by_document(self):
        """Prueba la recuperación de chunks por documento"""
        if not self.db_instance:
            self.skipTest("Base de datos no inicializada")
            return
            
        try:
            # Insertar documento con sus chunks
            document_id = self.db_instance.insert_document(self.test_document, self.test_chunks)
            
            # Obtener chunks del documento
            chunks = self.db_instance.get_chunks_by_document(document_id)
            
            # Verificar resultados
            self.assertIsNotNone(chunks)
            self.assertEqual(len(chunks), len(self.test_chunks))
            
            self.log_test_result(f"Recuperados {len(chunks)} chunks para documento {document_id}", True)
        except Exception as e:
            self.log_test_result(f"Error en recuperación de chunks: {e}", False)
            raise
    
    def test_vector_search(self):
        """Prueba la búsqueda vectorial"""
        if not self.db_instance:
            self.skipTest("Base de datos no inicializada")
            return
            
        try:
            # Insertar documento con sus chunks
            self.db_instance.insert_document(self.test_document, self.test_chunks)
            
            # Realizar búsqueda usando el mismo embedding del primer chunk
            results = self.db_instance.vector_search(self.test_embedding, n_results=2)
            
            # Verificar resultados
            self.assertIsNotNone(results)
            self.assertGreaterEqual(len(results), 1)
            
            self.log_test_result(f"Búsqueda vectorial exitosa con {len(results)} resultados", True)
        except Exception as e:
            self.log_test_result(f"Error en búsqueda vectorial: {e}", False)
            raise
    
    def test_vector_search_with_neighbors(self):
        """Prueba la búsqueda vectorial con inclusión de vecinos"""
        if not self.db_instance:
            self.skipTest("Base de datos no inicializada")
            return
            
        try:
            # Insertar documento con sus chunks
            self.db_instance.insert_document(self.test_document, self.test_chunks)
            
            # Realizar búsqueda con vecinos
            results = self.db_instance.vector_search(
                self.test_embedding, 
                n_results=1,
                include_neighbors=True
            )
            
            # Verificar que hay resultados y son válidos
            self.assertIsNotNone(results)
            self.assertGreaterEqual(len(results), 1)
            
            # Dependiendo de la implementación, puede incluir context o devolver más resultados
            self.log_test_result(f"Búsqueda vectorial con vecinos exitosa: {len(results)} resultados", True)
        except Exception as e:
            self.log_test_result(f"Error en búsqueda vectorial con vecinos: {e}", False)
            raise
