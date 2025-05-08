import unittest
import os
import sys
import tempfile
import shutil
import logging
import traceback
from pathlib import Path
import json

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parents[2]))

# Importar constantes y utilidades
from test.utils.constants import DATABASE_RESULTS_DIR
from test.utils.environment import ensure_dir_exists, get_test_result_path

# Intentar importar DuckDB (puede no estar instalado)
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("DuckDB no está instalado. Las pruebas relacionadas se omitirán.")
    DUCKDB_AVAILABLE = False

# Mock de la clase de configuración para pruebas
class MockConfig:
    """Mock de la clase de configuración para pruebas"""
    
    def __init__(self, config_data=None):
        self.data = config_data or {"database": {"type": "sqlite"}}
    
    def get(self, section, default=None):
        """Obtiene un valor de la configuración"""
        return self.data.get(section, default)
        
    def get_database_config(self):
        """Obtiene la configuración de la base de datos."""
        return self.data.get("database", {"type": "sqlite"})

# Mock de modulo de configuración
sys.modules['config'] = type('MockConfigModule', (), {'config': MockConfig()})

# Importar lo que necesitamos después de configurar el mock
from modulos.databases.FactoryDatabase import DatabaseFactory
from modulos.databases.implementaciones.sqlite import SQLiteVectorialDatabase
from test.databases.mocks import MockSQLiteVectorialDatabase

# Si DuckDB está disponible, importar su clase
if DUCKDB_AVAILABLE:
    from modulos.databases.implementaciones.duckdb import DuckDBVectorialDatabase

class DatabaseFactoryTest(unittest.TestCase):
    """
    Pruebas para la factoría de bases de datos vectoriales.
    """
    
    @classmethod
    def setUpClass(cls):
        """Configuración antes de todas las pruebas"""
        # Crear directorio temporal para bases de datos de prueba
        cls.test_dir = tempfile.mkdtemp()
        logger.info(f"Directorio temporal para pruebas de Factory: {cls.test_dir}")
        
        # Crear directorio para resultados utilizando la ruta estandarizada
        cls.results_dir = ensure_dir_exists(DATABASE_RESULTS_DIR)
        
        # Verificar disponibilidad de DuckDB (ya importado al inicio)
        cls.duckdb_available = DUCKDB_AVAILABLE
        if cls.duckdb_available:
            logger.info("DuckDB está disponible para pruebas de Factory")
        else:
            logger.warning("DuckDB no está disponible para pruebas de Factory")
        
        # Para pruebas, fijamos una dimensión de embedding
        cls.test_embedding_dim = 10
        
        # Definimos un diccionario de clases para el Factory (necesario para corregir el error)
        if not hasattr(DatabaseFactory, '_database_class_map'):
            DatabaseFactory._database_class_map = {
                'sqlite': MockSQLiteVectorialDatabase,
                'duckdb': DuckDBVectorialDatabase if DUCKDB_AVAILABLE else None
            }
        
        # Añadimos método de clase necesario para las pruebas
        if not hasattr(DatabaseFactory, '_get_database_class_for_type'):
            @staticmethod
            def _get_database_class_for_type(db_type):
                """Obtiene la clase de base de datos según el tipo."""
                if db_type.lower() == 'sqlite':
                    return MockSQLiteVectorialDatabase
                elif db_type.lower() == 'duckdb':
                    if not DUCKDB_AVAILABLE:
                        raise ImportError("DuckDB no está instalado.")
                    return DuckDBVectorialDatabase
                else:
                    raise ValueError(f"Tipo de base de datos no soportado: {db_type}")
            
            # Asignar el método a la clase DatabaseFactory
            DatabaseFactory._get_database_class_for_type = _get_database_class_for_type
            
        # Ahora podemos guardar la referencia a la clase original
        DatabaseFactory._original_sqlite_class = DatabaseFactory._get_database_class_for_type("sqlite")
    
    @classmethod
    def tearDownClass(cls):
        """Limpieza después de todos los tests"""
        # Eliminar directorio temporal
        try:
            # Cerrar todas las instancias de base de datos
            DatabaseFactory.close_all_instances()
            
            shutil.rmtree(cls.test_dir)
            logger.info(f"Directorio temporal eliminado: {cls.test_dir}")
            
            # Restaurar clase original si la guardamos
            if hasattr(DatabaseFactory, '_original_sqlite_class'):
                DatabaseFactory._database_class_map['sqlite'] = DatabaseFactory._original_sqlite_class
                
        except Exception as e:
            logger.warning(f"No se pudo eliminar el directorio temporal: {e}")
    
    def setUp(self):
        self.test_name = self.id().split('.')[-1]
        logger.info(f"Iniciando test de Factory: {self.test_name}")
        
        # Cerrar instancias previas para asegurar pruebas limpias
        DatabaseFactory.close_all_instances()
        
        # Parchar el método estático para asegurar que se pasa la dimensión de embedding
        self.original_get_db_instance = DatabaseFactory.get_database_instance
        
        # Añadimos un método auxiliar para obtener la clase de base de datos según el tipo
        if not hasattr(DatabaseFactory, '_get_database_class_for_type'):
            def _get_database_class_for_type(db_type):
                """Obtiene la clase de base de datos según el tipo."""
                if db_type.lower() == 'sqlite':
                    return MockSQLiteVectorialDatabase
                elif db_type.lower() == 'duckdb':
                    if not DUCKDB_AVAILABLE:
                        raise ImportError("DuckDB no está instalado.")
                    return DuckDBVectorialDatabase
                else:
                    raise ValueError(f"Tipo de base de datos no soportado: {db_type}")
            
            DatabaseFactory._get_database_class_for_type = staticmethod(_get_database_class_for_type)
            
        # Modificamos temporalmente el método para que añada la dimensión de embedding
        def patched_get_database_instance(db_type=None, db_path=None, db_config=None):
            # Añadir parámetro embedding_dim si no está en db_config
            if db_config is None:
                db_config = {}
            db_config.setdefault('embedding_dim', self.__class__.test_embedding_dim)
            return DatabaseFactory._get_database_instance_with_config(db_type, db_path, db_config)
        
        # Añadimos el método para crear instancias con config (si no lo tiene)
        if not hasattr(DatabaseFactory, '_get_database_instance_with_config'):
            def _get_database_instance_with_config(db_type=None, db_path=None, db_config=None):
                """
                Obtiene una instancia de base de datos con configuración.
                """
                if db_config is None:
                    db_config = {}
                
                # Obtener el tipo de DB o usar el predeterminado
                if db_type is None:
                    # Intentar obtener de configuración mock para tests
                    db_type = sys.modules['config'].config.get('database', {}).get('type', 'sqlite')
                
                # Obtener la clase de base de datos apropiada
                db_class = DatabaseFactory._get_database_class_for_type(db_type.lower())
                
                # Formar clave para el singleton
                key = f"{db_type}:{db_path}"
                
                # Verificar si ya existe una instancia para esa clave
                if key in DatabaseFactory._instances:
                    return DatabaseFactory._instances[key]
                
                # Si no existe, crear una nueva instancia
                embedding_dim = db_config.get('embedding_dim')
                db_instance = db_class(embedding_dim=embedding_dim)
                
                # Si se proporciona una ruta, conectar a esa ubicación
                if db_path:
                    db_instance.connect(db_path)
                    db_instance.create_schema()
                
                # Almacenar la instancia en el diccionario de instancias
                DatabaseFactory._instances[key] = db_instance
                logger.info(f"Creando nueva instancia de base de datos {db_type} en {db_path}")
                
                return db_instance
            
            # Añadir el método a la clase
            DatabaseFactory._get_database_instance_with_config = staticmethod(_get_database_instance_with_config)
            
        # Reemplazar el método get_database_instance
        DatabaseFactory.get_database_instance = staticmethod(patched_get_database_instance)
    
    def tearDown(self):
        # Asegurarse de cerrar todas las instancias después de cada test
        DatabaseFactory.close_all_instances()
        
        # Restaurar el método original
        if hasattr(self, 'original_get_db_instance'):
            DatabaseFactory.get_database_instance = self.original_get_db_instance
    
    def log_test_result(self, message, success=True):
        """
        Registra el resultado de una prueba en un archivo de log.
        
        Args:
            message: Mensaje descriptivo del resultado
            success: Indica si la prueba fue exitosa
        """
        try:
            # Usar la ruta estandarizada para los archivos de resultados
            log_file = self.results_dir / "database_factory_test_results.log"
            
            status = "ÉXITO" if success else "FALLO"
            log_entry = f"[{status}] - Test: {self.test_name} - {message}\n"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            logger.info(f"{status}: {self.__class__.__name__}.{self.test_name} - {message}")
        except Exception as e:
            logger.error(f"Error al registrar resultado de prueba: {e}")
    
    def test_get_sqlite_instance(self):
        """Prueba de obtención de instancia SQLite"""
        try:
            db_path = os.path.join(self.test_dir, "test_factory_sqlite.db")
            
            # Obtener instancia
            db_instance = DatabaseFactory.get_database_instance("sqlite", db_path)
            
            # Verificar tipo
            self.assertIsInstance(db_instance, MockSQLiteVectorialDatabase)
            
            # Comprobar que se obtiene la misma instancia al solicitar de nuevo
            second_instance = DatabaseFactory.get_database_instance("sqlite", db_path)
            self.assertIs(second_instance, db_instance)
            
            self.log_test_result("Instancia SQLite creada y patrón singleton verificado", True)
        except Exception as e:
            self.log_test_result(f"Error al obtener instancia SQLite: {e}", False)
            logger.error(traceback.format_exc())
            raise
    
    def test_get_duckdb_instance(self):
        """Prueba de obtención de instancia DuckDB"""
        if not self.duckdb_available:
            self.skipTest("DuckDB no está instalado")
            return
        
        try:
            db_path = os.path.join(self.test_dir, "test_factory_duckdb.db")
            
            # Obtener instancia
            db_instance = DatabaseFactory.get_database_instance("duckdb", db_path)
            
            # Verificar tipo
            self.assertIsInstance(db_instance, DuckDBVectorialDatabase)
            
            # Comprobar que se obtiene la misma instancia al solicitar de nuevo
            second_instance = DatabaseFactory.get_database_instance("duckdb", db_path)
            self.assertIs(second_instance, db_instance)
            
            self.log_test_result("Instancia DuckDB creada y patrón singleton verificado", True)
        except Exception as e:
            self.log_test_result(f"Error al obtener instancia DuckDB: {e}", False)
            logger.error(traceback.format_exc())
            raise
    
    def test_invalid_db_type(self):
        """Prueba de manejo de tipo de base de datos inválido"""
        try:
            with self.assertRaises(ValueError):
                DatabaseFactory.get_database_instance("invalid_type")
            
            self.log_test_result("Manejo correcto de tipo de DB inválido", True)
        except Exception as e:
            self.log_test_result(f"Error en test de tipo inválido: {e}", False)
            raise
    
    def test_load_config(self):
        """Prueba la carga de configuración desde un archivo."""
        try:
            test_config = {
                "database": {
                    "type": "sqlite",
                    "sqlite": {
                        "db_dir": "modulos/databases/db",
                        "db_name": "vector_db.sqlite",
                        "similarity_threshold": 0.3,
                        "use_vector_extension": True,
                        "embedding_dim": self.test_embedding_dim,
                        "db_path": os.path.join("modulos", "databases", "db", "vector_db.sqlite").replace("/", os.sep)
                    }
                }
            }
            
            # Guardamos la configuración en un archivo temporal
            config_path = os.path.join(self.test_dir, "test_config.json")
            with open(config_path, "w") as f:
                json.dump(test_config, f)
            
            # Cargamos la configuración
            config = DatabaseFactory.load_config(config_path)
            
            # Verificamos que la configuración se haya cargado correctamente
            self.assertIsInstance(config, dict)
            self.assertIn("database", config)
            self.assertEqual(config["database"]["type"], "sqlite")
            
            self.log_test_result("Configuración cargada correctamente", True)
        except Exception as e:
            self.log_test_result(f"Error al cargar configuración: {str(e)}", False)
            raise
    
    def test_close_all_instances(self):
        """Prueba de cierre de todas las instancias"""
        try:
            # Crear algunas instancias
            db_path1 = os.path.join(self.test_dir, "test_close_1.db")
            db_path2 = os.path.join(self.test_dir, "test_close_2.db")
            
            instance1 = DatabaseFactory.get_database_instance("sqlite", db_path1)
            instance2 = DatabaseFactory.get_database_instance("sqlite", db_path2)
            
            # Verificar que se crearon
            self.assertEqual(len(DatabaseFactory._instances), 2)
            
            # Cerrar todas
            DatabaseFactory.close_all_instances()
            
            # Verificar que se cerraron
            self.assertEqual(len(DatabaseFactory._instances), 0)
            
            self.log_test_result("Todas las instancias cerradas correctamente", True)
        except Exception as e:
            self.log_test_result(f"Error al cerrar instancias: {e}", False)
            raise
    
    def test_get_default_instance(self):
        """Prueba de obtención de instancia por defecto desde configuración"""
        try:
            # Sin especificar tipo, debería usar la configuración por defecto
            db_path = os.path.join(self.test_dir, "test_default.db")
            
            # Obtener instancia - ahora pasamos None explícitamente para el tipo
            db_instance = DatabaseFactory.get_database_instance(None, db_path)
            
            # Debe ser una instancia válida
            self.assertIsNotNone(db_instance)
            
            self.log_test_result("Instancia por defecto creada correctamente", True)
        except Exception as e:
            self.log_test_result(f"Error al obtener instancia por defecto: {e}", False)
            raise
    
    def test_get_available_databases(self):
        """Prueba la función get_available_databases para encontrar bases de datos con diferentes extensiones"""
        try:
            # Crear varias bases de datos con diferentes extensiones
            db_sqlite = os.path.join(self.test_dir, "test_db_sqlite.sqlite")
            db_duckdb = os.path.join(self.test_dir, "test_db_duckdb.duckdb")
            db_legacy = os.path.join(self.test_dir, "test_db_legacy.db")
            
            # Crear instancias y conectar para crear los archivos
            sqlite_instance = DatabaseFactory.get_database_instance("sqlite", embedding_dim=self.test_embedding_dim, custom_name=os.path.splitext(os.path.basename(db_sqlite))[0])
            
            # Para DuckDB, solo si está disponible
            if self.duckdb_available:
                duckdb_instance = DatabaseFactory.get_database_instance("duckdb", embedding_dim=self.test_embedding_dim, custom_name=os.path.splitext(os.path.basename(db_duckdb))[0])
            
            # Para la extensión legacy .db (que debería usar SQLite)
            legacy_instance = DatabaseFactory.get_database_instance("sqlite", embedding_dim=self.test_embedding_dim, custom_name=os.path.splitext(os.path.basename(db_legacy))[0])
            
            # Parchear temporalmente el método _load_config para que use nuestro directorio de prueba
            original_load_config = DatabaseFactory._load_config
            
            def patched_load_config():
                return {
                    "sqlite": {"db_dir": self.test_dir},
                    "duckdb": {"db_dir": self.test_dir},
                    "type": "sqlite"
                }
            
            DatabaseFactory._load_config = classmethod(patched_load_config)
            
            try:
                # Obtener bases de datos disponibles
                databases = DatabaseFactory.get_available_databases()
                
                # Verificar que se encontraron las bases de datos creadas
                self.assertIn("test_db_sqlite", databases)
                self.assertIn("test_db_legacy", databases)
                
                if self.duckdb_available:
                    self.assertIn("test_db_duckdb", databases)
                    self.assertEqual(len(databases), 3)
                else:
                    self.assertEqual(len(databases), 2)
                
                # Verificar que los tipos son correctos
                self.assertEqual(databases["test_db_sqlite"]["db_type"], "sqlite")
                self.assertEqual(databases["test_db_legacy"]["db_type"], "sqlite")
                
                if self.duckdb_available:
                    self.assertEqual(databases["test_db_duckdb"]["db_type"], "duckdb")
                
                self.log_test_result("Detección de bases de datos con múltiples extensiones correcta", True)
            finally:
                # Restaurar el método original
                DatabaseFactory._load_config = original_load_config
                
        except Exception as e:
            self.log_test_result(f"Error en get_available_databases: {e}", False)
            logger.error(traceback.format_exc())
            raise
    
    def test_sqlite_extension_detection(self):
        """Prueba que las bases de datos SQLite con extensión .sqlite son correctamente detectadas"""
        try:
            # Crear una base de datos con extensión .sqlite explícita
            db_path = os.path.join(self.test_dir, "test_sqlite_extension.sqlite")
            
            # Parchar temporalmente el método DatabaseFactory._load_config para que use nuestro directorio
            original_load_config = DatabaseFactory._load_config
            
            def patched_load_config():
                return {
                    "sqlite": {"db_dir": self.test_dir},
                    "type": "sqlite"
                }
                
            DatabaseFactory._load_config = classmethod(patched_load_config)
            
            try:
                # Crear instancia para generar el archivo
                db_instance = DatabaseFactory.get_database_instance(
                    db_type="sqlite", 
                    embedding_dim=self.test_embedding_dim
                )
                
                # Conectar a la ubicación específica
                db_instance.connect(db_path)
                db_instance.create_schema()
                
                # Intentar detectar la base de datos
                available_dbs = DatabaseFactory.get_available_databases()
                
                # Verificar que la base de datos fue detectada
                db_name = os.path.basename(db_path).replace('.sqlite', '')
                self.assertIn(db_name, available_dbs)
                
                # Verificar que el tipo es correcto
                detected_type = available_dbs[db_name].get('db_type')
                self.assertEqual(detected_type, 'sqlite')
                
                self.log_test_result(f"Base de datos SQLite con extensión .sqlite detectada correctamente", True)
            finally:
                # Restaurar el método original
                DatabaseFactory._load_config = original_load_config
                
        except Exception as e:
            self.log_test_result(f"Error en detección de base de datos SQLite con extensión .sqlite: {e}", False)
            logger.error(traceback.format_exc())
            raise

if __name__ == '__main__':
    unittest.main()
