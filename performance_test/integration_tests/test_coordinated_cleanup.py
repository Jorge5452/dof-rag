import unittest
from unittest.mock import patch, MagicMock, call
import time
import os

# Ajustar imports según la estructura del proyecto. Asumimos que está en el PYTHONPATH.
from modulos.resource_management.resource_manager import ResourceManager
from modulos.resource_management.memory_manager import MemoryManager
from modulos.resource_management.concurrency_manager import ConcurrencyManager
from modulos.session_manager.session_manager import SessionManager
from modulos.embeddings.embeddings_factory import EmbeddingFactory

# Simulación de la clase Config para tests de integración
class MockIntegrationConfig:
    def __init__(self, resource_management_config=None, general_config=None, sessions_config=None, embeddings_config=None):
        self.rm_config = resource_management_config or {
            "monitoring_interval": 0.01, # Muy corto para tests
            "aggressive_threshold_memory": 85.0,
            "warning_threshold_memory": 70.0,
            "warning_threshold_cpu": 80.0,
            "monitoring_enabled": True,
            "concurrency": {"default_cpu_workers": 1, "default_io_workers": 1, "max_total_workers": 2}
        }
        self.gen_config = general_config or {"log_level": "ERROR"} # Menos verboso para tests
        self.sess_config = sessions_config or {
            "sessions_dir": "./test_sessions_dir", 
            "session_timeout": 300, 
            "max_sessions": 10
        } # Necesario para SessionManager
        self.embed_config = embeddings_config or {} # Para EmbeddingFactory

    def get_resource_management_config(self):
        return self.rm_config

    def get_general_config(self):
        return self.gen_config

    def get_sessions_config(self): # Necesario para SessionManager
        return self.sess_config
    
    def get_embedding_config(self): # Para EmbeddingFactory
        return self.embed_config

class TestCoordinatedCleanup(unittest.TestCase):

    def setUp(self):
        # Limpiar singleton de ResourceManager para cada test
        ResourceManager._instance = None
        EmbeddingFactory._instances = {} # Limpiar instancias de modelos de EmbeddingFactory

        # Crear directorio de sesiones para SessionManager si no existe
        self.test_sessions_dir = "./test_sessions_dir"
        os.makedirs(self.test_sessions_dir, exist_ok=True)

        self.mock_config = MockIntegrationConfig()

        # Mockear las dependencias que no son el foco directo de la interacción RM <-> MM <-> SM/EF
        # pero que son necesarias para la instanciación o funcionamiento.
        # El objetivo es tener instancias REALES de RM, MM, SM y EF interactuando.

        # Para SessionManager, necesitamos que su __init__ no falle. Mockeamos config global.
        # Para EmbeddingFactory, su lógica de _cleanup_unused_models se llama desde MM.

        # Patchear config global usada por SessionManager y EmbeddingFactory si es necesario
        self.patcher_config_global_sm = patch('modulos.session_manager.session_manager.config', MagicMock(return_value=self.mock_config))
        self.patcher_config_global_ef = patch('modulos.embeddings.embeddings_factory.config', MagicMock(return_value=self.mock_config))
        
        self.mock_config_sm = self.patcher_config_global_sm.start()
        self.mock_config_ef = self.patcher_config_global_ef.start()

        # Instancias reales (o semi-reales) de los componentes clave
        # ResourceManager es un Singleton, se instancia con su config mockeada
        self.resource_manager = ResourceManager(config_instance=self.mock_config)
        
        # SessionManager - su constructor usa config global, que está mockeada
        self.session_manager_instance = SessionManager() 
        # Aseguramos que RM tenga la instancia correcta de SM
        self.resource_manager.session_manager = self.session_manager_instance

        # MemoryManager y ConcurrencyManager son creados por ResourceManager
        self.memory_manager_instance = self.resource_manager.memory_manager
        
        # Mockeamos los métodos de bajo nivel para observar llamadas, sin ejecutar su lógica pesada
        # Espiamos los métodos clave de las instancias reales
        self.memory_manager_instance.cleanup = MagicMock(wraps=self.memory_manager_instance.cleanup)
        self.session_manager_instance.clean_expired_sessions = MagicMock(wraps=self.session_manager_instance.clean_expired_sessions)
        EmbeddingFactory.release_inactive_models = MagicMock(wraps=EmbeddingFactory.release_inactive_models)
        
        # Mock para _release_model_resources dentro de MemoryManager para verificar su llamada
        # Esto es más específico que mockear EmbeddingFactory.release_inactive_models directamente en el test de MM
        self.memory_manager_instance._release_model_resources = MagicMock(wraps=self.memory_manager_instance._release_model_resources)

    def tearDown(self):
        if self.resource_manager:
            self.resource_manager.shutdown() # Detener el thread de RM
        ResourceManager._instance = None
        EmbeddingFactory._instances = {}
        self.patcher_config_global_sm.stop()
        self.patcher_config_global_ef.stop()
        # Limpiar directorio de sesiones de prueba
        if os.path.exists(self.test_sessions_dir):
            for f in os.listdir(self.test_sessions_dir):
                os.remove(os.path.join(self.test_sessions_dir, f))
            os.rmdir(self.test_sessions_dir)

    def test_cleanup_triggered_by_high_memory(self):
        """Testea que una alta memoria dispare request_cleanup y la cadena de llamadas."""
        # Simular alta memoria
        self.resource_manager.metrics["system_memory_percent"] = 90.0
        self.resource_manager.metrics["system_cpu_percent"] = 50.0 # CPU normal

        # Dar tiempo al ResourceMonitorThread para que reaccione
        # El intervalo es 0.01s. Esperamos un poco más para asegurar ejecución.
        time.sleep(0.1)

        # Verificaciones:
        # 1. ResourceManager.request_cleanup fue llamado (indirectamente, verificamos los efectos)
        #    MemoryManager.cleanup debe ser llamado con aggressive=True
        self.memory_manager_instance.cleanup.assert_called_with(aggressive=True)
        
        # 2. MemoryManager.cleanup llamó a sus sub-métodos
        #    Verificamos _release_model_resources específicamente
        self.memory_manager_instance._release_model_resources.assert_called_with(aggressive=True)
        
        # 3. _release_model_resources (en MM) llamó a EmbeddingFactory.release_inactive_models
        EmbeddingFactory.release_inactive_models.assert_called_with(aggressive=True)
        
        # 4. ResourceManager.request_cleanup llamó a SessionManager.clean_expired_sessions
        self.session_manager_instance.clean_expired_sessions.assert_called_with(called_aggressively=True)

    def test_cleanup_triggered_by_high_cpu(self):
        """Testea que una alta CPU dispare request_cleanup y la cadena de llamadas."""
        self.resource_manager.metrics["system_memory_percent"] = 50.0 # Memoria normal
        self.resource_manager.metrics["system_cpu_percent"] = 90.0 

        time.sleep(0.1)

        self.memory_manager_instance.cleanup.assert_called_with(aggressive=True)
        self.memory_manager_instance._release_model_resources.assert_called_with(aggressive=True)
        EmbeddingFactory.release_inactive_models.assert_called_with(aggressive=True)
        self.session_manager_instance.clean_expired_sessions.assert_called_with(called_aggressively=True)

    def test_cleanup_triggered_by_warning_memory(self):
        """Testea que memoria en warning dispare limpieza no agresiva."""
        self.resource_manager.metrics["system_memory_percent"] = 75.0 # Memoria en warning
        self.resource_manager.metrics["system_cpu_percent"] = 50.0

        time.sleep(0.1)
        
        self.memory_manager_instance.cleanup.assert_called_with(aggressive=False)
        self.memory_manager_instance._release_model_resources.assert_called_with(aggressive=False)
        EmbeddingFactory.release_inactive_models.assert_called_with(aggressive=False)
        self.session_manager_instance.clean_expired_sessions.assert_called_with(called_aggressively=False)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 