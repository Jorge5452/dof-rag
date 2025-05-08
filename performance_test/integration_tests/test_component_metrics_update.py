import unittest
from unittest.mock import patch, MagicMock

from modulos.resource_management.resource_manager import ResourceManager
# from .test_coordinated_cleanup import MockIntegrationConfig # No se puede importar así

class MockIntegrationConfig:
    def __init__(self, resource_management_config=None, general_config=None, 
                 sessions_config=None, embeddings_config=None): # Añadido embeddings_config
        self.rm_config = resource_management_config or {
            "monitoring_interval": 0.1,
            "monitoring_enabled": False, # Deshabilitar monitor para este test
            "concurrency": {"default_cpu_workers": 1, "default_io_workers": 1}
        }
        self.gen_config = general_config or {"log_level": "ERROR"}
        self.sess_config = sessions_config or {"sessions_dir": "./test_sessions_metrics"}
        self.embed_config = embeddings_config or {} # Para EmbeddingFactory

    def get_resource_management_config(self): return self.rm_config
    def get_general_config(self): return self.gen_config
    def get_sessions_config(self): return self.sess_config
    def get_embedding_config(self): return self.embed_config # Necesario para EF

class TestComponentMetricsUpdate(unittest.TestCase):

    def setUp(self):
        ResourceManager._instance = None
        self.mock_config = MockIntegrationConfig()

        # Patcheamos las clases SessionManager y EmbeddingFactory en el módulo resource_manager
        # para controlar las instancias que ResourceManager crea o los métodos de clase que llama.
        self.patcher_sm = patch('modulos.resource_management.resource_manager.SessionManager')
        self.patcher_ef = patch('modulos.resource_management.resource_manager.EmbeddingFactory')
        
        self.MockSessionManagerClass = self.patcher_sm.start()
        self.MockEmbeddingFactoryClass = self.patcher_ef.start()

        # Configurar los mocks de las CLASES para que sus instancias (si RM las crea) 
        # o sus métodos de clase devuelvan lo que necesitamos.
        self.mock_sm_instance = self.MockSessionManagerClass.return_value
        self.mock_sm_instance.get_active_sessions_count.return_value = 7

        self.MockEmbeddingFactoryClass.get_active_model_count.return_value = 3
        
        # También mockeamos psutil para aislar el test a las métricas de componentes RAG
        self.patcher_psutil = patch('modulos.resource_management.resource_manager.psutil')
        self.mock_psutil = self.patcher_psutil.start()
        self.mock_psutil.cpu_percent.return_value = 10.0
        self.mock_psutil.virtual_memory.return_value.percent = 20.0
        self.mock_psutil.virtual_memory.return_value.used = 1 * 1024 * 1024 * 1024
        self.mock_psutil.virtual_memory.return_value.total = 8 * 1024 * 1024 * 1024
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 50 * 1024 * 1024
        mock_process.cpu_percent.return_value = 5.0
        self.mock_psutil.Process.return_value = mock_process

        # Crear la instancia de ResourceManager DESPUÉS de que los patches estén activos
        self.resource_manager = ResourceManager(config_instance=self.mock_config)
        # El __init__ de ResourceManager intentará crear/obtener SessionManager y llamar a EF.
        # Si RM instancia SM, self.MockSessionManagerClass() será llamado.
        # RM llama directamente a EmbeddingFactory.get_active_model_count().

    def tearDown(self):
        if self.resource_manager and self.resource_manager.monitor_thread: # RM podría no tener monitor si está disabled
            self.resource_manager.shutdown()
        ResourceManager._instance = None
        self.patcher_sm.stop()
        self.patcher_ef.stop()
        self.patcher_psutil.stop()

    def test_update_metrics_from_rag_components(self):
        """Verifica que RM actualiza sus métricas con datos de SM y EF."""
        # update_metrics es llamado por el monitor, pero también podemos llamarlo directamente
        # si el monitor está deshabilitado o para un test más directo.
        self.resource_manager.update_metrics()

        # Verificar que los métodos de los mocks fueron llamados por update_metrics
        self.mock_sm_instance.get_active_sessions_count.assert_called_once()
        self.MockEmbeddingFactoryClass.get_active_model_count.assert_called_once()

        # Verificar que las métricas en ResourceManager se actualizaron
        self.assertEqual(self.resource_manager.metrics["active_sessions_rag"], 7)
        self.assertEqual(self.resource_manager.metrics["active_embedding_models"], 3)
        
        # Verificar también las métricas de psutil para asegurar que update_metrics corrió completamente
        self.assertEqual(self.resource_manager.metrics["system_cpu_percent"], 10.0)
        self.assertEqual(self.resource_manager.metrics["process_memory_mb"], 50.0)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 