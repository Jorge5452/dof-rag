import unittest
from unittest.mock import patch, MagicMock
import gc

# Suponiendo que los módulos del proyecto están en el path
from modulos.resource_management.memory_manager import MemoryManager
from modulos.resource_management.resource_manager import ResourceManager # Para mockear la instancia de RM
# from modulos.embeddings.embeddings_factory import EmbeddingFactory # Se mockeará directamente

class TestMemoryManager(unittest.TestCase):

    def setUp(self):
        # Mock de ResourceManager que MemoryManager espera
        self.mock_resource_manager = MagicMock(spec=ResourceManager)
        self.mock_resource_manager.metrics = {
            "system_memory_percent": 50.0, # Valor base
            # ... otras métricas que RM podría tener
        }
        # Mock para la instancia Config dentro de ResourceManager (si MemoryManager la accede)
        # self.mock_resource_manager.config_instance = MagicMock() 
        # self.mock_resource_manager.config_instance.get_resource_management_config.return_value = {}

        self.memory_manager = MemoryManager(resource_manager=self.mock_resource_manager)

    @patch('modulos.resource_management.memory_manager.gc')
    def test_run_garbage_collection_normal(self, mock_gc):
        self.memory_manager._run_garbage_collection(aggressive=False)
        mock_gc.collect.assert_called_once_with()
        # Verificar que se llama a la recolección normal (sin argumentos o gen 0,1,2)
        # La implementación actual solo llama a gc.collect() sin args para no agresivo

    @patch('modulos.resource_management.memory_manager.gc')
    def test_run_garbage_collection_aggressive(self, mock_gc):
        self.memory_manager._run_garbage_collection(aggressive=True)
        # La implementación actual llama a gc.collect(generation=2) para agresivo
        mock_gc.collect.assert_called_once_with(generation=2)

    @patch('modulos.resource_management.memory_manager.EmbeddingFactory')
    @patch('modulos.resource_management.memory_manager.torch') # Mock torch para torch.cuda.empty_cache
    @patch('modulos.resource_management.memory_manager.gc') # Mock gc para la llamada final en release_inactive_models
    def test_release_model_resources(self, mock_gc_in_ef, mock_torch, mock_embedding_factory):
        # Configurar el mock de EmbeddingFactory
        # release_inactive_models es un @classmethod
        mock_embedding_factory.release_inactive_models.return_value = (True, 1) # (True si GPU fue liberada, num_models_released)
        
        self.memory_manager._release_model_resources(aggressive=False)
        
        mock_embedding_factory.release_inactive_models.assert_called_once_with(aggressive=False)
        # La implementación de EmbeddingFactory.release_inactive_models ya llama a torch.cuda.empty_cache y gc.collect
        # así que no necesitamos verificarlo aquí directamente si confiamos en esa unidad.
        # Pero si queremos ser exhaustivos con _release_model_resources:
        # mock_torch.cuda.empty_cache.assert_called_once() # Si gpu_cleared es True
        # mock_gc_in_ef.collect.assert_called_once() # Al final de release_inactive_models

    @patch('modulos.resource_management.memory_manager.EmbeddingFactory')
    @patch('modulos.resource_management.memory_manager.torch')
    @patch('modulos.resource_management.memory_manager.gc')
    def test_release_model_resources_aggressive(self, mock_gc_in_ef, mock_torch, mock_embedding_factory):
        mock_embedding_factory.release_inactive_models.return_value = (False, 0)
        
        self.memory_manager._release_model_resources(aggressive=True)
        mock_embedding_factory.release_inactive_models.assert_called_once_with(aggressive=True)

    @patch.object(MemoryManager, '_run_garbage_collection')
    @patch.object(MemoryManager, '_release_model_resources')
    @patch.object(MemoryManager, '_clear_python_caches')
    @patch.object(MemoryManager, '_check_memory_fragmentation')
    def test_cleanup_normal(self, mock_check_frag, mock_clear_caches, mock_release_models, mock_run_gc):
        self.memory_manager.cleanup(aggressive=False)
        mock_run_gc.assert_called_once_with(aggressive=False)
        mock_release_models.assert_called_once_with(aggressive=False)
        mock_clear_caches.assert_called_once_with(aggressive=False)
        mock_check_frag.assert_called_once_with(aggressive=False)

    @patch.object(MemoryManager, '_run_garbage_collection')
    @patch.object(MemoryManager, '_release_model_resources')
    @patch.object(MemoryManager, '_clear_python_caches')
    @patch.object(MemoryManager, '_check_memory_fragmentation')
    def test_cleanup_aggressive(self, mock_check_frag, mock_clear_caches, mock_release_models, mock_run_gc):
        self.memory_manager.cleanup(aggressive=True)
        mock_run_gc.assert_called_once_with(aggressive=True)
        mock_release_models.assert_called_once_with(aggressive=True)
        mock_clear_caches.assert_called_once_with(aggressive=True)
        mock_check_frag.assert_called_once_with(aggressive=True)

    def test_optimize_batch_size_normal_memory(self):
        self.mock_resource_manager.metrics["system_memory_percent"] = 50.0
        # Asumiendo que ResourceManager tiene estos umbrales o se mockean en config
        self.mock_resource_manager.warning_threshold_memory = 70.0 
        self.mock_resource_manager.aggressive_threshold_memory = 85.0

        base_batch = 100
        min_batch = 10
        optimized_batch = self.memory_manager.optimize_batch_size(base_batch, min_batch)
        self.assertEqual(optimized_batch, base_batch)

    def test_optimize_batch_size_warning_memory(self):
        self.mock_resource_manager.metrics["system_memory_percent"] = 75.0 
        self.mock_resource_manager.warning_threshold_memory = 70.0
        self.mock_resource_manager.aggressive_threshold_memory = 85.0
        
        base_batch = 100
        min_batch = 10
        optimized_batch = self.memory_manager.optimize_batch_size(base_batch, min_batch)
        # Heurística: base_batch * 0.75
        self.assertEqual(optimized_batch, int(base_batch * 0.75))

    def test_optimize_batch_size_aggressive_memory(self):
        self.mock_resource_manager.metrics["system_memory_percent"] = 90.0
        self.mock_resource_manager.warning_threshold_memory = 70.0
        self.mock_resource_manager.aggressive_threshold_memory = 85.0
        
        base_batch = 100
        min_batch = 10
        optimized_batch = self.memory_manager.optimize_batch_size(base_batch, min_batch)
        # Heurística: base_batch * 0.5
        self.assertEqual(optimized_batch, int(base_batch * 0.5))

    def test_optimize_batch_size_respects_min_batch(self):
        self.mock_resource_manager.metrics["system_memory_percent"] = 90.0
        self.mock_resource_manager.warning_threshold_memory = 70.0
        self.mock_resource_manager.aggressive_threshold_memory = 85.0
        
        base_batch = 15 # int(15 * 0.5) = 7, que es menor que min_batch
        min_batch = 10
        optimized_batch = self.memory_manager.optimize_batch_size(base_batch, min_batch)
        self.assertEqual(optimized_batch, min_batch)

    def test_shutdown_method(self):
        # Actualmente shutdown en MemoryManager no hace nada, solo existe.
        try:
            self.memory_manager.shutdown()
        except Exception as e:
            self.fail(f"MemoryManager.shutdown() no debería lanzar excepciones: {e}")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 