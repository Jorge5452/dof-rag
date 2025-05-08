import unittest
from unittest.mock import patch, MagicMock, call
import os
import concurrent.futures

# Suponiendo que los módulos del proyecto están en el path
from modulos.resource_management.concurrency_manager import ConcurrencyManager
from modulos.resource_management.resource_manager import ResourceManager # Para mockear la instancia de RM

class MockConfigForConcurrency:
    def __init__(self, concurrency_config=None, general_config=None):
        self.concurrency_config_data = concurrency_config or {}
        self.general_config_data = general_config or {}

    def get_resource_management_config(self):
        # ConcurrencyManager accede a resource_management.concurrency
        return {"concurrency": self.concurrency_config_data}
    
    def get_general_config(self): # Por si ConcurrencyManager necesita algo de general
        return self.general_config_data

class TestConcurrencyManager(unittest.TestCase):

    def setUp(self):
        self.mock_rm_instance = MagicMock(spec=ResourceManager)
        
        # Configuración base para ConcurrencyManager
        self.base_concurrency_config = {
            "default_cpu_workers": "auto",
            "default_io_workers": "auto",
            "max_total_workers": None
        }
        self.mock_rm_instance.config_instance = MockConfigForConcurrency(self.base_concurrency_config)
        
        # Mock para system_static_info que ConcurrencyManager usa
        self.mock_rm_instance.get_system_static_info.return_value = {
            "cpu_cores": 4 # Valor por defecto para tests
        }

    @patch('modulos.resource_management.concurrency_manager.os.cpu_count')
    def test_calculate_workers_auto(self, mock_cpu_count):
        mock_cpu_count.return_value = 4
        cm = ConcurrencyManager(resource_manager=self.mock_rm_instance)

        # CPU workers auto
        self.assertEqual(cm._calculate_workers('cpu'), 4)
        # IO workers auto (min(32, (cpu_cores or 1) + 4))
        self.assertEqual(cm._calculate_workers('io'), 8) # (4+4)=8

        mock_cpu_count.return_value = 1
        cm = ConcurrencyManager(resource_manager=self.mock_rm_instance) # re-init con nuevo cpu_count via mock_rm
        self.mock_rm_instance.get_system_static_info.return_value = {"cpu_cores": 1}
        self.assertEqual(cm._calculate_workers('io'), 5) # (1+4)=5

    @patch('modulos.resource_management.concurrency_manager.os.cpu_count')
    def test_calculate_workers_fixed_values(self, mock_cpu_count):
        mock_cpu_count.return_value = 8 # No debería usarse si es fijo
        
        fixed_config = {
            "default_cpu_workers": 2,
            "default_io_workers": 10,
            "max_total_workers": None
        }
        self.mock_rm_instance.config_instance = MockConfigForConcurrency(fixed_config)
        cm = ConcurrencyManager(resource_manager=self.mock_rm_instance)

        self.assertEqual(cm._calculate_workers('cpu'), 2)
        self.assertEqual(cm._calculate_workers('io'), 10)

    @patch('modulos.resource_management.concurrency_manager.concurrent.futures.ThreadPoolExecutor')
    @patch('modulos.resource_management.concurrency_manager.concurrent.futures.ProcessPoolExecutor')
    @patch('modulos.resource_management.concurrency_manager.os.cpu_count')
    def test_initialize_pools(self, mock_cpu_count, mock_process_pool, mock_thread_pool):
        mock_cpu_count.return_value = 4
        # Expected: cpu_workers = 4, io_workers = 8

        cm = ConcurrencyManager(resource_manager=self.mock_rm_instance)
        cm._initialize_pools() # Normalmente llamado en __init__

        mock_thread_pool.assert_called_once_with(max_workers=8, thread_name_prefix='RAG_Thread')
        mock_process_pool.assert_called_once_with(max_workers=4)
        self.assertIsNotNone(cm.thread_pool_executor)
        self.assertIsNotNone(cm.process_pool_executor)

    @patch('modulos.resource_management.concurrency_manager.concurrent.futures.ThreadPoolExecutor')
    @patch('modulos.resource_management.concurrency_manager.concurrent.futures.ProcessPoolExecutor')
    @patch('modulos.resource_management.concurrency_manager.os.cpu_count')
    def test_initialize_pools_with_max_total_workers(self, mock_cpu_count, mock_process_pool, mock_thread_pool):
        mock_cpu_count.return_value = 8
        # Base: cpu_workers = 8, io_workers = 12. Total = 20
        
        max_total_config = {
            "default_cpu_workers": "auto",
            "default_io_workers": "auto",
            "max_total_workers": 10 # Límite inferior al calculado
        }
        self.mock_rm_instance.config_instance = MockConfigForConcurrency(max_total_config)
        self.mock_rm_instance.get_system_static_info.return_value={"cpu_cores": 8}
        
        cm = ConcurrencyManager(resource_manager=self.mock_rm_instance)
        # Proporción esperada: cpu_scaled = 10 * (8/20) = 4, io_scaled = 10 * (12/20) = 6
        mock_thread_pool.assert_called_once_with(max_workers=6, thread_name_prefix='RAG_Thread')
        mock_process_pool.assert_called_once_with(max_workers=4)

    def test_get_executors(self):
        cm = ConcurrencyManager(resource_manager=self.mock_rm_instance)
        self.assertIsInstance(cm.get_thread_pool_executor(), concurrent.futures.ThreadPoolExecutor)
        self.assertIsInstance(cm.get_process_pool_executor(), concurrent.futures.ProcessPoolExecutor)

    def test_run_in_executors(self):
        cm = ConcurrencyManager(resource_manager=self.mock_rm_instance)
        mock_thread_pool = MagicMock(spec=concurrent.futures.ThreadPoolExecutor)
        mock_process_pool = MagicMock(spec=concurrent.futures.ProcessPoolExecutor)
        cm.thread_pool_executor = mock_thread_pool
        cm.process_pool_executor = mock_process_pool

        def dummy_func(*args, **kwargs): return "done"

        cm.run_in_thread_pool(dummy_func, 1, b=2)
        mock_thread_pool.submit.assert_called_once_with(dummy_func, 1, b=2)

        cm.run_in_process_pool(dummy_func, 'foo')
        mock_process_pool.submit.assert_called_once_with(dummy_func, 'foo')

    def test_map_tasks_in_executors(self):
        cm = ConcurrencyManager(resource_manager=self.mock_rm_instance)
        mock_thread_pool = MagicMock(spec=concurrent.futures.ThreadPoolExecutor)
        mock_process_pool = MagicMock(spec=concurrent.futures.ProcessPoolExecutor)
        cm.thread_pool_executor = mock_thread_pool
        cm.process_pool_executor = mock_process_pool

        def dummy_func_map(x): return x * 2
        iterable = [1, 2, 3]

        cm.map_tasks_in_thread_pool(dummy_func_map, iterable, timeout=5)
        mock_thread_pool.map.assert_called_once_with(dummy_func_map, iterable, timeout=5, chunksize=1)

        cm.map_tasks_in_process_pool(dummy_func_map, iterable, chunksize=2)
        mock_process_pool.map.assert_called_once_with(dummy_func_map, iterable, timeout=None, chunksize=2)

    def test_shutdown_executors(self):
        cm = ConcurrencyManager(resource_manager=self.mock_rm_instance)
        
        # Mockeamos los métodos shutdown de los ejecutores *reales* que cm creó en su __init__.
        # Usamos `wraps` para que la funcionalidad original (cerrar el pool) aún se ejecute si es necesario,
        # aunque para este test unitario, solo nos importa que se llame a shutdown.
        # Si los pools reales interfieren con otros tests (poco probable aquí), podríamos no usar `wraps`.
        cm.thread_pool_executor.shutdown = MagicMock(wraps=cm.thread_pool_executor.shutdown)
        cm.process_pool_executor.shutdown = MagicMock(wraps=cm.process_pool_executor.shutdown)
            
        cm.shutdown_executors()
        
        cm.thread_pool_executor.shutdown.assert_called_once_with(wait=True, cancel_futures=False)
        cm.process_pool_executor.shutdown.assert_called_once_with(wait=True, cancel_futures=False)

    def test_get_config_value(self):
        # Prueba del método _get_config_value
        config_data = {
            "concurrency": {"test_key": 123, "another_key": "abc"},
            "general": {"log_level": "DEBUG"}
        }
        self.mock_rm_instance.config_instance = MockConfigForConcurrency(
            concurrency_config=config_data["concurrency"],
            general_config=config_data["general"]
        )
        cm = ConcurrencyManager(resource_manager=self.mock_rm_instance)
        
        self.assertEqual(cm._get_config_value("test_key", 456), 123)
        self.assertEqual(cm._get_config_value("non_existent_key", "default"), "default")
        self.assertEqual(cm._get_config_value("another_key"), "abc")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 