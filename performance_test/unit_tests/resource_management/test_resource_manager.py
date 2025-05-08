import unittest
from unittest.mock import patch, MagicMock, mock_open
import time
import os
import platform
import psutil
import threading

# Suponiendo que los módulos del proyecto están en el path o se ajusta el sys.path
from modulos.resource_management.resource_manager import ResourceManager
from modulos.resource_management.memory_manager import MemoryManager
from modulos.resource_management.concurrency_manager import ConcurrencyManager
# Necesitaremos mockear SessionManager y EmbeddingFactory si son importados directamente por ResourceManager
# Por ahora, asumimos que se pasan o se mockean en el __init__ si es necesario.

# Simulación de la clase Config
class MockConfig:
    def __init__(self, config_data=None):
        self.config_data = config_data or {}

    def get_resource_management_config(self):
        return self.config_data.get("resource_management", {})

    def get_general_config(self): # Añadido por si ResourceManager lo usa
        return self.config_data.get("general", {})

class TestResourceManager(unittest.TestCase):

    def setUp(self):
        # Asegurar que el Singleton se reinicie para cada test
        ResourceManager._instance = None
        self.mock_config_data = {
            "resource_management": {
                "monitoring_interval": 0.01, # Intervalo muy corto para tests
                "aggressive_threshold_memory": 85.0,
                "warning_threshold_memory": 70.0,
                "warning_threshold_cpu": 80.0,
                "monitoring_enabled": True, # Habilitar para probar el thread
                "concurrency": { # Añadir config de concurrencia
                    "default_cpu_workers": "auto",
                    "default_io_workers": "auto",
                    "max_total_workers": None
                }
            },
            "general": {"log_level": "INFO"} # Configuración general dummy
        }
        self.mock_config_instance = MockConfig(self.mock_config_data)

    def tearDown(self):
        # Limpiar la instancia Singleton después de cada test
        if ResourceManager._instance:
            if hasattr(ResourceManager._instance, 'monitor_thread') and ResourceManager._instance.monitor_thread:
                ResourceManager._instance.shutdown() # Asegurar que el thread se detiene
        ResourceManager._instance = None

    @patch('modulos.resource_management.resource_manager.MemoryManager')
    @patch('modulos.resource_management.resource_manager.ConcurrencyManager')
    @patch('modulos.resource_management.resource_manager.SessionManager')
    @patch('modulos.resource_management.resource_manager.EmbeddingFactory')
    @patch('modulos.resource_management.resource_manager.config', new_callable=lambda: MagicMock(spec=MockConfig))
    def test_singleton_instance(self, mock_global_config_class, mock_ef, mock_sm, mock_con_m, mock_mem_m):
        # Configurar el mock de la clase Config global para que devuelva nuestra instancia mock
        mock_global_config_class.return_value = self.mock_config_instance
        
        rm1 = ResourceManager(config_instance=self.mock_config_instance)
        rm2 = ResourceManager(config_instance=self.mock_config_instance)
        self.assertIs(rm1, rm2)
        self.assertTrue(rm1._initialized)

    @patch('modulos.resource_management.resource_manager.psutil')
    @patch('modulos.resource_management.resource_manager.platform')
    @patch('modulos.resource_management.resource_manager.os')
    @patch('modulos.resource_management.resource_manager.config', new_callable=lambda: MagicMock(spec=MockConfig))
    def test_get_system_static_info(self, mock_global_config_class, mock_os, mock_platform, mock_psutil):
        mock_global_config_class.return_value = self.mock_config_instance
        mock_os.cpu_count.return_value = 4
        mock_psutil.virtual_memory.return_value.total = 16 * 1024 * 1024 * 1024 # 16 GB
        mock_platform.system.return_value = "TestOS"
        mock_platform.release.return_value = "1.0"
        mock_platform.python_version.return_value = "3.9.0"

        rm = ResourceManager(config_instance=self.mock_config_instance)
        rm.shutdown() # Detener el monitor que se inicia en __init__
        
        stats = rm.get_system_static_info()
        
        self.assertEqual(stats["cpu_cores"], 4)
        self.assertEqual(stats["total_ram_gb"], 16.0)
        self.assertEqual(stats["os_version"], "TestOS 1.0")
        self.assertEqual(stats["python_version"], "3.9.0")
        mock_os.cpu_count.assert_called_once()
        mock_psutil.virtual_memory.assert_called_once()

    @patch('modulos.resource_management.resource_manager.psutil')
    @patch('modulos.resource_management.resource_manager.SessionManager')
    @patch('modulos.resource_management.resource_manager.EmbeddingFactory')
    @patch('modulos.resource_management.resource_manager.config', new_callable=lambda: MagicMock(spec=MockConfig))
    def test_update_metrics(self, mock_global_config_class, mock_embedding_factory, mock_session_manager, mock_psutil):
        mock_global_config_class.return_value = self.mock_config_instance
        
        # Configurar mocks para SessionManager y EmbeddingFactory
        # Suponiendo que ResourceManager los instancia o los obtiene de alguna manera.
        # Si se pasan al constructor, el mock de constructor se encargará.
        # Si se importan y se llama a un método de clase/estático, se necesita mockear el método en sí.
        mock_sm_instance = mock_session_manager.return_value # Si se instancia
        mock_sm_instance.get_active_sessions_count.return_value = 5
        
        # Para métodos de clase como EmbeddingFactory.get_active_model_count
        mock_embedding_factory.get_active_model_count.return_value = 2

        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value.percent = 60.0
        mock_psutil.virtual_memory.return_value.used = 6 * 1024 * 1024 * 1024
        mock_psutil.virtual_memory.return_value.total = 10 * 1024 * 1024 * 1024
        
        # Mockear el proceso actual
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024 # 100MB
        mock_process.cpu_percent.return_value = 10.0
        mock_psutil.Process.return_value = mock_process

        rm = ResourceManager(config_instance=self.mock_config_instance)
        rm.session_manager = mock_sm_instance # Asignar instancia mockeada
        rm.shutdown() # Detener el monitor

        rm.update_metrics()

        self.assertEqual(rm.metrics["system_cpu_percent"], 50.0)
        self.assertEqual(rm.metrics["system_memory_percent"], 60.0)
        self.assertEqual(rm.metrics["system_memory_used_gb"], 6.0)
        self.assertEqual(rm.metrics["system_memory_total_gb"], 10.0)
        self.assertEqual(rm.metrics["process_memory_mb"], 100.0)
        self.assertEqual(rm.metrics["process_cpu_percent"], 10.0)
        self.assertEqual(rm.metrics["active_sessions_rag"], 5)
        self.assertEqual(rm.metrics["active_embedding_models"], 2)
        mock_psutil.cpu_percent.assert_called_once_with(interval=None)
        mock_psutil.virtual_memory.assert_called_once()
        mock_sm_instance.get_active_sessions_count.assert_called_once()
        mock_embedding_factory.get_active_model_count.assert_called_once()


    @patch('modulos.resource_management.resource_manager.MemoryManager')
    @patch('modulos.resource_management.resource_manager.ConcurrencyManager')
    @patch('modulos.resource_management.resource_manager.SessionManager')
    @patch('modulos.resource_management.resource_manager.EmbeddingFactory')
    @patch('modulos.resource_management.resource_manager.threading.Thread') # Mock base Thread
    @patch('modulos.resource_management.resource_manager.config', new_callable=lambda: MagicMock(spec=MockConfig))
    def test_resource_monitor_thread_logic(self, mock_global_config_class, mock_thread_class, mock_ef, mock_sm, mock_con_m, mock_mem_m):
        mock_global_config_class.return_value = self.mock_config_instance
        
        # Mock para que la instancia de ResourceManager use
        mock_memory_manager_instance = mock_mem_m.return_value
        mock_session_manager_instance = mock_sm.return_value

        # Para controlar el bucle del thread
        self.side_effect_counter = 0
        def mock_update_metrics_side_effect():
            rm_instance = ResourceManager._instance # Obtener la instancia actual
            if self.side_effect_counter == 0: # Primera llamada, simular alta memoria
                rm_instance.metrics["system_memory_percent"] = 90.0
                rm_instance.metrics["system_cpu_percent"] = 50.0
            elif self.side_effect_counter == 1: # Segunda, simular alto CPU
                rm_instance.metrics["system_memory_percent"] = 50.0
                rm_instance.metrics["system_cpu_percent"] = 85.0
            elif self.side_effect_counter == 2: # Tercera, simular memoria warning
                rm_instance.metrics["system_memory_percent"] = 75.0
                rm_instance.metrics["system_cpu_percent"] = 50.0
            else: # Suficientes iteraciones, detener el thread
                if rm_instance and rm_instance.monitor_thread:
                    rm_instance.monitor_thread._stop_event.set()
            self.side_effect_counter += 1

        # Creamos la instancia de ResourceManager, lo que debería iniciar el thread
        # El __init__ de ResourceManager va a llamar a _start_monitoring
        # que a su vez crea e inicia ResourceMonitorThread.
        # Necesitamos que ResourceManager.update_metrics tenga el side_effect
        
        rm = ResourceManager(config_instance=self.mock_config_instance)
        rm.session_manager = mock_session_manager_instance
        rm.memory_manager = mock_memory_manager_instance
        
        # Sobrescribimos el update_metrics de la instancia con un mock que tiene side_effect
        rm.update_metrics = MagicMock(side_effect=mock_update_metrics_side_effect)
        
        # Dar tiempo al thread para que ejecute algunas iteraciones
        # El intervalo es 0.01s, 3 iteraciones deberían tomar ~0.03s + overhead
        # Damos un poco más de margen
        rm.monitor_thread.join(timeout=0.5) 
        
        # Verificar llamadas a request_cleanup (que llama a memory_manager.cleanup y session_manager.clean_expired_sessions)
        # Primera iteración: Memoria alta -> aggressive=True
        # Segunda iteración: CPU alto -> aggressive=True
        # Tercera iteración: Memoria warning -> aggressive=False
        
        # ResourceManager.request_cleanup llama a:
        # 1. self.memory_manager.cleanup(aggressive=aggressive_flag)
        # 2. self.session_manager.clean_expired_sessions(called_aggressively=aggressive_flag)

        calls_mem_cleanup = [
            unittest.mock.call(aggressive=True), # por memoria alta
            unittest.mock.call(aggressive=True), # por CPU alto
            unittest.mock.call(aggressive=False) # por memoria warning
        ]
        mock_memory_manager_instance.cleanup.assert_has_calls(calls_mem_cleanup, any_order=False)
        
        calls_sm_cleanup = [
            unittest.mock.call(called_aggressively=True),
            unittest.mock.call(called_aggressively=True),
            unittest.mock.call(called_aggressively=False)
        ]
        mock_session_manager_instance.clean_expired_sessions.assert_has_calls(calls_sm_cleanup, any_order=False)

        self.assertGreaterEqual(rm.update_metrics.call_count, 3) # Asegurar que el thread corrió
        
        # Importante: llamar a shutdown para limpiar el thread si no se detuvo por el side_effect
        rm.shutdown()


    @patch('modulos.resource_management.resource_manager.MemoryManager')
    @patch('modulos.resource_management.resource_manager.ConcurrencyManager')
    @patch('modulos.resource_management.resource_manager.SessionManager')
    @patch('modulos.resource_management.resource_manager.config', new_callable=lambda: MagicMock(spec=MockConfig))
    def test_shutdown(self, mock_global_config_class, mock_sm_class, mock_con_m_class, mock_mem_m_class):
        mock_global_config_class.return_value = self.mock_config_instance
        
        mock_memory_manager = mock_mem_m_class.return_value
        mock_concurrency_manager = mock_con_m_class.return_value
        # mock_session_manager = mock_sm_class.return_value # No se llama a shutdown en SM

        rm = ResourceManager(config_instance=self.mock_config_instance)
        # rm.session_manager = mock_session_manager # Asignar si es necesario
        
        # Verificar que el thread está vivo antes de shutdown
        self.assertTrue(rm.monitor_thread.is_alive())
        self.assertFalse(rm.monitor_thread._stop_event.is_set())

        rm.shutdown()

        mock_memory_manager.shutdown.assert_called_once()
        mock_concurrency_manager.shutdown_executors.assert_called_once()
        
        # El thread debería haberse detenido
        self.assertTrue(rm.monitor_thread._stop_event.is_set())
        rm.monitor_thread.join(timeout=0.1) # Esperar un poco a que termine
        self.assertFalse(rm.monitor_thread.is_alive())


    @patch('modulos.resource_management.resource_manager.MemoryManager')
    @patch('modulos.resource_management.resource_manager.ConcurrencyManager')
    @patch('modulos.resource_management.resource_manager.SessionManager')
    @patch('modulos.resource_management.resource_manager.config', new_callable=lambda: MagicMock(spec=MockConfig))
    def test_monitoring_disabled(self, mock_global_config_class, mock_sm_class, mock_con_m_class, mock_mem_m_class):
        self.mock_config_data["resource_management"]["monitoring_enabled"] = False
        mock_config_disabled = MockConfig(self.mock_config_data)
        mock_global_config_class.return_value = mock_config_disabled

        rm = ResourceManager(config_instance=mock_config_disabled)
        
        self.assertIsNone(rm.monitor_thread)
        # rm.shutdown() # No debería hacer nada o fallar si el thread es None
        # Probar que shutdown no falla
        try:
            rm.shutdown()
        except Exception as e:
            self.fail(f"rm.shutdown() falló cuando monitoring_enabled=False: {e}")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 