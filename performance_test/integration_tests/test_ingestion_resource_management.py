import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import os
import shutil # Para limpiar directorios de prueba
import time
from pathlib import Path

# Ajustar imports
from main import process_documents, process_single_document_wrapper # Necesitamos el wrapper para map_tasks
from modulos.resource_management.resource_manager import ResourceManager
# MockConfig similar a la de test_coordinated_cleanup.py
# from .test_coordinated_cleanup import MockIntegrationConfig # No se puede importar así directamente

class MockIntegrationConfig:
    def __init__(self, resource_management_config=None, general_config=None, 
                 chunks_config=None, embeddings_config=None, database_config=None,
                 sessions_config=None): # Añadido sessions_config
        self.rm_config = resource_management_config or {
            "monitoring_interval": 0.1,
            "aggressive_threshold_memory": 95.0, # Umbrales altos para no interferir
            "warning_threshold_memory": 90.0,
            "warning_threshold_cpu": 90.0,
            "monitoring_enabled": False, # Deshabilitar monitor para este test
            "concurrency": {"default_cpu_workers": 2, "default_io_workers": 2, "max_total_workers": 4}
        }
        self.gen_config = general_config or {"log_level": "ERROR"}
        self.chk_config = chunks_config or {
            "method": "character", 
            "memory_optimization": {"enabled": True, "batch_size": 10, "force_gc": False, "memory_monitor": False},
            "character": {"chunk_size": 100, "chunk_overlap": 10}
        }
        self.embed_config = embeddings_config or {"model": "mock_model", "mock_model": {"model_name": "mock/model"}}
        self.db_config = database_config or {"type": "sqlite", "sqlite": {"db_dir": "./test_db_dir_ingestion"}}
        self.sess_config = sessions_config or {"sessions_dir": "./test_sessions_dir_ingestion"}

    def get_resource_management_config(self): return self.rm_config
    def get_general_config(self): return self.gen_config
    def get_chunks_config(self): return self.chk_config
    def get_embedding_config(self): return self.embed_config
    def get_database_config(self): return self.db_config
    def get_sessions_config(self): return self.sess_config

class TestIngestionResourceManagement(unittest.TestCase):
    def setUp(self):
        ResourceManager._instance = None
        self.test_docs_dir = "./test_docs_ingestion"
        self.test_db_dir = "./test_db_dir_ingestion"
        self.test_sessions_dir = "./test_sessions_dir_ingestion"
        os.makedirs(self.test_docs_dir, exist_ok=True)
        os.makedirs(self.test_db_dir, exist_ok=True)
        os.makedirs(self.test_sessions_dir, exist_ok=True)

        # Crear archivos dummy markdown
        for i in range(3):
            with open(os.path.join(self.test_docs_dir, f"doc{i+1}.md"), "w") as f:
                f.write(f"# Documento {i+1}\nContenido de prueba.")

        self.mock_config = MockIntegrationConfig(database_config={"type": "sqlite", "sqlite": {"db_dir": self.test_db_dir}},
                                               sessions_config={"sessions_dir": self.test_sessions_dir})

        # Mockear dependencias profundas de process_documents y process_single_document
        self.patcher_config_main = patch('main.config', MagicMock(return_value=self.mock_config))
        self.patcher_embedding_factory = patch('main.EmbeddingFactory')
        self.patcher_chunker_factory = patch('main.ChunkerFactory')
        self.patcher_db_factory = patch('main.DatabaseFactory')
        self.patcher_markdown_processor = patch('main.MarkdownProcessor')
        self.patcher_session_manager_main = patch('main.SessionManager') # SessionManager en main.py
        
        self.mock_config_main = self.patcher_config_main.start()
        self.mock_embedding_factory = self.patcher_embedding_factory.start()
        self.mock_chunker_factory = self.patcher_chunker_factory.start()
        self.mock_db_factory = self.patcher_db_factory.start()
        self.mock_markdown_processor = self.patcher_markdown_processor.start()
        self.mock_session_manager_main = self.patcher_session_manager_main.start()

        # Configurar mocks para que devuelvan instancias mockeadas funcionales mínimamente
        self.mock_embed_manager = MagicMock()
        self.mock_embed_manager.embedding_dim = 100
        self.mock_embed_manager.get_document_embedding.return_value = [0.1] * 100
        self.mock_embedding_factory.get_embedding_manager.return_value = self.mock_embed_manager

        self.mock_chunker = MagicMock()
        # process_content_stream debe ser un generador
        def mock_stream_chunks(*args, **kwargs):
            yield {"text": "chunk1_text", "header": "h1"}
            yield {"text": "chunk2_text", "header": "h2"}
        self.mock_chunker.process_content_stream = MagicMock(side_effect=mock_stream_chunks)
        self.mock_chunker.model = self.mock_embed_manager # Chunker accede al modelo de embedding
        self.mock_chunker_factory.get_chunker.return_value = self.mock_chunker

        self.mock_db_instance = MagicMock()
        self.mock_db_instance.get_db_path.return_value = os.path.join(self.test_db_dir, "test.db")
        self.mock_db_instance.insert_document_metadata.return_value = 1 # Retorna un ID de documento
        self.mock_db_instance.optimize_database.return_value = True
        self.mock_db_factory.get_database_instance.return_value = self.mock_db_instance

        self.mock_md_processor_instance = MagicMock()
        self.mock_md_processor_instance.process_document.return_value = ({"title": "Test Doc"}, "Contenido del doc")
        self.mock_markdown_processor.return_value = self.mock_md_processor_instance
        
        self.mock_sm_instance_main = MagicMock()
        self.mock_session_manager_main.return_value = self.mock_sm_instance_main
        
        # Instancia real de ResourceManager (usa self.mock_config para su propia configuración)
        self.resource_manager = ResourceManager(config_instance=self.mock_config) 
        # Espiar los métodos de ConcurrencyManager y MemoryManager
        self.concurrency_manager_instance = self.resource_manager.concurrency_manager
        self.memory_manager_instance = self.resource_manager.memory_manager
        
        # Si map_tasks_in_process_pool no es mockeado, intentará usar ProcessPoolExecutor real.
        # Para este test de integración, podemos mockearlo para asegurar que es llamado.
        if self.concurrency_manager_instance:
             self.concurrency_manager_instance.map_tasks_in_process_pool = MagicMock(return_value=[1,1,1]) # Simula IDs devueltos
        
        self.memory_manager_instance.optimize_batch_size = MagicMock(wraps=self.memory_manager_instance.optimize_batch_size)

    def tearDown(self):
        if self.resource_manager:
             self.resource_manager.shutdown()
        ResourceManager._instance = None
        self.patcher_config_main.stop()
        self.patcher_embedding_factory.stop()
        self.patcher_chunker_factory.stop()
        self.patcher_db_factory.stop()
        self.patcher_markdown_processor.stop()
        self.patcher_session_manager_main.stop()

        if os.path.exists(self.test_docs_dir):
            shutil.rmtree(self.test_docs_dir)
        if os.path.exists(self.test_db_dir):
            shutil.rmtree(self.test_db_dir)
        if os.path.exists(self.test_sessions_dir):
            shutil.rmtree(self.test_sessions_dir)

    def test_ingestion_uses_concurrency_and_memory_managers(self):
        process_documents(file_path=self.test_docs_dir, session_name="test_ingestion_session")

        # Verificar que ConcurrencyManager fue usado (si hay más de 1 archivo)
        # En este caso, creamos 3 archivos de prueba.
        if self.concurrency_manager_instance:
            self.concurrency_manager_instance.map_tasks_in_process_pool.assert_called_once()
            # Verificar que los argumentos para process_single_document_wrapper son correctos
            # Esto es un poco más complejo de verificar directamente con map_tasks mockeado así.
            # Podríamos verificar el número de llamadas a md_processor.process_document o db.insert_single_chunk
            # si el map no estuviera completamente mockeado.

        # Verificar que MemoryManager.optimize_batch_size fue llamado.
        # process_single_document (llamado via wrapper) debería invocarlo.
        # Como map_tasks_in_process_pool está mockeado y devuelve [1,1,1], el wrapper no se ejecuta realmente.
        # Necesitamos una forma de verificar optimize_batch_size. 
        # Se llamará N veces (N=num_docs) si el map no está mockeado completamente.
        # Si el map_tasks está mockeado, debemos cambiar la estrategia o mockear process_single_document_wrapper.

        # Dado que `map_tasks_in_process_pool` está mockeado y no ejecuta realmente `process_single_document_wrapper`,
        # no podemos verificar `optimize_batch_size` de esta manera.
        # Vamos a testear `process_single_document` directamente con los mocks para verificar `optimize_batch_size`.

        # Testeamos una llamada a process_single_document_wrapper para asegurar que llama a optimize_batch_size
        # Necesitamos una instancia de ResourceManager para que process_single_document obtenga MemoryManager
        # Esto ya está configurado en setUp.
        
        # Para que el siguiente test funcione, debemos asegurar que ResourceManager._instance es el que queremos
        # y que main.ResourceManager() lo devuelva.
        with patch('main.ResourceManager', return_value=self.resource_manager): 
            # Construir los argumentos para el wrapper como lo haría process_documents
            doc_path_str = os.path.join(self.test_docs_dir, "doc1.md")
            args_for_wrapper = (doc_path_str, self.mock_md_processor_instance, self.mock_chunker, self.mock_db_instance)
            process_single_document_wrapper(args_for_wrapper)

        self.memory_manager_instance.optimize_batch_size.assert_called()
        # La aserción anterior verificará si fue llamado al menos una vez.
        # Para ser más precisos, si tuviéramos 3 documentos y el map no estuviera mockeado, esperaríamos 3 llamadas.
        # Con la llamada directa al wrapper, esperamos al menos una.
        self.assertGreaterEqual(self.memory_manager_instance.optimize_batch_size.call_count, 1)
        self.mock_db_instance.optimize_database.assert_called_once() # Al final de process_documents

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 