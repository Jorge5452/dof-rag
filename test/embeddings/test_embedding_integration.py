#!/usr/bin/env python3
"""
Pruebas de integración para el módulo de embeddings.
"""

import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import numpy as np
from typing import List, Dict, Any

# Añadir directorio raíz al path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modulos.embeddings.embeddings_factory import EmbeddingFactory
from modulos.embeddings.embeddings_manager import EmbeddingManager

class TestEmbeddingIntegration(unittest.TestCase):
    """Pruebas de integración para el módulo de embeddings"""
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial para todas las pruebas"""
        # Reiniciar cualquier instancia previa
        EmbeddingFactory.reset_instances()
        
        # Crear un directorio temporal para cache
        cls.temp_dir = tempfile.mkdtemp()
        
        # Configurar variables de entorno para especificar cache
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = cls.temp_dir
    
    @classmethod
    def tearDownClass(cls):
        """Limpieza después de todas las pruebas"""
        # Eliminar el directorio temporal
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
        
        # Reiniciar variables de entorno
        if "SENTENCE_TRANSFORMERS_HOME" in os.environ:
            del os.environ["SENTENCE_TRANSFORMERS_HOME"]
    
    def setUp(self):
        """Configuración inicial para cada prueba"""
        # Reiniciar las instancias para cada prueba
        EmbeddingFactory.reset_instances()
    
    def tearDown(self):
        """Limpieza después de cada prueba"""
        # Asegurar que se liberan las instancias
        EmbeddingFactory.reset_instances()
    
    @patch('modulos.embeddings.embeddings_manager.SentenceTransformer')
    @patch('modulos.embeddings.embeddings_factory.config')
    @patch('modulos.embeddings.embeddings_manager.config')
    def test_factory_manager_integration(self, mock_config_manager, mock_config_factory, mock_st):
        """Probar la integración entre Factory y Manager"""
        # Configurar mocks
        mock_config_manager.get_embedding_config.return_value = {
            "model": "modernbert",
            "trust_remote_code": True
        }
        mock_config_manager.get_specific_model_config.return_value = {
            "model_name": "nomic-ai/modernbert-embed-base",
            "device": "cpu",
            "batch_size": 32
        }
        
        mock_config_factory.get_embedding_config.return_value = {
            "model": "modernbert"
        }
        
        # Configurar mock de SentenceTransformer
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.ones(768)
        mock_st.return_value = mock_model
        
        # Obtener un gestor a través de la factoría
        manager1 = EmbeddingFactory.get_embedding_manager("modernbert")
        
        # Verificar que es una instancia de EmbeddingManager
        self.assertIsInstance(manager1, EmbeddingManager)
        
        # Acceder al modelo para activar la carga
        _ = manager1.model
        
        # Verificar dimensión del embedding
        self.assertEqual(manager1.embedding_dim, 768)
        
        # Obtener otro gestor del mismo tipo
        manager2 = EmbeddingFactory.get_embedding_manager("modernbert")
        
        # Verificar que es la misma instancia (singleton)
        self.assertIs(manager1, manager2)
        
        # Verificar que hay una sola instancia con dos referencias
        active_models = EmbeddingFactory.get_active_models()
        self.assertEqual(len(active_models), 1)
        model_info = active_models["embedding:modernbert"]
        self.assertEqual(model_info["reference_count"], 2)
        
        # Liberar una referencia
        remaining = EmbeddingFactory.release_embedding_manager(manager1)
        self.assertEqual(remaining, 1)
        
        # Liberar la otra referencia
        remaining = EmbeddingFactory.release_embedding_manager(manager2)
        self.assertEqual(remaining, 0)
    
    @patch('modulos.embeddings.embeddings_manager.SentenceTransformer')
    @patch('modulos.embeddings.embeddings_factory.config')
    @patch('modulos.embeddings.embeddings_manager.config')
    def test_multiple_model_types(self, mock_config_manager, mock_config_factory, mock_st):
        """Probar la gestión de múltiples tipos de modelos"""
        # Configurar mocks
        def get_embedding_config():
            return {
                "model": "modernbert",
                "trust_remote_code": True
            }
        
        def get_specific_model_config(model_type):
            config = {
                "modernbert": {
                    "model_name": "nomic-ai/modernbert-embed-base",
                    "device": "cpu",
                    "batch_size": 32
                },
                "e5-small": {
                    "model_name": "intfloat/e5-small",
                    "device": "cpu",
                    "batch_size": 16,
                    "prefix_queries": True
                }
            }
            return config.get(model_type, {})
        
        mock_config_manager.get_embedding_config.side_effect = get_embedding_config
        mock_config_manager.get_specific_model_config.side_effect = get_specific_model_config
        mock_config_factory.get_embedding_config.side_effect = get_embedding_config
        
        # Configurar mock de SentenceTransformer
        def create_mock_model(model_name):
            mock_model = MagicMock()
            
            if "modernbert" in model_name:
                mock_model.get_sentence_embedding_dimension.return_value = 768
                mock_model.encode.return_value = np.ones(768)
            elif "e5" in model_name:
                mock_model.get_sentence_embedding_dimension.return_value = 384
                mock_model.encode.return_value = np.ones(384)
            
            return mock_model
        
        mock_st.side_effect = create_mock_model
        
        # Obtener gestores para diferentes modelos
        modern_manager = EmbeddingFactory.get_embedding_manager("modernbert")
        e5_manager = EmbeddingFactory.get_embedding_manager("e5-small")
        
        # Acceder a los modelos para activar la carga
        _ = modern_manager.model
        _ = e5_manager.model
        
        # Verificar dimensiones diferentes
        self.assertEqual(modern_manager.embedding_dim, 768)
        self.assertEqual(e5_manager.embedding_dim, 384)
        
        # Verificar que hay dos instancias distintas
        active_models = EmbeddingFactory.get_active_models()
        self.assertEqual(len(active_models), 2)
        self.assertIn("embedding:modernbert", active_models)
        self.assertIn("embedding:e5-small", active_models)
        
        # Probar query_embedding con prefijo para e5
        query = "¿Qué es RAG?"
        e5_manager.get_query_embedding(query)
        
        # Verificar que se llamó con el prefijo correcto
        modern_manager._model.encode.assert_called_with(query)
        e5_manager._model.encode.assert_called_with(f"query: {query}")
    
    @patch('modulos.embeddings.embeddings_factory.time')
    @patch('modulos.embeddings.embeddings_manager.SentenceTransformer')
    @patch('modulos.embeddings.embeddings_factory.config')
    @patch('modulos.embeddings.embeddings_manager.config')
    def test_cleanup_mechanism(self, mock_config_manager, mock_config_factory, mock_st, mock_time):
        """Probar el mecanismo de limpieza de modelos inactivos"""
        # Configurar mocks de configuración
        mock_config_manager.get_embedding_config.return_value = {
            "model": "modernbert",
            "trust_remote_code": True
        }
        mock_config_manager.get_specific_model_config.return_value = {
            "model_name": "nomic-ai/modernbert-embed-base",
            "device": "cpu",
            "batch_size": 32
        }
        
        mock_config_factory.get_embedding_config.return_value = {
            "model": "modernbert"
        }
        
        # Configurar mock de SentenceTransformer
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_model
        
        # Configurar mock de time
        time_values = [1000.0, 1001.0, 1700.0]  # Initial, after get, after cleanup
        mock_time.time.side_effect = time_values
        
        # Establecer un timeout más corto para la prueba
        original_timeout = EmbeddingFactory._inactive_timeout
        EmbeddingFactory._inactive_timeout = 600  # 10 minutos
        
        try:
            # Obtener un gestor
            manager = EmbeddingFactory.get_embedding_manager("modernbert")
            
            # Verificar que hay una instancia
            self.assertEqual(len(EmbeddingFactory.get_active_models()), 1)
            
            # Liberar la referencia
            EmbeddingFactory.release_embedding_manager(manager)
            
            # Simular llamada al método de limpieza
            # La diferencia de tiempo simula 699 segundos (casi 10 minutos)
            EmbeddingFactory._cleanup_unused_models()
            
            # Verificar que aún existe la instancia (no ha pasado suficiente tiempo)
            self.assertEqual(len(EmbeddingFactory.get_active_models()), 1)
            
            # Cambiar tiempo a más de 10 minutos
            mock_time.time.return_value = 1700.0  # 699 segundos desde last_used
            
            # Ejecutar limpieza de nuevo
            EmbeddingFactory._cleanup_unused_models()
            
            # Verificar que se eliminó la instancia
            self.assertEqual(len(EmbeddingFactory.get_active_models()), 0)
            
        finally:
            # Restaurar timeout original
            EmbeddingFactory._inactive_timeout = original_timeout

if __name__ == '__main__':
    unittest.main() 