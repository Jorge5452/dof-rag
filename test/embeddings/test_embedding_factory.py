#!/usr/bin/env python3
"""
Pruebas unitarias para el EmbeddingFactory.
"""

import unittest
import os
import sys
import time
import threading
from unittest.mock import patch, MagicMock

# Añadir directorio raíz al path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modulos.embeddings.embeddings_factory import EmbeddingFactory, ModelReference
from modulos.embeddings.embeddings_manager import EmbeddingManager

class TestModelReference(unittest.TestCase):
    """Pruebas para la clase ModelReference que gestiona referencias a modelos"""
    
    def setUp(self):
        """Configuración inicial para las pruebas"""
        self.mock_model = MagicMock()
        self.model_ref = ModelReference(self.mock_model)
    
    def test_increment(self):
        """Probar el incremento del contador de referencias"""
        # Verificar que inicia en 0
        self.assertEqual(self.model_ref.reference_count, 0)
        
        # Incrementar y verificar
        count = self.model_ref.increment()
        self.assertEqual(count, 1)
        self.assertEqual(self.model_ref.reference_count, 1)
        
        # Incrementar de nuevo
        count = self.model_ref.increment()
        self.assertEqual(count, 2)
        self.assertEqual(self.model_ref.reference_count, 2)
    
    def test_decrement(self):
        """Probar el decremento del contador de referencias"""
        # Incrementar a 2
        self.model_ref.increment()
        self.model_ref.increment()
        self.assertEqual(self.model_ref.reference_count, 2)
        
        # Decrementar y verificar
        count = self.model_ref.decrement()
        self.assertEqual(count, 1)
        self.assertEqual(self.model_ref.reference_count, 1)
        
        # Decrementar de nuevo
        count = self.model_ref.decrement()
        self.assertEqual(count, 0)
        self.assertEqual(self.model_ref.reference_count, 0)
        
        # Verificar que no baja de 0
        count = self.model_ref.decrement()
        self.assertEqual(count, 0)
        self.assertEqual(self.model_ref.reference_count, 0)
    
    def test_update_last_used(self):
        """Probar la actualización del timestamp de último uso"""
        # Guardar el timestamp actual
        original_time = self.model_ref.last_used
        
        # Esperar un poco
        time.sleep(0.1)
        
        # Actualizar y verificar
        self.model_ref.update_last_used()
        self.assertGreater(self.model_ref.last_used, original_time)

class TestEmbeddingFactory(unittest.TestCase):
    """Pruebas para la factoría de embeddings"""
    
    def setUp(self):
        """Configuración inicial para las pruebas"""
        # Reiniciar las instancias para cada prueba
        EmbeddingFactory.reset_instances()
    
    @patch('modulos.embeddings.embeddings_factory.EmbeddingManager')
    @patch('modulos.embeddings.embeddings_factory.config')
    def test_get_embedding_manager(self, mock_config, mock_em):
        """Probar la obtención de gestores de embeddings"""
        # Configurar mocks
        mock_config.get_embedding_config.return_value = {
            "model": "modernbert"
        }
        
        # Crear una instancia mock del gestor
        mock_instance = MagicMock()
        mock_em.return_value = mock_instance
        
        # Obtener un gestor
        manager1 = EmbeddingFactory.get_embedding_manager("modernbert")
        
        # Verificar que se creó correctamente
        mock_em.assert_called_once_with("modernbert")
        self.assertEqual(manager1, mock_instance)
        
        # Obtener otro gestor del mismo tipo (debe reutilizar)
        manager2 = EmbeddingFactory.get_embedding_manager("modernbert")
        
        # Verificar que solo se llamó una vez al constructor
        mock_em.assert_called_once()
        self.assertEqual(manager2, mock_instance)
        
        # Verificar contador de referencias
        with EmbeddingFactory._lock:
            model_ref = EmbeddingFactory._instances["embedding:modernbert"]
            self.assertEqual(model_ref.get_ref_count(), 2)
    
    @patch('modulos.embeddings.embeddings_factory.EmbeddingManager')
    @patch('modulos.embeddings.embeddings_factory.config')
    def test_release_embedding_manager(self, mock_config, mock_em):
        """Probar la liberación de gestores de embeddings"""
        # Configurar mocks
        mock_config.get_embedding_config.return_value = {
            "model": "modernbert"
        }
        
        # Crear una instancia mock del gestor
        mock_instance = MagicMock()
        mock_em.return_value = mock_instance
        
        # Obtener un gestor
        manager = EmbeddingFactory.get_embedding_manager("modernbert")
        
        # Verificar el conteo inicial
        with EmbeddingFactory._lock:
            model_ref = EmbeddingFactory._instances["embedding:modernbert"]
            self.assertEqual(model_ref.get_ref_count(), 1)
        
        # Liberar la referencia
        count = EmbeddingFactory.release_embedding_manager(manager)
        
        # Verificar que se liberó correctamente
        self.assertEqual(count, 0)
        with EmbeddingFactory._lock:
            model_ref = EmbeddingFactory._instances["embedding:modernbert"]
            self.assertEqual(model_ref.get_ref_count(), 0)
    
    @patch('modulos.embeddings.embeddings_factory.EmbeddingManager')
    @patch('modulos.embeddings.embeddings_factory.config')
    def test_get_active_models(self, mock_config, mock_em):
        """Probar la obtención de modelos activos"""
        # Configurar mocks
        mock_config.get_embedding_config.return_value = {
            "model": "modernbert"
        }
        
        # Crear instancias mock de gestores con atributos necesarios
        mock_instance1 = MagicMock()
        mock_instance1.model_type = "modernbert"
        mock_instance2 = MagicMock()
        mock_instance2.model_type = "e5-small"
        
        # Configurar retornos diferentes según el modelo
        def side_effect(model_type):
            if model_type == "modernbert":
                return mock_instance1
            elif model_type == "e5-small":
                return mock_instance2
        
        mock_em.side_effect = side_effect
        
        # Obtener gestores
        EmbeddingFactory.get_embedding_manager("modernbert")
        EmbeddingFactory.get_embedding_manager("e5-small")
        
        # Obtener modelos activos
        active_models = EmbeddingFactory.get_active_models()
        
        # Verificar que hay dos modelos activos
        self.assertEqual(len(active_models), 2)
        self.assertIn("embedding:modernbert", active_models)
        self.assertIn("embedding:e5-small", active_models)
        
        # Verificar la estructura de la información
        modernbert_info = active_models["embedding:modernbert"]
        self.assertIn("reference_count", modernbert_info)
        self.assertIn("last_used", modernbert_info)
        self.assertIn("idle_time", modernbert_info)
        self.assertIn("model_type", modernbert_info)
        self.assertEqual(modernbert_info["model_type"], "modernbert")
    
    def test_reset_instances(self):
        """Probar el reinicio de instancias"""
        # Simular la creación de instancias
        instance_key = "embedding:test"
        mock_model = MagicMock()
        model_ref = ModelReference(mock_model)
        
        # Añadir manualmente la instancia
        with EmbeddingFactory._lock:
            EmbeddingFactory._instances[instance_key] = model_ref
        
        # Verificar que hay una instancia
        self.assertEqual(len(EmbeddingFactory._instances), 1)
        
        # Reiniciar instancias
        EmbeddingFactory.reset_instances()
        
        # Verificar que no hay instancias
        self.assertEqual(len(EmbeddingFactory._instances), 0)

if __name__ == '__main__':
    unittest.main() 