#!/usr/bin/env python3
"""
Pruebas unitarias para el EmbeddingManager.
"""

import unittest
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np
from typing import List

# Añadir directorio raíz al path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modulos.embeddings.embeddings_manager import EmbeddingManager

class TestEmbeddingManager(unittest.TestCase):
    """Pruebas para el gestor de embeddings"""
    
    @patch('modulos.embeddings.embeddings_manager.config')
    def setUp(self, mock_config):
        """Configuración inicial para las pruebas"""
        # Configurar el mock de config
        mock_config.get_embedding_config.return_value = {
            "model": "modernbert",
            "trust_remote_code": True
        }
        mock_config.get_specific_model_config.return_value = {
            "model_name": "nomic-ai/modernbert-embed-base",
            "device": "cpu",
            "batch_size": 32
        }
        
        # Inicializar el gestor de embeddings
        self.embedding_manager = EmbeddingManager("modernbert")
    
    @patch('modulos.embeddings.embeddings_manager.SentenceTransformer')
    def test_load_model(self, mock_st):
        """Probar la carga del modelo"""
        # Configurar el mock de SentenceTransformer
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_model
        
        # Cargar el modelo
        self.embedding_manager.load_model()
        
        # Verificar que se cargó correctamente
        mock_st.assert_called_once_with(
            "nomic-ai/modernbert-embed-base",
            device="cpu",
            trust_remote_code=True
        )
        
        # Verificar la dimensión
        self.assertEqual(self.embedding_manager.embedding_dim, 768)
    
    @patch('modulos.embeddings.embeddings_manager.SentenceTransformer')
    def test_get_document_embedding(self, mock_st):
        """Probar la generación de embeddings para documentos"""
        # Configurar el mock de SentenceTransformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones(768)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_model
        
        # Asegurar que el modelo está cargado
        self.embedding_manager._model = mock_model
        self.embedding_manager._embedding_dim = 768
        
        # Generar embedding con encabezado
        embedding = self.embedding_manager.get_document_embedding("Título", "Contenido del documento")
        
        # Verificar que se llamó encode con el texto combinado
        mock_model.encode.assert_called_with("Título - Contenido del documento")
        
        # Verificar que el resultado es una lista de floats (no un array numpy)
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 768)
        
        # Generar embedding sin encabezado
        embedding = self.embedding_manager.get_document_embedding(None, "Contenido del documento")
        
        # Verificar que se llamó encode solo con el contenido
        mock_model.encode.assert_called_with("Contenido del documento")
    
    @patch('modulos.embeddings.embeddings_manager.SentenceTransformer')
    def test_get_query_embedding(self, mock_st):
        """Probar la generación de embeddings para consultas"""
        # Configurar el mock de SentenceTransformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones(768)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_model
        
        # Asegurar que el modelo está cargado
        self.embedding_manager._model = mock_model
        self.embedding_manager._embedding_dim = 768
        
        # Generar embedding para una consulta
        embedding = self.embedding_manager.get_query_embedding("¿Qué es RAG?")
        
        # Verificar que se llamó encode con la consulta
        mock_model.encode.assert_called_with("¿Qué es RAG?")
        
        # Verificar que el resultado es una lista de floats
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 768)
    
    @patch('modulos.embeddings.embeddings_manager.SentenceTransformer')
    def test_batch_encode(self, mock_st):
        """Probar la codificación por lotes"""
        # Configurar el mock de SentenceTransformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((3, 768))
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_model
        
        # Asegurar que el modelo está cargado
        self.embedding_manager._model = mock_model
        self.embedding_manager._embedding_dim = 768
        
        # Crear una lista de textos
        texts = ["Texto 1", "Texto 2", "Texto 3"]
        
        # Generar embeddings en batch
        embeddings = self.embedding_manager.batch_encode(texts)
        
        # Verificar que se llamó encode con los textos y el batch_size
        mock_model.encode.assert_called_with(texts, batch_size=32)
        
        # Verificar que el resultado es una lista de listas
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), 3)
        
if __name__ == '__main__':
    unittest.main() 