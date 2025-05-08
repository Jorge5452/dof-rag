#!/usr/bin/env python3
"""
Pruebas unitarias para el pipeline RAG.
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock, Mock

# Añadir directorio raíz al path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Imports del módulo RAG
from modulos.rag.pipeline import RAGPipeline
from modulos.doc_processor.markdown_processor import Document
from test.utils.test_chunks import TestChunk

class TestRAGPipeline(unittest.TestCase):
    """Pruebas para el pipeline de RAG"""
    
    @patch('modulos.rag.pipeline.config')
    @patch('modulos.rag.pipeline.ChunkerFactory')
    @patch('modulos.rag.pipeline.EmbeddingFactory')
    @patch('modulos.rag.pipeline.DatabaseFactory')
    @patch('modulos.rag.pipeline.ClientFactory')
    def setUp(self, mock_client_factory, mock_db_factory, mock_embedding_factory, 
              mock_chunker_factory, mock_config):
        """Configuración inicial para las pruebas"""
        # Configurar mocks para las factorías
        self.mock_chunker = MagicMock()
        self.mock_embedding_manager = MagicMock()
        self.mock_db = MagicMock()
        self.mock_client = MagicMock()
        
        mock_chunker_factory.get_chunker.return_value = self.mock_chunker
        mock_embedding_factory.get_embedding_manager.return_value = self.mock_embedding_manager
        mock_db_factory.get_database_instance.return_value = self.mock_db
        mock_client_factory.get_client.return_value = self.mock_client
        
        # Configurar dimensiones del embedding
        self.mock_embedding_manager.get_dimensions.return_value = 768
        
        # Configurar respuesta simulada del cliente
        self.mock_client.generate_response.return_value = "Esta es una respuesta generada por el LLM."
        
        # Inicializar el pipeline RAG
        self.pipeline = RAGPipeline()
        
        # Preparar un documento de ejemplo
        self.test_document = Document(
            title="Documento de Prueba",
            content="# Documento de Prueba\n\nEste es un documento para pruebas de RAG.",
            url="test/document.md",
            path="test/document.md"
        )
        
        # Configurar chunks de muestra
        self.test_chunks = [
            TestChunk(text="Este es un documento para pruebas de RAG.", 
                 header="Documento de Prueba")
        ]
        
        # Configurar el chunker para devolver chunks de prueba
        self.mock_chunker.process.return_value = self.test_chunks
        
        # Configurar embeddings de prueba
        self.mock_embedding_manager.get_document_embedding.return_value = [0.1] * 768
        self.mock_embedding_manager.get_query_embedding.return_value = [0.2] * 768
    
    def test_ingest_document(self):
        """Probar la ingestión de un documento"""
        # Ejecutar ingestión
        self.pipeline.ingest_document(self.test_document)
        
        # Verificar que se llamó al chunker
        self.mock_chunker.process.assert_called_once_with(self.test_document.content)
        
        # Verificar que se generaron embeddings
        self.mock_embedding_manager.get_document_embedding.assert_called_once()
        
        # Verificar que se almacenó en la base de datos
        self.mock_db.insert_document.assert_called_once()
        self.mock_db.insert_chunk.assert_called_once()
    
    def test_query(self):
        """Probar la consulta al sistema RAG"""
        # Configurar la base de datos para devolver chunks relevantes
        self.mock_db.search_similar_chunks.return_value = self.test_chunks
        
        # Ejecutar consulta
        response = self.pipeline.query("¿Qué es RAG?")
        
        # Verificar el resultado
        self.assertEqual(response, "Esta es una respuesta generada por el LLM.")
        
        # Verificar que se generó embedding para la consulta
        self.mock_embedding_manager.get_query_embedding.assert_called_once()
        
        # Verificar que se buscaron chunks similares
        self.mock_db.search_similar_chunks.assert_called_once()
        
        # Verificar que se generó respuesta con el cliente
        self.mock_client.generate_response.assert_called_once()
    
    def test_batch_ingest(self):
        """Probar la ingestión por lotes"""
        # Crear varios documentos
        documents = [
            self.test_document,
            Document(
                title="Otro Documento",
                content="# Otro Documento\n\nEste es otro documento para pruebas.",
                url="test/another.md",
                path="test/another.md"
            )
        ]
        
        # Ejecutar ingestión por lotes
        self.pipeline.batch_ingest(documents)
        
        # Verificar que se procesaron todos los documentos
        self.assertEqual(self.mock_chunker.process.call_count, 2)
        self.assertEqual(self.mock_db.insert_document.call_count, 2)
    
    def test_pipeline_configuration(self):
        """Probar la configuración del pipeline"""
        # Verificar que se configuró correctamente
        self.assertIsNotNone(self.pipeline.chunker)
        self.assertIsNotNone(self.pipeline.embedding_manager)
        self.assertIsNotNone(self.pipeline.db)
        self.assertIsNotNone(self.pipeline.client)

if __name__ == '__main__':
    unittest.main() 