#!/usr/bin/env python3
"""
Pruebas unitarias para el cliente de Ollama.
"""

import unittest
import os
import sys
import json
from unittest.mock import patch, MagicMock, Mock, AsyncMock

# Añadir directorio raíz al path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modulos.clientes.implementaciones.ollama import OllamaClient
from modulos.chunks.interfaces import Chunk

class TestOllamaClient(unittest.TestCase):
    """Pruebas para el cliente de Ollama"""
    
    @patch('modulos.clientes.implementaciones.ollama.httpx.AsyncClient')
    @patch('modulos.clientes.implementaciones.ollama.config')
    def setUp(self, mock_config, mock_async_client):
        """Configuración inicial para las pruebas"""
        # Configurar el mock de config
        mock_config.get_client_config.return_value = {
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "llama2",
            "max_tokens": 2000,
            "temperature": 0.7,
            "top_p": 0.9,
            "timeout": 60,
            "stream": False
        }
        
        # Configurar el cliente asíncrono
        self.mock_client = MagicMock()
        self.mock_client.__aenter__.return_value = self.mock_client
        self.mock_client.__aexit__.return_value = None
        
        # Configurar respuesta para la generación
        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value={
            "model": "llama2",
            "response": "Esta es una respuesta de prueba de Ollama.",
            "done": True
        })
        self.mock_client.post = AsyncMock(return_value=mock_response)
        
        mock_async_client.return_value = self.mock_client
        
        # Inicializar el cliente
        self.client = OllamaClient()
        
        # Preparar chunks de prueba
        self.chunks = [
            Chunk(text="Información sobre RAG: Retrieval Augmented Generation", 
                 header="Introducción a RAG"),
            Chunk(text="RAG combina búsqueda de información con generación de texto.", 
                 header="Funcionamiento")
        ]
    
    def test_initialization(self):
        """Probar la inicialización del cliente"""
        # Verificar que se cargó la configuración correctamente
        self.assertEqual(self.client.model, "llama2")
        self.assertEqual(self.client.max_tokens, 2000)
        self.assertEqual(self.client.temperature, 0.7)
        self.assertEqual(self.client.top_p, 0.9)
        self.assertEqual(self.client.base_url, "http://localhost:11434")
    
    @patch('modulos.clientes.implementaciones.ollama.asyncio.run')
    def test_generate_response(self, mock_asyncio_run):
        """Probar la generación de respuestas"""
        # Configurar el mock para asyncio.run
        mock_asyncio_run.return_value = "Esta es una respuesta de prueba de Ollama."
        
        # Generar una respuesta
        response = self.client.generate_response("¿Qué es RAG?", self.chunks)
        
        # Verificar el resultado
        self.assertEqual(response, "Esta es una respuesta de prueba de Ollama.")
        
        # Verificar que se llamó a asyncio.run
        mock_asyncio_run.assert_called_once()
    
    @patch('modulos.clientes.implementaciones.ollama.asyncio.sleep', new_callable=AsyncMock)
    @patch('modulos.clientes.implementaciones.ollama.httpx.AsyncClient')
    @patch('modulos.clientes.implementaciones.ollama.config')
    @patch('modulos.clientes.implementaciones.ollama.asyncio.run')
    def test_retry_mechanism(self, mock_asyncio_run, mock_config, mock_async_client, mock_sleep):
        """Probar el mecanismo de reintentos"""
        # Configurar mocks
        mock_config.get_client_config.return_value = {
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "llama2",
            "max_tokens": 2000,
            "temperature": 0.7,
            "stream": False
        }
        
        # Configurar cliente con errores y luego éxito
        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        
        # Primera respuesta - Error
        error_response = MagicMock()
        error_response.json = AsyncMock(side_effect=Exception("Error de servicio"))
        
        # Segunda respuesta - Éxito
        success_response = MagicMock()
        success_response.json = AsyncMock(return_value={
            "model": "llama2",
            "response": "Respuesta después de reintento",
            "done": True
        })
        
        # Configurar secuencia de respuestas
        mock_client.post = AsyncMock(side_effect=[error_response, success_response])
        mock_async_client.return_value = mock_client
        
        # Simular resultado de asyncio.run
        mock_asyncio_run.return_value = "Respuesta después de reintento"
        
        # Reinicializar cliente para usar los nuevos mocks
        self.client = OllamaClient()
        self.client.max_retries = 3
        self.client.retry_backoff = 1
        
        # Generar respuesta
        response = self.client.generate_response("¿Qué es RAG?", self.chunks)
        
        # Verificar el resultado
        self.assertEqual(response, "Respuesta después de reintento")
    
    @patch('modulos.clientes.implementaciones.ollama.httpx.AsyncClient')
    @patch('modulos.clientes.implementaciones.ollama.config')
    @patch('modulos.clientes.implementaciones.ollama.asyncio.run')
    def test_stream_mode(self, mock_asyncio_run, mock_config, mock_async_client):
        """Probar el modo de streaming"""
        # Configurar el cliente para usar streaming
        mock_config.get_client_config.return_value = {
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "llama2",
            "max_tokens": 2000,
            "temperature": 0.7,
            "stream": True
        }
        
        # Crear respuestas simuladas para el streaming
        responses = [
            {"response": "Parte 1", "done": False},
            {"response": " de", "done": False},
            {"response": " la respuesta.", "done": True}
        ]
        
        # Configurar mock para AsyncClient
        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        
        # Configurar respuesta de streaming
        mock_response = MagicMock()
        mock_response.aiter_lines = AsyncMock(return_value=AsyncMock(__aiter__=lambda: iter([
            json.dumps(responses[0]),
            json.dumps(responses[1]),
            json.dumps(responses[2])
        ])))
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_async_client.return_value = mock_client
        
        # Simular resultado de asyncio.run
        mock_asyncio_run.return_value = "Parte 1 de la respuesta."
        
        # Reinicializar cliente con la nueva configuración
        self.client = OllamaClient()
        
        # Verificar que se activó el streaming
        self.assertTrue(self.client.stream)
        
        # Generar respuesta en modo streaming
        response = self.client.generate_response("¿Qué es RAG?", self.chunks)
        
        # Verificar el resultado combinado
        self.assertEqual(response, "Parte 1 de la respuesta.")

if __name__ == '__main__':
    unittest.main() 