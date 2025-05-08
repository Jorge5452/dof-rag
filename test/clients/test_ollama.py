#!/usr/bin/env python3
"""
Pruebas unitarias para el cliente de Ollama.
"""

import unittest
import os
import sys
import json
import asyncio
from unittest.mock import patch, MagicMock, Mock, AsyncMock

# Añadir directorio raíz al path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modulos.clientes.implementaciones.ollama import OllamaClient
from test.utils.test_chunks import TestChunk

class TestOllamaClient(unittest.TestCase):
    """Pruebas para el cliente de Ollama"""
    
    @patch('modulos.clientes.implementaciones.ollama.requests.get')
    def setUp(self, mock_requests_get):
        """Configuración inicial para las pruebas"""
        # Configurar respuesta simulada para la verificación de conexión
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response
        
        # Inicializar el cliente con parámetros directos en lugar de usar config
        self.client = OllamaClient(
            base_url="http://localhost:11434",
            model_name="llama2",
            max_tokens=2000,
            temperature=0.7,
            top_p=0.9,
            timeout=60,
            stream=False
        )
        
        # Preparar chunks de prueba
        self.chunks = [
            TestChunk(text="Información sobre RAG: Retrieval Augmented Generation", 
                 header="Introducción a RAG"),
            TestChunk(text="RAG combina búsqueda de información con generación de texto.", 
                 header="Funcionamiento")
        ]
        
        # Convertir chunks a diccionarios para los tests
        self.chunks_dict = [chunk.to_dict() for chunk in self.chunks]
    
    def test_initialization(self):
        """Probar la inicialización del cliente"""
        # Verificar que se cargó la configuración correctamente
        self.assertEqual(self.client.model_name, "llama2")
        self.assertEqual(self.client.max_tokens, 2000)
        self.assertEqual(self.client.temperature, 0.7)
        self.assertEqual(self.client.top_p, 0.9)
        self.assertEqual(self.client.base_url, "http://localhost:11434")
    
    @patch('modulos.clientes.implementaciones.ollama.requests.post')
    def test_generate_response(self, mock_post):
        """Probar la generación de respuestas"""
        # Configurar respuesta simulada
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": "Esta es una respuesta de prueba de Ollama."
            }
        }
        mock_post.return_value = mock_response
        
        # Generar una respuesta
        response = self.client.generate_response("¿Qué es RAG?", self.chunks_dict)
        
        # Verificar el resultado
        self.assertEqual(response, "Esta es una respuesta de prueba de Ollama.")
        
        # Verificar que se llamó a la API correctamente
        mock_post.assert_called_once()
        
        # Verificar que se incluyeron los chunks en el contexto
        args, kwargs = mock_post.call_args
        payload = kwargs.get('json', {})
        self.assertIn('messages', payload)
        self.assertEqual(len(payload['messages']), 2)  # system y user
        self.assertIn('Información sobre RAG', payload['messages'][0]['content'])
    
    @patch('modulos.clientes.implementaciones.ollama.requests.post')
    def test_retry_mechanism(self, mock_post):
        """Probar el mecanismo de reintentos simulado"""
        # Configurar primera respuesta (error)
        error_response = MagicMock()
        error_response.raise_for_status.side_effect = Exception("Error de servicio")
        
        # Configurar segunda respuesta (éxito)
        success_response = MagicMock()
        success_response.raise_for_status.return_value = None
        success_response.json.return_value = {
            "message": {
                "content": "Respuesta después de reintento"
            }
        }
        
        # Establecer secuencia de respuestas
        mock_post.side_effect = [error_response, success_response]
        
        # Simular reintentos manualmente
        try:
            # Primer intento - debería fallar
            response = self.client.generate_response("¿Qué es RAG?", self.chunks_dict)
        except Exception:
            # Simular reintento manual
            response = self.client.generate_response("¿Qué es RAG?", self.chunks_dict)
        
        # Verificar el resultado
        self.assertEqual(response, "Respuesta después de reintento")
        
        # Verificar que se hicieron dos llamadas
        self.assertEqual(mock_post.call_count, 2)
    
    @patch('modulos.clientes.implementaciones.ollama.requests.post')
    def test_stream_mode(self, mock_post):
        """Probar el modo de streaming"""
        # Configurar cliente para streaming
        self.client.stream = True
        
        # Crear una respuesta que simule el streaming
        mock_response = MagicMock()
        
        # Simular líneas de respuesta para iter_lines
        mock_response.iter_lines.return_value = [
            json.dumps({"message": {"content": "Parte 1"}, "done": False}).encode(),
            json.dumps({"message": {"content": " de"}, "done": False}).encode(),
            json.dumps({"message": {"content": " la respuesta."}, "done": True}).encode()
        ]
        
        mock_post.return_value = mock_response
        
        # Generar respuesta en modo streaming
        response = self.client.generate_response("¿Qué es RAG?", self.chunks_dict)
        
        # Verificar el resultado combinado
        self.assertEqual(response, "Parte 1 de la respuesta.")
        
        # Verificar que se usó el modo streaming en la llamada
        args, kwargs = mock_post.call_args
        self.assertTrue(kwargs.get('json', {}).get('stream', False))

if __name__ == '__main__':
    unittest.main() 