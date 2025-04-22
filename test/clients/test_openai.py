#!/usr/bin/env python3
"""
Pruebas unitarias para el cliente de OpenAI.
"""

import unittest
import os
import sys
import json
from unittest.mock import patch, MagicMock, Mock

# Añadir directorio raíz al path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modulos.clientes.implementaciones.openai import OpenAIClient
from modulos.chunks.interfaces import Chunk

class TestOpenAIClient(unittest.TestCase):
    """Pruebas para el cliente de OpenAI"""
    
    @patch('modulos.clientes.implementaciones.openai.OpenAI')
    @patch('modulos.clientes.implementaciones.openai.config')
    def setUp(self, mock_config, mock_openai):
        """Configuración inicial para las pruebas"""
        # Configurar el mock de config
        mock_config.get_client_config.return_value = {
            "provider": "openai",
            "api_key": "test-api-key",
            "model": "gpt-3.5-turbo",
            "max_tokens": 2000,
            "temperature": 0.7,
            "top_p": 0.9,
            "context_size": 16385,
            "timeout": 60,
            "stream": False
        }
        
        # Crear un mock para el cliente OpenAI
        self.mock_client = MagicMock()
        mock_openai.return_value = self.mock_client
        
        # Configurar ChatCompletion para simular respuestas
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Esta es una respuesta de prueba de OpenAI."
        self.mock_client.chat.completions.create.return_value = mock_response
        
        # Inicializar el cliente
        self.client = OpenAIClient()
        
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
        self.assertEqual(self.client.model, "gpt-3.5-turbo")
        self.assertEqual(self.client.max_tokens, 2000)
        self.assertEqual(self.client.temperature, 0.7)
        self.assertEqual(self.client.top_p, 0.9)
        
        # Verificar que se inicializó el cliente de OpenAI
        self.assertIsNotNone(self.client.client)
    
    def test_generate_response(self):
        """Probar la generación de respuestas"""
        # Generar una respuesta
        response = self.client.generate_response("¿Qué es RAG?", self.chunks)
        
        # Verificar el resultado
        self.assertEqual(response, "Esta es una respuesta de prueba de OpenAI.")
        
        # Verificar la llamada al cliente de OpenAI
        self.mock_client.chat.completions.create.assert_called_once()
        
        # Verificar los argumentos de la llamada
        args, kwargs = self.mock_client.chat.completions.create.call_args
        
        # Verificar que se pasaron los parámetros correctos
        self.assertEqual(kwargs["model"], "gpt-3.5-turbo")
        self.assertEqual(kwargs["max_tokens"], 2000)
        self.assertEqual(kwargs["temperature"], 0.7)
        self.assertEqual(kwargs["top_p"], 0.9)
        
        # Verificar que el sistema y los mensajes de usuario están incluidos
        messages = kwargs["messages"]
        self.assertTrue(any(msg["role"] == "system" for msg in messages))
        self.assertTrue(any(msg["role"] == "user" for msg in messages))
        
        # Verificar que se incluyeron los chunks en el contexto
        user_message = next(msg["content"] for msg in messages if msg["role"] == "user")
        self.assertIn("Introducción a RAG", user_message)
        self.assertIn("Funcionamiento", user_message)
    
    @patch('modulos.clientes.implementaciones.openai.time.sleep')
    def test_retry_mechanism(self, mock_sleep):
        """Probar el mecanismo de reintentos"""
        # Configurar el cliente para que falle en el primer intento
        self.mock_client.chat.completions.create.side_effect = [
            Exception("Error de servicio"),  # Primer intento: falla
            Exception("Error de servicio"),  # Segundo intento: falla
            MagicMock(choices=[MagicMock(message=MagicMock(content="Respuesta después de reintento"))])  # Tercer intento: éxito
        ]
        
        # Configurar retries y backoff
        self.client.max_retries = 3
        self.client.retry_backoff = 1
        
        # Generar respuesta (debería reintentar)
        response = self.client.generate_response("¿Qué es RAG?", self.chunks)
        
        # Verificar el resultado
        self.assertEqual(response, "Respuesta después de reintento")
        
        # Verificar que se hicieron 3 intentos
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 3)
        
        # Verificar que sleep fue llamado entre intentos
        self.assertEqual(mock_sleep.call_count, 2)
    
    @patch('modulos.clientes.implementaciones.openai.config')
    def test_stream_mode(self, mock_config):
        """Probar el modo de streaming"""
        # Configurar el cliente para usar streaming
        mock_config.get_client_config.return_value = {
            "provider": "openai",
            "api_key": "test-api-key",
            "model": "gpt-3.5-turbo",
            "max_tokens": 2000,
            "temperature": 0.7,
            "stream": True
        }
        
        # Crear un mock para el streaming de respuestas
        mock_stream_response = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Parte 1"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=" de"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=" la respuesta."))])
        ]
        self.mock_client.chat.completions.create.return_value = mock_stream_response
        
        # Reinicializar el cliente con la nueva configuración
        self.client = OpenAIClient()
        
        # Verificar que se activó el streaming
        self.assertTrue(self.client.stream)
        
        # Generar respuesta en modo streaming
        response = self.client.generate_response("¿Qué es RAG?", self.chunks)
        
        # Verificar el resultado combinado
        self.assertEqual(response, "Parte 1 de la respuesta.")
        
        # Verificar que se usó el streaming en la llamada
        args, kwargs = self.mock_client.chat.completions.create.call_args
        self.assertTrue(kwargs["stream"])

if __name__ == '__main__':
    unittest.main() 