#!/usr/bin/env python3
"""
Pruebas unitarias para el gestor de sesiones.
"""

import unittest
import os
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Añadir directorio raíz al path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modulos.session_manager.session_manager import SessionManager
from test.utils.test_chunks import TestChunk

class TestSessionManager(unittest.TestCase):
    """Pruebas para el gestor de sesiones"""
    
    def setUp(self):
        """Configuración inicial para las pruebas"""
        # Crear un directorio temporal para los archivos de sesión
        self.temp_dir = tempfile.TemporaryDirectory()
        self.session_dir = Path(self.temp_dir.name)
        
        # Configurar ruta para archivos de sesión
        self.session_path = self.session_dir / "sessions"
        self.session_path.mkdir(exist_ok=True)
        
        # Inicializar el gestor de sesiones
        self.session_manager = SessionManager(session_dir=str(self.session_path))
        
        # Crear algunas consultas y respuestas de muestra
        self.sample_queries = [
            "¿Qué es RAG?",
            "¿Cómo funciona el chunking?",
            "¿Qué modelos de embeddings son compatibles?"
        ]
        
        self.sample_responses = [
            "RAG (Retrieval Augmented Generation) es una técnica que combina...",
            "El chunking divide los documentos en fragmentos más pequeños...",
            "Los modelos compatibles incluyen ModernBERT, E5-small, CDE, entre otros..."
        ]
        
        # Crear chunks de ejemplo para el contexto
        self.sample_chunks = [
            TestChunk(text="RAG combina la recuperación de información con generación.", 
                 header="Definición de RAG"),
            TestChunk(text="Los chunkers dividen documentos en fragmentos semánticos.", 
                 header="Chunking")
        ]
        
        # Convertir chunks a diccionarios para los tests si es necesario
        self.sample_chunks_dict = [chunk.to_dict() for chunk in self.sample_chunks]
    
    def tearDown(self):
        """Limpieza después de las pruebas"""
        # Eliminar el directorio temporal
        self.temp_dir.cleanup()
    
    def test_create_session(self):
        """Probar la creación de una sesión"""
        # Crear una nueva sesión
        session_id = self.session_manager.create_session()
        
        # Verificar que la sesión existe
        self.assertTrue(self.session_manager.session_exists(session_id))
        
        # Verificar que el archivo de sesión existe
        session_file = self.session_path / f"{session_id}.json"
        self.assertTrue(session_file.exists())
        
        # Verificar que la sesión tiene la estructura correcta
        with open(session_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)
            self.assertIn("created_at", session_data)
            self.assertIn("updated_at", session_data)
            self.assertIn("interactions", session_data)
            self.assertEqual(len(session_data["interactions"]), 0)
    
    def test_add_interaction(self):
        """Probar la adición de interacciones a una sesión"""
        # Crear una sesión
        session_id = self.session_manager.create_session()
        
        # Añadir una interacción
        self.session_manager.add_interaction(
            session_id, 
            self.sample_queries[0], 
            self.sample_responses[0], 
            self.sample_chunks_dict
        )
        
        # Verificar que la interacción se añadió
        session_data = self.session_manager.get_session(session_id)
        self.assertEqual(len(session_data["interactions"]), 1)
        
        # Verificar el contenido de la interacción
        interaction = session_data["interactions"][0]
        self.assertEqual(interaction["query"], self.sample_queries[0])
        self.assertEqual(interaction["response"], self.sample_responses[0])
        self.assertEqual(len(interaction["context_chunks"]), len(self.sample_chunks))
        
        # Añadir otra interacción
        self.session_manager.add_interaction(
            session_id, 
            self.sample_queries[1], 
            self.sample_responses[1]
        )
        
        # Verificar que hay dos interacciones
        session_data = self.session_manager.get_session(session_id)
        self.assertEqual(len(session_data["interactions"]), 2)
    
    def test_get_interactions(self):
        """Probar la obtención de interacciones"""
        # Crear una sesión y añadir varias interacciones
        session_id = self.session_manager.create_session()
        
        for i in range(3):
            self.session_manager.add_interaction(
                session_id,
                self.sample_queries[i],
                self.sample_responses[i]
            )
        
        # Obtener todas las interacciones
        interactions = self.session_manager.get_interactions(session_id)
        self.assertEqual(len(interactions), 3)
        
        # Verificar el orden (las más recientes primero)
        self.assertEqual(interactions[0]["query"], self.sample_queries[2])
        self.assertEqual(interactions[2]["query"], self.sample_queries[0])
    
    def test_delete_session(self):
        """Probar la eliminación de sesiones"""
        # Crear una sesión
        session_id = self.session_manager.create_session()
        
        # Verificar que existe
        self.assertTrue(self.session_manager.session_exists(session_id))
        
        # Eliminar la sesión
        self.session_manager.delete_session(session_id)
        
        # Verificar que ya no existe
        self.assertFalse(self.session_manager.session_exists(session_id))
        
        # Verificar que el archivo se eliminó
        session_file = self.session_path / f"{session_id}.json"
        self.assertFalse(session_file.exists())
    
    def test_list_sessions(self):
        """Probar el listado de sesiones"""
        # Crear varias sesiones
        session_ids = [
            self.session_manager.create_session(),
            self.session_manager.create_session(),
            self.session_manager.create_session()
        ]
        
        # Listar las sesiones
        sessions = self.session_manager.list_sessions()
        
        # Verificar que se encontraron todas
        self.assertEqual(len(sessions), 3)
        
        # Verificar que los IDs coinciden
        for session_id in session_ids:
            self.assertIn(session_id, [s["id"] for s in sessions])
    
    def test_get_session_history(self):
        """Probar la obtención del historial de sesión"""
        # Crear una sesión y añadir interacciones
        session_id = self.session_manager.create_session()
        
        for i in range(3):
            self.session_manager.add_interaction(
                session_id,
                self.sample_queries[i],
                self.sample_responses[i]
            )
        
        # Obtener el historial como lista de mensajes
        history = self.session_manager.get_session_history(session_id)
        
        # Verificar estructura
        self.assertEqual(len(history), 6)  # 3 pares de consulta-respuesta
        
        # Verificar alternancia de mensajes
        for i in range(0, 6, 2):
            self.assertEqual(history[i]["role"], "user")
            self.assertEqual(history[i+1]["role"], "assistant")

if __name__ == '__main__':
    unittest.main() 