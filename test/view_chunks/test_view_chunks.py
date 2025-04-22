#!/usr/bin/env python3
"""
Pruebas unitarias para el visualizador de chunks.
"""

import unittest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Añadir directorio raíz al path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modulos.view_chunks.chunk_exporter import ChunkExporter
from modulos.chunks.interfaces import Chunk

class TestChunkExporter(unittest.TestCase):
    """Pruebas para el exportador de chunks"""
    
    def setUp(self):
        """Configuración inicial para las pruebas"""
        # Crear un directorio temporal para los archivos de salida
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        
        # Inicializar el exportador
        self.exporter = ChunkExporter(output_dir=str(self.output_dir))
        
        # Crear chunks de prueba
        self.chunks = [
            Chunk(text="Este es el primer chunk de prueba.", header="Sección 1"),
            Chunk(text="Este es el segundo chunk de prueba.", header="Sección 1"),
            Chunk(text="Este es el tercer chunk de prueba.", header="Sección 2"),
            Chunk(text="Este es el cuarto chunk de prueba sin encabezado."),
        ]
    
    def tearDown(self):
        """Limpieza después de las pruebas"""
        # Eliminar el directorio temporal
        self.temp_dir.cleanup()
    
    def test_export_single_chunk(self):
        """Probar la exportación de un único chunk"""
        # Exportar el primer chunk
        chunk = self.chunks[0]
        filename = self.exporter.export_chunk(chunk, "test_single")
        
        # Verificar que se creó el archivo
        self.assertTrue(os.path.exists(filename))
        
        # Verificar el contenido
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Sección 1", content)
            self.assertIn("Este es el primer chunk de prueba", content)
    
    def test_export_chunk_collection(self):
        """Probar la exportación de una colección de chunks"""
        # Exportar todos los chunks
        filenames = self.exporter.export_chunks(self.chunks, "test_collection")
        
        # Verificar que se crearon los archivos
        self.assertEqual(len(filenames), len(self.chunks))
        for filename in filenames:
            self.assertTrue(os.path.exists(filename))
        
        # Verificar que los archivos contienen los contenidos esperados
        for i, filename in enumerate(filenames):
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn(self.chunks[i].text, content)
                if self.chunks[i].header:
                    self.assertIn(self.chunks[i].header, content)
    
    def test_export_with_metadata(self):
        """Probar la exportación con metadatos adicionales"""
        # Crear metadatos
        metadata = {
            "document_title": "Documento de Prueba",
            "document_url": "test/document.md",
            "embedding_model": "modernbert",
            "chunking_method": "context"
        }
        
        # Exportar con metadatos
        filename = self.exporter.export_chunk(self.chunks[0], "test_metadata", metadata)
        
        # Verificar que se incluyeron los metadatos
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
            for key, value in metadata.items():
                self.assertIn(key, content)
                self.assertIn(value, content)
    
    def test_export_formatting(self):
        """Probar las opciones de formato en la exportación"""
        # Configurar el exportador con opciones de formato
        self.exporter.include_separators = True
        self.exporter.include_chunk_number = True
        
        # Exportar chunks
        filenames = self.exporter.export_chunks(self.chunks, "test_formatting")
        
        # Verificar formato
        for i, filename in enumerate(filenames):
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()
                # Verificar separadores
                self.assertIn("="*40, content)
                # Verificar numeración
                self.assertIn(f"Chunk #{i+1}", content)
    
    def test_custom_output_directory(self):
        """Probar la exportación a un directorio personalizado"""
        # Crear un subdirectorio
        subdir = self.output_dir / "custom_subdir"
        subdir.mkdir(exist_ok=True)
        
        # Configurar exportador para usar ese directorio
        exporter = ChunkExporter(output_dir=str(subdir))
        
        # Exportar un chunk
        filename = exporter.export_chunk(self.chunks[0], "test_subdir")
        
        # Verificar que se guardó en el subdirectorio
        self.assertTrue(str(subdir) in filename)
        self.assertTrue(os.path.exists(filename))

if __name__ == '__main__':
    unittest.main() 