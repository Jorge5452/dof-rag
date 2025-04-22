#!/usr/bin/env python3
"""
Pruebas unitarias para el procesador de documentos.
"""

import unittest
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

# Añadir directorio raíz al path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modulos.doc_processor.markdown_processor import MarkdownProcessor, Document

class TestMarkdownProcessor(unittest.TestCase):
    """Pruebas para el procesador de documentos Markdown"""
    
    def setUp(self):
        """Configuración inicial para las pruebas"""
        # Crear un directorio temporal para los archivos de prueba
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Crear algunos archivos Markdown de prueba
        self.create_test_files()
        
        # Inicializar el procesador
        self.processor = MarkdownProcessor()
    
    def tearDown(self):
        """Limpieza después de las pruebas"""
        # Eliminar el directorio temporal
        self.temp_dir.cleanup()
    
    def create_test_files(self):
        """Crear archivos Markdown de prueba"""
        # Archivo simple
        simple_md = self.test_dir / "simple.md"
        with open(simple_md, "w", encoding="utf-8") as f:
            f.write("# Documento Simple\n\nEste es un documento de prueba simple.")
        
        # Archivo con múltiples secciones
        sections_md = self.test_dir / "sections.md"
        with open(sections_md, "w", encoding="utf-8") as f:
            f.write("# Documento con Secciones\n\n")
            f.write("## Sección 1\n\nContenido de la sección 1.\n\n")
            f.write("## Sección 2\n\nContenido de la sección 2.\n\n")
            f.write("### Subsección 2.1\n\nContenido de la subsección 2.1.\n\n")
        
        # Archivo en subdirectorio
        subdir = self.test_dir / "subdir"
        subdir.mkdir(exist_ok=True)
        nested_md = subdir / "nested.md"
        with open(nested_md, "w", encoding="utf-8") as f:
            f.write("# Documento Anidado\n\nEste archivo está en un subdirectorio.")
        
        # Archivo que no es Markdown
        non_md = self.test_dir / "not_markdown.txt"
        with open(non_md, "w", encoding="utf-8") as f:
            f.write("Este archivo no es Markdown.")
    
    def test_process_markdown_file(self):
        """Probar el procesamiento de un archivo Markdown individual"""
        # Procesar el archivo simple
        simple_path = self.test_dir / "simple.md"
        document = self.processor.process_markdown_file(simple_path)
        
        # Verificar que se extrajo correctamente
        self.assertIsInstance(document, Document)
        self.assertEqual(document.title, "Documento Simple")
        self.assertEqual(document.url, str(simple_path))
        self.assertIn("Este es un documento de prueba simple", document.content)
    
    def test_process_markdown_directory(self):
        """Probar el procesamiento recursivo de un directorio"""
        # Procesar todo el directorio
        documents = self.processor.process_markdown_directory(self.test_dir)
        
        # Verificar que se encontraron todos los archivos Markdown
        self.assertEqual(len(documents), 3)  # simple.md, sections.md, nested.md
        
        # Verificar que se extrajo correctamente la información
        titles = [doc.title for doc in documents]
        self.assertIn("Documento Simple", titles)
        self.assertIn("Documento con Secciones", titles)
        self.assertIn("Documento Anidado", titles)
        
        # Verificar que no se incluyó el archivo no Markdown
        for doc in documents:
            self.assertNotIn("not_markdown.txt", doc.url)
    
    def test_document_parsing(self):
        """Probar el análisis detallado de un documento con secciones"""
        # Procesar el archivo con secciones
        sections_path = self.test_dir / "sections.md"
        document = self.processor.process_markdown_file(sections_path)
        
        # Verificar título y contenido
        self.assertEqual(document.title, "Documento con Secciones")
        
        # Verificar que se capturaron todas las secciones
        self.assertIn("## Sección 1", document.content)
        self.assertIn("Contenido de la sección 1", document.content)
        self.assertIn("## Sección 2", document.content)
        self.assertIn("Contenido de la sección 2", document.content)
        self.assertIn("### Subsección 2.1", document.content)
        self.assertIn("Contenido de la subsección 2.1", document.content)
    
    def test_document_metadata(self):
        """Probar la extracción de metadatos del documento"""
        # Procesar el archivo simple
        simple_path = self.test_dir / "simple.md"
        document = self.processor.process_markdown_file(simple_path)
        
        # Verificar metadatos
        self.assertEqual(document.title, "Documento Simple")
        self.assertEqual(document.url, str(simple_path))
        self.assertIsNotNone(document.created_at)
        self.assertIsNotNone(document.updated_at)
        
        # Verificar que la ruta es correcta
        self.assertEqual(document.path, str(simple_path))

if __name__ == '__main__':
    unittest.main() 