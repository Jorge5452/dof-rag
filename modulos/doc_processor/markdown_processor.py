import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import time

from config import config

# Configurar logging
logger = logging.getLogger(__name__)

class MarkdownProcessor:
    """
    Clase para procesar archivos Markdown.
    
    Esta clase se encarga de:
    1. Leer archivos Markdown
    2. Extraer metadatos
    3. Preparar el documento para el chunking
    """
    
    def __init__(self):
        """
        Inicializa el procesador de archivos Markdown.
        """
        self.processing_config = config.get_processing_config()
    
    def read_markdown_file(self, document_path: str) -> str:
        """
        Lee un archivo Markdown y retorna su contenido.
        
        Args:
            document_path: Ruta del archivo Markdown.
            
        Returns:
            Contenido del archivo como string.
            
        Raises:
            FileNotFoundError: Si el archivo no existe.
        """
        try:
            path = Path(document_path)
            if not path.exists():
                raise FileNotFoundError(f"El archivo {document_path} no existe")
                
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            logger.info(f"Archivo Markdown leído correctamente: {document_path}")
            return content
        except Exception as e:
            logger.error(f"Error al leer el archivo Markdown {document_path}: {e}")
            raise
    
    def extract_metadata(self, document_path: str, content: Optional[str] = None) -> Dict[str, Any]:
        """
        Extrae metadatos básicos de un documento Markdown.
        
        Args:
            document_path: Ruta del archivo Markdown.
            content: Contenido del archivo (opcional, si ya se ha leído).
        
        Returns:
            Diccionario con metadatos del documento:
            {
                'title': str,       # Título extraído o nombre del archivo
                'url': str,         # URL construida como 'file://' + ruta absoluta
                'file_path': str,   # Ruta completa del archivo
                'file_name': str,   # Nombre del archivo sin extensión
                'file_size': int,   # Tamaño en bytes
                'created_at': str,  # Fecha de creación (ISO format)
                'modified_at': str, # Fecha de modificación (ISO format)
            }
        """
        path = Path(document_path)
        
        # Leer el contenido si no se proporciona
        if content is None:
            content = self.read_markdown_file(document_path)
        
        # Metadatos básicos del sistema de archivos
        stat = path.stat()
        
        # Intentar extraer un título del contenido
        title = self._extract_title_from_content(content) or path.stem
        
        # Construir metadatos
        metadata = {
            'title': title,
            'url': f"file://{path.absolute()}",
            'file_path': str(path.absolute()),
            'file_name': path.stem,
            'file_size': stat.st_size,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_ctime)),
            'modified_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
            'content_length': len(content)
        }
        
        return metadata
    
    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """
        Intenta extraer un título del contenido del archivo Markdown.
        Busca encabezados de nivel 1 (# Título) o metadatos YAML frontmatter.
        
        Args:
            content: Contenido del archivo Markdown.
            
        Returns:
            Título extraído o None si no se encuentra.
        """
        import re
        
        # Buscar encabezado H1 (# Título)
        h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()
        
        # Buscar título en frontmatter YAML
        frontmatter_match = re.search(r'^---\s+(?:.|\n)+?title:\s*"?([^"\n]+)"?(?:.|\n)+?---', content)
        if frontmatter_match:
            return frontmatter_match.group(1).strip()
        
        return None
    
    def process_document(self, document_path: str) -> Tuple[Dict[str, Any], str]:
        """
        Procesa un documento Markdown completo.
        
        Args:
            document_path: Ruta del archivo Markdown.
            
        Returns:
            Tupla con (metadatos, contenido) del documento.
        """
        # Leer el contenido
        content = self.read_markdown_file(document_path)
        
        # Extraer metadatos
        metadata = self.extract_metadata(document_path, content)
        
        return metadata, content
    
    def process_batch(self, document_paths: List[str]) -> List[Tuple[Dict[str, Any], str]]:
        """
        Procesa un lote de documentos Markdown.
        
        Args:
            document_paths: Lista de rutas de archivos Markdown.
            
        Returns:
            Lista de tuplas (metadatos, contenido) para cada documento.
        """
        results = []
        
        for path in document_paths:
            try:
                result = self.process_document(path)
                results.append(result)
                logger.info(f"Documento procesado correctamente: {path}")
            except Exception as e:
                logger.error(f"Error al procesar el documento {path}: {e}")
                # Continuar con el siguiente documento
        
        return results
    
    def find_markdown_files(self, directory_path: str) -> List[str]:
        """
        Busca todos los archivos Markdown en un directorio.
        
        Args:
            directory_path: Ruta del directorio a explorar.
            
        Returns:
            Lista de rutas de archivos Markdown encontrados.
        """
        markdown_files = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.warning(f"El directorio {directory_path} no existe")
            return []
            
        if not directory.is_dir():
            logger.warning(f"{directory_path} no es un directorio")
            return []
        
        # Buscar archivos .md y .markdown
        for ext in ["*.md", "*.markdown"]:
            markdown_files.extend([str(f) for f in directory.glob(ext)])
        
        logger.info(f"Encontrados {len(markdown_files)} archivos Markdown en {directory_path}")
        return markdown_files
