#!/usr/bin/env python3
"""
Utilidades para pruebas relacionadas con chunks.
Este módulo proporciona implementaciones concretas de ChunkAbstract para pruebas.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple

# Añadir directorio raíz al path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modulos.chunks.ChunkAbstract import ChunkAbstract


class TestChunk(ChunkAbstract):
    """
    Implementación concreta de ChunkAbstract para uso en pruebas unitarias.
    Esta clase implementa los métodos abstractos requeridos y proporciona
    funcionalidad básica para simular el comportamiento de un chunker real.
    """

    def __init__(self, text: str, header: Optional[str] = None, 
                 embedding: Optional[bytes] = None, embedding_dim: int = 0, 
                 page: Optional[str] = None):
        """
        Constructor que inicializa un chunk de prueba con valores específicos.
        
        Parámetros:
            text: Contenido del chunk
            header: Encabezado asociado al chunk
            embedding: Representación vectorial del chunk (opcional)
            embedding_dim: Dimensión del embedding (opcional)
            page: Número o etiqueta de página (opcional)
        """
        # Mockear la configuración antes de llamar a super().__init__
        import sys
        from unittest.mock import MagicMock
        
        # Guardar el módulo config original
        original_config = None
        if 'config' in sys.modules:
            original_config = sys.modules['config']
        
        # Crear un mock de config para la inicialización
        mock_config = MagicMock()
        mock_config.get_chunks_config.return_value = {
            "method": "context",
            "header_format": "standard"
        }
        sys.modules['config'] = MagicMock()
        sys.modules['config'].config = mock_config
        
        try:
            # Llamar al constructor de la clase padre con el embedding_model=None
            super().__init__(embedding_model=None)
        finally:
            # Restaurar el módulo config original
            if original_config:
                sys.modules['config'] = original_config
        
        self.text = text
        self.header = header
        self.embedding = embedding
        self.embedding_dim = embedding_dim
        self.page = page
        
        # Diccionario interno para compatibilidad con acceso por índice/clave
        self._data = {
            "text": text,
            "header": header,
            "embedding": embedding,
            "embedding_dim": embedding_dim,
            "page": page
        }

    def __getitem__(self, key):
        """
        Permite que la instancia sea accesible mediante índices o claves
        como si fuera un diccionario.
        
        Parámetros:
            key: Clave o índice para acceder al valor
            
        Retorna:
            El valor asociado a la clave
        """
        if isinstance(key, str):
            return self._data.get(key)
        else:
            raise TypeError("TestChunk solo acepta claves de tipo string")
    
    def __contains__(self, key):
        """
        Permite verificar si una clave está presente en el chunk.
        
        Parámetros:
            key: Clave a verificar
            
        Retorna:
            True si la clave existe, False en caso contrario
        """
        return key in self._data
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el chunk a un diccionario para facilitar su uso en pruebas.
        
        Retorna:
            Diccionario con los atributos del chunk
        """
        return self._data.copy()

    def extract_headers(self, content: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Implementación simplificada para pruebas que extrae encabezados básicos.
        
        Parámetros:
            content: Contenido del texto
            **kwargs: Parámetros adicionales
            
        Retorna:
            Lista de diccionarios con información de encabezados
        """
        headers = []
        lines = content.split('\n')
        position = 0
        
        for line in lines:
            level, text = self.get_heading_level(line)
            if level is not None:
                headers.append({
                    "header_text": text,
                    "level": level,
                    "start_index": position,
                    "end_index": position + len(line)
                })
            position += len(line) + 1  # +1 por el caracter de nueva línea
            
        return headers

    def chunk(self, content: str, headers: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Implementación simplificada para pruebas que divide el texto en chunks básicos.
        
        Parámetros:
            content: Contenido del texto
            headers: Lista de encabezados extraídos
            **kwargs: Parámetros adicionales
            
        Retorna:
            Lista de diccionarios con chunks
        """
        doc_title = kwargs.get("doc_title", "Documento de prueba")
        page = kwargs.get("page", "1")
        
        # Si no se especifica chunk_size, usar un valor predeterminado
        chunk_size = kwargs.get("chunk_size", 1000)
        
        # Dividir el contenido en chunks simples por tamaño
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk_text = content[i:i+chunk_size]
            header = self.find_header_for_position(i, headers)
            chunks.append({
                "text": chunk_text,
                "header": header or doc_title,
                "page": page
            })
            
        return chunks

    def process(self, text: str) -> List[Any]:
        """
        Método requerido para compatibilidad con los tests actuales.
        Procesa el texto y devuelve una lista de chunks.
        
        Parámetros:
            text: Texto a procesar
            
        Retorna:
            Lista que contiene este mismo chunk (para compatibilidad con tests)
        """
        return [self]


def create_test_chunks(count: int = 5, with_embeddings: bool = False) -> List[TestChunk]:
    """
    Crea una lista de chunks de prueba con datos simulados.
    
    Parámetros:
        count: Número de chunks a crear
        with_embeddings: Si se deben generar embeddings aleatorios
        
    Retorna:
        Lista de objetos TestChunk
    """
    chunks = []
    
    for i in range(count):
        # Generar embedding aleatorio si se solicita
        embedding = None
        embedding_dim = 0
        
        if with_embeddings:
            import numpy as np
            embedding_dim = 768  # Dimensión común en modelos de embedding
            embedding = np.random.rand(embedding_dim).astype(np.float32).tobytes()
        
        # Crear chunk con datos simulados
        chunk = TestChunk(
            text=f"Este es el contenido del chunk de prueba #{i+1}",
            header=f"Sección de prueba {i//2 + 1}",
            embedding=embedding,
            embedding_dim=embedding_dim,
            page=str(i//3 + 1)  # Agrupar chunks en "páginas"
        )
        
        chunks.append(chunk)
        
    return chunks 