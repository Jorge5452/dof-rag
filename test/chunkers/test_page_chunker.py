#!/usr/bin/env python3
"""
Script de prueba para demostrar el uso del PageChunker.

Uso:
    python -m test.chunkers.test_page_chunker
"""
import os
import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Añadir el directorio raíz al path para permitir importaciones relativas
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from config import config
from modulos.chunks.ChunkerFactory import ChunkerFactory

def create_test_file():
    """
    Crea un archivo Markdown de ejemplo con marcadores de página para pruebas.
    """
    test_content = """# Documento de Prueba para PageChunker

Este es un documento de prueba que contiene marcadores de página para probar el PageChunker.

## Sección 1

Esta es la primera sección del documento que pertenece a la primera página.

{1}---------------------

## Sección 2

Esta es la segunda sección del documento que pertenece a la segunda página.

### Subsección 2.1

Esta es una subsección de la segunda sección.

{2}---------------------

## Sección 3

Esta es la tercera sección del documento que pertenece a la tercera página.

### Subsección 3.1

Esta es una subsección de la tercera sección.

#### Subsección 3.1.1

Esta es una subsección aún más anidada.

{3}---------------------

## Conclusión

Esta es la sección final del documento en la cuarta página.
"""
    # Crear directorio para pruebas si no existe
    test_dir = Path(root_dir) / "test_files"
    test_dir.mkdir(exist_ok=True)
    
    # Escribir el contenido al archivo
    test_file = test_dir / "test_page_chunker.md"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    logger.info(f"Archivo de prueba creado: {test_file}")
    return str(test_file)

def test_page_chunker(file_path):
    """
    Prueba el PageChunker con el archivo especificado.
    
    Args:
        file_path: Ruta al archivo Markdown con marcadores de página.
    """
    # Leer el contenido del archivo
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Obtener una instancia del PageChunker
    page_chunker = ChunkerFactory.get_chunker("page")
    
    # Procesar el contenido
    logger.info("Procesando documento con PageChunker...")
    doc_title = Path(file_path).stem
    chunks = page_chunker.process_content(content, doc_title=doc_title)
    
    # Mostrar información sobre los chunks generados
    logger.info(f"Se generaron {len(chunks)} chunks")
    
    # Mostrar detalles de cada chunk
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk #{i+1} (Página {chunk['page']}):")
        logger.info(f"Header: {chunk['header']}")
        logger.info(f"Texto ({len(chunk['text'])} caracteres): {chunk['text'][:50]}...")
        logger.info("-" * 50)

def main():
    """
    Función principal que ejecuta la prueba.
    """
    # Crear archivo de prueba
    file_path = create_test_file()
    
    # Modificar la configuración para usar el PageChunker
    logger.info("Configurando el sistema para usar PageChunker...")
    
    # Probar el PageChunker
    test_page_chunker(file_path)
    
    logger.info("Prueba completada.")

if __name__ == "__main__":
    main() 