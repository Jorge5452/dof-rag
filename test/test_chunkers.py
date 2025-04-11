import os
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import glob
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json
import textwrap
import sys
import yaml

# Añadir el directorio raíz al path para permitir importaciones relativas
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from modulos.chunks.ChunkerFactory import ChunkerFactory

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_markdown_files(base_dir: str) -> List[Path]:
    """
    Busca recursivamente todos los archivos Markdown en un directorio.
    
    Args:
        base_dir: Directorio base para iniciar la búsqueda
        
    Returns:
        Lista de rutas a archivos Markdown encontrados
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.error(f"El directorio {base_dir} no existe")
        return []
    
    # Buscar archivos .md recursivamente
    md_files = list(base_path.glob("**/*.md"))
    
    if not md_files:
        logger.warning(f"No se encontraron archivos .md en {base_dir}")
    else:
        logger.info(f"Se encontraron {len(md_files)} archivos .md en {base_dir}")
    
    return md_files

def load_embedding_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """
    Carga un modelo de embedding de sentence-transformers.
    
    Args:
        model_name: Nombre del modelo. Si es None, se toma de la configuración.
        
    Returns:
        Modelo de embedding inicializado
    """
    # Obtener la configuración de embeddings
    embedding_config = config.get_embedding_config()
    selected_model = model_name or embedding_config.get("model", "cde-small")
    
    # Obtener la configuración específica del modelo
    model_config = config.get_specific_model_config(selected_model)
    model_name = model_config.get("model_name")
    
    if not model_name:
        # Si no se encuentra el nombre del modelo en la configuración específica,
        # usar un valor predeterminado
        logger.warning(f"No se encontró configuración para el modelo {selected_model}, "
                      f"usando modelo predeterminado")
        model_name = "nomic-ai/modernbert-embed-base"
    
    logger.info(f"Cargando modelo de embedding: {model_name}")
    return SentenceTransformer(model_name)

def get_chunker_config(chunker_type: str) -> Dict[str, Any]:
    """
    Obtiene la configuración específica para un tipo de chunker.
    
    Args:
        chunker_type: Tipo de chunker ('character', 'token', 'context')
        
    Returns:
        Diccionario con la configuración específica del chunker
    """
    chunks_config = config.get_chunks_config()
    specific_config = chunks_config.get(chunker_type, {})
    
    # Añadir la configuración general relevante
    result = {
        "method": chunker_type,
        "general": {
            "config_global": chunks_config.get("method", "context")
        }
    }
    
    # Añadir configuración específica
    result.update({chunker_type: specific_config})
    
    return result

def process_document(document_path: str, chunker_type: str, embedding_model) -> List[Dict[str, Any]]:
    """
    Procesa un documento con el chunker especificado.
    
    Args:
        document_path: Ruta al documento Markdown
        chunker_type: Tipo de chunker ('character', 'token', 'context')
        embedding_model: Modelo de embedding inicializado
        
    Returns:
        Lista de chunks generados
    """
    # Obtener el chunker del factory
    chunker = ChunkerFactory.get_chunker(chunker_type, embedding_model)
    
    # Procesar el documento
    start_time = time.time()
    chunks = chunker.integrate(document_path)
    processing_time = time.time() - start_time
    
    logger.info(f"Documento procesado con {chunker_type} en {processing_time:.2f} segundos: "
               f"{len(chunks)} chunks generados")
    
    return chunks

def print_chunk_stats(chunks: List[Dict[str, Any]], chunker_type: str, max_preview: int = 3):
    """
    Muestra estadísticas sobre los chunks generados.
    
    Args:
        chunks: Lista de chunks
        chunker_type: Tipo de chunker utilizado
        max_preview: Número máximo de chunks a mostrar en la vista previa
    """
    if not chunks:
        logger.warning("No hay chunks para mostrar estadísticas")
        return
    
    # Obtener configuración del chunker
    chunker_config = get_chunker_config(chunker_type)
    
    # Calcular estadísticas
    chunk_lengths = [len(chunk["text"]) for chunk in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    min_length = min(chunk_lengths) if chunk_lengths else 0
    max_length = max(chunk_lengths) if chunk_lengths else 0
    
    # Mostrar estadísticas generales
    print("\n" + "=" * 60)
    print(f"CHUNKER: {chunker_type.upper()}")
    print(f"CONFIGURACIÓN:")
    for key, value in chunker_config.get(chunker_type, {}).items():
        print(f"  - {key}: {value}")
    print("\nESTADÍSTICAS:")
    print(f"- Total de chunks: {len(chunks)}")
    print(f"- Longitud promedio: {avg_length:.2f} caracteres")
    print(f"- Longitud mínima: {min_length} caracteres")
    print(f"- Longitud máxima: {max_length} caracteres")
    print(f"- Dimensión de embedding: {chunks[0]['embedding_dim'] if chunks else 'N/A'}")
    
    # Mostrar vista previa de algunos chunks
    print("\nVISTA PREVIA DE CHUNKS:")
    for i, chunk in enumerate(chunks[:max_preview]):
        print(f"\nChunk {i+1}:")
        print(f"- Header: {chunk['header']}")
        print(f"- Page: {chunk['page']}")
        # Mostrar una versión recortada del texto para evitar salidas muy largas en consola
        preview_text = chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
        print(f"- Text ({len(chunk['text'])} caracteres): {preview_text}")
        print(f"- Embedding shape: {len(chunk['embedding']) if chunk['embedding'] is not None else 'None'}")
    
    # Si hay más chunks de los que se muestran, indicarlo
    if len(chunks) > max_preview:
        print(f"\n... y {len(chunks) - max_preview} chunks más.")
    
    print("=" * 60 + "\n")

def save_results_to_file(results_dir: Path, chunker_type: str, document_path: str, chunks: List[Dict[str, Any]]):
    """
    Guarda los resultados de un chunker para un documento específico en un archivo de texto.
    
    Args:
        results_dir: Directorio donde guardar los resultados
        chunker_type: Tipo de chunker utilizado
        document_path: Ruta del documento procesado
        chunks: Lista de chunks generados
    """
    # Crear la carpeta para los resultados si no existe
    os.makedirs(results_dir, exist_ok=True)
    
    # Extraer el nombre base del archivo original sin extensión
    doc_path = Path(document_path)
    doc_name = doc_path.stem
    
    # Crear un nombre de archivo único que incluya el tipo de chunker y el nombre del documento
    result_file = results_dir / f"{chunker_type}_{doc_name}_results.txt"
    
    # Obtener configuración del chunker
    chunker_config = get_chunker_config(chunker_type)
    
    # Calcular estadísticas
    chunk_lengths = [len(chunk["text"]) for chunk in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    min_length = min(chunk_lengths) if chunk_lengths else 0
    max_length = max(chunk_lengths) if chunk_lengths else 0
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Abrir el archivo en modo write (no append) para cada documento/chunker
    with open(result_file, 'w', encoding='utf-8') as f:
        # Escribir encabezado para este documento
        f.write(f"{'=' * 80}\n")
        f.write(f"MÉTODO DE CHUNKING: {chunker_type.upper()}\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"FECHA Y HORA: {timestamp}\n")
        f.write(f"DOCUMENTO: {document_path}\n")
        f.write(f"NOMBRE DEL ARCHIVO: {doc_name}\n\n")
        
        # Escribir configuración detallada del chunker
        f.write("CONFIGURACIÓN DETALLADA DEL CHUNKER:\n")
        f.write("-" * 40 + "\n")
        
        # Según el tipo de chunker, mostrar información específica relevante
        if chunker_type == 'character':
            character_config = chunker_config.get('character', {})
            f.write(f"- Método: Chunking por caracteres\n")
            f.write(f"- Tamaño del chunk: {character_config.get('chunk_size', 1000)} caracteres\n")
            f.write(f"- Solapamiento: {character_config.get('chunk_overlap', 200)} caracteres\n")
            f.write(f"- Extracción de encabezados: {'Activada' if character_config.get('header_extraction_enabled', True) else 'Desactivada'}\n")
            f.write(f"- Nivel mínimo de encabezado: {character_config.get('min_header_length', 1)}\n")
            f.write(f"- Nivel máximo de encabezado: {character_config.get('max_header_length', 3)}\n")
        
        elif chunker_type == 'token':
            token_config = chunker_config.get('token', {})
            f.write(f"- Método: Chunking por tokens\n")
            f.write(f"- Tokenizador: {token_config.get('tokenizer', 'intfloat/multilingual-e5-small')}\n")
            f.write(f"- Máximo de tokens por chunk: {token_config.get('max_tokens', 512)}\n")
            f.write(f"- Solapamiento: {token_config.get('token_overlap', 100)} tokens\n")
        
        elif chunker_type == 'context':
            context_config = chunker_config.get('context', {})
            f.write(f"- Método: Chunking por contexto\n")
            f.write(f"- Uso de encabezados: {'Activado' if context_config.get('use_headers', True) else 'Desactivado'}\n")
            f.write(f"- Nivel máximo de encabezado: {context_config.get('max_header_level', 3)}\n")
            f.write(f"- Tamaño máximo de chunk: {context_config.get('max_chunk_size', 1500)} caracteres\n")
        
        # Escribir configuración general adicional
        f.write(f"- Método global configurado: {chunker_config.get('general', {}).get('config_global', 'context')}\n")
        
        # Añadir información sobre el embedding
        if chunks and 'embedding_dim' in chunks[0]:
            f.write(f"- Dimensión de embedding: {chunks[0]['embedding_dim']}\n")
        
        f.write("\n")
        
        # Escribir métricas
        f.write("MÉTRICAS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"- Total de chunks: {len(chunks)}\n")
        f.write(f"- Longitud promedio: {avg_length:.2f} caracteres\n")
        f.write(f"- Longitud mínima: {min_length} caracteres\n")
        f.write(f"- Longitud máxima: {max_length} caracteres\n")
        if chunks:
            f.write(f"- Dimensión de embedding: {chunks[0]['embedding_dim']}\n")
        f.write("\n")
        
        # Escribir detalles de cada chunk
        f.write("DETALLES DE CHUNKS:\n")
        f.write("=" * 80 + "\n\n")
        
        for i, chunk in enumerate(chunks):
            f.write(f"CHUNK {i+1}:\n")
            f.write("-" * 40 + "\n")
            f.write(f"- Header: {chunk['header']}\n")
            f.write(f"- Page: {chunk['page']}\n")
            f.write(f"- Longitud: {len(chunk['text'])} caracteres\n")
            f.write(f"- Embedding dim: {chunk['embedding_dim']}\n")
            
            # Escribir el texto completo del chunk (sin truncar)
            f.write(f"\nTEXTO COMPLETO:\n")
            f.write("-" * 40 + "\n")
            # Texto indentado para mejor legibilidad
            wrapped_text = textwrap.fill(
                chunk["text"], 
                width=80, 
                initial_indent="  ", 
                subsequent_indent="  "
            )
            f.write(f"{wrapped_text}\n\n")
            
            # Separador entre chunks
            f.write("=" * 80 + "\n\n")
        
    logger.info(f"Resultados guardados en {result_file}")

def main():
    """Función principal que ejecuta el test de chunkers."""
    parser = argparse.ArgumentParser(description="Test de chunkers para archivos Markdown")
    parser.add_argument("--dir", type=str, default="pruebas", 
                        help="Directorio donde buscar archivos Markdown (predeterminado: 'pruebas')")
    parser.add_argument("--file", type=str, 
                        help="Ruta específica a un archivo Markdown para procesar")
    parser.add_argument("--chunkers", type=str, default="character,token,context",
                        help="Lista separada por comas de los chunkers a utilizar")
    parser.add_argument("--model", type=str, 
                        help="Nombre del modelo de embedding a utilizar (opcional)")
    parser.add_argument("--results-dir", type=str, default="test/resultados_pruebas",
                        help="Directorio donde guardar los resultados de las pruebas")
    
    args = parser.parse_args()
    
    # Convertir rutas relativas a absolutas si es necesario
    base_dir = Path(__file__).parent.parent
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = base_dir / results_dir
    
    # Cargar el modelo de embedding
    embedding_model = load_embedding_model(args.model)
    
    # Determinar los chunkers a utilizar
    chunker_types = [c.strip() for c in args.chunkers.split(",")]
    
    # Determinar los archivos a procesar
    if args.file:
        # Procesar un archivo específico
        if not os.path.exists(args.file):
            logger.error(f"El archivo {args.file} no existe")
            return
        md_files = [Path(args.file)]
    else:
        # Buscar archivos en el directorio
        md_files = find_markdown_files(args.dir)
    
    if not md_files:
        logger.error("No se encontraron archivos Markdown para procesar")
        return
    
    # Procesar cada archivo con cada chunker
    for md_file in md_files:
        print(f"\nProcesando archivo: {md_file}")
        
        for chunker_type in chunker_types:
            print(f"\nUtilizando chunker: {chunker_type}")
            try:
                chunks = process_document(str(md_file), chunker_type, embedding_model)
                print_chunk_stats(chunks, chunker_type)
                
                # Guardar resultados en archivo
                save_results_to_file(results_dir, chunker_type, str(md_file), chunks)
                
            except Exception as e:
                logger.error(f"Error al procesar {md_file} con chunker {chunker_type}: {e}")

if __name__ == "__main__":
    main()
