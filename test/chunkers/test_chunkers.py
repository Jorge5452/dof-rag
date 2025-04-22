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
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

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
    
    # Crear nombre del archivo de resultados
    result_file = results_dir / f"{chunker_type}_{doc_name}_results.txt"
    
    # Obtener configuración del chunker
    chunker_config = get_chunker_config(chunker_type)
    
    # Guardar resultados en un archivo de texto
    with open(result_file, 'w', encoding='utf-8') as f:
        # Cabecera con información general
        f.write(f"RESULTADOS DE CHUNKING: {chunker_type.upper()}\n")
        f.write(f"Documento: {document_path}\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de chunks: {len(chunks)}\n\n")
        
        # Información de configuración
        f.write("CONFIGURACIÓN DEL CHUNKER:\n")
        for key, value in chunker_config.get(chunker_type, {}).items():
            f.write(f"  - {key}: {value}\n")
        f.write("\n")
        
        # Calcular estadísticas
        chunk_lengths = [len(chunk["text"]) for chunk in chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        min_length = min(chunk_lengths) if chunk_lengths else 0
        max_length = max(chunk_lengths) if chunk_lengths else 0
        
        # Escribir estadísticas generales
        f.write("ESTADÍSTICAS:\n")
        f.write(f"- Total de chunks: {len(chunks)}\n")
        f.write(f"- Longitud promedio: {avg_length:.2f} caracteres\n")
        f.write(f"- Longitud mínima: {min_length} caracteres\n")
        f.write(f"- Longitud máxima: {max_length} caracteres\n")
        f.write(f"- Dimensión de embedding: {chunks[0]['embedding_dim'] if chunks else 'N/A'}\n\n")
        
        # Escribir detalles de cada chunk
        f.write("DETALLES DE CHUNKS:\n")
        for i, chunk in enumerate(chunks):
            f.write(f"\n{'=' * 40}\n")
            f.write(f"CHUNK #{i+1}\n")
            f.write(f"- Header: {chunk['header']}\n")
            f.write(f"- Page: {chunk['page']}\n")
            f.write(f"- Embedding dimension: {chunk['embedding_dim']}\n")
            f.write(f"- Text ({len(chunk['text'])} caracteres):\n\n")
            
            # Formatear el texto con indentación para mejorar la legibilidad
            wrapped_text = textwrap.fill(chunk["text"], width=80, 
                                        initial_indent="  ", subsequent_indent="  ")
            f.write(f"{wrapped_text}\n")
    
    logger.info(f"Resultados guardados en {result_file}")
    return result_file

def run_multiple_chunkers(document_path: str, chunker_types: List[str], model_name: Optional[str] = None, 
                         results_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Ejecuta múltiples chunkers en un documento y compara los resultados.
    
    Args:
        document_path: Ruta al documento a procesar
        chunker_types: Lista de tipos de chunker a usar
        model_name: Nombre del modelo de embedding (opcional)
        results_dir: Directorio para guardar resultados (opcional)
        
    Returns:
        Diccionario con resultados comparativos
    """
    # Validar que el archivo existe
    if not os.path.exists(document_path):
        logger.error(f"El archivo {document_path} no existe")
        return {}
    
    # Cargar el modelo de embedding
    embedding_model = load_embedding_model(model_name)
    
    # Directorio de resultados
    if results_dir:
        results_path = Path(results_dir)
    else:
        # Crear directorio de resultados basado en la fecha y hora actual
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path(f"test_results_{timestamp}")
    
    # Resultados comparativos
    results = {
        "document": document_path,
        "model": embedding_model.get_submodule("_modules.0._model_id") 
                if hasattr(embedding_model, "get_submodule") else str(embedding_model),
        "timestamp": datetime.now().isoformat(),
        "chunkers": {}
    }
    
    # Procesar con cada chunker
    for chunker_type in chunker_types:
        logger.info(f"Procesando documento con chunker: {chunker_type}")
        
        try:
            # Procesar el documento
            start_time = time.time()
            chunks = process_document(document_path, chunker_type, embedding_model)
            processing_time = time.time() - start_time
            
            # Mostrar estadísticas
            print_chunk_stats(chunks, chunker_type)
            
            # Guardar resultados a archivo
            result_file = save_results_to_file(results_path, chunker_type, document_path, chunks)
            
            # Guardar estadísticas comparativas
            chunk_lengths = [len(chunk["text"]) for chunk in chunks]
            
            results["chunkers"][chunker_type] = {
                "processing_time": processing_time,
                "num_chunks": len(chunks),
                "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
                "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
                "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
                "result_file": str(result_file)
            }
            
        except Exception as e:
            logger.error(f"Error al procesar con chunker {chunker_type}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            results["chunkers"][chunker_type] = {
                "error": str(e)
            }
    
    # Guardar resultados comparativos en formato JSON
    summary_file = results_path / f"summary_{Path(document_path).stem}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Resumen comparativo guardado en {summary_file}")
    return results

def main():
    """Función principal que ejecuta las pruebas de chunkers."""
    parser = argparse.ArgumentParser(description="Prueba de chunkers con diferentes configuraciones")
    parser.add_argument("--dir", type=str, default="pruebas", 
                      help="Directorio donde buscar archivos Markdown")
    parser.add_argument("--file", type=str, 
                      help="Ruta específica a un archivo Markdown para procesar")
    parser.add_argument("--chunkers", type=str, default="character,token,context",
                      help="Lista separada por comas de los chunkers a utilizar")
    parser.add_argument("--model", type=str, 
                      help="Nombre del modelo de embedding a utilizar (opcional)")
    parser.add_argument("--results-dir", type=str, default="test/results/chunker_tests",
                      help="Directorio donde guardar los resultados de las pruebas")
    
    args = parser.parse_args()
    
    # Crear lista de chunkers a probar
    chunker_types = [c.strip() for c in args.chunkers.split(",")]
    
    # Si se especificó un archivo, procesar ese archivo
    if args.file:
        logger.info(f"Procesando archivo específico: {args.file}")
        run_multiple_chunkers(args.file, chunker_types, args.model, args.results_dir)
    else:
        # Si se especificó un directorio, buscar todos los archivos Markdown
        logger.info(f"Buscando archivos Markdown en: {args.dir}")
        md_files = find_markdown_files(args.dir)
        
        # Procesar cada archivo encontrado
        for file_path in md_files:
            logger.info(f"Procesando archivo: {file_path}")
            run_multiple_chunkers(str(file_path), chunker_types, args.model, args.results_dir)
    
    logger.info("Pruebas de chunking completadas")

if __name__ == "__main__":
    main() 