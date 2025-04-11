"""
Lógica principal del sistema RAG.

Este módulo contiene las funciones para:
1. Procesar documentos (ingestar)
2. Procesar consultas (query)

Orquesta todos los componentes del sistema RAG para trabajar juntos.
"""
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

# Configurar logging
logger = logging.getLogger(__name__)

# Importación de lo necesario para colorama
from colorama import init, Fore, Style

# Inicializar colorama para que funcione en todas las plataformas
init(autoreset=True)

# Definir colores y estilos para mejorar la legibilidad
C_SUCCESS = Fore.GREEN + Style.BRIGHT
C_ERROR = Fore.RED + Style.BRIGHT
C_WARNING = Fore.YELLOW + Style.BRIGHT
C_INFO = Fore.WHITE
C_HIGHLIGHT = Fore.MAGENTA + Style.BRIGHT
C_VALUE = Fore.CYAN

def process_documents(file_path: str, session_name: Optional[str] = None) -> None:
    """
    Procesa documentos Markdown para la ingestión en el sistema RAG.
    
    Args:
        file_path: Ruta al directorio o archivo a procesar
        session_name: Nombre personalizado para la sesión (opcional)
    """
    import os
    from pathlib import Path
    import time
    import logging
    from typing import List, Dict, Any
    
    # Importaciones de módulos del sistema
    from config import config
    from modulos.chunks import ChunkerFactory
    from modulos.embeddings.embeddings_factory import EmbeddingFactory
    from modulos.databases.FactoryDatabase import DatabaseFactory
    from modulos.session_manager.session_manager import SessionManager
    from modulos.doc_processor.markdown_processor import MarkdownProcessor
    
    logger = logging.getLogger(__name__)
    
    # Medir tiempo de inicio
    start_time = time.time()
    
    # Obtener configuraciones
    chunks_config = config.get_chunks_config()
    embeddings_config = config.get_embedding_config()
    database_config = config.get_database_config()
    
    chunking_method = chunks_config.get("method", "character")
    embedding_model = embeddings_config.get("model", "modernbert")
    db_type = database_config.get("type", "sqlite")
    
    logger.info(f"Iniciando ingestión con chunking: {chunking_method}, embeddings: {embedding_model}, db: {db_type}")
    
    # Inicializar componentes
    embedding_manager = EmbeddingFactory.get_embedding_manager(embedding_model)
    # Obtener dimensiones del embedding
    embedding_dim = embedding_manager.embedding_dim
    
    # Inicializar chunker y asignarle el modelo de embeddings
    chunker = ChunkerFactory.get_chunker(chunking_method, embedding_manager)
    
    # Crear un nombre de base de datos único basado en timestamp
    db_name = f"ragdb_{int(time.time())}"
    if session_name:
        # Sanitizar el nombre personalizado para usarlo como nombre de archivo
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in session_name)
        db_name = f"{safe_name}_{int(time.time())}"
    
    # Inicializar la base de datos utilizando una interfaz más completa
    db = DatabaseFactory.get_database_instance(
        db_type=db_type,
        embedding_dim=embedding_dim,
        embedding_model=embedding_model,
        chunking_method=chunking_method,
        custom_name=db_name
    )
    
    # Recuperar la ruta de la base de datos del objeto db
    db_path = db.get_db_path()
    
    # Registrar la base de datos y sus metadatos en el SessionManager
    session_manager = SessionManager()
    
    # Preparar metadatos completos de la base de datos
    db_metadata = {
        "db_type": db_type,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "chunking_method": chunking_method,
        "created_at": time.time(),
        "last_used": time.time(),
        "custom_name": session_name,
        "db_path": db_path
    }
    
    # Registrar la base de datos con todos sus metadatos
    session_manager.register_database(db_name, db_metadata)
    
    logger.info(f"Base de datos creada: {db_name}")
    
    # Inicializar el procesador de Markdown
    markdown_processor = MarkdownProcessor()
    
    # Obtener la lista de archivos a procesar
    file_path_obj = Path(file_path)
    if file_path_obj.is_dir():
        # Si es un directorio, buscar todos los archivos markdown
        md_files = list(file_path_obj.glob("**/*.md"))
        logger.info(f"Se encontraron {len(md_files)} archivos Markdown para procesar")
    else:
        # Si es un archivo, procesarlo directamente
        md_files = [file_path_obj] if file_path_obj.suffix.lower() == '.md' else []
        if not md_files:
            logger.error(f"El archivo especificado no es un archivo Markdown válido: {file_path}")
            return
    
    # Procesar cada documento
    successful_docs = 0
    failed_docs = 0
    
    for doc_path in md_files:
        logger.info(f"Procesando documento: {doc_path}")
        doc_id = process_single_document(str(doc_path), markdown_processor, chunker, db)
        
        if doc_id:
            successful_docs += 1
            logger.info(f"Documento procesado correctamente, ID: {doc_id}")
        else:
            failed_docs += 1
            logger.error(f"Error al procesar documento: {doc_path}")
    
    # Mostrar resumen de procesamiento
    elapsed_time = time.time() - start_time
    logger.info(f"Procesamiento completo en {elapsed_time:.2f} segundos")
    logger.info(f"Documentos procesados correctamente: {successful_docs}")
    
    if failed_docs:
        logger.warning(f"Documentos con errores: {failed_docs}")
    
    # Optimizar la base de datos después de insertar todos los documentos
    logger.info("Optimizando base de datos...")
    db.optimize_database()
    logger.info("Proceso de ingestión completado")

def process_single_document(file_path: str, 
                           markdown_processor: Any, 
                           chunker: Any, 
                           db: Any) -> Optional[int]:
    """
    Procesa un único documento Markdown y lo inserta en la base de datos.
    
    Args:
        file_path: Ruta al archivo Markdown
        markdown_processor: Instancia del procesador de Markdown
        chunker: Instancia del chunker
        db: Instancia de la base de datos
        
    Returns:
        ID del documento insertado o None si falla
    """
    try:
        # Procesar el documento
        metadata, content = markdown_processor.process_document(file_path)
        
        # Generar chunks
        raw_chunks = chunker.process_content(content)
        
        # Preparar chunks con embeddings
        prepared_chunks = []
        for chunk in raw_chunks:
            # Obtener embedding combinando encabezado y texto
            embedding = chunker.model.get_document_embedding(chunk['header'], chunk['text'])
            
            # Crear chunk final con embedding
            prepared_chunk = {
                'text': chunk['text'],
                'header': chunk['header'],
                'page': chunk['page'],
                'embedding': embedding,
                'embedding_dim': chunker.model.get_dimensions()
            }
            
            prepared_chunks.append(prepared_chunk)
        
        # Insertar en la base de datos
        document_id = db.insert_document(metadata, prepared_chunks)
        
        return document_id
    
    except Exception as e:
        logger.error(f"{C_ERROR}Error al procesar documento {file_path}: {e}")
        return None

def process_query(query: str, n_chunks: int = 5, model: Optional[str] = None, 
                 session_id: Optional[str] = None, db_index: Optional[int] = None) -> str:
    """
    Procesa una consulta utilizando el sistema RAG.
    
    Args:
        query: Texto de la consulta
        n_chunks: Número de chunks a recuperar
        model: Modelo de IA a utilizar (opcional)
        session_id: ID de sesión específica a utilizar (opcional)
        db_index: Índice de la base de datos a utilizar (opcional)
        
    Returns:
        Respuesta generada
    """
    # Importaciones bajo demanda
    from config import config  
    from modulos.session_manager.session_manager import SessionManager
    from modulos.embeddings.embeddings_factory import EmbeddingFactory
    
    logger.info(f"{C_INFO}Recibida consulta: {C_HIGHLIGHT}'{query}'")
    logger.info(f"{C_INFO}Se recuperarán {C_VALUE}{n_chunks} {C_INFO}chunks más relevantes")
    
    # Usar session_manager para obtener la configuración correcta
    session_manager = SessionManager()
    
    try:
        # Obtener la base de datos y configuración
        if db_index is not None:
            # Si se especificó un índice, usamos ese índice
            db, session = session_manager.get_database_by_index(db_index)
            logger.info(f"{C_INFO}Usando base de datos con índice: {C_VALUE}{db_index}")
        elif session_id:
            # Si se especificó un ID de sesión, usamos esa sesión
            db, session = session_manager.get_session_database(session_id)
            logger.info(f"{C_INFO}Usando sesión especificada: {C_VALUE}{session_id}")
        else:
            # Si no se especificó nada, usamos la sesión más reciente
            db, session = session_manager.get_session_database()
            logger.info(f"{C_INFO}Usando sesión más reciente: {C_VALUE}{session['id']}")
        
        # Cargar el modelo de embeddings correcto para esta sesión
        embedding_model = session["embedding_model"]
        embedding_manager = EmbeddingFactory.get_embedding_manager(embedding_model)
        embedding_manager.load_model()
        
        # Ahora que tenemos la configuración correcta, podríamos implementar el proceso completo:
        # 1. Crear embedding para la consulta
        # 2. Buscar chunks relevantes
        # 3. Generar respuesta con un modelo de IA
        
        if model:
            logger.info(f"{C_INFO}Usando modelo específico: {C_VALUE}{model}")
        
        # Por ahora, solo devolvemos un mensaje informativo
        return (
            f"Esta funcionalidad no está implementada todavía.\n"
            f"Se procesaría la consulta '{query}' usando la sesión '{session['id']}'\n"
            f"- Modelo de embedding: {embedding_model}\n"
            f"- Método de chunking: {session['chunking_method']}\n"
            f"- Base de datos: {session['db_type']}\n"
            f"- Dimensión de embedding: {embedding_manager.embedding_dim}\n"
            f"- Chunks a recuperar: {n_chunks}\n"
            f"- Modelo de IA: {model or 'predeterminado'}"
        )
    
    except ValueError as e:
        return f"{C_ERROR}Error: {str(e)}"
    except Exception as e:
        logger.error(f"{C_ERROR}Error al procesar consulta: {e}")
        return f"{C_ERROR}Error inesperado al procesar la consulta: {str(e)}"

def verify_database_file(db_path: str) -> bool:
    """
    Verifica que el archivo físico de la base de datos existe.
    
    Args:
        db_path: Ruta al archivo de la base de datos
        
    Returns:
        bool: True si el archivo existe, False en caso contrario
    """
    if db_path == ":memory:":
        return True
    
    file_path = Path(db_path)
    exists = file_path.exists()
    
    if exists:
        size_kb = file_path.stat().st_size / 1024
        logger.info(f"{C_SUCCESS}✓ Base de datos verificada: {C_VALUE}{db_path} ({C_SUCCESS}{size_kb:.2f} KB{C_INFO})")
    else:
        logger.error(f"{C_ERROR}✗ Archivo de base de datos no encontrado: {C_VALUE}{db_path}")
        
    return exists
