"""
Lógica principal del sistema RAG.

Este módulo contiene las funciones para:
1. Procesar documentos (ingestar)
2. Procesar consultas (query)

Orquesta todos los componentes del sistema RAG para trabajar juntos.
"""
import logging
import time
import os # Añadido para concurrencia
import sys # Añadido para stderr en wrapper
from pathlib import Path
from typing import Optional, Any
import concurrent.futures # Añadido para concurrencia
import modulos.session_manager.session_manager # Importación añadida para resolver el warning

# Configurar logging
logger = logging.getLogger(__name__)

# Importación de lo necesario para colorama
from colorama import init, Fore, Style, Back
import gc  # Añadimos importación del garbage collector

# Añadir importación de DatabaseFactory
from modulos.databases.FactoryDatabase import DatabaseFactory

# Inicializar colorama para que funcione en todas las plataformas
init(autoreset=True)

# Definir colores y estilos para mejorar la legibilidad
C_TITLE = Back.BLUE + Fore.WHITE + Style.BRIGHT
C_SUBTITLE = Fore.BLUE + Style.BRIGHT
C_SUCCESS = Fore.GREEN + Style.BRIGHT
C_ERROR = Fore.RED + Style.BRIGHT
C_WARNING = Fore.YELLOW
C_HIGHLIGHT = Fore.MAGENTA + Style.BRIGHT
C_COMMAND = Fore.YELLOW
C_INFO = Style.RESET_ALL
C_VALUE = Fore.CYAN + Style.BRIGHT
C_PROMPT = Style.BRIGHT + Fore.GREEN
C_RESET = Style.RESET_ALL
C_SEPARATOR = Style.DIM + Fore.BLUE

# Reducir verbosidad de logging en algunos módulos
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('modulos.databases').setLevel(logging.ERROR)
logging.getLogger('modulos.databases.implementaciones').setLevel(logging.ERROR)
logging.getLogger('modulos.databases.implementaciones.sqlite').setLevel(logging.ERROR)
logging.getLogger('modulos.databases.implementaciones.duckdb').setLevel(logging.ERROR)
logging.getLogger('modulos.session_manager').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('filelock').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

def process_documents(file_path: str, session_name: Optional[str] = None) -> None:
    """
    Procesa documentos desde una ruta y los ingiere en la base de datos.
    
    Args:
        file_path: Ruta al archivo o directorio a procesar
        session_name: Nombre opcional para la sesión (usado para configurar la BD)
    """
    start_time = time.time()
    
    # Obtener el modelo de embeddings primero para conocer las dimensiones
    try:
        from modulos.embeddings.embeddings_factory import EmbeddingFactory
        embedding_manager = EmbeddingFactory().get_embedding_manager()
        embedding_dim = embedding_manager.get_dimensions()
        logger.info(f"Dimensiones de embeddings: {C_VALUE}{embedding_dim}{C_RESET}")
    except Exception as e:
        logger.error(f"{C_ERROR}Error al inicializar el modelo de embeddings: {e}")
        return
    
    # Crear o actualizar la base de datos
    db = None
    try:
        db_config = None
        if session_name:
            session_manager = modulos.session_manager.session_manager.SessionManager()
            try:
                db_config = session_manager.get_session_database_config(session_name)
                logger.info(f"Usando configuración de base de datos de la sesión '{session_name}'")
            except Exception as e:
                logger.error(f"{C_ERROR}Error al obtener configuración de BD para la sesión '{session_name}': {e}")
                return
        db = DatabaseFactory().get_database_instance(embedding_dim=embedding_dim)
    except Exception as e:
        logger.error(f"{C_ERROR}Error al crear o conectar con la base de datos: {e}")
        return
    
    # Obtener chunker y procesador Markdown
    chunker = None
    try:
        from modulos.chunks.ChunkerFactory import ChunkerFactory
        chunker = ChunkerFactory().get_chunker(embedding_model=embedding_manager)
    except Exception as e:
        logger.error(f"{C_ERROR}Error al inicializar el chunker: {e}")
        return
    
    try:
        from modulos.doc_processor.markdown_processor import MarkdownProcessor
        markdown_processor = MarkdownProcessor()
    except Exception as e:
        logger.error(f"{C_ERROR}Error al inicializar el procesador de Markdown: {e}")
        return
    
    # Buscar archivos Markdown para procesamiento
    md_files = []
    file_path_obj = Path(file_path).resolve()
    
    if file_path_obj.is_dir():
        # Si es un directorio, buscar todos los archivos .md recursivamente
        logger.info(f"Buscando archivos Markdown en {file_path_obj}...")
        md_files = list(file_path_obj.glob("**/*.md"))
    elif file_path_obj.suffix.lower() == '.md':
        # Si es un archivo .md específico
        md_files = [file_path_obj]
    else:
        if not md_files:
            logger.error(f"El archivo especificado no es un archivo Markdown válido: {file_path}")
            return
    
    # Obtener ResourceManager y ConcurrencyManager - Una sola instancia para todo el proceso
    from modulos.resource_management.resource_manager import ResourceManager
    resource_manager = ResourceManager()
    concurrency_manager = resource_manager.concurrency_manager
    
    # Actualizar ResourceManager sobre la operación en proceso
    resource_manager.metrics["operation_in_progress"] = "document_processing"
    resource_manager.metrics["total_documents"] = len(md_files)
    resource_manager.update_metrics() # Forzar actualización inicial de métricas
 
    # Procesar cada documento
    successful_docs = 0
    failed_docs = 0
    total_files = len(md_files)
    logger.info(f"Preparando para procesar {total_files} documentos...")

    # Lista para almacenar los archivos procesados exitosamente y sus metadatos
    processed_files = []
    file_metadata = {}  # Diccionario para guardar metadatos ricos para cada archivo

    # Usar concurrencia solo si hay manager y más de 1 archivo
    if concurrency_manager and total_files > 1:
        logger.info(f"Utilizando ConcurrencyManager para procesamiento paralelo de {total_files} documentos.")
        
        # Determinar chunksize óptimo para distribuir archivos entre workers
        iterable_length = len(md_files)
        chunksize = concurrency_manager.get_optimal_chunksize(
            task_type="default", 
            iterable_length=iterable_length
        )
        logger.info(f"Tamaño de lote (chunksize) calculado para procesamiento: {chunksize}")
        
        # Crear el iterador de argumentos
        args_iterator = ((str(doc_path), markdown_processor, chunker, db, resource_manager) 
                        for doc_path in md_files)
        
        # Usar el nuevo método map_tasks que selecciona automáticamente el mejor pool
        results = concurrency_manager.map_tasks(
            process_single_document_wrapper,
            args_iterator,
            chunksize=chunksize,
            task_type="default",  # El tipo de tarea determinará el pool óptimo
            # Sugerencia: usar ProcessPool para más de 10 archivos
            prefer_process=total_files >= 10
        )
        
        # Procesar los resultados de la ejecución paralela
        if results:
            processed_count = 0
            for doc_path, result in zip(md_files, results):
                processed_count += 1
                if isinstance(result, dict) and result.get("document_id") is not None:
                    successful_docs += 1
                    doc_id = result.get("document_id")
                    # Agregar a la lista de archivos procesados
                    processed_files.append(str(doc_path))
                    
                    # Guardar metadatos enriquecidos
                    file_path_str = str(doc_path)
                    file_metadata[file_path_str] = {
                        "size": doc_path.stat().st_size,
                        "chunks": result.get("chunk_count", 0),
                        "processing_time": result.get("processing_time", 0)
                    }
                    
                    logger.info(f"({processed_count}/{total_files}) Documento procesado OK: {doc_path} -> ID: {doc_id}")
                else:
                    failed_docs += 1
                    logger.error(f"({processed_count}/{total_files}) Error procesando: {doc_path}")
        else:
            logger.error(f"{C_ERROR}No se recibieron resultados del procesamiento paralelo.")
            failed_docs = total_files  # Marcar todos como fallidos si no hay resultados
    else:
        # Procesamiento secuencial si no hay ConcurrencyManager o es un solo archivo
        logger.info("Procesando secuencialmente.")
        for i, doc_path in enumerate(md_files):
            logger.info(f"({i+1}/{total_files}) Procesando: {doc_path}")
            result = process_single_document(str(doc_path), markdown_processor, chunker, db, resource_manager)
            if isinstance(result, dict) and result.get("document_id") is not None:
                successful_docs += 1
                processed_files.append(str(doc_path))
                
                # Guardar metadatos enriquecidos
                file_path_str = str(doc_path)
                file_metadata[file_path_str] = {
                    "size": doc_path.stat().st_size,
                    "chunks": result.get("chunk_count", 0),
                    "processing_time": result.get("processing_time", 0)
                }
            else:
                failed_docs += 1

    # Actualizar metadatos de sesión con la lista de archivos procesados y sus metadatos
    if session_name and processed_files:
        try:
            # Actualizar la lista de archivos usando el nuevo método mejorado
            session_manager.update_session_file_list(session_name, processed_files, file_metadata)
            logger.info(f"Metadatos de sesión {session_name} actualizados con {len(processed_files)} archivos procesados.")
        except Exception as e:
            logger.warning(f"No se pudo actualizar la lista de archivos en los metadatos de sesión: {e}")
    
    # Mostrar resumen de procesamiento
    elapsed_time = time.time() - start_time
    logger.info(f"Procesamiento completo en {C_VALUE}{elapsed_time:.2f}{C_RESET} segundos")
    logger.info(f"Documentos procesados correctamente: {C_VALUE}{successful_docs}{C_RESET}")
    
    if failed_docs:
        logger.warning(f"Documentos con errores: {C_VALUE}{failed_docs}{C_RESET}")
    
    # Optimizar la base de datos después de insertar todos los documentos
    # (Nota: La concurrencia se gestiona via ResourceManager/ConcurrencyManager)
    # (Nota: La optimización de memoria (batch_size) se gestiona via ResourceManager/MemoryManager)
    logger.info("Optimizando base de datos...")
    db.optimize_database()
    
    # Ahora que hemos terminado con todos los documentos, liberar recursos
    if resource_manager and resource_manager.memory_manager:
        logger.info("Realizando limpieza final de memoria...")
        resource_manager.memory_manager.cleanup(reason="processing_completed")
    
    logger.info("Proceso de ingestión completado")

# Wrapper para usar con map, ya que process_single_document toma múltiples args
def process_single_document_wrapper(args_tuple):
    # Desempaquetar argumentos
    file_path, markdown_processor, chunker, db, resource_manager = args_tuple
    # Llamar a la función original
    # Añadir manejo de excepciones aquí por si la función falla dentro del proceso worker
    try:
        return process_single_document(file_path, markdown_processor, chunker, db, resource_manager)
    except Exception as e:
        # Loggear el error desde el proceso worker puede ser complicado dependiendo de la config de logging.
        # Es más seguro retornar None y loggear el error en el bucle principal que recoge resultados.
        # logger.error(...) # Podría no funcionar como se espera
        print(f"[Worker Error] Error procesando {os.path.basename(file_path)}: {e}", file=sys.stderr) # Imprimir a stderr
        return None 

def process_single_document(file_path: str, 
                           markdown_processor: Any, 
                           chunker: Any, 
                           db: Any,
                           resource_manager: Any = None) -> Optional[int]:
    """
    Procesa un único documento Markdown y lo inserta en la base de datos.
    Utiliza un enfoque de streaming para procesar documentos grandes sin
    cargar todos los chunks en memoria simultáneamente.
    
    Args:
        file_path: Ruta al archivo Markdown
        markdown_processor: Instancia del procesador de Markdown
        chunker: Instancia del chunker
        db: Instancia de la base de datos
        resource_manager: Instancia de ResourceManager
        
    Returns:
        ID del documento insertado o None si falla
    """
    if resource_manager is None:
        # Solo si no recibimos el resource_manager como parámetro
        try:
            from modulos.resource_management.resource_manager import ResourceManager
            resource_manager = ResourceManager()
        except Exception as e:
            logger.error(f"Error al inicializar ResourceManager: {e}")
            resource_manager = None
    
    memory_manager = None
    if resource_manager:
        memory_manager = resource_manager.memory_manager

    try:
        # Obtener configuración de optimización de memoria
        from config import config
        chunks_config = config.get_chunks_config()
        memory_config = chunks_config.get("memory_optimization", {})
        
        # memory_optimization_enabled = memory_config.get("enabled", True) # Ya no se usa directamente aquí
        base_batch_size = memory_config.get("batch_size", 50) # Mantenemos la config como base
        
        # Obtener batch_size optimizado desde MemoryManager y verificar si están suspendidas las verificaciones
        verification_suspended = resource_manager.is_verification_suspended() if resource_manager else False
        if memory_manager:
            batch_size = memory_manager.optimize_batch_size(
                base_batch_size=base_batch_size, 
                min_batch_size=10,
                verification_suspended=verification_suspended
            )
            logger.info(f"Batch size optimizado por MemoryManager a: {C_VALUE}{batch_size}{C_RESET}")
        else:
            batch_size = base_batch_size
            logger.warning(f"MemoryManager no disponible, usando batch_size de configuración: {C_VALUE}{batch_size}{C_RESET}")

        # Procesar el documento para obtener metadatos y contenido
        metadata, content = markdown_processor.process_document(file_path)
        
        # Obtener el título del documento desde los metadatos
        doc_title = metadata.get('title', 'Documento')
        
        # Determinar el tamaño del documento para posible suspensión de verificaciones
        document_size_kb = len(content) / 1024  # Tamaño aproximado en KB
        
        # Estimar si es un documento grande basado en tamaño
        is_large_document = document_size_kb > 500  # Documentos > 500KB son considerados grandes
        
        # Solo considerar suspensión para documentos pequeños
        if resource_manager and not is_large_document:
            logger.info(f"Documento pequeño detectado ({document_size_kb:.1f}KB). Considerando suspensión de verificaciones.")
            resource_manager.auto_suspend_if_needed(document_size_kb=document_size_kb, duration_seconds=300)
        else:
            logger.info(f"Documento de tamaño considerable ({document_size_kb:.1f}KB). Manteniendo verificaciones activas.")
        
        # Método optimizado: streaming de chunks (Asumiendo que es el modo por defecto ahora)
        document_id = db.insert_document_metadata(metadata)
        
        if not document_id:
            logger.error(f"{C_ERROR}Error al insertar metadatos del documento {file_path}")
            return None
        
        # Iniciar una transacción para mejorar rendimiento
        db.begin_transaction()
        
        # Variables para estadísticas y control
        processed_chunks = 0
        start_time = time.time()
        
        # Determinar el tamaño óptimo de lote para procesamiento
        if resource_manager and resource_manager.concurrency_manager:
            # Usar el tamaño óptimo de lote para embeddings
            concurrency_manager = resource_manager.concurrency_manager
            batch_size = concurrency_manager.get_optimal_chunksize("embeddings")
            logger.info(f"Usando batch_size optimizado para embeddings: {batch_size}")
        else:
            # Si no hay ConcurrencyManager, usar el valor de la configuración
            batch_size = base_batch_size
            logger.info(f"Usando batch_size de configuración: {batch_size}")
        
        # Si es un documento grande, considerar incrementar el batch size
        if document_size_kb > 1000:  # Más de 1MB
            batch_size = max(batch_size, 32)  # Usar al menos 32 para documentos grandes
            logger.info(f"Documento grande detectado, usando batch_size aumentado: {batch_size}")
        
        try:
            # Lista temporal para acumular chunks durante el procesamiento por lotes
            chunks_buffer = []
            
            # Generar, procesar e insertar chunks uno por uno usando streaming
            for chunk in chunker.process_content_stream(content, doc_title=doc_title):
                logger.debug(f"Procesando chunk con header: {chunk.get('header', 'Sin header')} y longitud de texto: {len(chunk.get('text', ''))} caracteres")
                
                # Verificar si el modelo está inicializado
                if chunker.model is None:
                    logger.error(f"{C_ERROR}Error: El modelo de embeddings en chunker es None. No se puede generar embedding.")
                    db.rollback_transaction()
                    return None
                
                try:
                    # Agregar el chunk al buffer de procesamiento
                    chunks_buffer.append(chunk)
                    
                    # Cuando el buffer alcanza el tamaño de lote, procesar todos juntos
                    if len(chunks_buffer) >= batch_size:
                        _process_chunks_batch(chunks_buffer, chunker, document_id, db)
                        processed_chunks += len(chunks_buffer)
                        chunks_buffer = []  # Reiniciar el buffer
                        
                        # Mostrar progreso
                        elapsed = time.time() - start_time
                        rate = processed_chunks / elapsed if elapsed > 0 else 0
                        logger.info(f"Procesados {C_VALUE}{processed_chunks}{C_RESET} chunks ({C_VALUE}{rate:.2f}{C_RESET} chunks/seg)")
                        
                        # Forzar garbage collection periódicamente para evitar acumulación de memoria
                        if processed_chunks % (batch_size * 2) == 0:
                            gc.collect()
                            
                            # Actualizar batch_size dinámicamente según uso de recursos
                            if memory_manager:
                                verification_suspended = resource_manager.is_verification_suspended() if resource_manager else False
                                new_batch_size = memory_manager.optimize_batch_size(
                                    base_batch_size=batch_size, 
                                    min_batch_size=10,
                                    verification_suspended=verification_suspended
                                )
                                
                                if new_batch_size != batch_size:
                                    batch_size = new_batch_size
                                    logger.debug(f"Batch size re-optimizado a: {batch_size}")
                                
                except Exception as chunk_e:
                    logger.error(f"{C_ERROR}Error al procesar/insertar chunk individual: {chunk_e}", exc_info=True)
            
            # Procesar los chunks restantes en el buffer
            if chunks_buffer:
                _process_chunks_batch(chunks_buffer, chunker, document_id, db)
                processed_chunks += len(chunks_buffer)
                chunks_buffer = []
            
            # Confirmar transacción
            db.commit_transaction()
            
            # Ahora que hemos terminado de procesar todos los chunks, podemos liberar memoria
            if memory_manager:
                # Forzar una recolección de basura pero sin intentar liberar el modelo de embedding
                # ya que lo necesitaremos para futuros documentos
                gc.collect()
            
            # Evaluar al final del procesamiento si debemos reconsiderar la suspensión
            # basado en el número total de chunks procesados
            if resource_manager and is_large_document and processed_chunks < 10:
                # Si teníamos un documento teóricamente grande pero generó pocos chunks,
                # reconsiderar y posiblemente suspender verificaciones
                logger.info(f"Documento grande ({document_size_kb:.1f}KB) pero generó pocos chunks ({processed_chunks}). Reconsiderando suspensión.")
                resource_manager.auto_suspend_if_needed(
                    document_size_kb=document_size_kb, 
                    chunk_count=processed_chunks,
                    duration_seconds=300
                )
                
            logger.info(f"Completado: {file_path} -> {C_VALUE}{processed_chunks}{C_RESET} chunks procesados en {C_VALUE}{time.time() - start_time:.2f}{C_RESET}s")
            return {
                "document_id": document_id, 
                "chunk_count": processed_chunks,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            # Rollback en caso de error
            logger.error(f"{C_ERROR}Error procesando chunks de {file_path}: {e}", exc_info=True)
            try:
                db.rollback_transaction()
                logger.info("Transacción revertida.")
            except Exception as rollback_e:
                logger.error(f"Error adicional al intentar rollback: {rollback_e}")
            return None
    except Exception as e:
        logger.error(f"{C_ERROR}Error procesando documento {file_path}: {e}", exc_info=True)
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
    from modulos.clientes.FactoryClient import ClientFactory
    
    try:
        # Usar session_manager para obtener la configuración correcta
        session_manager = SessionManager()
    
        # Obtener la base de datos y configuración
        if db_index is not None:
            # Si se especificó un índice, usamos ese índice (pasando el session_id si está disponible)
            db, session = session_manager.get_database_by_index(db_index, session_id=session_id)
        elif session_id:
            # Si se especificó un ID de sesión pero no un índice, usamos la base de datos más reciente asociada a la sesión
            session_dbs = session_manager.get_session_databases(session_id)
            if session_dbs:
                # Usar la primera base de datos asociada a la sesión (la más reciente)
                db, session = session_manager.get_database_by_index(0, session_id=session_id)
            else:
                # Si no hay bases de datos asociadas, usar la más reciente en general
                db, session = session_manager.get_database_by_index(0)
        else:
            # Si no se especificó nada, usamos la base de datos más reciente
            db, session = session_manager.get_database_by_index(0)
        
        # Inicialización del modelo específico que usa esta base de datos
        embedding_model = session.get("embedding_model", "modernbert")
        embedding_manager = EmbeddingFactory.get_embedding_manager(embedding_model)
        
        try:
            # Cargar modelo con manejo de excepciones específico
            embedding_manager.load_model()
        except Exception as e:
            logger.error(f"Error al cargar el modelo de embeddings: {e}")
            return "No se pudo cargar el modelo de embeddings. Por favor, verifica la configuración o intenta con otra base de datos."
        
        # Generar embedding de la consulta
        query_embedding = embedding_manager.get_query_embedding(query)
        
        # Buscar los chunks más relevantes con manejo de errores mejorado
        try:
            search_results = db.vector_search(query_embedding, n_results=n_chunks)
        except Exception as e:
            logger.error(f"Error en la búsqueda vectorial: {e}")
            return "Hubo un problema al buscar información relevante. Es posible que la base de datos seleccionada no sea compatible con la consulta actual."
        
        if not search_results:
            return "No se encontró información relevante para responder a esta consulta."
        
        # Usar el modelo de IA especificado o el predeterminado
        if model is None:
            ai_config = config.get_ai_client_config()
            model = ai_config.get("type", "openai")
            
        # Obtener cliente de IA con manejo de errores mejorado
        try:
            ai_client = ClientFactory.get_client(client_type=model)
        except Exception as e:
            logger.error(f"Error al crear cliente IA: {e}")
            return "No se pudo inicializar el modelo de IA. Por favor, verifica tu configuración y API keys."
        
        # Preparar contexto para la respuesta
        context_chunks = []
        for chunk in search_results:
            context_chunks.append({
                "text": chunk["text"],
                "header": chunk.get("header", ""),
                "similarity": chunk.get("similarity", 0.0),
                "page": chunk.get("page", "N/A")  # Asegurar que se incluya el número de página
            })
        
        # Generar respuesta con manejo de errores mejorado
        try:
            # Siempre mostrar el contexto usado para la respuesta
            response = ai_client.generate_response(query, context=context_chunks, show_context=True)
            
            # Si estamos usando streaming, necesitamos asegurarnos de que se muestre el resultado completo
            if hasattr(response, '__iter__') and not isinstance(response, str):
                # Consumir el generador y devolver el texto completo
                try:
                    # Intentar iterar sobre la respuesta
                    chunks = list(response)
                    
                    # Unir los chunks en una respuesta completa
                    full_response = "".join(chunks)
                    
                    if full_response:
                        return full_response
                    else:
                        # Si la respuesta está vacía después de iterar, significa que el generador no produjo nada
                        # En este caso, vamos a obtener la respuesta original de Gemini directamente
                        if hasattr(ai_client, 'last_response_text') and ai_client.last_response_text:
                            return ai_client._format_response_with_context(ai_client.last_response_text, query)
                        else:
                            # Si todo lo demás falla, usar una respuesta genérica
                            return "La respuesta se generó correctamente pero no se pudo mostrar. Por favor, intenta nuevamente."
                except Exception as e:
                    # Si hay un error al iterar, intentar usar la respuesta como string
                    if response:
                        return str(response)
                    else:
                        return "Error al procesar la respuesta. Por favor, intenta nuevamente."
            
            # Si llegamos aquí, ya tenemos la respuesta como texto
            # Nos aseguramos de que la respuesta no sea None
            if response is None:
                return "No se recibió respuesta del modelo. Esto puede ser un problema de conexión o API."
                
            # Retornamos la respuesta (ya formateada si show_context=True)
            return response
            
        except Exception as e:
            logger.error(f"Error al generar respuesta: {e}")
            return "Hubo un problema al generar la respuesta. Verifica tu configuración y conexión a internet."
        
    except Exception as e:
        # Actualizar el timestamp de último uso de la base de datos si es posible
        try:
            if 'session' in locals() and 'db_name' in session:
                db_name = session.get("id", "")
                # Usar register_database para actualizar el timestamp
                session_manager.register_database(db_name, {
                    "last_used": time.time(),
                    **session  # Mantener el resto de metadatos
                })
        except Exception as update_err:
            # No mostramos este error al usuario final
            logger.debug(f"Error al actualizar metadatos: {update_err}")
    
        # Manejo de errores generales
        try:
            error_msg = f"{e.__class__.__name__}: {str(e)}"
        except Exception:
            error_msg = "Error desconocido durante el procesamiento"
            
        logger.error(f"Error al procesar consulta: {error_msg}")
        
        # Mensaje de error más amigable
        if "database" in error_msg.lower():
            return "Error al acceder a la base de datos. Por favor, verifica que la base de datos seleccionada existe y es accesible."
        elif "embedding" in error_msg.lower() or "model" in error_msg.lower():
            return "Error con el modelo de embeddings. Por favor, verifica la configuración o intenta con otro modelo."
        else:
            return f"Se produjo un error al procesar tu consulta: {error_msg}. Por favor, intenta nuevamente o selecciona otra base de datos."

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

def _process_chunks_batch(chunks_batch, chunker, document_id, db):
    """
    Procesa un lote de chunks, generando embeddings e insertándolos en la base de datos.
    
    Esta función se encarga de procesar múltiples chunks en una sola operación,
    lo que mejora el rendimiento al reducir la sobrecarga.
    
    Args:
        chunks_batch (List[Dict]): Lista de chunks a procesar
        chunker: Instancia del chunker que contiene el modelo de embeddings
        document_id: ID del documento al que pertenecen los chunks
        db: Instancia de la base de datos para inserción
    """
    if not chunks_batch:
        return
    
    # Extraer textos y encabezados para procesamiento por lotes
    headers = [chunk['header'] for chunk in chunks_batch]
    texts = [chunk['text'] for chunk in chunks_batch]
    
    try:
        # Verificar si el modelo tiene capacidad de procesamiento por lotes
        if hasattr(chunker.model, 'get_document_embeddings_batch'):
            # Procesar el lote completo en una sola llamada (muy eficiente)
            embeddings = chunker.model.get_document_embeddings_batch(headers, texts)
        else:
            # Fallback: procesar uno por uno si no hay método de lotes
            embeddings = []
            for header, text in zip(headers, texts):
                embedding = chunker.model.get_document_embedding(header, text)
                embeddings.append(embedding)
        
        # Preparar chunks para inserción
        prepared_chunks = []
        for i, chunk in enumerate(chunks_batch):
            if i < len(embeddings):  # Verificación de seguridad
                prepared_chunks.append({
                    'text': chunk['text'],
                    'header': chunk['header'],
                    'page': chunk.get('page', ''),
                    'embedding': embeddings[i],
                    'embedding_dim': chunker.model.get_dimensions()
                })
        
        # Intentar inserción por lotes si está disponible
        if hasattr(db, 'insert_chunks_batch'):
            # Insertar todo el lote en una sola operación
            chunk_ids = db.insert_chunks_batch(document_id, prepared_chunks)
            if chunk_ids and len(chunk_ids) > 0:
                logger.debug(f"Batch de {len(prepared_chunks)} chunks insertado correctamente")
            else:
                logger.warning(f"La inserción del batch de chunks no devolvió IDs")
        else:
            # Fallback: insertar uno por uno
            for prepared_chunk in prepared_chunks:
                chunk_id = db.insert_single_chunk(document_id, prepared_chunk)
                if not chunk_id:
                    logger.warning(f"La inserción del chunk retornó None o 0")
        
        # Liberar memoria explícitamente
        del embeddings
        del prepared_chunks
        
    except Exception as e:
        logger.error(f"{C_ERROR}Error al procesar lote de chunks: {e}", exc_info=True)
        raise
