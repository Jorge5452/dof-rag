"""
Punto de entrada para el sistema RAG.

Este script maneja los argumentos de línea de comandos y ejecuta las funciones principales del sistema,
ya sea para ingestión de documentos o para consultas.

Uso:
    Ingestión: python run.py --ingest --files [directorio]
    Consulta: python run.py --query "¿Tu pregunta aquí?"
    Listar sesiones: python run.py --list-sessions
    Listar bases de datos: python run.py --list-dbs
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

# Importación global necesaria para evitar errores de referencia
from config import config

# Importar utilidades del sistema
from modulos.utils.formatting import (
    C_TITLE, C_SUBTITLE, C_SUCCESS, C_ERROR, C_WARNING, C_HIGHLIGHT, 
    C_COMMAND, C_PARAM, C_INFO, C_VALUE, C_PROMPT, C_RESET, C_SEPARATOR, Style,
    print_header, print_separator, print_status, print_formatted_response,
    print_command_help, print_useful_commands
)
from modulos.utils.logging_utils import setup_logging, silence_verbose_loggers

# Configurar logging
setup_logging(level=logging.INFO, log_file="/logs/rag_system.log")

# Silenciar módulos verbosos
silence_verbose_loggers()

# Obtener logger para este módulo
logger = logging.getLogger(__name__)

def main() -> int:
    """
    Función principal que procesa los argumentos y ejecuta la lógica correspondiente.
    
    Returns:
        int: Código de salida (0 para éxito, 1 para error)
    """
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(
        description="Sistema RAG para ingestión de documentos y consultas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Grupo para el modo de operación (mutuamente exclusivos)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--ingest", action="store_true", help="Modo de ingestión de documentos")
    mode_group.add_argument("--query", type=str, nargs='?', const='', help="Consulta para el sistema RAG. Sin argumento inicia modo interactivo.")
    mode_group.add_argument("--list-sessions", action="store_true", help="Lista las sesiones disponibles")
    mode_group.add_argument("--list-dbs", action="store_true", help="Lista las bases de datos disponibles")
    mode_group.add_argument("--optimize-db", type=int, help="Optimiza una base de datos específica por índice")
    mode_group.add_argument("--optimize-all", action="store_true", help="Optimiza todas las bases de datos")
    mode_group.add_argument("--db-stats", action="store_true", help="Muestra estadísticas de bases de datos")
    
    # Argumentos para el modo de ingestión
    parser.add_argument("--files", type=str, help="Directorio o archivo Markdown a procesar")
    parser.add_argument("--session-name", type=str, help="Nombre personalizado para la sesión")
    
    # Argumentos de procesamiento y análisis
    parser.add_argument("--export-chunks", action="store_true", help="Exporta chunks a archivos TXT en las mismas ubicaciones que los Markdown")
    
    # Argumentos para el modo de consulta
    processing_config = config.get_processing_config()
    default_chunks = processing_config.get("max_chunks_to_retrieve", 5)
    
    parser.add_argument("--chunks", type=int, default=default_chunks, 
                      help=f"Número de chunks a recuperar para la consulta (default: {default_chunks})")
    parser.add_argument("--model", type=str, help="Modelo de IA a utilizar para la consulta")
    parser.add_argument("--session", type=str, help="ID de sesión específica a utilizar")
    parser.add_argument("--db-index", type=int, help="Índice de la base de datos a utilizar (0 es la más reciente)")
    parser.add_argument("--show-dbs", action="store_true", help="Mostrar bases de datos disponibles antes de la consulta")
    
    # Argumentos generales
    parser.add_argument("--debug", action="store_true", help="Activar modo de depuración (logs verbose)")
    
    args = parser.parse_args()
    
    # Configurar nivel de logging
    if args.debug or config.get_general_config().get("debug", False):
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug(C_SUCCESS + "Modo de depuración activado")
    
    try:
        # Ejecutar el modo correspondiente
        if args.ingest:
            result = handle_ingest_mode(args)
            # Exportar chunks si se solicitó
            if args.export_chunks:
                handle_export_chunks_mode(args)
            return result
        elif args.query is not None:
            return handle_query_mode(args)
        elif args.list_sessions:
            return list_sessions()
        elif args.list_dbs:
            show_available_databases()
            return 0
        elif args.optimize_db is not None:
            optimize_database(args.optimize_db)
            return 0
        elif args.optimize_all:
            optimize_all_databases()
            return 0
        elif args.db_stats:
            show_database_statistics()
            return 0
        elif args.export_chunks:
            return handle_export_chunks_mode(args)
        
        return 0
    except Exception as e:
        logger.error(f"{C_ERROR}Error inesperado: {e}")
        return 1

def handle_ingest_mode(args: argparse.Namespace) -> int:
    """
    Maneja el modo de ingestión de documentos.
    
    Args:
        args: Argumentos de línea de comandos
        
    Returns:
        int: Código de salida (0 para éxito, 1 para error)
    """
    # Cargar módulos solo cuando se necesiten para ingestión
    from main import process_documents
    
    if not args.files:
        logger.error(f"{C_ERROR}El argumento --files es requerido para el modo --ingest")
        return 1
        
    # Verificar que el directorio o archivo existe
    if not os.path.exists(args.files):
        logger.error(f"{C_ERROR}El directorio o archivo {args.files} no existe")
        return 1
        
    # Procesar documentos
    logger.info(f"{C_HIGHLIGHT}Iniciando ingestión de documentos desde: {C_VALUE}{args.files}")
    process_documents(args.files, session_name=args.session_name)
    return 0

def handle_query_mode(args: argparse.Namespace) -> int:
    """
    Maneja el modo de consulta.
    
    Args:
        args: Argumentos de línea de comandos
        
    Returns:
        int: Código de salida (0 para éxito, 1 para error)
    """
    # Cargar módulos solo cuando se necesiten para consulta
    from main import process_query
    
    # Configurar nivel de logging basado en el modo de depuración
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # Reducir verbosidad de logging en modo normal
        logging.getLogger().setLevel(logging.WARNING)
        silence_verbose_loggers()
    
    # Verificar si estamos en modo interactivo (consulta vacía)
    interactive_mode = args.query == ''
    
    # En modo consulta, verificar si se solicita mostrar las bases de datos
    if args.show_dbs or args.db_index is None:
        # Mostrar un resumen simple de las bases de datos
        print("\n" + C_TITLE + " BASES DE DATOS DISPONIBLES " + C_RESET)
        print_separator()
        
        sorted_dbs = show_available_databases(show_output=False)
        
        # Mostrar lista simplificada de bases de datos (solo las más recientes)
        for i, (name, db) in enumerate(sorted_dbs[:5]):  # Limitar a 5 bases de datos
            created_date = datetime.fromtimestamp(db.get('created_at', 0)).strftime('%Y-%m-%d')
            model = db.get('embedding_model', 'desconocido')
            
            if i == 0:  # Resaltar la más reciente
                print(f"{C_SUCCESS}[{i}] {C_HIGHLIGHT}{name}{C_RESET} - {created_date} - Modelo: {C_VALUE}{model}{C_RESET}")
            else:
                print(f"[{i}] {C_HIGHLIGHT}{name}{C_RESET} - {created_date} - Modelo: {model}")
        
        print_separator()
        
        # Si solo se pidió mostrar las bases de datos y no se proporcionó índice, preguntar
        if args.db_index is None:
            try:
                db_index = input(f"\n{C_PROMPT}Seleccione índice de base de datos (Enter para usar la más reciente): ")
                if db_index.strip():
                    args.db_index = int(db_index)
            except ValueError:
                print_status("warning", "Entrada inválida. Usando la base de datos más reciente.")
    
    # Mostrar un mensaje de inicio
    if interactive_mode:
        # Modo interactivo
        run_interactive_mode(args.chunks, args.model, args.session, args.db_index)
    else:
        # Modo de consulta única - Interfaz simplificada
        print("\n" + C_SUBTITLE + " CONSULTA: " + C_VALUE + f"{args.query}" + C_RESET)
        
        # Timer para medir tiempo de respuesta
        start_time = time.time()
        
        # Reducir temporalmente el nivel de log durante la consulta si no estamos en modo debug
        original_log_level = logging.getLogger().level
        if not args.debug:
            logging.getLogger().setLevel(logging.WARNING)
            
        try:
            # Procesar la consulta
            response = process_query(
                args.query, 
                n_chunks=args.chunks, 
                model=args.model, 
                session_id=args.session,
                db_index=args.db_index
            )
        except Exception as e:
            # En caso de error, mostrar mensaje claro
            response = f"Error al procesar consulta: {str(e)}"
        finally:
            # Restaurar nivel de log original
            logging.getLogger().setLevel(original_log_level)
        
        # Mostrar la respuesta con mejor formato
        print("\n" + C_TITLE + " RESPUESTA " + C_RESET)
        print_separator()
        
        # Extraer solo la sección de respuesta si contiene separadores de formato
        if "=======================  RESPUESTA  =======================" in response:
            parts = response.split("=======================  RESPUESTA  =======================")
            if len(parts) > 1:
                # Extraer la parte de respuesta (sin encabezado)
                response_text = parts[1].split("=======================  CONTEXTO  =======================")[0].strip()
                context_text = response.split("=======================  CONTEXTO  =======================")
                
                # Imprimir solo la respuesta primero
                print(response_text)
                
                # Mostrar tiempo de respuesta después de la respuesta principal
                elapsed_time = time.time() - start_time
                print_separator()
                print_status("info", f"Tiempo de respuesta: {elapsed_time:.2f} segundos")
                
                # Imprimir contexto si existe - Ahora siempre mostramos el contexto
                if len(context_text) > 1:
                    print("\n" + C_TITLE + " CONTEXTO UTILIZADO " + C_RESET)
                    print_separator()
                    print(context_text[1].strip())
            else:
                print(response)
                
                # Mostrar tiempo de respuesta
                elapsed_time = time.time() - start_time
                print_separator()
                print_status("info", f"Tiempo de respuesta: {elapsed_time:.2f} segundos")
        else:
            print(response)
            
            # Mostrar tiempo de respuesta
            elapsed_time = time.time() - start_time
            print_separator()
            print_status("info", f"Tiempo de respuesta: {elapsed_time:.2f} segundos")
    
    return 0

def list_sessions() -> int:
    """
    Lista las sesiones disponibles.
    
    Returns:
        int: Código de salida (0 para éxito)
    """
    # Importar solo cuando sea necesario
    from modulos.session_manager.session_manager import SessionManager
    
    session_manager = SessionManager()
    sessions = session_manager.list_sessions()
    
    print_header(f"SESIONES DISPONIBLES ({len(sessions)})")
    
    for i, session in enumerate(sessions, 1):
        print(f"{C_HIGHLIGHT}{i}. ID: {session['id']} - Modelo: {session['embedding_model']} - Chunking: {session['chunking_method']} - DB: {session['db_type']}")
        if session.get('custom_name'):
            print(f"   Nombre: {session['custom_name']}")
    
    print(Style.BRIGHT + "=" * 80 + "\n")
    return 0

def run_interactive_mode(n_chunks: int = 5, model: Optional[str] = None, 
                        session_id: Optional[str] = None, db_index: Optional[int] = None) -> None:
    """
    Ejecuta el sistema RAG en modo interactivo, permitiendo consultas consecutivas.
    
    Args:
        n_chunks: Número de chunks a recuperar para cada consulta
        model: Modelo de IA a utilizar (opcional)
        session_id: ID de sesión específica a utilizar (opcional)
        db_index: Índice de la base de datos a utilizar (opcional)
    """
    # Importar solo cuando sea necesario para el modo interactivo
    from main import process_query
    from colorama import Style  # Asegurar que Style está disponible
    
    # Configurar nivel de logging para el modo interactivo
    logging.getLogger().setLevel(logging.WARNING)
    silence_verbose_loggers()
    
    print("\n" + C_TITLE + " MODO INTERACTIVO " + C_RESET)
    print_separator()
    
    # Mostrar instrucciones simplificadas
    print_status("info", "Escribe tus preguntas y presiona Enter para obtener respuestas.")
    print_status("info", f"Para salir: {C_COMMAND}salir{C_RESET}, {C_COMMAND}exit{C_RESET} o {C_COMMAND}q{C_RESET}")
    print_status("info", f"Para ver bases de datos: {C_COMMAND}dbs{C_RESET}")
    print_status("info", f"Para cambiar base de datos: {C_COMMAND}cambiar <n>{C_RESET}")
    print_status("info", f"Para ver ayuda: {C_COMMAND}ayuda{C_RESET} o {C_COMMAND}help{C_RESET}")
    print_separator()
    
    # Variables para mantener estado
    current_db_index = db_index
    current_session_id = session_id
    current_model = model
    history = []
    
    # Bucle principal del modo interactivo
    while True:
        try:
            # Obtener la consulta del usuario
            query = input(f"\n{C_PROMPT}> ")
            
            # Verificar comandos especiales
            if query.lower() in ["salir", "exit", "q", "quit"]:
                print_status("success", "Sesión finalizada.")
                break
                
            elif query.lower() == "dbs":
                # Mostrar bases de datos disponibles
                sorted_dbs = show_available_databases()
                continue
                
            elif query.lower().startswith("cambiar "):
                # Cambiar la base de datos activa
                try:
                    new_index = int(query.split()[1])
                    current_db_index = new_index
                    print_status("success", f"Base de datos cambiada a índice {C_VALUE}{new_index}")
                except (IndexError, ValueError):
                    print_status("error", "Formato inválido. Uso: cambiar <número>")
                continue
                
            elif query.lower() in ["ayuda", "help", "?"]:
                # Mostrar comandos de ayuda
                print("\n" + C_TITLE + " COMANDOS DISPONIBLES " + C_RESET)
                print_separator()
                print(f"{C_INFO}• {C_COMMAND}salir{C_RESET}, {C_COMMAND}exit{C_RESET}, {C_COMMAND}q{C_RESET} - Salir del modo interactivo")
                print(f"{C_INFO}• {C_COMMAND}dbs{C_RESET} - Mostrar bases de datos disponibles")
                print(f"{C_INFO}• {C_COMMAND}cambiar <n>{C_RESET} - Cambiar a la base de datos con índice <n>")
                print(f"{C_INFO}• {C_COMMAND}ayuda{C_RESET}, {C_COMMAND}help{C_RESET}, {C_COMMAND}?{C_RESET} - Mostrar esta ayuda")
                print_separator()
                continue
                
            elif not query.strip():
                # Ignorar consultas vacías
                continue
            
            # Procesar consulta normal - Reducir verbosidad
            # No mostramos mensaje de procesamiento para una experiencia más limpia
            
            # Medir tiempo de respuesta
            start_time = time.time()
            
            # Procesar la consulta con manejo de errores
            try:
                response = process_query(
                    query, 
                    n_chunks=n_chunks, 
                    model=current_model,
                    session_id=current_session_id,
                    db_index=current_db_index
                )
                
                # Agregar a historial
                history.append((query, response))
                
                # Mostrar respuesta con formato mejorado
                # Extraer solo la sección de respuesta si contiene separadores de formato
                if "=======================  RESPUESTA  =======================" in response:
                    parts = response.split("=======================  RESPUESTA  =======================")
                    if len(parts) > 1:
                        # Extraer la parte de respuesta (sin encabezado)
                        response_text = parts[1].split("=======================  CONTEXTO  =======================")[0].strip()
                        context_text = response.split("=======================  CONTEXTO  =======================")
                        
                        # Imprimir solo la respuesta
                        print("\n" + C_TITLE + " RESPUESTA " + C_RESET)
                        print_separator()
                        print(response_text)
                        
                        # Mostrar tiempo de respuesta de forma discreta
                        elapsed_time = time.time() - start_time
                        print_separator()
                        print_status("info", f"Tiempo: {elapsed_time:.2f} segundos")
                        
                        # Imprimir contexto si existe
                        if len(context_text) > 1:
                            print("\n" + C_TITLE + " CONTEXTO UTILIZADO " + C_RESET)
                            print_separator()
                            print(context_text[1].strip())
                    else:
                        # Si no podemos separar la respuesta, mostrar todo
                        print("\n" + C_TITLE + " RESPUESTA " + C_RESET)
                        print_separator()
                        print(response)
                        
                        # Mostrar tiempo de respuesta
                        elapsed_time = time.time() - start_time
                        print_separator()
                        print_status("info", f"Tiempo: {elapsed_time:.2f} segundos")
                else:
                    # Si no está formateada, mostrar la respuesta completa
                    print("\n" + C_TITLE + " RESPUESTA " + C_RESET)
                    print_separator()
                    print(response)
                    
                    # Mostrar tiempo de respuesta
                    elapsed_time = time.time() - start_time
                    print_separator()
                    print_status("info", f"Tiempo: {elapsed_time:.2f} segundos")
                
            except Exception as e:
                print_status("error", f"Error al procesar consulta: {str(e)}")
            
        except KeyboardInterrupt:
            print_status("info", "\nSesión interrumpida por el usuario.")
            break
            
        except Exception as e:
            print_status("error", f"Error: {str(e)}")
            
    print_status("info", "¡Gracias por usar el sistema RAG!")
    print()

def show_available_databases(show_output: bool = True) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Lista las bases de datos disponibles.
    
    Args:
        show_output: Si se debe mostrar la salida en pantalla
        
    Returns:
        Lista ordenada de bases de datos (nombre, metadatos)
    """
    # Importar el gestor de sesiones
    from modulos.session_manager.session_manager import SessionManager
    from colorama import Style  # Asegurar que Style está disponible
    
    # Usar el gestor de sesiones para listar bases de datos
    session_manager = SessionManager()
    databases = session_manager.list_available_databases()
    
    # Convertir el diccionario a una lista ordenada por último uso (más reciente primero)
    sorted_dbs = []
    for name, metadata in databases.items():
        sorted_dbs.append((name, metadata))
    
    sorted_dbs.sort(key=lambda x: x[1].get('last_used', 0), reverse=True)
    
    # Si se solicita mostrar en pantalla
    if show_output:
        # Mostrar título
        print("\n" + C_TITLE + " BASES DE DATOS DISPONIBLES " + C_RESET)
        print_separator()
        
        if not sorted_dbs:
            print_status("warning", "No hay bases de datos disponibles")
            return sorted_dbs
            
        # Mostrar información resumida de cada base de datos
        for i, (name, db) in enumerate(sorted_dbs):
            created_date = datetime.fromtimestamp(db.get('created_at', 0)).strftime('%Y-%m-%d')
            model = db.get('embedding_model', 'desconocido')
            chunks = db.get('chunking_method', 'desconocido')
            db_type = db.get('db_type', 'desconocido').upper()  # Tipo de base de datos
            
            # Formato más compacto y visual, incluyendo el tipo de base de datos
            if i == 0:  # Resaltar la más reciente
                print(f"{C_SUCCESS}[{i}] {C_HIGHLIGHT}{name}{C_RESET} - {created_date} - DB: {C_VALUE}{db_type}{C_RESET} - Modelo: {C_VALUE}{model}{C_RESET} - Chunking: {chunks}")
            else:
                print(f"[{i}] {C_HIGHLIGHT}{name}{C_RESET} - {created_date} - DB: {db_type} - Modelo: {model} - Chunking: {chunks}")
        
        print_separator()
        
        # Mostrar comandos útiles
        print_useful_commands()
    
    return sorted_dbs

def optimize_database(db_index: int) -> None:
    """
    Optimiza una base de datos específica.
    
    Args:
        db_index: Índice de la base de datos a optimizar
    """
    # Importar solo cuando sea necesario
    from modulos.session_manager.session_manager import SessionManager
    
    try:
        # Obtener lista de bases de datos
        sorted_dbs = show_available_databases(show_output=False)
        
        if not sorted_dbs or db_index >= len(sorted_dbs):
            print(f"{C_ERROR}Índice de base de datos inválido: {db_index}")
            return
        
        # Obtener información de la base de datos
        name, metadata = sorted_dbs[db_index]
        db_path = metadata.get('db_path')
        
        if not db_path or not os.path.exists(db_path):
            print(f"{C_ERROR}No se pudo localizar el archivo de base de datos en: {db_path}")
            return
        
        print(f"\n{C_INFO}Optimizando base de datos: {C_VALUE}{name}")
        
        # Obtener instancia de base de datos
        session_manager = SessionManager()
        # Pasamos None como session_id ya que estamos optimizando una base de datos específica por índice
        db, metadata = session_manager.get_database_by_index(db_index, session_id=None)
        
        # Medir tiempo de optimización
        start_time = time.time()
        
        # Ejecutar optimización
        success = db.optimize_database()
        
        # Calcular tiempo
        elapsed_time = time.time() - start_time
        
        if success:
            print(f"{C_SUCCESS}✓ Optimización completada exitosamente en {elapsed_time:.2f} segundos")
        else:
            print(f"{C_ERROR}✗ Error durante la optimización")
        
    except Exception as e:
        print(f"{C_ERROR}Error: {str(e)}")

def optimize_all_databases() -> None:
    """
    Optimiza todas las bases de datos disponibles.
    """
    # Importar solo cuando sea necesario
    from modulos.databases.FactoryDatabase import DatabaseFactory
    
    print_header("OPTIMIZACIÓN DE TODAS LAS BASES DE DATOS")
    
    # Obtener lista de bases de datos
    sorted_dbs = show_available_databases(show_output=False)
    
    if not sorted_dbs:
        print(f"{C_WARNING}No hay bases de datos disponibles para optimizar.")
        return
    
    # Extraer nombres de bases de datos para optimización
    db_names = [name for name, _ in sorted_dbs]
    
    print(f"{C_INFO}Se optimizarán {len(db_names)} bases de datos...")
    
    # Iniciar optimización
    start_time = time.time()
    results = DatabaseFactory.optimize_all_databases(db_names)  # Pasar los nombres de bases de datos
    elapsed_time = time.time() - start_time
    
    # Mostrar resultados
    success_count = sum(1 for success in results.values() if success)
    print(f"\n{C_SUBTITLE}RESULTADOS DE OPTIMIZACIÓN:")
    print(f"{C_SUCCESS}✓ Bases de datos optimizadas exitosamente: {success_count}")
    print(f"{C_ERROR}✗ Bases de datos con errores: {len(results) - success_count}")
    print(f"{C_INFO}Tiempo total: {elapsed_time:.2f} segundos")

def show_database_statistics() -> None:
    """
    Muestra estadísticas detalladas de todas las bases de datos.
    """
    # Importar solo cuando sea necesario
    from modulos.databases.FactoryDatabase import DatabaseFactory
    
    print_header("ESTADÍSTICAS DE BASES DE DATOS")
    
    # Obtener estadísticas
    stats = DatabaseFactory.get_db_statistics()
    
    # Mostrar estadísticas globales
    print(f"{C_SUBTITLE}ESTADÍSTICAS GLOBALES:")
    print(f"{C_INFO}• Total de bases de datos: {C_VALUE}{stats['total_databases']}")
    print(f"{C_INFO}• Total de documentos: {C_VALUE}{stats['total_documents']}")
    print(f"{C_INFO}• Total de chunks: {C_VALUE}{stats['total_chunks']}")
    
    # Mostrar estadísticas por base de datos
    if stats["databases"]:
        print(f"\n{C_SUBTITLE}ESTADÍSTICAS POR BASE DE DATOS:")
        
        for name, db_stats in stats["databases"].items():
            if "error" in db_stats:
                print(f"\n{C_HIGHLIGHT}{name}: {C_ERROR}Error al obtener estadísticas: {db_stats['error']}")
                continue
                
            print(f"\n{C_HIGHLIGHT}{name}:")
            print(f"{C_INFO}• Documentos: {C_VALUE}{db_stats.get('total_documents', 'N/A')}")
            print(f"{C_INFO}• Chunks: {C_VALUE}{db_stats.get('total_chunks', 'N/A')}")
            
            # Información sobre documento más reciente
            if "latest_document" in db_stats:
                doc = db_stats["latest_document"]
                print(f"{C_INFO}• Documento más reciente: {C_VALUE}{doc.get('title', 'Sin título')} (ID: {doc.get('id', 'N/A')})")
            
            # Tamaño de la base de datos
            if "db_size_mb" in db_stats:
                print(f"{C_INFO}• Tamaño: {C_VALUE}{db_stats['db_size_mb']:.2f} MB")
            
            # Fecha de creación
            if "db_created" in db_stats and db_stats["db_created"] != "unknown":
                try:
                    created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(db_stats["db_created"]))
                    print(f"{C_INFO}• Creada: {C_VALUE}{created_time}")
                except (TypeError, ValueError):
                    print(f"{C_INFO}• Creada: {C_VALUE}{db_stats['db_created']}")
    
    print(Style.BRIGHT + "=" * 80)

def query_database(db_index: int, query: str, session_id: str = None) -> None:
    """
    Ejecuta una consulta en una base de datos específica.
    
    Args:
        db_index: Índice de la base de datos a consultar
        query: Consulta a realizar
        session_id: ID de sesión (opcional)
    """
    from modulos.rag.main import process_query
    
    try:
        # Si tenemos un ID de sesión, lo usamos directamente
        if session_id:
            print(f"{C_INFO}Consultando en la sesión: {C_VALUE}{session_id}")
        
        # Procesar la consulta
        response = process_query(
            query=query, 
            db_index=db_index,
            session_id=session_id,
            stream=True
        )
        
        # La respuesta ya se ha mostrado al usuario a través del streaming
        
    except Exception as e:
        print(f"{C_ERROR}Error al consultar: {str(e)}")

def handle_export_chunks_mode(args: argparse.Namespace) -> int:
    """
    Maneja el modo de exportación de chunks a archivos TXT.
    
    Args:
        args: Argumentos de línea de comandos
        
    Returns:
        int: Código de salida (0 para éxito, 1 para error)
    """
    if not args.files:
        logger.error(f"{C_ERROR}El argumento --files es requerido para el modo --export-chunks")
        return 1
        
    # Verificar que el directorio o archivo existe
    if not os.path.exists(args.files):
        logger.error(f"{C_ERROR}El directorio o archivo {args.files} no existe")
        return 1
    
    try:
        # Importar el módulo de exportación
        from modulos.view_chunks.chunk_exporter import export_chunks_for_files
        
        # Obtener la base de datos más reciente o la especificada por índice
        from modulos.session_manager.session_manager import SessionManager
        session_manager = SessionManager()
        
        if args.db_index is not None:
            db, session = session_manager.get_database_by_index(args.db_index, session_id=args.session)
        else:
            # Usar la base de datos más reciente
            db, session = session_manager.get_database_by_index(0, session_id=args.session)
        
        if not db:
            logger.error(f"{C_ERROR}No se pudo obtener una conexión a la base de datos")
            return 1
        
        print(f"\n{C_TITLE} EXPORTANDO CHUNKS {C_RESET}")
        print_separator()
        
        # Mostrar información sobre la base de datos que se está utilizando
        print(f"{C_INFO}Base de datos: {C_VALUE}{session.get('id', 'unknown')}")
        print(f"{C_INFO}Modelo: {C_VALUE}{session.get('embedding_model', 'unknown')}")
        print(f"{C_INFO}Método de chunking: {C_VALUE}{session.get('chunking_method', 'unknown')}")
        print_separator()
        
        # Exportar chunks
        start_time = time.time()
        results = export_chunks_for_files(args.files, db)
        elapsed_time = time.time() - start_time
        
        # Mostrar resultados
        successful = sum(1 for result in results.values() if result)
        failed = sum(1 for result in results.values() if not result)
        
        print_separator()
        print(f"{C_SUCCESS}Exportación completada en {elapsed_time:.2f} segundos")
        print(f"{C_SUCCESS}Archivos procesados correctamente: {successful}")
        if failed > 0:
            print(f"{C_ERROR}Archivos con errores: {failed}")
        print_separator()
        
        # Liberar recursos
        db.close()
        import gc
        gc.collect()
        
        return 0
        
    except Exception as e:
        logger.error(f"{C_ERROR}Error al exportar chunks: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
