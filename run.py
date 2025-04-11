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

# Importar colorama para salida con colores
from colorama import init, Fore, Style

# Inicializar colorama para que funcione en todas las plataformas
init(autoreset=True)

# Configuración de colores para el formato de salida
C_TITLE = Fore.CYAN + Style.BRIGHT
C_SUBTITLE = Fore.BLUE + Style.BRIGHT
C_SUCCESS = Fore.GREEN + Style.BRIGHT
C_ERROR = Fore.RED + Style.BRIGHT
C_WARNING = Fore.YELLOW + Style.BRIGHT
C_HIGHLIGHT = Fore.MAGENTA + Style.BRIGHT
C_COMMAND = Fore.YELLOW
C_PARAM = Fore.GREEN
C_INFO = Fore.WHITE
C_VALUE = Fore.CYAN
C_PROMPT = Style.BRIGHT + Fore.GREEN
C_RESET = Style.RESET_ALL

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main() -> int:
    """
    Función principal que procesa los argumentos y ejecuta la lógica correspondiente.
    
    Returns:
        int: Código de salida (0 para éxito, 1 para error)
    """
    # Importaciones aquí para evitar cargarlas cuando no sean necesarias
    from config import config
    
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
            return handle_ingest_mode(args)
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
    logger.info(f"{C_SUCCESS}Iniciando ingestión de documentos desde: {args.files}")
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
    
    # Verificar si estamos en modo interactivo (consulta vacía)
    interactive_mode = args.query == ''
    
    # En modo consulta, verificar si se solicita mostrar las bases de datos
    if args.show_dbs:
        sorted_dbs = show_available_databases()
        # Si solo se pidió mostrar las bases de datos y no se proporcionó índice, preguntar
        if args.db_index is None:
            try:
                db_index = input(f"\n{C_PROMPT}Seleccione el índice de la base de datos a utilizar (Enter para usar la más reciente): ")
                if db_index.strip():
                    args.db_index = int(db_index)
            except ValueError:
                print(f"{C_WARNING}Entrada inválida. Usando la base de datos más reciente.")
    
    if interactive_mode:
        # Modo interactivo
        run_interactive_mode(args.chunks, args.model, args.session, args.db_index)
    else:
        # Modo de consulta única
        logger.info(f"{C_SUCCESS}Procesando consulta: {args.query}")
        response = process_query(
            args.query, 
            n_chunks=args.chunks, 
            model=args.model, 
            session_id=args.session,
            db_index=args.db_index
        )
        print_formatted_response("RESPUESTA", response)
    
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
    
    print_header("MODO INTERACTIVO DEL SISTEMA RAG")
    
    print_command_help([
        f"{C_INFO}• Escribe tus preguntas y presiona Enter para obtener respuestas.",
        f"{C_INFO}• Escribe {C_COMMAND}'salir'{C_INFO}, {C_COMMAND}'exit'{C_INFO} o {C_COMMAND}'q'{C_INFO} para terminar la sesión.",
        f"{C_INFO}• Escribe {C_COMMAND}'dbs'{C_INFO} para mostrar las bases de datos disponibles.",
        f"{C_INFO}• Escribe {C_COMMAND}'cambiar <n>'{C_INFO} para cambiar a la base de datos con índice <n>."
    ])
    
    # Variables para mantener estado
    current_db_index = db_index
    current_session_id = session_id
    current_model = model
    
    # Bucle principal del modo interactivo
    while True:
        try:
            # Obtener la consulta del usuario
            query = input(f"\n{C_PROMPT}> ")
            
            # Verificar comandos especiales
            query = query.strip()
            if not query:
                continue
                
            # Comandos para salir
            if query.lower() in ['salir', 'exit', 'q', 'quit']:
                print(f"{C_SUCCESS}¡Hasta luego! Saliendo del modo interactivo.")
                break
                
            # Comando para mostrar bases de datos
            if query.lower() == 'dbs':
                show_available_databases()
                continue
                
            # Comando para cambiar de base de datos
            if query.lower().startswith('cambiar '):
                try:
                    new_db_index = int(query.lower().split('cambiar ')[1])
                    print(f"{C_SUCCESS}Cambiando a base de datos con índice {new_db_index}")
                    current_db_index = new_db_index
                    continue
                except (ValueError, IndexError):
                    print(f"{C_ERROR}Índice de base de datos inválido. Formato: 'cambiar <número>'")
                    continue
            
            # Procesar la consulta
            logger.info(f"[Modo Interactivo] Procesando consulta: {query}")
            
            # Medir tiempo de respuesta
            start_time = time.time()
            
            response = process_query(
                query, 
                n_chunks=n_chunks, 
                model=current_model, 
                session_id=current_session_id,
                db_index=current_db_index
            )
            
            # Calcular tiempo transcurrido
            elapsed_time = time.time() - start_time
            
            # Mostrar respuesta
            print(f"\n{Style.BRIGHT}{Fore.CYAN}" + "-"*80)
            print(f"{C_SUBTITLE}RESPUESTA (tiempo: {elapsed_time:.2f}s):")
            print(f"{Style.BRIGHT}{Fore.CYAN}" + "-"*80)
            print(response)
            print(f"{Style.BRIGHT}{Fore.CYAN}" + "-"*80)
            
        except KeyboardInterrupt:
            print(f"\n\n{C_WARNING}Interrupción detectada. Saliendo del modo interactivo.")
            break
        except Exception as e:
            logger.error(f"{C_ERROR}Error en modo interactivo: {e}")
            print(f"{C_ERROR}Error: {str(e)}")

def show_available_databases(show_output: bool = True) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Muestra las bases de datos disponibles de forma numerada.
    
    Args:
        show_output: Si es True, muestra información en pantalla
        
    Returns:
        Lista ordenada de tuplas (nombre, metadata) de bases de datos
    """
    # Importar solo cuando sea necesario
    from modulos.session_manager.session_manager import SessionManager
    from config import config
    
    # Obtener las bases de datos disponibles
    session_manager = SessionManager()
    databases = session_manager.list_available_databases()
    
    # Crear una lista ordenada para presentar las bases de datos
    sorted_dbs = []
    for name, metadata in databases.items():
        sorted_dbs.append((name, metadata))
    
    # Ordenar por último uso si está disponible
    sorted_dbs.sort(key=lambda x: x[1].get('last_used', 0), reverse=True)
    
    if show_output:
        print_header(f"BASES DE DATOS DISPONIBLES ({len(databases)})")
        
        if not sorted_dbs:
            print(f"{C_WARNING}No se encontraron bases de datos. Asegúrate de ingestar documentos primero.")
            print(f"{C_INFO}Directorio buscado: " + str(Path(config.get_database_config().get("sqlite", {}).get("db_dir", "modulos/databases/db")).absolute()))
            print(Style.BRIGHT + "="*80 + "\n")
            return sorted_dbs
        
        # Mostrar la información numerada
        for i, (name, metadata) in enumerate(sorted_dbs):
            print(f"{C_HIGHLIGHT}{i}. {C_INFO}Nombre: {name}")
            print(f"   {C_INFO}Tipo: {metadata.get('db_type', 'desconocido')}")
            print(f"   {C_INFO}Modelo: {metadata.get('embedding_model', 'desconocido')}")
            print(f"   {C_INFO}Chunking: {metadata.get('chunking_method', 'desconocido')}")
            
            # Mostrar la ruta de manera adaptativa según el tipo
            db_path = metadata.get('db_path', 'desconocida')
            if db_path and os.path.exists(db_path):
                file_size = os.path.getsize(db_path)
                print(f"   {C_INFO}Ruta: {db_path} ({file_size/1024/1024:.2f} MB)")
            else:
                print(f"   {C_INFO}Ruta: {db_path} ({C_ERROR}No encontrado)")
            
            if metadata.get('custom_name'):
                print(f"   {C_INFO}Nombre personalizado: {metadata['custom_name']}")
                
            # Mostrar el comando de ejemplo para utilizar esta base de datos
            print(f"   {C_INFO}Para usar esta base: python run.py --query \"tu pregunta\" --db-index {i}")
            print()
        
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
        db, metadata = session_manager.get_database_by_index(db_index)
        
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

# Funciones de utilidad para formateo de la salida
def print_header(title: str) -> None:
    """
    Imprime un encabezado formateado.
    
    Args:
        title: Título del encabezado
    """
    print("\n" + Style.BRIGHT + "="*80)
    print(C_TITLE + title)
    print(Style.BRIGHT + "="*80)

def print_formatted_response(title: str, response: str) -> None:
    """
    Imprime una respuesta con formato.
    
    Args:
        title: Título de la respuesta
        response: Texto de la respuesta
    """
    print("\n" + Style.BRIGHT + "="*80)
    print(C_TITLE + title + ":")
    print(Style.BRIGHT + "="*80)
    print(response)
    print(Style.BRIGHT + "="*80 + "\n")

def print_command_help(commands: List[str]) -> None:
    """
    Imprime la ayuda de comandos.
    
    Args:
        commands: Lista de comandos con formato
    """
    for command in commands:
        print(command)
    print(Style.BRIGHT + "=" * 80 + "\n")

def print_useful_commands() -> None:
    """
    Imprime una lista de comandos útiles.
    """
    print(Style.BRIGHT + "="*80)
    print(f"{C_SUBTITLE}COMANDOS ÚTILES:")
    print("  • Para consultar usando la base de datos más reciente:")
    print(f"    {C_COMMAND}python run.py --query {C_PARAM}\"tu pregunta\"")
    print("  • Para consultar usando una base de datos específica por índice:")
    print(f"    {C_COMMAND}python run.py --query {C_PARAM}\"tu pregunta\" --db-index {C_PARAM}<número>")
    print("  • Para ver esta lista de nuevo:")
    print(f"    {C_COMMAND}python run.py --list-dbs")
    print("  • Para mostrar las bases de datos antes de preguntar:")
    print(f"    {C_COMMAND}python run.py --query {C_PARAM}\"tu pregunta\" --show-dbs")
    print("  • Para optimizar una base de datos específica:")
    print(f"    {C_COMMAND}python run.py --optimize-db {C_PARAM}<número>")
    print("  • Para optimizar todas las bases de datos:")
    print(f"    {C_COMMAND}python run.py --optimize-all")
    print("  • Para ver estadísticas de las bases de datos:")
    print(f"    {C_COMMAND}python run.py --db-stats")
    print("  • Para modo interactivo:")
    print(f"    {C_COMMAND}python run.py --query")
    print(Style.BRIGHT + "="*80 + "\n")

if __name__ == "__main__":
    sys.exit(main())
