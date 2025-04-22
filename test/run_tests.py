#!/usr/bin/env python3
"""
Script unificado para ejecutar pruebas del sistema RAG.

Este script permite ejecutar diferentes tipos de pruebas del sistema RAG
utilizando una interfaz unificada y generando informes estandarizados.

Uso:
    python -m test.run_tests [--type <tipo_prueba>] [opciones específicas]

Ejemplos:
    python -m test.run_tests --type databases --db-type sqlite
    python -m test.run_tests --type chunkers --chunkers character,token
    python -m test.run_tests --type all
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path
import time
from datetime import datetime

# Añadir el directorio raíz al path para permitir importaciones relativas
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar módulos de utilidades
from test.utils.environment import (
    ensure_dir_exists, get_test_result_path, prepare_test_environment
)
from test.utils.reporting import run_tests_with_reporting
from test.utils.discovery import (
    discover_tests, get_test_suite_by_type, list_available_tests
)

def run_database_tests(args):
    """
    Ejecuta pruebas de bases de datos.
    
    Args:
        args: Argumentos de línea de comandos
    """
    print("\n" + "=" * 60)
    print("PRUEBAS DE BASES DE DATOS VECTORIALES")
    print("=" * 60)
    
    # Preparar entorno
    results_dir, logger = prepare_test_environment(
        "databases", args.results_dir
    )
    
    # Determinar el patrón de prueba según el tipo de base de datos
    filter_pattern = None
    if args.db_type and args.db_type != "all":
        filter_pattern = f"test_{args.db_type}_*.py"
        print(f"Ejecutando pruebas para bases de datos tipo: {args.db_type}")
    else:
        print("Ejecutando todas las pruebas de bases de datos")
    
    # Obtener la suite de pruebas
    test_suite = get_test_suite_by_type("databases", filter_pattern)
    
    # Ejecutar pruebas y generar informes
    metadata = {"db_type": args.db_type} if args.db_type else {}
    run_tests_with_reporting(test_suite, "databases", results_dir, metadata=metadata)
    
    print("\n¡Pruebas de bases de datos completadas!")

def run_chunker_tests(args):
    """
    Ejecuta pruebas de chunkers y analiza los resultados.
    
    Args:
        args: Argumentos de línea de comandos
    """
    print("\n" + "=" * 60)
    print("PRUEBAS DE CHUNKING CON CONFIGURACIONES DETALLADAS")
    print("=" * 60)
    
    # Preparar entorno
    results_dir, logger = prepare_test_environment(
        "chunkers", args.results_dir
    )
    analysis_dir = ensure_dir_exists(args.analysis_dir)
    
    print("\nEjecutando pruebas con los siguientes parámetros:")
    print(f"- Directorio/archivo: {args.dir if not args.file else args.file}")
    print(f"- Métodos de chunking: {args.chunkers}")
    print(f"- Modelo de embedding: {args.model if args.model else 'Por defecto (configuración)'}")
    print(f"- Directorio de resultados: {results_dir}")
    print(f"- Directorio de análisis: {analysis_dir}")
    
    # Construir el comando para ejecutar las pruebas
    test_cmd = ["python", "-m", "test.chunkers.test_chunkers"]
    
    if args.dir:
        test_cmd.extend(["--dir", args.dir])
    if args.file:
        test_cmd.extend(["--file", args.file])
    if args.chunkers:
        test_cmd.extend(["--chunkers", args.chunkers])
    if args.model:
        test_cmd.extend(["--model", args.model])
    
    test_cmd.extend(["--results-dir", str(results_dir)])
    
    # Ejecutar las pruebas
    print("Ejecutando pruebas de chunkers...")
    subprocess.run(test_cmd)
    
    # Construir el comando para analizar los resultados
    analysis_cmd = ["python", "-m", "test.analizar_resultados"]
    analysis_cmd.extend(["--dir", str(results_dir)])
    analysis_cmd.extend(["--out", str(analysis_dir)])
    
    # Ejecutar el análisis
    print("\nAnalizando resultados...")
    subprocess.run(analysis_cmd)
    
    print("\n¡Proceso completo! Revisa los resultados en los directorios especificados.")

def run_client_tests(args):
    """
    Ejecuta pruebas de clientes de IA.
    
    Args:
        args: Argumentos de línea de comandos
    """
    print("\n" + "=" * 60)
    print("PRUEBAS DE CLIENTES DE IA")
    print("=" * 60)
    
    # Preparar entorno
    results_dir, logger = prepare_test_environment(
        "clients", args.results_dir
    )
    
    # Ejecutar pruebas según el cliente especificado
    if args.client == "all":
        clients = ["gemini", "openai", "ollama", "config"]
    else:
        clients = [args.client]
    
    for client in clients:
        print(f"\nProbando cliente: {client}")
        
        # Seleccionar el módulo de prueba apropiado
        if client == "config":
            module = "test.clients.test_client_configs"
        else:
            module = f"test.clients.test_{client}"
        
        # Ejecutar el módulo de prueba
        subprocess.run(["python", "-m", module])
    
    print("\n¡Pruebas de clientes completadas!")

def run_embedding_tests(args):
    """
    Ejecuta pruebas de embeddings.
    
    Args:
        args: Argumentos de línea de comandos
    """
    print("\n" + "=" * 60)
    print("PRUEBAS DE EMBEDDINGS")
    print("=" * 60)
    
    # Preparar entorno
    results_dir, logger = prepare_test_environment(
        "embeddings", args.results_dir
    )
    
    # Ejecutar pruebas según el componente especificado
    if args.embedding_component == "all":
        components = ["manager", "factory", "integration"]
    else:
        components = [args.embedding_component]
    
    for component in components:
        print(f"\nProbando componente de embeddings: {component}")
        subprocess.run(["python", "-m", f"test.embeddings.test_embedding_{component}"])
    
    print("\n¡Pruebas de embeddings completadas!")

def run_single_module_tests(args, test_type):
    """
    Ejecuta pruebas de un tipo específico que no requieren parámetros especiales.
    
    Args:
        args: Argumentos de línea de comandos
        test_type: Tipo de prueba (doc_processor, rag, session_manager, view_chunks)
    """
    print("\n" + "=" * 60)
    print(f"PRUEBAS DE {test_type.upper()}")
    print("=" * 60)
    
    # Preparar entorno
    results_dir, logger = prepare_test_environment(
        test_type, args.results_dir
    )
    
    # Obtener la suite de pruebas
    test_suite = get_test_suite_by_type(test_type)
    
    # Ejecutar pruebas y generar informes
    run_tests_with_reporting(test_suite, test_type, results_dir)
    
    print(f"\n¡Pruebas de {test_type} completadas!")

def run_all_tests(args):
    """
    Ejecuta todas las pruebas disponibles.
    
    Args:
        args: Argumentos de línea de comandos
    """
    print("\n" + "=" * 60)
    print("EJECUTANDO TODAS LAS PRUEBAS DEL SISTEMA RAG")
    print("=" * 60)
    
    # Mostrar información sobre la ejecución
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Fecha y hora: {timestamp}")
    print(f"Directorio base de resultados: {args.results_dir}")
    print(f"Directorio de análisis: {args.analysis_dir}")
    print("=" * 60)
    
    # Ejecutar pruebas de chunkers
    chunker_args = argparse.Namespace(
        dir=args.dir,
        file=args.file,
        chunkers="character,token,context,page",
        model=args.model,
        results_dir=args.results_dir,
        analysis_dir=args.analysis_dir
    )
    run_chunker_tests(chunker_args)
    
    # Ejecutar pruebas de bases de datos
    db_args = argparse.Namespace(
        db_type=None,  # Probar todas las bases de datos
        results_dir=args.results_dir
    )
    run_database_tests(db_args)
    
    # Ejecutar pruebas de clientes
    client_args = argparse.Namespace(
        client="all",
        results_dir=args.results_dir
    )
    run_client_tests(client_args)
    
    # Ejecutar pruebas de embeddings
    embedding_args = argparse.Namespace(
        embedding_component="all",
        results_dir=args.results_dir
    )
    run_embedding_tests(embedding_args)
    
    # Ejecutar pruebas de módulos individuales
    for test_type in ["doc_processor", "rag", "session_manager", "view_chunks"]:
        module_args = argparse.Namespace(
            results_dir=args.results_dir
        )
        run_single_module_tests(module_args, test_type)
    
    print("\n" + "=" * 60)
    print("TODAS LAS PRUEBAS COMPLETADAS")
    print("=" * 60)

def list_tests(args):
    """
    Lista las pruebas disponibles.
    
    Args:
        args: Argumentos de línea de comandos
    """
    print("\n" + "=" * 60)
    print("LISTADO DE PRUEBAS DISPONIBLES")
    print("=" * 60)
    
    if args.type and args.type != "all":
        available_tests = list_available_tests(args.type)
    else:
        available_tests = list_available_tests()
    
    # Mostrar las pruebas disponibles
    for test_type, tests in available_tests.items():
        print(f"\n{test_type.upper()}:")
        print("-" * 50)
        
        if not tests:
            print("  (No se encontraron pruebas)")
            continue
        
        for test in tests:
            print(f"  - {test}")
    
    print("\n¡Listado de pruebas completado!")

def main():
    """Función principal que orquesta la ejecución de las pruebas."""
    parser = argparse.ArgumentParser(description="Orquestador unificado de pruebas para el sistema RAG")
    
    # Argumentos generales
    parser.add_argument("--type", type=str, 
                      choices=["databases", "chunkers", "clients", "embeddings", 
                               "doc_processor", "rag", "session_manager", "view_chunks", 
                               "list", "all"],
                      default="all", help="Tipo de prueba a ejecutar")
    
    # Argumentos para chunkers
    parser.add_argument("--dir", type=str, default="pruebas", 
                      help="Directorio donde buscar archivos Markdown (predeterminado: 'pruebas')")
    parser.add_argument("--file", type=str, 
                      help="Ruta específica a un archivo Markdown para procesar")
    parser.add_argument("--chunkers", type=str, default="character,token,context,page",
                      help="Lista separada por comas de los chunkers a utilizar")
    parser.add_argument("--model", type=str, 
                      help="Nombre del modelo de embedding a utilizar (opcional)")
    
    # Argumentos para bases de datos
    parser.add_argument("--db-type", type=str, choices=["sqlite", "duckdb", "all"],
                      help="Tipo de base de datos a probar")
    
    # Argumentos para clientes
    parser.add_argument("--client", type=str, choices=["gemini", "openai", "ollama", "config", "all"],
                      default="all", help="Cliente específico a probar")
    
    # Argumentos para embeddings
    parser.add_argument("--embedding-component", type=str, 
                      choices=["manager", "factory", "integration", "all"],
                      default="all", help="Componente de embeddings a probar")
    
    # Argumentos para resultados
    parser.add_argument("--results-dir", type=str, default="test/results",
                      help="Directorio base donde guardar los resultados de las pruebas")
    parser.add_argument("--analysis-dir", type=str, default="test/results/analysis",
                      help="Directorio donde guardar los análisis")
    
    # Argumento para pruebas específicas
    parser.add_argument("--tests", type=str,
                      help="Lista separada por comas de pruebas específicas a ejecutar")
    
    args = parser.parse_args()
    
    # Asegurar que existen los directorios
    ensure_dir_exists(args.results_dir)
    ensure_dir_exists(args.analysis_dir)
    
    # Ejecutar el tipo de prueba seleccionado
    if args.type == "databases":
        run_database_tests(args)
    elif args.type == "chunkers":
        run_chunker_tests(args)
    elif args.type == "clients":
        run_client_tests(args)
    elif args.type == "embeddings":
        run_embedding_tests(args)
    elif args.type in ["doc_processor", "rag", "session_manager", "view_chunks"]:
        run_single_module_tests(args, args.type)
    elif args.type == "list":
        list_tests(args)
    elif args.type == "all":
        run_all_tests(args)
    else:
        print(f"Tipo de prueba no reconocido: {args.type}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 