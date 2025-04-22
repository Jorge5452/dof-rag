#!/usr/bin/env python3
"""
Script orquestador para ejecutar diferentes tipos de pruebas del sistema RAG.

Uso:
    python run_tests.py [--type <tipo_prueba>] [opciones específicas]
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

def ensure_dir_exists(dir_path):
    """
    Asegura que un directorio exista, creándolo si es necesario.
    
    Args:
        dir_path: Ruta del directorio
    """
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def get_test_result_path(base_dir, test_type):
    """
    Obtiene la ruta para los resultados de un tipo de prueba específico.
    
    Args:
        base_dir: Directorio base de resultados
        test_type: Tipo de prueba (chunkers, databases, etc.)
        
    Returns:
        Ruta completa al directorio de resultados
    """
    result_dir = Path(base_dir) / test_type
    ensure_dir_exists(result_dir)
    return str(result_dir)

def run_chunker_tests(args):
    """
    Ejecuta pruebas de chunkers y analiza los resultados.
    
    Args:
        args: Argumentos de línea de comandos
    """
    print("\n" + "=" * 60)
    print("PRUEBAS DE CHUNKING CON CONFIGURACIONES DETALLADAS")
    print("=" * 60)
    
    # Asegurar que existan los directorios de resultados
    results_dir = get_test_result_path(args.results_dir, "chunkers")
    analysis_dir = Path(args.analysis_dir)
    ensure_dir_exists(analysis_dir)
    
    print("\nEjecutando pruebas con los siguientes parámetros:")
    print(f"- Directorio/archivo: {args.dir if not args.file else args.file}")
    print(f"- Métodos de chunking: {args.chunkers}")
    print(f"- Modelo de embedding: {args.model if args.model else 'Por defecto (configuración)'}")
    print(f"- Directorio de resultados: {results_dir}")
    print(f"- Directorio de análisis: {analysis_dir}")
    
    print("\nIMPORTANTE: Los archivos de resultados se guardarán con el formato:")
    print("  [método_chunking]_[nombre_documento]_results.txt")
    print("Ejemplo: context_documento1_results.txt\n")
    
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
    if results_dir:
        test_cmd.extend(["--results-dir", results_dir])
    
    # Ejecutar las pruebas
    print("Ejecutando pruebas de chunkers...")
    subprocess.run(test_cmd)
    
    # Construir el comando para analizar los resultados
    analysis_cmd = ["python", "-m", "test.analizar_resultados"]
    
    if results_dir:
        analysis_cmd.extend(["--dir", results_dir])
    if analysis_dir:
        analysis_cmd.extend(["--out", str(analysis_dir)])
    
    # Ejecutar el análisis
    print("\nAnalizando resultados...")
    subprocess.run(analysis_cmd)
    
    print("\n¡Proceso completo! Revisa los resultados en los directorios especificados.")

def run_database_tests(args):
    """
    Ejecuta pruebas de bases de datos.
    
    Args:
        args: Argumentos de línea de comandos
    """
    print("\n" + "=" * 60)
    print("PRUEBAS DE BASES DE DATOS VECTORIALES")
    print("=" * 60)
    
    # Asegurar que exista el directorio de resultados
    results_dir = get_test_result_path(args.results_dir, "databases")
    
    db_test_cmd = ["python", "-m", "test.run_all_database_tests"]
    
    if args.db_type:
        db_test_cmd.extend(["--type", args.db_type])
    if results_dir:
        db_test_cmd.extend(["--results-dir", results_dir])
    
    print(f"Ejecutando pruebas para bases de datos: {args.db_type if args.db_type else 'todas'}")
    print(f"Resultados en: {results_dir}")
    subprocess.run(db_test_cmd)
    
    print("\n¡Pruebas de bases de datos completadas!")

def run_client_tests(args):
    """
    Ejecuta pruebas de clientes de IA.
    
    Args:
        args: Argumentos de línea de comandos
    """
    print("\n" + "=" * 60)
    print("PRUEBAS DE CLIENTES DE IA")
    print("=" * 60)
    
    # Asegurar que exista el directorio de resultados
    results_dir = get_test_result_path(args.results_dir, "clients")
    print(f"Resultados en: {results_dir}")
    
    # Configurar variables de entorno para los resultados
    os.environ["TEST_RESULTS_DIR"] = results_dir
    
    if args.client == "gemini" or args.client == "all":
        print("\nProbando cliente Gemini...")
        subprocess.run(["python", "-m", "test.clients.test_gemini"])
    
    if args.client == "openai" or args.client == "all":
        print("\nProbando cliente OpenAI...")
        subprocess.run(["python", "-m", "test.clients.test_openai"])
    
    if args.client == "ollama" or args.client == "all":
        print("\nProbando cliente Ollama...")
        subprocess.run(["python", "-m", "test.clients.test_ollama"])
    
    if args.client == "config" or args.client == "all":
        print("\nProbando configuración de clientes...")
        subprocess.run(["python", "-m", "test.clients.test_client_configs"])
    
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
    
    # Asegurar que exista el directorio de resultados
    results_dir = get_test_result_path(args.results_dir, "embeddings")
    print(f"Resultados en: {results_dir}")
    
    # Configurar variables de entorno para los resultados
    os.environ["TEST_RESULTS_DIR"] = results_dir
    
    if args.embedding_component == "manager" or args.embedding_component == "all":
        print("\nProbando EmbeddingManager...")
        subprocess.run(["python", "-m", "test.embeddings.test_embedding_manager"])
    
    if args.embedding_component == "factory" or args.embedding_component == "all":
        print("\nProbando EmbeddingFactory...")
        subprocess.run(["python", "-m", "test.embeddings.test_embedding_factory"])
    
    if args.embedding_component == "integration" or args.embedding_component == "all":
        print("\nProbando integración de embeddings...")
        subprocess.run(["python", "-m", "test.embeddings.test_embedding_integration"])
    
    print("\n¡Pruebas de embeddings completadas!")

def run_doc_processor_tests(args):
    """
    Ejecuta pruebas del procesador de documentos.
    
    Args:
        args: Argumentos de línea de comandos
    """
    print("\n" + "=" * 60)
    print("PRUEBAS DEL PROCESADOR DE DOCUMENTOS")
    print("=" * 60)
    
    # Asegurar que exista el directorio de resultados
    results_dir = get_test_result_path(args.results_dir, "doc_processor")
    print(f"Resultados en: {results_dir}")
    
    test_cmd = ["python", "-m", "test.doc_processor.test_doc_processor"]
    
    if results_dir:
        test_cmd.extend(["--results-dir", results_dir])
    
    print("Ejecutando pruebas del procesador de documentos...")
    subprocess.run(test_cmd)
    
    print("\n¡Pruebas de procesamiento de documentos completadas!")

def run_rag_tests(args):
    """
    Ejecuta pruebas del módulo RAG principal.
    
    Args:
        args: Argumentos de línea de comandos
    """
    print("\n" + "=" * 60)
    print("PRUEBAS DEL MÓDULO RAG")
    print("=" * 60)
    
    # Asegurar que exista el directorio de resultados
    results_dir = get_test_result_path(args.results_dir, "rag")
    print(f"Resultados en: {results_dir}")
    
    test_cmd = ["python", "-m", "test.rag.test_rag_pipeline"]
    
    if results_dir:
        test_cmd.extend(["--results-dir", results_dir])
    
    print("Ejecutando pruebas del pipeline RAG...")
    subprocess.run(test_cmd)
    
    print("\n¡Pruebas del módulo RAG completadas!")

def run_session_manager_tests(args):
    """
    Ejecuta pruebas del gestor de sesiones.
    
    Args:
        args: Argumentos de línea de comandos
    """
    print("\n" + "=" * 60)
    print("PRUEBAS DEL GESTOR DE SESIONES")
    print("=" * 60)
    
    # Asegurar que exista el directorio de resultados
    results_dir = get_test_result_path(args.results_dir, "session_manager")
    print(f"Resultados en: {results_dir}")
    
    test_cmd = ["python", "-m", "test.session_manager.test_session_manager"]
    
    if results_dir:
        test_cmd.extend(["--results-dir", results_dir])
    
    print("Ejecutando pruebas del gestor de sesiones...")
    subprocess.run(test_cmd)
    
    print("\n¡Pruebas del gestor de sesiones completadas!")

def run_view_chunks_tests(args):
    """
    Ejecuta pruebas del visualizador de chunks.
    
    Args:
        args: Argumentos de línea de comandos
    """
    print("\n" + "=" * 60)
    print("PRUEBAS DEL VISUALIZADOR DE CHUNKS")
    print("=" * 60)
    
    # Asegurar que exista el directorio de resultados
    results_dir = get_test_result_path(args.results_dir, "view_chunks")
    print(f"Resultados en: {results_dir}")
    
    test_cmd = ["python", "-m", "test.view_chunks.test_view_chunks"]
    
    if results_dir:
        test_cmd.extend(["--results-dir", results_dir])
    
    print("Ejecutando pruebas del visualizador de chunks...")
    subprocess.run(test_cmd)
    
    print("\n¡Pruebas del visualizador de chunks completadas!")

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
    
    # Ejecutar pruebas del procesador de documentos
    doc_processor_args = argparse.Namespace(
        results_dir=args.results_dir
    )
    run_doc_processor_tests(doc_processor_args)
    
    # Ejecutar pruebas del módulo RAG
    rag_args = argparse.Namespace(
        results_dir=args.results_dir
    )
    run_rag_tests(rag_args)
    
    # Ejecutar pruebas del gestor de sesiones
    session_args = argparse.Namespace(
        results_dir=args.results_dir
    )
    run_session_manager_tests(session_args)
    
    # Ejecutar pruebas del visualizador de chunks
    view_chunks_args = argparse.Namespace(
        results_dir=args.results_dir
    )
    run_view_chunks_tests(view_chunks_args)
    
    print("\n" + "=" * 60)
    print("TODAS LAS PRUEBAS COMPLETADAS")
    print("=" * 60)

def main():
    """Función principal que orquesta la ejecución de las pruebas."""
    parser = argparse.ArgumentParser(description="Orquestador de pruebas para el sistema RAG")
    
    # Argumentos generales
    parser.add_argument("--type", type=str, 
                      choices=["chunkers", "databases", "clients", "embeddings", 
                               "doc_processor", "rag", "session_manager", "view_chunks", "all"],
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
    parser.add_argument("--db-type", type=str, choices=["sqlite", "duckdb"],
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
    
    args = parser.parse_args()
    
    # Asegurar que existen los directorios
    ensure_dir_exists(args.results_dir)
    ensure_dir_exists(args.analysis_dir)
    
    # Ejecutar el tipo de prueba seleccionado
    if args.type == "chunkers":
        run_chunker_tests(args)
    elif args.type == "databases":
        run_database_tests(args)
    elif args.type == "clients":
        run_client_tests(args)
    elif args.type == "embeddings":
        run_embedding_tests(args)
    elif args.type == "doc_processor":
        run_doc_processor_tests(args)
    elif args.type == "rag":
        run_rag_tests(args)
    elif args.type == "session_manager":
        run_session_manager_tests(args)
    elif args.type == "view_chunks":
        run_view_chunks_tests(args)
    elif args.type == "all":
        run_all_tests(args)
    else:
        print(f"Tipo de prueba no reconocido: {args.type}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
