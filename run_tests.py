#!/usr/bin/env python3
"""
Script para ejecutar pruebas de chunkers y analizar los resultados en secuencia.
"""

import argparse
import subprocess
from pathlib import Path

def main():
    """Función principal que ejecuta el test de chunkers y analiza los resultados."""
    parser = argparse.ArgumentParser(description="Ejecuta pruebas y análisis de chunkers")
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
    parser.add_argument("--analysis-dir", type=str, default="test/analisis_resultados",
                        help="Directorio donde guardar los análisis")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("PRUEBAS DE CHUNKING CON CONFIGURACIONES DETALLADAS")
    print("=" * 60)
    print("\nEjecutando pruebas con los siguientes parámetros:")
    print(f"- Directorio/archivo: {args.dir if not args.file else args.file}")
    print(f"- Métodos de chunking: {args.chunkers}")
    print(f"- Modelo de embedding: {args.model if args.model else 'Por defecto (configuración)'}")
    print(f"- Directorio de resultados: {args.results_dir}")
    print(f"- Directorio de análisis: {args.analysis_dir}")
    
    print("\nIMPORTANTE: Los archivos de resultados se guardarán con el formato:")
    print("  [método_chunking]_[nombre_documento]_results.txt")
    print("Ejemplo: context_documento1_results.txt\n")
    
    # Construir el comando para ejecutar las pruebas
    test_cmd = ["python", "test/test_chunkers.py"]
    
    if args.dir:
        test_cmd.extend(["--dir", args.dir])
    if args.file:
        test_cmd.extend(["--file", args.file])
    if args.chunkers:
        test_cmd.extend(["--chunkers", args.chunkers])
    if args.model:
        test_cmd.extend(["--model", args.model])
    if args.results_dir:
        test_cmd.extend(["--results-dir", args.results_dir])
    
    # Ejecutar las pruebas
    print("Ejecutando pruebas de chunkers...")
    subprocess.run(test_cmd)
    
    # Construir el comando para analizar los resultados
    analysis_cmd = ["python", "test/analizar_resultados.py"]
    
    if args.results_dir:
        analysis_cmd.extend(["--dir", args.results_dir])
    if args.analysis_dir:
        analysis_cmd.extend(["--out", args.analysis_dir])
    
    # Ejecutar el análisis
    print("\nAnalizando resultados...")
    subprocess.run(analysis_cmd)
    
    print("\n¡Proceso completo! Revisa los resultados en los directorios especificados.")

if __name__ == "__main__":
    main()
