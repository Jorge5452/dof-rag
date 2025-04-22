#!/usr/bin/env python3
"""
Script de compatibilidad para redirigir a run_tests.py.
DEPRECATED: Este script será eliminado en versiones futuras.
Utilice 'python -m test.run_tests --type databases' en su lugar.
"""

import sys
import warnings
import argparse
from pathlib import Path
import subprocess

# Mostrar advertencia
warnings.warn(
    "Este script está obsoleto y será eliminado en futuras versiones. "
    "Use 'python -m test.run_tests --type databases' en su lugar.",
    DeprecationWarning
)

def main():
    # Analizar argumentos actuales
    parser = argparse.ArgumentParser(description="Ejecuta pruebas de bases de datos (REDIRECCIÓN)")
    parser.add_argument('--type', help="Tipo de base de datos a probar")
    parser.add_argument('--results-dir', help="Directorio donde guardar los resultados")
    args, unknown = parser.parse_known_args()
    
    # Construir comando equivalente para run_tests.py
    cmd = [sys.executable, "-m", "test.run_tests", "--type", "databases"]
    
    if args.type:
        cmd.extend(["--db-type", args.type])
    
    if args.results_dir:
        cmd.extend(["--results-dir", args.results_dir])
    
    # Añadir argumentos desconocidos (si hay)
    if unknown:
        cmd.extend(unknown)
    
    # Ejecutar comando equivalente
    print("\nRedirigiendo a run_tests.py...")
    print("Comando: " + " ".join(cmd))
    print("=" * 60 + "\n")
    return subprocess.call(cmd)

if __name__ == "__main__":
    sys.exit(main()) 