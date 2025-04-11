"""
Script para ejecutar todas las pruebas de bases de datos y generar un informe.

Este script descubre y ejecuta todas las pruebas en el directorio databases,
genera un informe detallado y guarda los resultados en resultados_db_tests.
"""

import unittest
import sys
import os
import logging
from pathlib import Path
import shutil
import time
import datetime

# Añadir el directorio raíz al path para permitir importaciones relativas
sys.path.insert(0, str(Path(__file__).parent.parent))

def prepare_test_environment():
    """
    Prepara el entorno para las pruebas, incluyendo la creación de directorios.
    """
    # Crear directorio para resultados si no existe
    results_dir = Path(__file__).parent / "resultados_db_tests"
    os.makedirs(results_dir, exist_ok=True)
    
    # Configurar archivo de log
    log_file = results_dir / "run_tests.log"
    
    # Configurar logging principal
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Limpiar archivos de log previos (excepto el actual)
    for old_log in results_dir.glob("*.log"):
        if old_log != log_file and old_log.name != "run_tests.log":
            try:
                old_log.unlink()
                logger.info(f"Archivo de log antiguo eliminado: {old_log}")
            except Exception as e:
                logger.warning(f"No se pudo eliminar {old_log}: {e}")
    
    logger.info(f"Entorno de pruebas preparado. Resultados en: {results_dir}")
    return results_dir, logger

def run_tests():
    """
    Ejecuta todas las pruebas de bases de datos y muestra un resumen.
    
    Returns:
        int: 0 si todas las pruebas pasaron, 1 en caso contrario
    """
    start_time = time.time()
    
    # Preparar entorno y obtener logger
    results_dir, logger = prepare_test_environment()
    
    print("\n" + "=" * 70)
    print("EJECUTANDO PRUEBAS DE BASES DE DATOS VECTORIALES")
    print("=" * 70)
    print("Incluye pruebas actualizadas para:")
    print("- Detección de diferentes extensiones de bases de datos (.db, .sqlite, .duckdb)")
    print("- Compatibilidad con múltiples motores de bases de datos")
    print("- Correcto manejo de metadatos en bases de datos")
    print("-" * 70)
    
    try:
        # Descubrir y cargar todas las pruebas desde el directorio databases
        loader = unittest.TestLoader()
        start_dir = Path(__file__).parent / 'databases'
        
        # Verificar que el directorio existe
        if not start_dir.exists():
            logger.error(f"Directorio de pruebas no encontrado: {start_dir}")
            print(f"ERROR: Directorio de pruebas no encontrado: {start_dir}")
            return 1
            
        suite = loader.discover(str(start_dir), pattern='test_*.py')
        
        # Verificar que se encontraron pruebas
        if not suite.countTestCases():
            logger.warning(f"No se encontraron pruebas en {start_dir}")
            print(f"ADVERTENCIA: No se encontraron pruebas en {start_dir}")
        
        # Configurar el runner con verbosidad adecuada
        runner = unittest.TextTestRunner(verbosity=2)
        
        # Ejecutar las pruebas
        result = runner.run(suite)
        
        # Calcular tiempo transcurrido
        elapsed_time = time.time() - start_time
        
        # Guardar resumen en archivo
        summary_file = results_dir / "test_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("RESUMEN DE PRUEBAS DE BASES DE DATOS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Fecha y hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tiempo de ejecución: {elapsed_time:.2f} segundos\n\n")
            f.write(f"Total de pruebas ejecutadas: {result.testsRun}\n")
            f.write(f"Pruebas exitosas: {result.testsRun - len(result.errors) - len(result.failures)}\n")
            f.write(f"Pruebas fallidas: {len(result.failures)}\n")
            f.write(f"Errores: {len(result.errors)}\n\n")
            
            if result.failures:
                f.write("FALLOS:\n")
                f.write("-" * 50 + "\n")
                for failure in result.failures:
                    f.write(f"- {failure[0]}\n")
                    f.write(f"  {str(failure[1])[:500]}...\n\n")  # Limitar longitud
            
            if result.errors:
                f.write("ERRORES:\n")
                f.write("-" * 50 + "\n")
                for error in result.errors:
                    f.write(f"- {error[0]}\n")
                    f.write(f"  {str(error[1])[:500]}...\n\n")  # Limitar longitud
        
        # Mostrar resumen en pantalla
        print("\n" + "=" * 70)
        print(f"RESULTADOS: {result.testsRun} pruebas ejecutadas en {elapsed_time:.2f} segundos")
        print(f"- Éxitos: {result.testsRun - len(result.errors) - len(result.failures)}")
        print(f"- Fallos: {len(result.failures)}")
        print(f"- Errores: {len(result.errors)}")
        print(f"Resumen guardado en: {summary_file}")
        print("=" * 70)
        
        # Devolver código de salida adecuado
        return 0 if result.wasSuccessful() else 1
    
    except Exception as e:
        logger.error(f"Error durante la ejecución de las pruebas: {e}", exc_info=True)
        print(f"ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
