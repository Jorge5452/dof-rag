import unittest
import sys
import os
import time
import logging
from pathlib import Path
import shutil

# Añadir el directorio raíz al path para permitir importaciones relativas
sys.path.insert(0, str(Path(__file__).parent.parent))

def prepare_test_environment():
    """
    Prepara el entorno para las pruebas, creando directorios necesarios.
    """
    # Crear directorios para resultados
    results_dirs = {
        "db_tests": Path(__file__).parent / "resultados_db_tests",
        "chunker_tests": Path(__file__).parent / "resultados_pruebas"
    }
    
    for name, dir_path in results_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directorio de resultados para {name}: {dir_path}")
    
    return results_dirs

def run_tests():
    """
    Ejecuta todas las pruebas del proyecto y muestra un resumen.
    """
    start_time = time.time()
    
    # Preparar entorno
    results_dirs = prepare_test_environment()
    
    print("\n" + "=" * 70)
    print("EJECUTANDO PRUEBAS COMPLETAS DEL SISTEMA RAG")
    print("=" * 70)
    
    # Configurar logging principal
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(results_dirs["db_tests"] / "all_tests.log"),
            logging.StreamHandler()
        ]
    )
    
    # Descubrir y cargar todas las pruebas desde el directorio test
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Configurar el runner con verbosidad
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Ejecutar las pruebas
    result = runner.run(suite)
    
    # Calcular tiempo transcurrido
    elapsed_time = time.time() - start_time
    
    # Mostrar resumen
    print("\n" + "=" * 70)
    print(f"RESULTADOS TOTALES: {result.testsRun} pruebas ejecutadas en {elapsed_time:.2f} segundos")
    print(f"- Éxitos: {result.testsRun - len(result.errors) - len(result.failures)}")
    print(f"- Errores: {len(result.errors)}")
    print(f"- Fallos: {len(result.failures)}")
    print("=" * 70)
    
    # Guardar resumen en archivo
    summary_file = results_dirs["db_tests"] / "all_tests_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE TODAS LAS PRUEBAS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Fecha y hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
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
    
    print(f"Resumen guardado en: {summary_file}")
    
    # Devolver código de salida adecuado
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests())
