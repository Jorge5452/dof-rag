import os
import sys
import unittest
import datetime
import logging
from pathlib import Path

# Añadir el directorio raíz al path para permitir importaciones relativas
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(Path(__file__).parent / "resultados_db_tests" / "run_tests.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_all_database_tests():
    """
    Ejecuta todas las pruebas de bases de datos y genera un informe de resultados.
    """
    try:
        # Crear directorio para resultados si no existe
        results_dir = Path(__file__).parent / "resultados_db_tests"
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info(f"Entorno de pruebas preparado. Resultados en: {results_dir}")
        
        # Mostrar información sobre las pruebas
        print("\n" + "=" * 70)
        print("EJECUTANDO TODAS LAS PRUEBAS DE BASES DE DATOS")
        print("=" * 70)
        print("Incluye pruebas actualizadas para:")
        print("- Correcta detección de archivos .sqlite y .duckdb")
        print("- Compatibilidad con múltiples motores de bases de datos")
        print("- Manejo adecuado de metadatos y extensiones de archivo")
        print("-" * 70)
        
        # Descubrir y ejecutar todas las pruebas en el directorio databases
        test_loader = unittest.TestLoader()
        test_suite = test_loader.discover(str(Path(__file__).parent / "databases"), pattern="test_*.py")
        
        # Ejecutar las pruebas
        test_runner = unittest.TextTestRunner(verbosity=2)
        test_results = test_runner.run(test_suite)
        
        # Generar informe de resultados
        generate_test_summary(test_results, results_dir)
        
        # Devolver el código de estado (0 si todo está bien, 1 si hay fallos)
        return 0 if test_results.wasSuccessful() else 1
        
    except Exception as e:
        logger.error(f"Error al ejecutar las pruebas: {str(e)}")
        return 1

def generate_test_summary(test_results, results_dir):
    """
    Genera un informe de resumen de las pruebas.
    
    Args:
        test_results: Resultados de las pruebas
        results_dir: Directorio donde se almacenarán los resultados
    """
    try:
        # Crear archivo de resumen
        summary_file = results_dir / "test_summary.txt"
        
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("RESUMEN DE PRUEBAS DE BASES DE DATOS\n")
            f.write("==================================================\n\n")
            
            # Fecha y hora
            now = datetime.datetime.now()
            f.write(f"Fecha y hora: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Estadísticas generales
            success_count = test_results.testsRun - len(test_results.failures) - len(test_results.errors)
            f.write(f"Total de pruebas ejecutadas: {test_results.testsRun}\n")
            f.write(f"Pruebas exitosas: {success_count}\n")
            f.write(f"Pruebas fallidas: {len(test_results.failures)}\n")
            f.write(f"Errores: {len(test_results.errors)}\n\n")
            
            # Detalle de fallos
            if test_results.failures:
                f.write("FALLOS:\n")
                f.write("--------------------------------------------------\n")
                for test, error in test_results.failures:
                    f.write(f"- {test}\n  {error[:500]}...\n\n")
            
            # Detalle de errores
            if test_results.errors:
                f.write("ERRORES:\n")
                f.write("--------------------------------------------------\n")
                for test, error in test_results.errors:
                    f.write(f"- {test}\n  {error[:500]}...\n\n")
                    
        logger.info(f"Resumen de pruebas generado en {summary_file}")
        
    except Exception as e:
        logger.error(f"Error al generar resumen de pruebas: {str(e)}")

if __name__ == "__main__":
    sys.exit(run_all_database_tests())
