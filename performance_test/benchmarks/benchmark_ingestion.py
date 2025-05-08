import argparse
import time
import logging
import csv
import os
import sys
import statistics
from datetime import datetime

# Asegurarse de que los módulos del proyecto están en el path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
try:
    from main import process_documents
    from modulos.resource_management.resource_manager import ResourceManager # Para inicializarlo si es necesario
    from modulos.utils.logging_utils import setup_logging, silence_verbose_loggers
except ImportError as e:
    print(f"Error al importar módulos: {e}")
    print("Asegúrate de que el script se ejecuta desde la raíz del proyecto o ajusta el PYTHONPATH.")
    sys.exit(1)

# Configurar logging básico para el script de benchmark
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT) # Nivel WARNING para no saturar
logger = logging.getLogger(__name__)

def run_ingestion_benchmark(corpus_dir, session_name_prefix, num_runs):
    """Ejecuta la ingesta N veces y devuelve los tiempos de duración."""
    durations = []
    logger.info(f"Iniciando benchmark de ingesta ({num_runs} ejecuciones) para: {corpus_dir}")
    
    # Silenciar loggers del RAG para no afectar la medición del tiempo
    silence_verbose_loggers()
    logging.getLogger('main').setLevel(logging.CRITICAL)
    # Podríamos necesitar silenciar más loggers específicos del flujo de ingesta

    for i in range(num_runs):
        run_session_name = f"{session_name_prefix}_{i+1}"
        logger.warning(f"Ejecución {i+1}/{num_runs}, Sesión: {run_session_name}")
        start_time = time.perf_counter() # Usar perf_counter para mayor precisión
        try:
            # Limpiar instancia de ResourceManager antes de cada ingesta? 
            # Podría ser necesario si el estado interno afecta el rendimiento.
            # ResourceManager._instance = None 
            # rm = ResourceManager() # Reinicializar
            
            process_documents(file_path=corpus_dir, session_name=run_session_name)
            end_time = time.perf_counter()
            duration = end_time - start_time
            durations.append(duration)
            logger.warning(f"Ejecución {i+1} completada en {duration:.4f} segundos.")
        except Exception as e:
            logger.error(f"Error en la ejecución {i+1}: {e}", exc_info=True)
            durations.append(None) # Marcar como fallida
        
        # Pequeña pausa y quizás limpieza de memoria entre ejecuciones?
        time.sleep(1)
        # import gc
        # gc.collect()

    return durations

def save_results(output_file, corpus_dir, num_runs, durations):
    """Guarda los resultados detallados y el resumen en un archivo CSV."""
    valid_durations = [d for d in durations if d is not None]
    summary = {}
    if valid_durations:
        summary = {
            'corpus_dir': corpus_dir,
            'num_runs': num_runs,
            'num_successful_runs': len(valid_durations),
            'avg_duration_s': round(statistics.mean(valid_durations), 4),
            'median_duration_s': round(statistics.median(valid_durations), 4),
            'stdev_duration_s': round(statistics.stdev(valid_durations), 4) if len(valid_durations) > 1 else 0,
            'min_duration_s': round(min(valid_durations), 4),
            'max_duration_s': round(max(valid_durations), 4),
            'timestamp': datetime.now().isoformat()
        }

    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            # Escribir resumen primero
            if summary:
                summary_writer = csv.writer(csvfile)
                summary_writer.writerow(["Benchmark Summary - Ingestion"]) 
                writer = csv.DictWriter(csvfile, fieldnames=summary.keys())
                writer.writeheader()
                writer.writerow(summary)
                summary_writer.writerow([]) # Línea vacía
                summary_writer.writerow(["Detailed Run Durations (seconds)"]) # Encabezado para detalles
                header_row = [f'Run_{i+1}' for i in range(len(durations))]
                data_row = [round(d, 4) if d is not None else 'FAILED' for d in durations]
                summary_writer.writerow(header_row)
                summary_writer.writerow(data_row)
            else:
                 csvfile.write("Benchmark fallido: No se completaron ejecuciones exitosas.\n")
                 csvfile.write(f"Corpus: {corpus_dir}\n")
                 csvfile.write(f"Runs intentados: {num_runs}\n")

        logger.warning(f"Resultados del benchmark de ingesta guardados en: {output_file}")
        if summary:
            logger.warning(f"Resumen: Avg={summary['avg_duration_s']}s, Median={summary['median_duration_s']}s, Stdev={summary['stdev_duration_s']}s")
        else:
            logger.error("El benchmark de ingesta no produjo resultados válidos.")
            
    except Exception as e:
        logger.error(f"Error al guardar los resultados del benchmark: {e}")

def main():
    parser = argparse.ArgumentParser(description="Script de benchmark para ingesta de documentos.")
    parser.add_argument("--corpus-dir", required=True, help="Directorio con el corpus estándar para ingesta.")
    parser.add_argument("-n", "--num-runs", type=int, default=5, help="Número de ejecuciones del benchmark.")
    parser.add_argument("--session-prefix", default="ingestion_benchmark", help="Prefijo para los nombres de sesión de cada ejecución.")
    parser.add_argument("--output-csv", default="../results/benchmarks/ingestion_benchmark_results.csv", help="Archivo CSV para guardar los resultados.")
    parser.add_argument("--debug", action="store_true", help="Activar logging DEBUG del script.")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Verificar que el directorio del corpus existe
    if not os.path.isdir(args.corpus_dir):
        logger.error(f"El directorio del corpus especificado no existe: {args.corpus_dir}")
        sys.exit(1)
        
    # Inicializar ResourceManager (solo para asegurar que existe, aunque su estado podría afectar)
    try:
        _ = ResourceManager()
    except Exception as e:
        logger.error(f"Error al inicializar ResourceManager: {e}")
        # Podríamos decidir continuar o salir
        # sys.exit(1)

    # Ejecutar benchmark
    durations = run_ingestion_benchmark(args.corpus_dir, args.session_prefix, args.num_runs)

    # Guardar resultados
    save_results(args.output_csv, args.corpus_dir, args.num_runs, durations)

    logger.warning("Benchmark de ingesta finalizado.")

if __name__ == "__main__":
    main() 