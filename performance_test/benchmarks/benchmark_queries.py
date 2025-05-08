import argparse
import time
import logging
import csv
import os
import sys
import statistics
import numpy as np # Para percentiles
from datetime import datetime

# Asegurarse de que los módulos del proyecto están en el path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
try:
    from main import process_query
    from modulos.resource_management.resource_manager import ResourceManager # Para inicializarlo si es necesario
    from modulos.utils.logging_utils import setup_logging, silence_verbose_loggers
except ImportError as e:
    print(f"Error al importar módulos: {e}")
    print("Asegúrate de que el script se ejecuta desde la raíz del proyecto o ajusta el PYTHONPATH.")
    sys.exit(1)

# Configurar logging básico
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def run_query_benchmark(queries_file, db_index, session_id, num_runs_per_query):
    """Ejecuta cada consulta N veces y devuelve una lista de resultados detallados."""
    results = []
    
    try:
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        if not queries:
            logger.error("El archivo de consultas está vacío.")
            return None
        logger.warning(f"Iniciando benchmark de consultas para {len(queries)} queries (x{num_runs_per_query} runs cada una).")
    except FileNotFoundError:
        logger.error(f"Archivo de consultas no encontrado: {queries_file}")
        return None

    # Silenciar loggers para medición precisa
    silence_verbose_loggers()
    logging.getLogger('main').setLevel(logging.CRITICAL)
    # Silenciar otros loggers relevantes del flujo de consulta
    logging.getLogger('modulos.clientes').setLevel(logging.CRITICAL) 
    logging.getLogger('modulos.embeddings').setLevel(logging.CRITICAL)
    logging.getLogger('modulos.databases').setLevel(logging.CRITICAL) 

    total_queries_run = 0
    start_benchmark_time = time.perf_counter()

    for i, query_text in enumerate(queries):
        query_latencies = []
        logger.warning(f"Benchmarking query {i+1}/{len(queries)}: '{query_text[:60]}...'")
        for j in range(num_runs_per_query):
            start_time = time.perf_counter()
            response = None
            error = None
            try:
                # Asegurar que ResourceManager esté inicializado
                # _ = ResourceManager() 
                
                response = process_query(query=query_text, db_index=db_index, session_id=session_id)
                # Nota: process_query puede tener su propio logging, que silenciamos arriba.
                success = True if response else False
            except Exception as e:
                logger.error(f"Error ejecutando query '{query_text[:60]}...' (Run {j+1}): {e}")
                success = False
                error = str(e)
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            query_latencies.append(latency_ms if success else None)
            results.append({
                'query_index': i + 1,
                'query_text': query_text,
                'run_index': j + 1,
                'success': success,
                'latency_ms': round(latency_ms, 2) if success else None,
                'error': error,
                'response_length': len(response) if success and response else 0
            })
            total_queries_run += 1
            # Pequeña pausa?
            # time.sleep(0.1)
            
        valid_latencies = [lat for lat in query_latencies if lat is not None]
        if valid_latencies:
             avg_lat = statistics.mean(valid_latencies)
             logger.warning(f"Query {i+1} Avg Latency: {avg_lat:.2f} ms")
        else:
             logger.error(f"Query {i+1} falló en todas las ejecuciones.")

    end_benchmark_time = time.perf_counter()
    total_duration = end_benchmark_time - start_benchmark_time
    logger.warning(f"Benchmark completado. {total_queries_run} ejecuciones en {total_duration:.2f} segundos.")

    return results, total_duration, total_queries_run

def calculate_summary_stats(results):
    """Calcula estadísticas resumen de los resultados detallados."""
    all_latencies = [r['latency_ms'] for r in results if r['success'] and r['latency_ms'] is not None]
    if not all_latencies:
        return None

    summary = {
        'total_executions': len(results),
        'successful_executions': len(all_latencies),
        'failed_executions': len(results) - len(all_latencies),
        'avg_latency_ms': round(statistics.mean(all_latencies), 2),
        'median_latency_ms': round(statistics.median(all_latencies), 2),
        'stdev_latency_ms': round(statistics.stdev(all_latencies), 2) if len(all_latencies) > 1 else 0,
        'min_latency_ms': round(min(all_latencies), 2),
        'max_latency_ms': round(max(all_latencies), 2),
        'p95_latency_ms': round(np.percentile(all_latencies, 95), 2),
        'p99_latency_ms': round(np.percentile(all_latencies, 99), 2)
    }
    return summary

def save_query_results(output_file, results, summary, total_duration, total_queries_run):
    """Guarda los resultados detallados y el resumen en CSV."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            if not results:
                csvfile.write("Benchmark de consultas fallido: No se obtuvieron resultados.")
                return
                
            # Escribir Resumen General
            summary_writer = csv.writer(csvfile)
            summary_writer.writerow(["Benchmark Summary - Queries"])
            if summary:
                dict_writer_summary = csv.DictWriter(csvfile, fieldnames=summary.keys())
                dict_writer_summary.writeheader()
                dict_writer_summary.writerow(summary)
            else:
                 summary_writer.writerow(["No successful executions to calculate summary stats."])
            # Añadir info global
            summary_writer.writerow([])
            summary_writer.writerow(["Total Benchmark Duration (s)", round(total_duration, 2)])
            qps = total_queries_run / total_duration if total_duration > 0 else 0
            summary_writer.writerow(["Overall QPS (executions/s)", round(qps, 2)])
            
            # Escribir Detalles
            summary_writer.writerow([])
            summary_writer.writerow(["Detailed Query Results"])
            fieldnames_detail = results[0].keys()
            dict_writer_detail = csv.DictWriter(csvfile, fieldnames=fieldnames_detail)
            dict_writer_detail.writeheader()
            dict_writer_detail.writerows(results)

        logger.warning(f"Resultados del benchmark de consultas guardados en: {output_file}")
        if summary:
            logger.warning(f"Resumen Latencia: Avg={summary['avg_latency_ms']}ms, Median={summary['median_latency_ms']}ms, P95={summary['p95_latency_ms']}ms")
            
    except Exception as e:
        logger.error(f"Error al guardar los resultados del benchmark de consultas: {e}")

def main():
    parser = argparse.ArgumentParser(description="Script de benchmark para consultas RAG.")
    parser.add_argument("--queries-file", required=True, help="Archivo con consultas estándar.")
    parser.add_argument("--db-index", type=int, default=0, help="Índice de la DB a usar.")
    parser.add_argument("--session-id", type=str, default=None, help="ID de sesión a usar (opcional).")
    parser.add_argument("-n", "--num-runs", type=int, default=3, help="Número de ejecuciones por consulta.")
    parser.add_argument("--output-csv", default="../results/benchmarks/queries_benchmark_results.csv", help="Archivo CSV para guardar los resultados.")
    parser.add_argument("--debug", action="store_true", help="Activar logging DEBUG del script.")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    # Inicializar ResourceManager (para que los componentes internos puedan usarlo si es necesario)
    try:
        _ = ResourceManager()
    except Exception as e:
        logger.error(f"Error al inicializar ResourceManager: {e}")

    # Ejecutar benchmark
    results, total_duration, total_queries_run = run_query_benchmark(args.queries_file, args.db_index, args.session_id, args.num_runs)

    # Calcular resumen
    summary = calculate_summary_stats(results) if results else None

    # Guardar resultados
    if results:
        save_query_results(args.output_csv, results, summary, total_duration, total_queries_run)
    else:
        logger.error("Benchmark de consultas no produjo resultados.")

    logger.warning("Benchmark de consultas finalizado.")

if __name__ == "__main__":
    main() 