import argparse
import time
import logging
import csv
import threading
import os
import sys
import random
from datetime import datetime
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

# Asegurarse de que los módulos del proyecto están en el path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
try:
    from main import process_query
    from modulos.resource_management.resource_manager import ResourceManager
    from modulos.utils.logging_utils import setup_logging
except ImportError as e:
    print(f"Error al importar módulos: {e}")
    print("Asegúrate de que el script se ejecuta desde la raíz del proyecto o ajusta el PYTHONPATH.")
    sys.exit(1)

# Configurar logging básico
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Variables globales para monitoreo y resultados
monitoring_active = False
metrics_data = []
query_results = []
queries_processed = 0
queries_succeeded = 0
queries_failed = 0
result_lock = threading.Lock()

def monitor_resources_query(resource_manager, interval, output_csv):
    """Función para monitorear recursos durante la prueba de consultas."""
    global monitoring_active, metrics_data
    logger.info("Inicio del monitoreo de recursos (consultas)...")
    start_time = time.time()
    fieldnames = [
        'timestamp', 'elapsed_seconds', 'system_cpu_percent', 'system_memory_percent',
        'process_cpu_percent', 'process_memory_mb', 'active_sessions_rag', 'active_embedding_models'
    ]
    metrics_data.append(fieldnames)

    while monitoring_active:
        try:
            current_time = time.time()
            elapsed = current_time - start_time
            metrics = resource_manager.metrics
            row = {
                'timestamp': datetime.now().isoformat(), 'elapsed_seconds': round(elapsed, 2),
                'system_cpu_percent': metrics.get('system_cpu_percent', 'N/A'),
                'system_memory_percent': metrics.get('system_memory_percent', 'N/A'),
                'process_cpu_percent': metrics.get('process_cpu_percent', 'N/A'),
                'process_memory_mb': metrics.get('process_memory_mb', 'N/A'),
                'active_sessions_rag': metrics.get('active_sessions_rag', 'N/A'),
                'active_embedding_models': metrics.get('active_embedding_models', 'N/A')
            }
            metrics_data.append([row.get(field, 'N/A') for field in fieldnames])
            time.sleep(interval)
        except Exception as e:
            logger.error(f"Error en monitoreo (consultas): {e}")
            time.sleep(interval)
    
    logger.info("Fin del monitoreo de recursos (consultas).")
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(metrics_data)
        logger.info(f"Métricas de consulta guardadas en: {output_csv}")
    except Exception as e:
        logger.error(f"Error al guardar métricas de consulta: {e}")

def query_worker(query_queue, db_index, session_id, results_list):
    """Trabajador que procesa consultas de la cola."""
    global queries_processed, queries_succeeded, queries_failed
    while True:
        try:
            query_text = query_queue.get()
            if query_text is None: # Señal para terminar
                query_queue.task_done()
                break
            
            start_query_time = time.time()
            error_msg = None
            response = None
            try:
                # Llamar a la función de procesamiento de consulta
                response = process_query(query=query_text, db_index=db_index, session_id=session_id)
                # Aquí response ya incluye el contexto si se configuró así
            except Exception as e:
                logger.warning(f"Error procesando query '{query_text[:50]}...': {e}")
                error_msg = str(e)
            
            end_query_time = time.time()
            duration = end_query_time - start_query_time
            success = error_msg is None and response is not None
            
            # Registrar resultado de forma thread-safe
            with result_lock:
                results_list.append({
                    'query': query_text,
                    'success': success,
                    'duration_ms': round(duration * 1000, 2),
                    'error': error_msg,
                    'response_length': len(response) if response else 0
                })
                queries_processed += 1
                if success:
                    queries_succeeded += 1
                else:
                    queries_failed += 1
            
            query_queue.task_done()
            time.sleep(0.05) # Pequeña pausa para no saturar CPU si las consultas son muy rápidas

        except Exception as e:
            logger.error(f"Error fatal en query_worker: {e}")
            query_queue.task_done() # Asegurar que la tarea se marque como hecha
            # Continuar con la siguiente consulta si es posible

def main():
    global monitoring_active, metrics_data, query_results
    global queries_processed, queries_succeeded, queries_failed

    parser = argparse.ArgumentParser(description="Script de prueba de carga para consultas concurrentes.")
    parser.add_argument("--db-index", type=int, default=0, help="Índice de la DB a usar (0=más reciente).")
    parser.add_argument("--session-id", type=str, default=None, help="ID de sesión existente a usar (opcional).")
    parser.add_argument("--queries-file", required=True, help="Archivo con consultas de prueba (una por línea).")
    parser.add_argument("--num-threads", type=int, default=10, help="Número de hilos concurrentes.")
    parser.add_argument("--num-queries", type=int, default=100, help="Número total de consultas a enviar.")
    parser.add_argument("--duration", type=int, default=None, help="Duración de la prueba en segundos (ignora num-queries si se especifica).")
    parser.add_argument("--monitor-interval", type=float, default=2.0, help="Intervalo (s) para monitoreo de recursos.")
    parser.add_argument("--output-metrics-csv", default="../results/concurrent_queries_metrics.csv", help="CSV para métricas de recursos.")
    parser.add_argument("--output-results-csv", default="../results/concurrent_queries_results.csv", help="CSV para resultados de consultas.")
    parser.add_argument("--debug", action="store_true", help="Activar logging DEBUG.")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    logger.info(f"Iniciando prueba de carga de consultas concurrentes")
    logger.info(f"Usando DB índice: {args.db_index}",) 
    if args.session_id: logger.info(f"Usando Session ID: {args.session_id}")
    logger.info(f"Archivo de consultas: {args.queries_file}")
    logger.info(f"Hilos concurrentes: {args.num_threads}")
    if args.duration:
        logger.info(f"Duración de la prueba: {args.duration} segundos")
    else:
        logger.info(f"Número de consultas: {args.num_queries}")

    # Leer consultas del archivo
    try:
        with open(args.queries_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        if not queries:
            logger.error("El archivo de consultas está vacío o no contiene consultas válidas.")
            sys.exit(1)
        logger.info(f"Cargadas {len(queries)} consultas de prueba.")
    except FileNotFoundError:
        logger.error(f"Archivo de consultas no encontrado: {args.queries_file}")
        sys.exit(1)

    # Asegurarse de que los directorios de resultados existan
    os.makedirs(os.path.dirname(args.output_metrics_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_results_csv), exist_ok=True)

    # Obtener ResourceManager
    try:
        resource_manager = ResourceManager()
    except Exception as e:
        logger.error(f"Error al inicializar ResourceManager: {e}")
        sys.exit(1)

    # Iniciar monitoreo
    monitoring_active = True
    metrics_data = []
    monitor_thread = threading.Thread(target=monitor_resources_query, args=(resource_manager, args.monitor_interval, args.output_metrics_csv), daemon=True)
    monitor_thread.start()

    # Preparar cola y resultados
    query_queue = Queue()
    query_results = []
    queries_processed = 0
    queries_succeeded = 0
    queries_failed = 0

    # Crear y iniciar hilos trabajadores
    workers = []
    for _ in range(args.num_threads):
        worker = threading.Thread(target=query_worker, args=(query_queue, args.db_index, args.session_id, query_results), daemon=True)
        worker.start()
        workers.append(worker)

    # Llenar la cola de consultas
    test_start_time = time.time()
    queries_sent = 0
    if args.duration:
        logger.info(f"Enviando consultas durante {args.duration} segundos...")
        while time.time() - test_start_time < args.duration:
            query_queue.put(random.choice(queries))
            queries_sent += 1
            time.sleep(0.01) # Pequeño delay para no llenar la cola instantáneamente
    else:
        logger.info(f"Enviando {args.num_queries} consultas...")
        for i in range(args.num_queries):
            query_queue.put(random.choice(queries))
            queries_sent = i + 1

    logger.info(f"Se enviaron {queries_sent} consultas a la cola.")

    # Esperar a que la cola se vacíe
    logger.info("Esperando a que todas las consultas sean procesadas...")
    query_queue.join()
    logger.info("Todas las consultas procesadas.")

    # Señalizar a los workers para que terminen
    for _ in range(args.num_threads):
        query_queue.put(None)
    
    # Esperar a que los workers terminen
    for worker in workers:
        worker.join(timeout=10)
        if worker.is_alive():
            logger.warning(f"Worker thread {worker.name} no finalizó limpiamente.")
            
    test_end_time = time.time()
    test_duration = test_end_time - test_start_time

    # Detener monitoreo
    monitoring_active = False
    if monitor_thread.is_alive():
        monitor_thread.join(timeout=args.monitor_interval * 2)
        if monitor_thread.is_alive():
            logger.warning("Hilo de monitoreo de consultas no finalizó limpiamente.")

    # Guardar resultados de consultas
    logger.info("Guardando resultados de consultas...")
    if query_results:
        try:
            with open(args.output_results_csv, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames_results = query_results[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames_results)
                writer.writeheader()
                writer.writerows(query_results)
            logger.info(f"Resultados de consulta guardados en: {args.output_results_csv}")
        except Exception as e:
            logger.error(f"Error al guardar los resultados de consulta: {e}")
    else:
        logger.warning("No se registraron resultados de consultas.")

    # Imprimir resumen
    logger.info("Resumen de la prueba de carga de consultas:")
    logger.info(f"- Duración total: {test_duration:.2f} segundos")
    logger.info(f"- Hilos concurrentes: {args.num_threads}")
    logger.info(f"- Consultas enviadas: {queries_sent}")
    logger.info(f"- Consultas procesadas: {queries_processed}")
    logger.info(f"- Consultas exitosas: {queries_succeeded}")
    logger.info(f"- Consultas fallidas: {queries_failed}")
    if queries_processed > 0:
        avg_duration = sum(r['duration_ms'] for r in query_results if r['success']) / queries_succeeded if queries_succeeded else 0
        logger.info(f"- Tiempo promedio por consulta exitosa: {avg_duration:.2f} ms")
        qps = queries_processed / test_duration if test_duration > 0 else 0
        logger.info(f"- Consultas por segundo (QPS): {qps:.2f}")

    logger.info("Prueba de carga de consultas finalizada.")
    sys.exit(0)

if __name__ == "__main__":
    main() 