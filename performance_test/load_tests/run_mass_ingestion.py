import argparse
import time
import logging
import csv
import threading
import os
import sys
from datetime import datetime

# Asegurarse de que los módulos del proyecto están en el path
# Esto podría requerir añadir la raíz del proyecto al PYTHONPATH
# o ajustar el sys.path aquí.
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
try:
    from main import process_documents
    from modulos.resource_management.resource_manager import ResourceManager
    from modulos.utils.logging_utils import setup_logging
except ImportError as e:
    print(f"Error al importar módulos: {e}")
    print("Asegúrate de que el script se ejecuta desde la raíz del proyecto o ajusta el PYTHONPATH.")
    sys.exit(1)

# Configurar logging básico para el script de prueba
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Variable global para controlar el hilo de monitoreo
monitoring_active = False
metrics_data = []

def monitor_resources(resource_manager, interval, output_csv):
    """Función ejecutada en un hilo para recolectar métricas periódicamente."""
    global monitoring_active, metrics_data
    logger.info("Inicio del monitoreo de recursos...")
    start_time = time.time()
    
    # Encabezados del CSV
    fieldnames = [
        'timestamp', 'elapsed_seconds', 'system_cpu_percent', 'system_memory_percent',
        'system_memory_used_gb', 'system_memory_total_gb', 'process_cpu_percent',
        'process_memory_mb', 'active_sessions_rag', 'active_embedding_models'
        # Añadir más métricas de resource_manager.metrics si es necesario
    ]
    metrics_data.append(fieldnames) # Añadir encabezado a los datos en memoria

    while monitoring_active:
        try:
            current_time = time.time()
            elapsed = current_time - start_time
            # Actualizar métricas llamando al método del ResourceManager
            # (El propio RM actualiza internamente con su thread, pero podemos forzar una lectura aquí si queremos)
            # resource_manager.update_metrics() # Opcional, RM ya debería tener métricas actualizadas
            
            metrics = resource_manager.metrics # Leer las métricas actuales
            
            row = {
                'timestamp': datetime.now().isoformat(),
                'elapsed_seconds': round(elapsed, 2),
                'system_cpu_percent': metrics.get('system_cpu_percent', 'N/A'),
                'system_memory_percent': metrics.get('system_memory_percent', 'N/A'),
                'system_memory_used_gb': metrics.get('system_memory_used_gb', 'N/A'),
                'system_memory_total_gb': metrics.get('system_memory_total_gb', 'N/A'),
                'process_cpu_percent': metrics.get('process_cpu_percent', 'N/A'),
                'process_memory_mb': metrics.get('process_memory_mb', 'N/A'),
                'active_sessions_rag': metrics.get('active_sessions_rag', 'N/A'),
                'active_embedding_models': metrics.get('active_embedding_models', 'N/A')
            }
            metrics_data.append([row.get(field, 'N/A') for field in fieldnames]) # Guardar como lista
            
            # Esperar el intervalo especificado
            time.sleep(interval)
        except Exception as e:
            logger.error(f"Error en el hilo de monitoreo: {e}")
            # Continuar monitoreando si es posible
            time.sleep(interval)

    logger.info("Fin del monitoreo de recursos.")
    # Escribir datos al CSV al finalizar
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(metrics_data)
        logger.info(f"Métricas guardadas en: {output_csv}")
    except Exception as e:
        logger.error(f"Error al guardar las métricas en CSV: {e}")

def main():
    global monitoring_active, metrics_data

    parser = argparse.ArgumentParser(description="Script de prueba de carga para ingesta masiva de documentos.")
    parser.add_argument("--corpus-dir", required=True, help="Directorio que contiene los archivos Markdown a ingerir.")
    parser.add_argument("--session-name", default="mass_ingestion_test", help="Nombre para la sesión de ingesta.")
    parser.add_argument("--monitor-interval", type=float, default=5.0, help="Intervalo (segundos) para el monitoreo de recursos.")
    parser.add_argument("--output-csv", default="../results/mass_ingestion_metrics.csv", help="Archivo CSV para guardar las métricas.")
    parser.add_argument("--debug", action="store_true", help="Activar logging DEBUG.")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        # Configurar logging del sistema RAG a DEBUG también si es necesario
        # setup_logging(level=logging.DEBUG)

    logger.info(f"Iniciando prueba de carga de ingesta masiva desde: {args.corpus_dir}")
    logger.info(f"Nombre de sesión: {args.session_name}")
    logger.info(f"Intervalo de monitoreo: {args.monitor_interval}s")
    logger.info(f"Archivo de salida de métricas: {args.output_csv}")

    # Verificar que el directorio del corpus existe
    if not os.path.isdir(args.corpus_dir):
        logger.error(f"El directorio del corpus especificado no existe: {args.corpus_dir}")
        sys.exit(1)

    # Asegurarse de que el directorio de resultados exista
    results_dir = os.path.dirname(args.output_csv)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    # Obtener instancia de ResourceManager
    try:
        resource_manager = ResourceManager() # Usar Singleton
    except Exception as e:
        logger.error(f"Error al inicializar ResourceManager: {e}")
        sys.exit(1)

    # Iniciar el monitoreo en un hilo separado
    monitoring_active = True
    metrics_data = [] # Reiniciar datos
    monitor_thread = threading.Thread(target=monitor_resources, args=(resource_manager, args.monitor_interval, args.output_csv), daemon=True)
    monitor_thread.start()

    # Ejecutar la ingesta
    ingestion_start_time = time.time()
    success = True
    try:
        # Llamar a la función principal de ingesta del sistema RAG
        process_documents(file_path=args.corpus_dir, session_name=args.session_name)
        logger.info("Proceso de ingesta completado.")
    except Exception as e:
        logger.error(f"Error durante la ingesta masiva: {e}", exc_info=True)
        success = False
    finally:
        ingestion_end_time = time.time()
        ingestion_duration = ingestion_end_time - ingestion_start_time
        logger.info(f"Duración total de la ingesta: {ingestion_duration:.2f} segundos")

        # Detener el monitoreo
        monitoring_active = False
        if monitor_thread.is_alive():
            logger.info("Esperando a que el hilo de monitoreo finalice...")
            monitor_thread.join(timeout=args.monitor_interval * 2) # Esperar un poco más
            if monitor_thread.is_alive():
                 logger.warning("El hilo de monitoreo no finalizó limpiamente.")

        # Asegurarse de que ResourceManager se cierre limpiamente (opcional aquí, se podría hacer al final de todo)
        # resource_manager.shutdown()

    if success:
        logger.info("Prueba de carga de ingesta finalizada con éxito.")
        sys.exit(0)
    else:
        logger.error("Prueba de carga de ingesta finalizada con errores.")
        sys.exit(1)

if __name__ == "__main__":
    main() 