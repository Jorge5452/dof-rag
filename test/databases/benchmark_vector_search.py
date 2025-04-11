#!/usr/bin/env python
"""
Benchmark para evaluar el rendimiento de búsquedas vectoriales en diferentes bases de datos.

Este script ejecuta una serie de consultas vectoriales en bases de datos SQLite y/o DuckDB
y mide los tiempos de ejecución para comparar el rendimiento. También genera gráficos 
comparativos si lo solicita.

Uso:
    python benchmark_vector_search.py --sqlite path/to/sqlite.db --duckdb path/to/duckdb.db 
    --queries 50 --dim 384 --plot results.png

Antes de usar este script, asegúrese de tener bases de datos pobladas con datos de prueba
ejecutando primero create_test_data.py.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate

# Añadir el directorio raíz al path para permitir importaciones relativas
sys.path.insert(0, str(Path(__file__).parents[2]))

from modulos.databases.implementaciones.sqlite import SQLiteVectorialDatabase
try:
    from modulos.databases.implementaciones.duckdb import DuckDBVectorialDatabase
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    print("DuckDB no está disponible. Instalarlo con: pip install duckdb")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_random_query(dim: int = 384) -> list:
    """Genera un vector de consulta aleatorio normalizado."""
    # Generar un vector aleatorio
    query = np.random.rand(dim).astype(np.float32)
    # Normalizar el vector (para simular embeddings realistas)
    norm = np.linalg.norm(query)
    if norm > 0:
        query = query / norm
    return query.tolist()

def benchmark_database(db_instance, num_queries: int = 10, embedding_dim: int = 384) -> dict:
    """
    Realiza pruebas de rendimiento en la base de datos.
    
    Args:
        db_instance: Instancia conectada de la base de datos vectorial
        num_queries: Número de consultas a realizar
        embedding_dim: Dimensión de los embeddings
        
    Returns:
        Diccionario con los resultados del benchmark
    """
    results = {
        "total_time": 0,
        "avg_time": 0,
        "min_time": float('inf'),
        "max_time": 0,
        "num_results": [],
        "times": []
    }
    
    logger.info(f"Iniciando benchmark con {num_queries} consultas...")
    
    for i in range(num_queries):
        # Generar consulta aleatoria
        query_embedding = generate_random_query(embedding_dim)
        
        # Medir tiempo de búsqueda
        start_time = time.time()
        search_results = db_instance.vector_search(query_embedding, n_results=5)
        end_time = time.time()
        
        # Calcular tiempo de ejecución
        exec_time = end_time - start_time
        
        # Actualizar estadísticas
        results["total_time"] += exec_time
        results["min_time"] = min(results["min_time"], exec_time)
        results["max_time"] = max(results["max_time"], exec_time)
        results["num_results"].append(len(search_results))
        results["times"].append(exec_time)
        
        logger.info(f"Consulta {i+1}/{num_queries}: {exec_time:.4f}s, {len(search_results)} resultados")
    
    # Calcular promedios
    if num_queries > 0:
        results["avg_time"] = results["total_time"] / num_queries
    
    return results

def run_benchmark(db_type: str, db_path: str, queries: int = 10, embedding_dim: int = 384) -> dict:
    """
    Ejecuta un benchmark completo para el tipo de base de datos especificado.
    
    Args:
        db_type: Tipo de base de datos ('sqlite' o 'duckdb')
        db_path: Ruta a la base de datos
        queries: Número de consultas a realizar
        embedding_dim: Dimensión de los embeddings
        
    Returns:
        Resultados del benchmark
    """
    logger.info(f"Preparando benchmark para {db_type} en {db_path}")
    
    # Verificar si el archivo existe
    if not os.path.exists(db_path):
        logger.error(f"La base de datos {db_path} no existe")
        return None
    
    # Inicializar y conectar la base de datos
    try:
        if db_type.lower() == 'sqlite':
            db = SQLiteVectorialDatabase(embedding_dim=embedding_dim)
        elif db_type.lower() == 'duckdb':
            if not DUCKDB_AVAILABLE:
                logger.error("DuckDB no está disponible")
                return None
            db = DuckDBVectorialDatabase(embedding_dim=embedding_dim)
        else:
            logger.error(f"Tipo de base de datos no soportado: {db_type}")
            return None
        
        # Conectar a la base de datos
        db.connect(db_path)
        
        # Ejecutar benchmark
        results = benchmark_database(db, queries, embedding_dim)
        results["db_type"] = db_type
        results["db_path"] = db_path
        
        # Cerrar conexión
        db.close_connection()
        
        return results
        
    except Exception as e:
        logger.error(f"Error durante el benchmark: {e}")
        return None

def print_results(results: dict) -> None:
    """Imprime los resultados del benchmark en formato tabular."""
    if not results:
        print("No hay resultados para mostrar")
        return
    
    print(f"\nRESULTADOS DEL BENCHMARK PARA {results['db_type'].upper()}")
    print("-" * 50)
    
    table = [
        ["Base de datos", results["db_type"]],
        ["Ruta", results["db_path"]],
        ["Tiempo total", f"{results['total_time']:.4f}s"],
        ["Tiempo promedio", f"{results['avg_time']:.4f}s"],
        ["Tiempo mínimo", f"{results['min_time']:.4f}s"],
        ["Tiempo máximo", f"{results['max_time']:.4f}s"],
        ["Promedio de resultados", f"{sum(results['num_results'])/len(results['num_results']):.2f}"]
    ]
    
    print(tabulate(table, tablefmt="pretty"))

def plot_results(results_list: list, output_file: str = None) -> None:
    """
    Genera gráficos con los resultados del benchmark.
    
    Args:
        results_list: Lista de resultados de benchmark
        output_file: Ruta donde guardar el gráfico (opcional)
    """
    if not results_list:
        logger.error("No hay resultados para graficar")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Gráfico 1: Tiempos de ejecución por consulta
    plt.subplot(2, 1, 1)
    for results in results_list:
        db_type = results["db_type"]
        plt.plot(range(1, len(results["times"]) + 1), results["times"], 
                 marker='o', linestyle='-', label=f"{db_type}")
    
    plt.title('Tiempos de ejecución por consulta')
    plt.xlabel('Número de consulta')
    plt.ylabel('Tiempo (segundos)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Gráfico 2: Comparación de tiempos promedio
    plt.subplot(2, 1, 2)
    db_types = [r["db_type"] for r in results_list]
    avg_times = [r["avg_time"] for r in results_list]
    
    bars = plt.bar(db_types, avg_times, color=['#1f77b4', '#ff7f0e'])
    plt.title('Comparación de tiempos promedio')
    plt.xlabel('Base de datos')
    plt.ylabel('Tiempo promedio (segundos)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Añadir etiquetas en las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.4f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Guardar o mostrar el gráfico
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Gráfico guardado en {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark de bases de datos vectoriales",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sqlite", type=str, default=None,
                        help="Ruta a la base de datos SQLite")
    parser.add_argument("--duckdb", type=str, default=None,
                        help="Ruta a la base de datos DuckDB")
    parser.add_argument("--queries", type=int, default=10,
                        help="Número de consultas a realizar")
    parser.add_argument("--dim", type=int, default=384,
                        help="Dimensión de los embeddings")
    parser.add_argument("--plot", type=str, default=None,
                        help="Ruta donde guardar el gráfico de resultados")
    
    args = parser.parse_args()
    
    # Verificar que al menos se proporciona una base de datos
    if not args.sqlite and not args.duckdb:
        parser.error("Debe especificar al menos una base de datos (--sqlite o --duckdb)")
    
    results_list = []
    
    # Ejecutar benchmark para SQLite si se proporciona
    if args.sqlite:
        sqlite_results = run_benchmark('sqlite', args.sqlite, args.queries, args.dim)
        if sqlite_results:
            results_list.append(sqlite_results)
            print_results(sqlite_results)
    
    # Ejecutar benchmark para DuckDB si se proporciona
    if args.duckdb:
        if not DUCKDB_AVAILABLE:
            logger.error("DuckDB no está disponible. Instálelo primero.")
        else:
            duckdb_results = run_benchmark('duckdb', args.duckdb, args.queries, args.dim)
            if duckdb_results:
                results_list.append(duckdb_results)
                print_results(duckdb_results)
    
    # Generar gráficos si hay resultados
    if len(results_list) > 0:
        plot_results(results_list, args.plot)
