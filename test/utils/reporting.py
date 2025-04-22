"""
Funciones para generar informes de resultados de pruebas.

Este módulo contiene funciones para formatear y guardar
informes de resultados de pruebas en diferentes formatos.
"""
import os
from pathlib import Path
from datetime import datetime
import time
import unittest
import json

from test.utils.constants import (
    SUMMARY_HEADER, SUMMARY_SEPARATOR, DATETIME_FORMAT,
    FILENAME_DATETIME_FORMAT
)

def format_failures_and_errors(failures=None, errors=None, max_length=500):
    """
    Formatea los fallos y errores para los informes.
    
    Args:
        failures: Lista de fallos (opcional)
        errors: Lista de errores (opcional)
        max_length: Longitud máxima del mensaje de error a incluir
        
    Returns:
        str: Texto formateado con los fallos y errores
    """
    report = ""
    
    if failures:
        report += "FALLOS:\n"
        report += SUMMARY_SEPARATOR + "\n"
        for test, error in failures:
            report += f"- {test}\n"
            error_text = str(error)
            if len(error_text) > max_length:
                error_text = error_text[:max_length] + "...\n"
            report += f"  {error_text}\n\n"
    
    if errors:
        report += "ERRORES:\n"
        report += SUMMARY_SEPARATOR + "\n"
        for test, error in errors:
            report += f"- {test}\n"
            error_text = str(error)
            if len(error_text) > max_length:
                error_text = error_text[:max_length] + "...\n"
            report += f"  {error_text}\n\n"
    
    return report

def generate_test_summary(result, test_type, elapsed_time=None):
    """
    Genera un informe de resumen de las pruebas ejecutadas.
    
    Args:
        result: Resultado de unittest.TestRunner
        test_type: Tipo de prueba (databases, chunkers, etc.)
        elapsed_time: Tiempo transcurrido en segundos (opcional)
        
    Returns:
        str: Texto con el resumen de las pruebas
    """
    # Título del informe
    report = f"RESUMEN DE PRUEBAS DE {test_type.upper()}\n"
    report += SUMMARY_HEADER + "\n\n"
    
    # Fecha y tiempo
    timestamp = datetime.now().strftime(DATETIME_FORMAT)
    report += f"Fecha y hora: {timestamp}\n"
    
    if elapsed_time is not None:
        report += f"Tiempo de ejecución: {elapsed_time:.2f} segundos\n"
    
    report += "\n"
    
    # Estadísticas generales
    success_count = result.testsRun - len(result.failures) - len(result.errors)
    report += f"Total de pruebas ejecutadas: {result.testsRun}\n"
    report += f"Pruebas exitosas: {success_count}\n"
    report += f"Pruebas fallidas: {len(result.failures)}\n"
    report += f"Errores: {len(result.errors)}\n\n"
    
    # Detalles de fallos y errores
    report += format_failures_and_errors(result.failures, result.errors)
    
    return report

def save_test_report(report_text, output_dir, filename=None, test_type=None):
    """
    Guarda el informe de resultados en un archivo.
    
    Args:
        report_text: Texto del informe
        output_dir: Directorio donde guardar el informe
        filename: Nombre del archivo (opcional)
        test_type: Tipo de prueba (opcional, para generar nombre de archivo)
        
    Returns:
        Path: Ruta al archivo de informe
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar nombre de archivo si no se proporciona
    if filename is None:
        timestamp = datetime.now().strftime(FILENAME_DATETIME_FORMAT)
        type_prefix = f"{test_type}_" if test_type else ""
        filename = f"{type_prefix}test_summary_{timestamp}.txt"
    
    # Asegurar que la extensión sea .txt
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    # Guardar el informe
    file_path = output_dir / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return file_path

def save_test_results_json(result, output_dir, test_type=None, metadata=None):
    """
    Guarda los resultados de las pruebas en formato JSON.
    
    Args:
        result: Resultado de unittest.TestRunner
        output_dir: Directorio donde guardar el informe
        test_type: Tipo de prueba
        metadata: Datos adicionales sobre la prueba (diccionario)
        
    Returns:
        Path: Ruta al archivo JSON
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar nombre de archivo
    timestamp = datetime.now().strftime(FILENAME_DATETIME_FORMAT)
    type_prefix = f"{test_type}_" if test_type else ""
    filename = f"{type_prefix}test_results_{timestamp}.json"
    
    # Crear diccionario de resultados
    success_count = result.testsRun - len(result.failures) - len(result.errors)
    
    results_data = {
        "test_type": test_type,
        "timestamp": datetime.now().strftime(DATETIME_FORMAT),
        "total_tests": result.testsRun,
        "success_count": success_count,
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "was_successful": result.wasSuccessful(),
        "failures": [
            {"test": str(test), "message": str(error)[:500]}
            for test, error in result.failures
        ],
        "errors": [
            {"test": str(test), "message": str(error)[:500]}
            for test, error in result.errors
        ]
    }
    
    # Añadir metadatos si se proporcionan
    if metadata:
        results_data["metadata"] = metadata
    
    # Guardar en formato JSON
    file_path = output_dir / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    
    return file_path

def run_tests_with_reporting(
    test_suite, test_type, output_dir=None, verbosity=2, metadata=None
):
    """
    Ejecuta pruebas y genera informes de resultados.
    
    Args:
        test_suite: Suite de pruebas unittest
        test_type: Tipo de prueba (databases, chunkers, etc.)
        output_dir: Directorio de salida (opcional)
        verbosity: Nivel de detalle (1-3)
        metadata: Metadatos adicionales (opcional)
        
    Returns:
        tuple: (resultado de las pruebas, ruta al informe TXT, ruta al informe JSON)
    """
    from test.utils.environment import get_test_result_path
    
    # Verificar que hay pruebas en la suite
    if test_suite.countTestCases() == 0:
        print(f"ADVERTENCIA: No se encontraron pruebas para: {test_type}")
        return None, None, None
    
    # Determinar directorio de salida
    if output_dir is None:
        output_dir = get_test_result_path(test_type)
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Configurar el runner con verbosidad adecuada
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    # Medir el tiempo de ejecución
    start_time = time.time()
    
    # Ejecutar las pruebas
    print(f"\nEjecutando pruebas de {test_type}...")
    result = runner.run(test_suite)
    
    # Calcular tiempo transcurrido
    elapsed_time = time.time() - start_time
    
    # Generar informe de texto
    report_text = generate_test_summary(result, test_type, elapsed_time)
    txt_path = save_test_report(report_text, output_dir, test_type=test_type)
    
    # Guardar resultados en formato JSON
    json_path = save_test_results_json(
        result, output_dir, test_type, 
        metadata={"elapsed_time": elapsed_time, **(metadata or {})}
    )
    
    # Mostrar resumen en consola
    print("\n" + SUMMARY_HEADER)
    print(f"RESULTADOS: {result.testsRun} pruebas ejecutadas en {elapsed_time:.2f} segundos")
    print(f"- Éxitos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"- Fallos: {len(result.failures)}")
    print(f"- Errores: {len(result.errors)}")
    print(f"Resumen guardado en: {txt_path}")
    print(f"Resultados JSON en: {json_path}")
    print(SUMMARY_HEADER)
    
    return result, txt_path, json_path 