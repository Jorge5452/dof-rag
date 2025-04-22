"""
Funciones para descubrir y ejecutar pruebas.

Este módulo contiene funciones para descubrir pruebas
en directorios específicos y organizarlas para su ejecución.
"""
import os
import sys
import unittest
from pathlib import Path

from test.utils.constants import (
    TEST_DIR, TEST_PATTERN, DATABASE_TEST_PATTERN,
    CHUNKER_TEST_PATTERN, CLIENT_TEST_PATTERN, EMBEDDING_TEST_PATTERN
)

def discover_tests(start_dir, pattern=None, test_type=None):
    """
    Descubre pruebas en el directorio especificado con el patrón dado.
    
    Args:
        start_dir: Directorio donde buscar pruebas
        pattern: Patrón de archivos a buscar (opcional)
        test_type: Tipo de prueba para determinar el patrón (opcional)
        
    Returns:
        unittest.TestSuite: Suite con las pruebas descubiertas
    """
    # Asegurar que el directorio es un Path
    start_dir = Path(start_dir)
    
    # Determinar el patrón según el tipo de prueba si no se especifica
    if pattern is None and test_type is not None:
        patterns = {
            "databases": DATABASE_TEST_PATTERN,
            "chunkers": CHUNKER_TEST_PATTERN,
            "clients": CLIENT_TEST_PATTERN,
            "embeddings": EMBEDDING_TEST_PATTERN
        }
        pattern = patterns.get(test_type, TEST_PATTERN)
    
    # Usar patrón predeterminado si no se especifica
    if pattern is None:
        pattern = TEST_PATTERN
    
    # Verificar que el directorio existe
    if not start_dir.exists():
        print(f"ERROR: Directorio de pruebas no encontrado: {start_dir}")
        return unittest.TestSuite()
    
    # Descubrir las pruebas
    loader = unittest.TestLoader()
    return loader.discover(str(start_dir), pattern=pattern)

def filter_tests_by_name(suite, test_names):
    """
    Filtra una suite de pruebas por nombres específicos.
    
    Args:
        suite: Suite de pruebas completa
        test_names: Lista de nombres de pruebas a incluir
        
    Returns:
        unittest.TestSuite: Suite filtrada con las pruebas seleccionadas
    """
    if not test_names:
        return suite
    
    # Convertir los nombres a un conjunto para búsqueda rápida
    test_names_set = set(test_names)
    
    # Función recursiva para filtrar suites anidadas
    def filter_suite(suite):
        filtered_suite = unittest.TestSuite()
        
        for test in suite:
            if isinstance(test, unittest.TestSuite):
                # Recursivamente filtrar suites anidadas
                filtered_subsuite = filter_suite(test)
                if filtered_subsuite.countTestCases() > 0:
                    filtered_suite.addTest(filtered_subsuite)
            else:
                # Comprobar si el nombre de la prueba está en el conjunto
                test_id = test.id()
                test_name = test_id.split('.')[-1]
                class_name = test_id.split('.')[-2]
                
                # Incluir si coincide con el nombre de la prueba o de la clase
                if (test_name in test_names_set or 
                    class_name in test_names_set or 
                    f"{class_name}.{test_name}" in test_names_set):
                    filtered_suite.addTest(test)
        
        return filtered_suite
    
    return filter_suite(suite)

def get_test_suite_by_type(test_type, filter_pattern=None, specific_tests=None):
    """
    Obtiene una suite de pruebas según el tipo especificado.
    
    Args:
        test_type: Tipo de prueba (databases, chunkers, clients, etc.)
        filter_pattern: Patrón para filtrar nombres de archivo (opcional)
        specific_tests: Lista de nombres de pruebas específicas (opcional)
        
    Returns:
        unittest.TestSuite: Suite con las pruebas seleccionadas
    """
    # Determinar el directorio de pruebas según el tipo
    type_to_dir = {
        "databases": TEST_DIR / "databases",
        "chunkers": TEST_DIR / "chunkers",
        "clients": TEST_DIR / "clients",
        "embeddings": TEST_DIR / "embeddings",
        "doc_processor": TEST_DIR / "doc_processor",
        "rag": TEST_DIR / "rag",
        "session_manager": TEST_DIR / "session_manager",
        "view_chunks": TEST_DIR / "view_chunks",
        "integration": TEST_DIR / "integration"
    }
    
    start_dir = type_to_dir.get(test_type)
    
    # Si no se encuentra el directorio para el tipo, usar el directorio de test
    if start_dir is None or not start_dir.exists():
        print(f"ADVERTENCIA: No se encontró directorio para pruebas de tipo '{test_type}'")
        print(f"Buscando pruebas con patrón 'test_{test_type}_*.py' en el directorio principal")
        start_dir = TEST_DIR
        pattern = f"test_{test_type}_*.py" if filter_pattern is None else filter_pattern
    else:
        pattern = filter_pattern
    
    # Descubrir las pruebas
    suite = discover_tests(start_dir, pattern, test_type)
    
    # Filtrar por pruebas específicas si se solicita
    if specific_tests:
        suite = filter_tests_by_name(suite, specific_tests)
    
    return suite

def list_available_tests(test_type=None):
    """
    Lista las pruebas disponibles para un tipo específico o todas.
    
    Args:
        test_type: Tipo de prueba (opcional)
        
    Returns:
        dict: Diccionario con los tipos de prueba y sus pruebas disponibles
    """
    available_tests = {}
    
    # Si se especifica un tipo, listar solo ese tipo
    if test_type:
        suite = get_test_suite_by_type(test_type)
        tests = []
        
        def extract_test_info(suite):
            for test in suite:
                if isinstance(test, unittest.TestSuite):
                    extract_test_info(test)
                else:
                    test_id = test.id()
                    tests.append(test_id)
        
        extract_test_info(suite)
        available_tests[test_type] = tests
    else:
        # Listar todos los tipos de prueba
        test_types = [
            "databases", "chunkers", "clients", "embeddings",
            "doc_processor", "rag", "session_manager", "view_chunks", "integration"
        ]
        
        for t_type in test_types:
            suite = get_test_suite_by_type(t_type)
            tests = []
            
            def extract_test_info(suite):
                for test in suite:
                    if isinstance(test, unittest.TestSuite):
                        extract_test_info(test)
                    else:
                        test_id = test.id()
                        tests.append(test_id)
            
            extract_test_info(suite)
            
            if tests:
                available_tests[t_type] = tests
    
    return available_tests 