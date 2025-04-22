"""
Constantes compartidas para el sistema de pruebas.
"""
from pathlib import Path

# Directorios base
TEST_DIR = Path(__file__).parent.parent
PROJECT_ROOT = TEST_DIR.parent
RESULTS_DIR = TEST_DIR / "results"

# Subdirectorios de resultados estandarizados
DATABASE_RESULTS_DIR = RESULTS_DIR / "database_tests"
CHUNKER_RESULTS_DIR = RESULTS_DIR / "chunker_tests"
CLIENT_RESULTS_DIR = RESULTS_DIR / "client_tests"
EMBEDDING_RESULTS_DIR = RESULTS_DIR / "embedding_tests"
INTEGRATION_RESULTS_DIR = RESULTS_DIR / "integration_tests"
DOC_PROCESSOR_RESULTS_DIR = RESULTS_DIR / "doc_processor_tests"
RAG_RESULTS_DIR = RESULTS_DIR / "rag_tests"
SESSION_MANAGER_RESULTS_DIR = RESULTS_DIR / "session_manager_tests"
VIEW_CHUNKS_RESULTS_DIR = RESULTS_DIR / "view_chunks_tests"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

# Patrones de descubrimiento de pruebas
TEST_PATTERN = "test_*.py"
DATABASE_TEST_PATTERN = "test_*_database.py"
CHUNKER_TEST_PATTERN = "test_*_chunker.py"
CLIENT_TEST_PATTERN = "test_*.py"  # En el directorio clients
EMBEDDING_TEST_PATTERN = "test_embedding_*.py"

# Formatos de fecha y tiempo
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
FILENAME_DATETIME_FORMAT = "%Y%m%d_%H%M%S"

# Configuración de informes
SUMMARY_HEADER = "=" * 80
SUMMARY_SEPARATOR = "-" * 80 