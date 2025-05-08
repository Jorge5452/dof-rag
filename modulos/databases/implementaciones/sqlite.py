import os
import sqlite3
import numpy as np
import logging
import struct
import time
import math
from typing import List, Dict, Any, Optional, Tuple
import json
import uuid

try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
    SQLITE_VEC_VERSION = None
except ImportError:
    SQLITE_VEC_AVAILABLE = False
    SQLITE_VEC_VERSION = None

from modulos.databases.VectorialDatabase import VectorialDatabase

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SQLiteVectorialDatabase(VectorialDatabase):
    """
    Implementación de una base de datos vectorial usando SQLite.
    Proporciona métodos para inserción, búsqueda y recuperación de documentos y chunks,
    además de soporte para búsqueda vectorial eficiente.
    
    Esta implementación utiliza la extensión sqlite-vss si está disponible para mejorar
    el rendimiento de las búsquedas vectoriales.
    """
    
    def __init__(self, embedding_dim: int = None):
        """
        Inicializa la base de datos SQLite.
        
        Args:
            embedding_dim: Dimensión de los embeddings.
        """
        super().__init__()  # Asegurarse de inicializar la clase base
        logger.debug("Inicializando SQLiteVectorialDatabase")
        
        self._conn = None
        self._cursor = None
        self._db_path = None
        
        # Establecer la dimensión de los embeddings
        if embedding_dim is None:
            raise ValueError("Se requiere la dimensión de embeddings para inicializar la base de datos SQLite")
        self._embedding_dim = embedding_dim
        
        # Configuración para extensión vectorial
        self._extension_loaded = False
        self._extension_path = None
        self._vector_table_name = 'chunks_vectors'
        self._use_vector_extension = True  # Por defecto intentar usar la extensión
        
        # Inicializar metadatos
        self._metadata = {}
        
        # Inicializar umbral de similitud
        self._similarity_threshold = 0.3  # Valor por defecto
    
    def connect(self, db_path: str) -> None:
        """
        Establece la conexión a la base de datos SQLite.
        
        Parámetros:
            db_path (str): Ruta al archivo de la base de datos SQLite.
        
        Raises:
            sqlite3.Error: Si ocurre un error al conectar con la base de datos.
        """
        try:
            # Asegurar que el directorio existe
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            
            # Guardar la ruta para referencia
            self._db_path = db_path
            
            # Establecer conexión
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            
            # Configurar la conexión para devolver filas como diccionarios
            self._conn.row_factory = sqlite3.Row
            
            # Crear cursor
            self._cursor = self._conn.cursor()
            
            # Habilitar claves foráneas
            self._cursor.execute("PRAGMA foreign_keys = ON;")
            
            # Actualizar la configuración desde config.py
            from config import config
            sqlite_config = config.get_database_config().get('sqlite', {})
            self._similarity_threshold = sqlite_config.get('similarity_threshold', 0.3)
            logger.debug(f"Umbral de similitud establecido en: {self._similarity_threshold}")
            
            # Cargar extensiones vectoriales
            self._extension_loaded = self.load_extensions()
            
            # Si la extensión no se cargó, crear funciones manuales
            if not self._extension_loaded:
                self._create_manual_vector_functions()
            
            # Crear el esquema si no existe
            self.create_schema()
            
            logger.info(f"Conexión establecida correctamente a la base de datos SQLite: {db_path}")
        
        except sqlite3.Error as e:
            logger.error(f"Error al conectar a la base de datos SQLite: {e}")
            if self._conn:
                self._conn.close()
            raise
    
    def _create_manual_vector_functions(self):
        """
        Crea funciones personalizadas en SQLite para operaciones vectoriales cuando 
        la extensión sqlite-vec no está disponible.
        """
        logger.info("Creando funciones vectoriales manuales en SQLite...")
        
        def unpack_vector(blob):
            """Desempaqueta un BLOB en un vector numpy"""
            if not blob:
                return None
            try:
                # Determinar la dimensión a partir del tamaño del blob
                dim = len(blob) // 4  # 4 bytes por float (float32)
                fmt = f"{dim}f"  # formato para struct.unpack
                return np.array(struct.unpack(fmt, blob))
            except Exception as e:
                logger.error(f"Error al desempaquetar vector: {e}")
                return None
        
        def cosine_similarity(blob1, blob2):
            """Calcula la similitud coseno entre dos vectores almacenados como BLOBs"""
            vec1 = unpack_vector(blob1)
            vec2 = unpack_vector(blob2)
            
            if vec1 is None or vec2 is None:
                return 0.0
                
            # Calcular similitud coseno
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(dot_product / (norm1 * norm2))
            
        # Registrar las funciones en SQLite
        self._conn.create_function("vec_cosine_similarity", 2, cosine_similarity)
        logger.info("Funciones vectoriales manuales creadas exitosamente")
    
    def load_extensions(self) -> bool:
        """
        Carga la extensión sqlite-vec para operaciones vectoriales eficientes.
        
        Returns:
            bool: True si la extensión se cargó correctamente, False en caso contrario.
        """
        global SQLITE_VEC_VERSION
        
        if not self._use_vector_extension:
            logger.info("Extensión vectorial deshabilitada por configuración.")
            return False
            
        if not SQLITE_VEC_AVAILABLE:
            logger.warning("Paquete sqlite-vec no disponible. Intente instalarlo con: pip install sqlite-vec")
            self._use_vector_extension = False
            return False
        
        try:
            # Habilitar la carga de extensiones
            self._conn.enable_load_extension(True)
            
            # Método 1: Usando el módulo sqlite_vec para cargar la extensión
            try:
                sqlite_vec.load(self._conn)
                # Verificar la carga
                try:
                    vec_version = self._conn.execute("SELECT vec_version()").fetchone()[0]
                    SQLITE_VEC_VERSION = vec_version
                    logger.info(f"Extensión sqlite-vec cargada correctamente con método 1. Versión: {vec_version}")
                    
                    # Deshabilitar la carga de extensiones por seguridad
                    self._conn.enable_load_extension(False)
                    
                    # Guardar información de la extensión
                    self.store_metadata("vector_extension_version", vec_version)
                    self.store_metadata("vector_extension_enabled", "true")
                    
                    # Si la versión es v0.1.6, establecer tabla y funciones específicas
                    if vec_version == "v0.1.6":
                        self._vector_table_name = "vec_index"  # Nombre simple para la tabla
                        
                    return True
                except sqlite3.OperationalError:
                    # Si no podemos obtener la versión, es posible que la extensión no se haya cargado completamente
                    logger.warning("Extensión sqlite-vec se cargó pero la función vec_version() no está disponible")
                    
                    # Verificar si otras funciones como vec_cosine_similarity están disponibles
                    try:
                        # Crear vectores de prueba
                        test_vec1 = np.ones(5, dtype=np.float32).tobytes()
                        test_vec2 = np.ones(5, dtype=np.float32).tobytes()
                        
                        # Intentar usar vec_cosine_similarity
                        result = self._conn.execute("SELECT vec_cosine_similarity(?, ?)", (test_vec1, test_vec2)).fetchone()[0]
                        logger.info(f"Función vec_cosine_similarity disponible, resultado de prueba: {result}")
                        self.store_metadata("vector_extension_enabled", "true")
                        self.store_metadata("vector_extension_version", "unknown")
                        self._conn.enable_load_extension(False)
                        return True
                    except sqlite3.OperationalError as e:
                        logger.warning(f"La función vec_cosine_similarity no está disponible: {e}")
            except Exception as e:
                logger.warning(f"Error al cargar sqlite-vec con método 1: {e}")
            
            # Método 2: Intenta otra forma de cargar la extensión (directa con .so/.dll)
            try:
                # Obtiene la ruta de la biblioteca del módulo
                import os.path
                import sqlite_vec
                ext_path = os.path.dirname(os.path.abspath(sqlite_vec.__file__))
                
                # Intenta cargar en diferentes ubicaciones posibles
                possible_paths = [
                    os.path.join(ext_path, "sqlite_vec"),
                    os.path.join(ext_path, "sqlite_vec.so"),
                    os.path.join(ext_path, "sqlite_vec.dll"),
                    "sqlite_vec",
                ]
                
                for path in possible_paths:
                    try:
                        self._conn.enable_load_extension(True)
                        self._conn.load_extension(path)
                        logger.info(f"Extensión sqlite-vec cargada desde: {path}")
                        self.store_metadata("vector_extension_enabled", "true")
                        self.store_metadata("vector_extension_path", path)
                        self._conn.enable_load_extension(False)
                        return True
                    except Exception as e:
                        logger.debug(f"No se pudo cargar extensión desde {path}: {e}")
            except Exception as e:
                logger.warning(f"Error en método alternativo de carga: {e}")
                
            # Si llegamos aquí, no se pudo cargar la extensión
            logger.warning("No se pudo cargar la extensión sqlite-vec. Se utilizarán funciones vectoriales manuales.")
            self._use_vector_extension = False
            self.store_metadata("vector_extension_enabled", "false")
            
            # Deshabilitar la carga de extensiones por seguridad
            try:
                self._conn.enable_load_extension(False)
            except:
                pass
                
            return False
            
        except sqlite3.Error as e:
            logger.error(f"Error al gestionar extensiones en SQLite: {e}")
            self._use_vector_extension = False
            return False
    
    def close_connection(self) -> None:
        """
        Cierra la conexión a la base de datos SQLite.
        """
        if self._conn:
            try:
                self._conn.close()
                self._conn = None
                self._cursor = None
                logger.info("Conexión a la base de datos SQLite cerrada correctamente.")
            except sqlite3.Error as e:
                logger.error(f"Error al cerrar la conexión a la base de datos SQLite: {e}")
    
    def create_schema(self) -> None:
        """
        Crea el esquema de la base de datos si no existe.
        
        Crea las tablas documents y chunks con sus índices correspondientes.
        """
        try:
            # Tabla de documentos
            self._cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Índice para búsqueda por URL
            self._cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);
            """)
            
            # Tabla de chunks
            self._cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    document_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    header TEXT,
                    page TEXT,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );
            """)
            
            # Índice para buscar chunks por documento
            self._cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
            """)
            
            # Confirmar los cambios
            self._conn.commit()
            logger.info("Esquema de la base de datos creado correctamente.")
            
            # Crear índices vectoriales usando la dimensión fija
            if self._extension_loaded:
                self.create_vector_index()
        
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error(f"Error al crear el esquema de la base de datos: {e}")
            raise
    
    def create_vector_index(self, force_rebuild: bool = False) -> bool:
        """
        Crea o reconstruye el índice vectorial para optimizar las búsquedas.
        
        Utiliza la dimensión fija de embedding establecida en la inicialización.
        Compatible con sqlite-vec v0.1.6.
        
        Args:
            force_rebuild: Si es True, fuerza la reconstrucción del índice incluso si ya existe.
                                  
        Returns:
            bool: True si el índice se creó/reconstruyó correctamente, False en caso contrario.
        """
        if not self._extension_loaded:
            logger.warning("No se puede crear índice vectorial sin la extensión sqlite-vec")
            return False
            
        try:
            # Verificar si ya existe el índice vectorial
            self._cursor.execute(f"""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='{self._vector_table_name}';
            """)
            table_exists = self._cursor.fetchone() is not None
            
            if table_exists and not force_rebuild:
                logger.info(f"Índice vectorial '{self._vector_table_name}' ya existe. No se reconstruirá.")
                return True
                
            # Si existe y queremos reconstruir, eliminarlo primero
            if table_exists:
                self._cursor.execute(f"DROP TABLE IF EXISTS {self._vector_table_name}")
                logger.info(f"Índice vectorial '{self._vector_table_name}' eliminado para reconstrucción.")
            
            # Usar exactamente la sintaxis proporcionada en la documentación oficial
            sql = f"CREATE VIRTUAL TABLE {self._vector_table_name} USING vec0(embedding float[{self._embedding_dim}])"
            
            try:
                logger.info(f"Intentando crear índice vectorial con: {sql}")
                self._cursor.execute(sql)
                
                # Verificar si se creó correctamente
                self._cursor.execute(f"""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='{self._vector_table_name}';
                """)
                
                if self._cursor.fetchone() is not None:
                    logger.info(f"Índice vectorial '{self._vector_table_name}' creado correctamente.")
                    self._conn.commit()
                    return True
                else:
                    logger.warning(f"No se pudo verificar la creación del índice vectorial '{self._vector_table_name}'.")
                    return False
            except sqlite3.Error as e:
                logger.warning(f"Error al crear el índice vectorial: {e}")
                # Desactivar la funcionalidad vectorial si no se puede crear el índice
                self._use_vector_extension = False
                self.store_metadata("vector_extension_enabled", "false")
                return False
                
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error(f"Error al crear el índice vectorial: {e}")
            return False
    
    def serialize_vector(self, vector: List[float]) -> bytes:
        """
        Serializa un vector de floats a un blob binario.
        
        Utiliza la dimensión fija establecida en la inicialización. Si el vector
        proporcionado tiene una dimensión diferente, se trunca o se rellena con ceros.
        
        Parámetros:
            vector (List[float]): Vector a serializar.
            
        Returns:
            bytes: Representación binaria del vector.
        """
        # Adaptar el vector a la dimensión configurada
        if len(vector) != self._embedding_dim:
            if len(vector) > self._embedding_dim:
                # Truncar si es más grande
                vector = vector[:self._embedding_dim]
                logger.debug(f"Vector truncado a {self._embedding_dim} dimensiones")
            else:
                # Rellenar con ceros si es más pequeño
                vector = vector + [0.0] * (self._embedding_dim - len(vector))
                logger.debug(f"Vector rellenado a {self._embedding_dim} dimensiones")
                
        return struct.pack(f"{self._embedding_dim}f", *vector)
    
    def deserialize_vector(self, blob: bytes, dim: int = None) -> List[float]:
        """
        Deserializa un vector desde su formato bytes a una lista de flotantes.
        
        Args:
            blob: Blob con el vector serializado
            dim: Dimensiones del vector a deserializar. Si es None, se usa la
                 dimensión configurada en la instancia.
                 
        Returns:
            List[float]: Vector deserializado.
        """
        # Si no se proporciona dim, usar la dimensión configurada en la instancia
        embedding_dim = dim if dim is not None else self._embedding_dim
        return list(struct.unpack(f"{embedding_dim}f", blob))
    
    def insert_document(self, document: Dict[str, Any], chunks: List[Dict[str, Any]]) -> int:
        """
        Inserta un documento y sus chunks embebidos en la base de datos.
        
        Los embeddings se adaptan automáticamente a la dimensión fija configurada.
        
        Args:
            document: Contiene campos como title, url, file_path.
            chunks: Lista de diccionarios para cada chunk con campos como text, header, page, embedding.
        
        Returns:
            int: ID del documento insertado
            
        Raises:
            sqlite3.Error: Si ocurre un error durante la inserción.
        """
        try:
            # Iniciar transacción
            self._conn.execute("BEGIN TRANSACTION;")
            
            # Insertar el documento
            self._cursor.execute("""
                INSERT INTO documents (title, url, file_path)
                VALUES (?, ?, ?)
            """, (
                document.get('title', 'Sin título'),
                document.get('url', f"local://{document.get('file_path', 'unknown')}"),
                document.get('file_path', '')
            ))
            
            # Obtener el ID del documento insertado
            document_id = self._cursor.lastrowid
            
            if document_id is None:
                # Si no se obtuvo un ID, buscar el último ID insertado
                self._cursor.execute("SELECT last_insert_rowid()")
                document_id = self._cursor.fetchone()[0]
            
            logger.debug(f"Documento insertado con ID: {document_id}")
            
            # Verificar el estado de la extensión vectorial
            vector_extension_enabled = self._extension_loaded and self._use_vector_extension
            vector_table_exists = False
            
            if vector_extension_enabled:
                # Verificar si la tabla existe
                self._cursor.execute(f"""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='{self._vector_table_name}';
                """)
                vector_table_exists = self._cursor.fetchone() is not None
                
                if not vector_table_exists:
                    # Intentar crear la tabla si no existe
                    vector_table_exists = self.create_vector_index()
                    if not vector_table_exists:
                        logger.warning("No se pudo crear la tabla vectorial durante la inserción.")
            
            # Insertar los chunks asociados
            for chunk in chunks:
                # Convertir el embedding a bytes para almacenamiento eficiente
                embedding = chunk.get('embedding')
                embedding_bytes = None
                
                if embedding is not None:
                    if isinstance(embedding, list):
                        embedding_bytes = self.serialize_vector(embedding)
                    elif isinstance(embedding, np.ndarray):
                        embedding_bytes = self.serialize_vector(embedding.tolist())
                
                # IMPORTANTE: Usar el document_id como entero
                self._cursor.execute("""
                    INSERT INTO chunks (document_id, text, header, page, embedding, embedding_dim)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    int(document_id),  # Asegurar que es un entero
                    chunk.get('text', ''),
                    chunk.get('header', None),
                    chunk.get('page', None),
                    embedding_bytes,
                    self._embedding_dim  # Usar la dimensión fija
                ))
                
                chunk_id = self._cursor.lastrowid
                
                # Solo intentar actualizar el índice vectorial si todo está configurado
                if vector_extension_enabled and vector_table_exists and chunk_id and embedding_bytes:
                    try:
                        # Usar la sintaxis exacta del ejemplo proporcionado
                        self._cursor.execute(
                            f"INSERT INTO {self._vector_table_name}(rowid, embedding) VALUES (?, ?)",
                            (chunk_id, embedding_bytes)
                        )
                    except sqlite3.Error as e:
                        logger.warning(f"Error al actualizar índice vectorial (no crítico): {e}")
            
            # Confirmar la transacción
            self._conn.commit()
            
            logger.info(f"Documento insertado correctamente con ID: {document_id}, con {len(chunks)} chunks.")
            return document_id
            
        except Exception as e:
            # Si hay error, hacer rollback
            if self._conn:
                self._conn.rollback()
            logger.error(f"Error al insertar documento y chunks: {e}")
            raise
    
    def get_chunks_by_document(self, document_id, offset=0, limit=100):
        """
        Recupera los chunks asociados a un documento.
        
        Args:
            document_id: ID del documento
            offset: Inicio de la paginación
            limit: Número máximo de resultados
            
        Returns:
            Lista de chunks con sus metadatos y embeddings
        """
        try:
            cursor = self._conn.cursor()
            
            query = """
            SELECT c.id, c.text, c.document_id, c.header, c.page, c.embedding
            FROM chunks c
            WHERE c.document_id = ?
            ORDER BY c.id
            LIMIT ? OFFSET ?
            """
            
            cursor.execute(query, (document_id, limit, offset))
            
            chunks = []
            for row in cursor.fetchall():
                chunk = {
                    "id": row[0],
                    "text": row[1],
                    "document_id": row[2],
                    "header": row[3],
                    "page": row[4],
                    "embedding": self.deserialize_vector(row[5])
                }
                chunks.append(chunk)
            
            logger.info(f"Recuperados {len(chunks)} chunks para el documento ID: {document_id} (offset: {offset}, limit: {limit})")
            return chunks
        
        except Exception as e:
            logger.error(f"Error al recuperar chunks por documento: {e}")
            raise
    
    def _vector_search_with_extension(self, query_embedding, filters=None, n_results=5, include_neighbors=False):
        """
        Realiza búsqueda vectorial utilizando la extensión sqlite-vec.
        
        Args:
            query_embedding: Vector de consulta
            filters: Filtros adicionales
            n_results: Número máximo de resultados
            include_neighbors: Si se incluyen chunks vecinos
            
        Returns:
            Lista de chunks ordenados por similitud
        """
        # Verificar si tenemos la extensión y la tabla
        self._cursor.execute(f"""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='{self._vector_table_name}';
        """)
        vector_table_exists = self._cursor.fetchone() is not None
        
        if not vector_table_exists:
            logger.warning(f"La tabla vectorial {self._vector_table_name} no existe. Usando búsqueda manual.")
            return self._vector_search_manual(query_embedding, filters, n_results, include_neighbors)
            
        # Serializar el embedding de consulta
        query_blob = self.serialize_vector(query_embedding)
        
        # Construir la consulta SQL adaptada a v0.1.6
        # En v0.1.6, la tabla de vectores tiene columnas rowid y embedding implícitas
        base_query = """
        SELECT c.id, c.document_id, c.text, c.header, c.page, c.embedding_dim,
            d.title, d.url, d.file_path,
            vec_cosine_similarity(c.embedding, ?) AS similarity
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        """
        
        # Agregar filtros si existen
        where_clauses = []
        params = [query_blob]
        
        if filters:
            if 'document_id' in filters:
                where_clauses.append("c.document_id = ?")
                params.append(filters['document_id'])
            
            if 'min_similarity' in filters:
                where_clauses.append("similarity >= ?")
                params.append(filters['min_similarity'])
            else:
                # Usar umbral de similitud por defecto
                where_clauses.append("similarity >= ?")
                params.append(self._similarity_threshold)
        else:
            # Usar umbral de similitud por defecto
            where_clauses.append("similarity >= ?")
            params.append(self._similarity_threshold)
        
        # Completar la consulta
        if where_clauses:
            base_query += " WHERE " + " AND ".join(where_clauses)
        
        base_query += " ORDER BY similarity DESC LIMIT ?"
        params.append(n_results)
        
        try:
            # Ejecutar la consulta
            cursor = self._conn.cursor()
            cursor.execute(base_query, params)
            
            # Procesar resultados
            results = []
            for row in cursor.fetchall():
                chunk_id, doc_id, text, header, page, embedding_dim, title, url, file_path, similarity = row
                
                # Crear objeto de resultado
                chunk_result = {
                    'id': chunk_id,
                    'document_id': doc_id,
                    'text': text,
                    'header': header,
                    'page': page,
                    'title': title,
                    'url': url,
                    'file_path': file_path,
                    'similarity': float(similarity)
                }
                results.append(chunk_result)
            
            # Incluir chunks vecinos si se solicita
            if include_neighbors and results:
                best_match = results[0]
                neighbors = self._get_adjacent_chunks(best_match["document_id"], best_match["id"])
                
                if neighbors:
                    # Insertar vecinos al principio
                    results = neighbors + results
            
            return results
        except sqlite3.Error as e:
            logger.error(f"Error durante la búsqueda vectorial: {e}")
            # Fallback a búsqueda manual
            return self._vector_search_manual(query_embedding, filters, n_results, include_neighbors)
    
    def optimize_database(self) -> bool:
        """
        Optimiza la base de datos SQLite ejecutando VACUUM y ANALYZE.
        
        Returns:
            bool: True si la optimización fue exitosa, False en caso contrario
        """
        try:
            logger.info(f"Optimizando base de datos SQLite: {self._db_path}")
            # Ejecutar VACUUM para compactar la base de datos
            self._conn.execute("VACUUM;")
            # Ejecutar ANALYZE para actualizar estadísticas
            self._conn.execute("ANALYZE;")
            # Ejecutar PRAGMA optimize para optimizaciones adicionales
            self._conn.execute("PRAGMA optimize;")
            logger.info("Optimización de SQLite completada exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error al optimizar base de datos SQLite: {e}")
            return False

    def vector_search(self, query_embedding, filters=None, n_results=5, include_neighbors=False):
        """
        Realiza una búsqueda vectorial.
        
        Args:
            query_embedding: Vector de embedding de consulta
            filters: Filtros adicionales para la búsqueda
            n_results: Número máximo de resultados
            include_neighbors: Si se incluyen chunks vecinos
            
        Returns:
            list: Lista de chunks ordenados por similitud
        """
        # Verificar si podemos usar la extensión vectorial
        if self._extension_loaded and self._use_vector_extension:
            return self._vector_search_with_extension(
                query_embedding, filters, n_results, include_neighbors
            )
        else:
            # Fallback a búsqueda manual si no está disponible la extensión
            return self._vector_search_manual(
                query_embedding, filters, n_results, include_neighbors
            )

    def insert_document_metadata(self, document: Dict[str, Any]) -> int:
        """
        Inserta solo los metadatos de un documento sin los chunks.
        Diseñado para procesamiento en streaming donde los chunks se insertarán después.
        
        Args:
            document: Diccionario con los datos del documento (title, url, file_path, etc.)
            
        Returns:
            int: ID del documento insertado, None si falla
        """
        try:
            # Insertar el documento
            self._cursor.execute("""
                INSERT INTO documents (title, url, file_path)
                VALUES (?, ?, ?)
            """, (
                document.get('title', 'Sin título'),
                document.get('url', f"local://{document.get('file_path', 'unknown')}"),
                document.get('file_path', '')
            ))
            
            # Obtener el ID del documento insertado
            document_id = self._cursor.lastrowid
            
            if document_id is None:
                # Si no se obtuvo un ID, buscar el último ID insertado
                self._cursor.execute("SELECT last_insert_rowid()")
                document_id = self._cursor.fetchone()[0]
            
            # Hacer commit de esta operación para asegurar que el documento existe
            self._conn.commit()
            
            logger.debug(f"Documento (solo metadatos) insertado con ID: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error al insertar metadatos del documento: {e}")
            self._conn.rollback()
            return None
    
    def insert_single_chunk(self, document_id: int, chunk: Dict[str, Any]) -> int:
        """
        Inserta un único chunk asociado a un documento en la base de datos.
        Diseñado para procesamiento streaming de documentos grandes.
        
        Args:
            document_id (int): ID del documento al que pertenece el chunk
            chunk (dict): Diccionario con los datos del chunk
                Debe contener: 'text', 'header', 'page', 'embedding', 'embedding_dim'
            
        Returns:
            int: ID del chunk insertado, None si falla
        """
        try:
            logger.debug(f"Iniciando inserción de chunk para documento ID {document_id}")
            # Verificar el estado de la extensión vectorial
            vector_extension_enabled = self._extension_loaded and self._use_vector_extension
            logger.debug(f"Estado de extensión vectorial: {vector_extension_enabled}")
            
            # Convertir el embedding a bytes para almacenamiento eficiente
            embedding = chunk.get('embedding')
            embedding_bytes = None
            
            if embedding is not None:
                if isinstance(embedding, list):
                    embedding_bytes = self.serialize_vector(embedding)
                    logger.debug(f"Embedding serializado: {len(embedding)} dimensiones -> {len(embedding_bytes)} bytes")
                elif isinstance(embedding, np.ndarray):
                    embedding_bytes = self.serialize_vector(embedding.tolist())
                    logger.debug(f"Embedding (numpy) serializado: {embedding.shape} -> {len(embedding_bytes)} bytes")
            else:
                logger.warning(f"Embedding es None para chunk: {chunk.get('header', 'Sin encabezado')[:50]}")
            
            # Insertar el chunk en la tabla principal
            sql_insert = """
                INSERT INTO chunks (document_id, text, header, page, embedding, embedding_dim)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            
            params = (
                int(document_id),  # Asegurar que es un entero
                chunk.get('text', ''),
                chunk.get('header', None),
                chunk.get('page', None),
                embedding_bytes,
                chunk.get('embedding_dim', self._embedding_dim)  # Usar la dimensión propuesta o la fija
            )
            
            logger.debug(f"Ejecutando SQL: {sql_insert}")
            logger.debug(f"Parámetros: doc_id={document_id}, header_len={len(chunk.get('header', '')) if chunk.get('header') else 0}, text_len={len(chunk.get('text', ''))}, embedding_dim={chunk.get('embedding_dim', self._embedding_dim)}")
            
            self._cursor.execute(sql_insert, params)
            
            # Obtener el ID del chunk insertado
            chunk_id = self._cursor.lastrowid
            logger.debug(f"Chunk insertado con ID: {chunk_id}")
            
            # Solo intentar actualizar el índice vectorial si todo está configurado
            if vector_extension_enabled and chunk_id and embedding_bytes:
                try:
                    # Verificar si la tabla existe
                    self._cursor.execute(f"""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name='{self._vector_table_name}';
                    """)
                    vector_table_exists = self._cursor.fetchone() is not None
                    logger.debug(f"Tabla vectorial existe: {vector_table_exists}")
                    
                    if vector_table_exists:
                        # Usar la sintaxis exacta del ejemplo proporcionado
                        sql_vector = f"INSERT INTO {self._vector_table_name}(rowid, embedding) VALUES (?, ?)"
                        logger.debug(f"Actualizando índice vectorial con SQL: {sql_vector}")
                        self._cursor.execute(sql_vector, (chunk_id, embedding_bytes))
                        logger.debug("Índice vectorial actualizado correctamente")
                    else:
                        # Si no existe, intentar crearla
                        logger.warning(f"Tabla vectorial {self._vector_table_name} no existe. Intentando crear...")
                        if self.create_vector_index():
                            logger.debug(f"Tabla vectorial creada. Insertando datos...")
                            self._cursor.execute(
                                f"INSERT INTO {self._vector_table_name}(rowid, embedding) VALUES (?, ?)",
                                (chunk_id, embedding_bytes)
                            )
                        else:
                            logger.error("No se pudo crear el índice vectorial")
                except sqlite3.Error as e:
                    logger.warning(f"Error al actualizar índice vectorial para chunk {chunk_id} (no crítico): {e}")
            
            return chunk_id
            
        except Exception as e:
            logger.error(f"Error al insertar chunk individual: {e}", exc_info=True)
            return None

    def begin_transaction(self) -> bool:
        """
        Inicia una transacción manual para inserción masiva.
        Útil para mejorar rendimiento con muchas inserciones.
        
        Returns:
            bool: True si se inició correctamente la transacción
        """
        try:
            if self._conn:
                logger.debug("Iniciando transacción en SQLite")
                self._conn.execute("BEGIN TRANSACTION;")
                return True
            else:
                logger.error("No se puede iniciar transacción: conexión no disponible")
                return False
        except Exception as e:
            logger.error(f"Error al iniciar transacción en SQLite: {e}", exc_info=True)
            return False
    
    def commit_transaction(self) -> bool:
        """
        Confirma una transacción en curso.
        
        Returns:
            bool: True si se confirmó correctamente la transacción
        """
        try:
            if self._conn:
                logger.debug("Confirmando transacción en SQLite")
                self._conn.commit()
                return True
            else:
                logger.error("No se puede confirmar transacción: conexión no disponible")
                return False
        except Exception as e:
            logger.error(f"Error al confirmar transacción en SQLite: {e}", exc_info=True)
            return False
    
    def rollback_transaction(self) -> bool:
        """
        Revierte una transacción en curso.
        
        Returns:
            bool: True si se revirtió correctamente la transacción
        """
        try:
            if self._conn:
                logger.debug("Revirtiendo transacción en SQLite")
                self._conn.rollback()
                return True
            else:
                logger.error("No se puede revertir transacción: conexión no disponible")
                return False
        except Exception as e:
            logger.error(f"Error al revertir transacción en SQLite: {e}", exc_info=True)
            return False