import os
import logging
import struct
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import uuid

try:
    import duckdb
except ImportError:
    duckdb = None

from modulos.databases.VectorialDatabase import VectorialDatabase

logger = logging.getLogger(__name__)

class DuckDBVectorialDatabase(VectorialDatabase):
    """
    Implementación de VectorialDatabase usando DuckDB.
    
    La dimensión del embedding se fija en la inicialización y se utiliza para todas las operaciones
    vectoriales posteriores, garantizando consistencia y mejor rendimiento.
    """
    
    def __init__(self, embedding_dim: int):
        """
        Inicializa la base de datos DuckDB con una dimensión de embedding específica.
        
        Args:
            embedding_dim: Dimensión fija de los embeddings que se utilizarán.
                           Este valor debe venir del modelo de embeddings y es obligatorio.
        """
        super().__init__()  # Inicializar la superclase para heredar atributos como _logger
        
        if duckdb is None:
            raise ImportError("DuckDB no está instalado. Instálalo con 'pip install duckdb'.")
        
        if embedding_dim is None or embedding_dim <= 0:
            raise ValueError("La dimensión del embedding debe ser un número positivo")
            
        self._conn = None
        self._ext_loaded = False
        self._schema_created = False
        self._similarity_threshold = 0.3
        self._db_path = None  # Añadido para guardar la ruta de la base de datos
        
        # Fijar la dimensión del embedding
        self._embedding_dim = embedding_dim
        logger.info(f"Dimensión de embedding fijada en: {self._embedding_dim}")
    
    def connect(self, db_path: str) -> bool:
        """
        Conecta a la base de datos DuckDB.
        
        Args:
            db_path: Ruta al archivo de base de datos
            
        Returns:
            True si la conexión fue exitosa, False en caso contrario
        """
        try:
            # Guardar la ruta de la base de datos para uso posterior
            self._db_path = db_path
            
            # Leer configuración específica para DuckDB
            from config import config
            db_config = config.get_database_config()
            duckdb_config = db_config.get('duckdb', {})
            
            # Obtener parámetros de configuración
            memory_limit = duckdb_config.get('memory_limit', '2GB')
            threads_config = duckdb_config.get('threads', 4)  # Default a 4 threads
            self._similarity_threshold = duckdb_config.get('similarity_threshold', 0.3)
            
            # Procesar el valor de threads: DuckDB requiere al menos 1 thread
            import multiprocessing
            if threads_config == 'auto' or threads_config == 0:
                # Usar el número de núcleos disponibles, al menos 1
                threads = max(1, multiprocessing.cpu_count())
            else:
                # Intentar convertir a entero, asegurando que sea al menos 1
                try:
                    threads = max(1, int(threads_config))
                except (ValueError, TypeError):
                    # Si no es posible convertir, usar 1 como valor seguro por defecto
                    logger.warning(f"Valor inválido para threads: '{threads_config}'. Usando 1 como valor por defecto.")
                    threads = 1
            
            # Validación adicional para asegurar que sea un número válido
            if not isinstance(threads, int) or threads < 1:
                threads = 1
                logger.warning(f"Valor corregido de threads a {threads} para cumplir con los requisitos de DuckDB")
            
            # Configuración para conexión DuckDB
            connection_config = {
                'memory_limit': memory_limit,
                'threads': threads  # Debe ser un número entero, no string
            }
            
            logger.debug(f"Configuración para conexión DuckDB: {connection_config}")
            
            if db_path == ':memory:':
                self._conn = duckdb.connect(database=':memory:', config=connection_config)
            else:
                # Asegurarse de que el directorio existe
                db_dir = os.path.dirname(os.path.abspath(db_path))
                if not os.path.exists(db_dir):
                    try:
                        os.makedirs(db_dir, exist_ok=True)
                        logger.info(f"Directorio creado para la base de datos: {db_dir}")
                    except Exception as e:
                        logger.error(f"Error al crear directorio para la base de datos: {e}")
                        return False
                
                # Configuraciones mejoradas para DuckDB
                self._conn = duckdb.connect(database=db_path, config=connection_config)
            
            # Verificar que la conexión se haya establecido correctamente
            if self._conn is None:
                logger.error("La conexión a DuckDB se creó como None")
                return False
                
            # Ejecutar una consulta sencilla para verificar que la conexión funciona
            try:
                self._conn.execute("SELECT 1")
                self._conn.fetchone()
            except Exception as e:
                logger.error(f"La conexión a DuckDB se creó pero no es funcional: {e}")
                return False
            
            # Inicializar el cursor para compatibilidad con la clase abstracta
            self._cursor = self._conn
            
            # Cargar extensiones necesarias inmediatamente después de conectar
            # Pero no fallar si no se pueden cargar
            self.load_extensions()
            
            logger.info(f"Conexión exitosa a DuckDB: {db_path} (memoria: {memory_limit}, hilos: {threads})")
            
            # Crear el esquema tras conectar exitosamente
            if not self.create_schema():
                logger.warning("No se pudo crear el esquema completo en DuckDB, pero se continuará el proceso")
            
            return True
        except Exception as e:
            logger.error(f"Error al conectar a DuckDB: {e}")
            self._conn = None
            self._cursor = None
            return False
    
    def close_connection(self) -> bool:
        """
        Cierra la conexión a la base de datos.
        
        Returns:
            True si se cerró correctamente, False en caso contrario
        """
        try:
            if self._conn:
                self._conn.close()
                self._conn = None
                logger.info("Conexión a DuckDB cerrada correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al cerrar conexión DuckDB: {e}")
            return False
    
    def create_schema(self) -> bool:
        """
        Crea el esquema de la base de datos si no existe.
        
        Returns:
            True si se creó correctamente, False en caso contrario
        """
        if not self._conn:
            logger.error("No hay conexión a la base de datos")
            return False
        
        try:
            # Verificar si las tablas ya existen para evitar errores
            try:
                self._conn.execute("CREATE SCHEMA IF NOT EXISTS main")
            except Exception as e:
                logger.warning(f"No se pudo crear el esquema main (puede ser ignorado): {e}")
            
            # Verificar si ya existen las tablas principales
            try:
                self._conn.execute("SELECT 1 FROM documents LIMIT 1")
                self._conn.execute("SELECT 1 FROM chunks LIMIT 1")
                logger.info("Las tablas ya existen en la base de datos, omitiendo creación de esquema")
                self._schema_created = True
                return True
            except Exception:
                # Si hay un error al consultar las tablas, es porque no existen y debemos crearlas
                pass
            
            # Crear secuencia para IDs de documentos
            try:
                self._conn.execute("""
                    CREATE SEQUENCE IF NOT EXISTS doc_id_seq;
                """)
            except Exception as e:
                logger.warning(f"Error al crear secuencia doc_id_seq: {e}")
                # Continuar a pesar del error
            
            # Crear tabla de documentos con ID autogenerado usando la secuencia
            # Usando CAST para asegurar que el tipo sea INTEGER
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY DEFAULT CAST(nextval('doc_id_seq') AS INTEGER),
                    title TEXT,
                    url TEXT,
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Crear secuencia para IDs de chunks
            try:
                self._conn.execute("""
                    CREATE SEQUENCE IF NOT EXISTS chunk_id_seq;
                """)
            except Exception as e:
                logger.warning(f"Error al crear secuencia chunk_id_seq: {e}")
                # Continuar a pesar del error
            
            # Crear tabla de chunks con ID autogenerado usando la secuencia
            # Asegurando que document_id sea INTEGER para coincidir con la clave primaria
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY DEFAULT CAST(nextval('chunk_id_seq') AS INTEGER),
                    document_id INTEGER,
                    text TEXT NOT NULL,
                    header TEXT,
                    page TEXT,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                )
            """)
            
            # Crear tabla de metadatos si no existe
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS db_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Crear índice para búsquedas rápidas por document_id
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
            
            # Guardar metadato sobre la dimensión de embeddings
            try:
                self.store_metadata("embedding_dim", str(self._embedding_dim))
            except Exception as e:
                logger.warning(f"No se pudo guardar metadata de embedding_dim: {e}")
            
            self._schema_created = True
            logger.info("Esquema de DuckDB creado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al crear esquema en DuckDB: {e}")
            return False
    
    def load_extensions(self) -> bool:
        """
        Carga extensiones para búsqueda vectorial si están disponibles.
        
        Returns:
            True si las extensiones se cargaron correctamente, False en caso contrario.
        """
        # Evitar cargar extensiones si ya están cargadas
        if self._ext_loaded:
            logger.debug("Extensiones DuckDB ya están cargadas")
            return True
            
        # Evitar intentar cargar extensiones si no hay conexión
        if not self._conn:
            logger.error("No hay conexión a la base de datos para cargar extensiones")
            return False
            
        try:
            # Leer las extensiones configuradas
            from config import config
            db_config = config.get_database_config()
            duckdb_config = db_config.get('duckdb', {})
            extensions = duckdb_config.get('extensions', [])  # Lista vacía por defecto
            
            # Filtrar explícitamente las extensiones no deseadas
            excluded_extensions = ['httpfs', 'json']
            extensions = [ext for ext in extensions if ext not in excluded_extensions]
            
            logger.info(f"Extensiones a cargar: {extensions}")
            
            # Instalar y cargar las extensiones configuradas
            for extension in extensions:
                try:
                    # Comprobar si la extensión ya está cargada
                    try:
                        self._conn.execute(f"SELECT 1 FROM pragma_installed_extensions() WHERE extension_name = '{extension}'")
                        already_installed = len(self._conn.fetchall()) > 0
                    except:
                        already_installed = False
                    
                    if not already_installed:
                        logger.debug(f"Instalando extensión DuckDB '{extension}'...")
                        self._conn.execute(f"INSTALL {extension};")
                    
                    logger.debug(f"Cargando extensión DuckDB '{extension}'...")
                    self._conn.execute(f"LOAD {extension};")
                    logger.info(f"Extensión DuckDB '{extension}' cargada correctamente")
                except Exception as e:
                    logger.debug(f"No se pudo cargar extensión '{extension}' (no es crítico): {e}")
            
            self._ext_loaded = True
            logger.info("Proceso de carga de extensiones DuckDB completado")
            return True
        except Exception as e:
            logger.warning(f"No se pudieron cargar todas las extensiones DuckDB: {e}")
            # No fallar completamente si las extensiones no se cargan
            self._ext_loaded = True
            return True  # Devolver True para permitir que el proceso continúe incluso si hay problemas con las extensiones
    
    def serialize_vector(self, vector: List[float]) -> bytes:
        """
        Serializa un vector para almacenamiento en la base de datos.
        
        Utiliza la dimensión fija establecida en la inicialización. Si el vector
        proporcionado tiene una dimensión diferente, se trunca o se rellena con ceros.
        
        Args:
            vector: Lista de valores float que representan el vector
            
        Returns:
            Vector serializado como bytes
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
        
        # Convertir lista a array numpy y luego a bytes
        return struct.pack(f'{self._embedding_dim}f', *vector)
    
    def deserialize_vector(self, blob: bytes, dim: int = None) -> List[float]:
        """
        Deserializa un vector desde la base de datos.
        
        Args:
            blob: Vector serializado como bytes
            dim: Dimensión del vector (se ignora y se usa la dimensión fija configurada)
            
        Returns:
            Lista de valores float que representan el vector
        """
        # Usar la dimensión fija de la clase
        return list(struct.unpack(f'{self._embedding_dim}f', blob))
    
    def insert_document(self, document: Dict[str, Any], chunks: List[Dict[str, Any]]) -> int:
        """
        Inserta un documento y sus chunks en la base de datos.
        
        Los embeddings se adaptan automáticamente a la dimensión fija configurada.
        
        Args:
            document: Diccionario con información del documento
            chunks: Lista de chunks generados a partir del documento
            
        Returns:
            ID del documento insertado
            
        Raises:
            Exception: Si hay un error durante la inserción
        """
        if not self._conn:
            raise ValueError("No hay conexión a la base de datos")
        
        if not self._schema_created:
            self.create_schema()
        
        try:
            # Comenzar una transacción para garantizar consistencia
            self.begin_transaction()
            
            # Insertar el documento y obtener su ID
            self._conn.execute("""
                INSERT INTO documents (title, url, file_path, created_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                RETURNING id
            """, (
                document.get('title', ''),
                document.get('url', ''),
                document.get('file_path', ''),
            ))
            
            # Obtener el ID del documento insertado usando RETURNING
            document_id = self._conn.fetchone()[0]
            
            # Insertar cada chunk
            for chunk in chunks:
                # Procesar el embedding
                embedding = chunk.get('embedding')
                
                # Serializar el embedding si existe
                serialized_embedding = None
                if embedding is not None:
                    serialized_embedding = self.serialize_vector(embedding)
                
                # Insertar el chunk con su embedding serializado
                self._conn.execute("""
                    INSERT INTO chunks (document_id, text, header, page, embedding, embedding_dim, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    document_id,
                    chunk.get('text', ''),
                    chunk.get('header', None),
                    chunk.get('page', None),
                    serialized_embedding,
                    self._embedding_dim if serialized_embedding else None
                ))
            
            # Confirmar la transacción
            self.commit_transaction()
            
            logger.info(f"Documento insertado con ID {document_id} y {len(chunks)} chunks")
            
            return document_id
            
        except Exception as e:
            # Revertir los cambios en caso de error
            self.rollback_transaction()
            logger.error(f"Error al insertar documento: {e}")
            raise
    
    def insert_document_metadata(self, document: Dict[str, Any]) -> int:
        """
        Inserta solo los metadatos de un documento en la base de datos.
        Implementación para procesamiento en streaming de documentos grandes.
        
        Args:
            document: Diccionario con los datos del documento
            
        Returns:
            int: ID del documento insertado, None si falla
        """
        if not self._conn:
            logger.error("No hay conexión a la base de datos")
            return None
        
        if not self._schema_created:
            self.create_schema()
        
        try:
            # Iniciar una transacción explícita
            self.begin_transaction()
            
            # Insertar el documento
            self._conn.execute("""
                INSERT INTO documents (title, url, file_path, created_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                RETURNING id
            """, (
                document.get('title', ''),
                document.get('url', ''),
                document.get('file_path', ''),
            ))
            
            # Obtener el ID del documento insertado usando RETURNING
            document_id = self._conn.fetchone()[0]
            
            # Hacer commit usando el método de la clase padre en lugar del comando SQL directo
            self.commit_transaction()
            
            logger.debug(f"Documento (solo metadatos) insertado con ID: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error al insertar metadatos del documento: {e}")
            # Hacer rollback en caso de error usando el método de la clase padre
            self.rollback_transaction()
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
        if not self._conn:
            logger.error("No hay conexión a la base de datos")
            return None
        
        try:
            # Procesar el embedding
            embedding = chunk.get('embedding')
            
            # Serializar el embedding si existe
            serialized_embedding = None
            if embedding is not None:
                serialized_embedding = self.serialize_vector(embedding)
            
            # Insertar el chunk individual
            self._conn.execute("""
                INSERT INTO chunks (document_id, text, header, page, embedding, embedding_dim, created_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                RETURNING id
            """, (
                document_id,
                chunk.get('text', ''),
                chunk.get('header', None),
                chunk.get('page', None),
                serialized_embedding,
                self._embedding_dim if serialized_embedding else None
            ))
            
            # Obtener el ID del chunk insertado
            chunk_id = self._conn.fetchone()[0]
            
            return chunk_id
            
        except Exception as e:
            logger.error(f"Error al insertar chunk individual: {e}")
            return None
    
    def optimize_database(self) -> bool:
        """
        Optimiza la base de datos realizando operaciones de mantenimiento.
        
        Returns:
            True si la optimización fue exitosa, False en caso contrario
        """
        if not self._conn:
            logger.error("No hay conexión a la base de datos")
            return False
        
        try:
            # Ejecutar vacuum para liberar espacio no utilizado
            self._conn.execute("VACUUM")
            
            # Analizar las tablas para mejorar el plan de consultas
            self._conn.execute("ANALYZE chunks")
            self._conn.execute("ANALYZE documents")
            
            logger.info("Base de datos DuckDB optimizada correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al optimizar la base de datos: {e}")
            return False
    
    def vector_search(self, 
                     query_embedding: List[float], 
                     filters: Optional[Dict[str, Any]] = None, 
                     n_results: int = 5, 
                     include_neighbors: bool = False) -> List[Dict[str, Any]]:
        """
        Realiza una búsqueda por similitud vectorial entre el embedding de consulta y los embeddings almacenados.
        
        Args:
            query_embedding: Vector de embedding de la consulta
            filters: Filtros adicionales para la búsqueda (ej. {'document_id': 123})
            n_results: Número máximo de resultados a retornar
            include_neighbors: Si se incluyen chunks vecinos en los resultados
            
        Returns:
            Lista de chunks ordenados por similitud
        """
        if not self._conn:
            logger.error("No hay conexión a la base de datos")
            return []
        
        # Verificar que el embedding no sea None o vacío
        if query_embedding is None or len(query_embedding) == 0:
            logger.error("El embedding de consulta es None o está vacío")
            return []
            
        try:
            # Convertir a numpy array para cálculos más eficientes
            query_embedding_np = np.array(query_embedding, dtype=np.float32)
            
            # Verificar dimensiones
            if len(query_embedding_np) != self._embedding_dim:
                logger.warning(f"Dimensión de embedding de consulta ({len(query_embedding_np)}) difiere "
                              f"de la configurada ({self._embedding_dim}). Ajustando dimensiones.")
                # Adaptar dimensiones
                if len(query_embedding_np) > self._embedding_dim:
                    # Truncar si es más grande
                    query_embedding_np = query_embedding_np[:self._embedding_dim]
                else:
                    # Rellenar con ceros si es más pequeño
                    pad_width = self._embedding_dim - len(query_embedding_np)
                    query_embedding_np = np.pad(query_embedding_np, (0, pad_width), 'constant')
            
            # Normalizar para calcular similitud coseno
            query_norm = np.linalg.norm(query_embedding_np)
            if query_norm > 0:
                query_embedding_np = query_embedding_np / query_norm
            
            # Decidir qué método de búsqueda usar
            # Para búsquedas más grandes, usar el método por lotes
            total_chunks = 0
            try:
                self._cursor.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
                total_chunks = self._cursor.fetchone()[0]
            except Exception as e:
                logger.warning(f"No se pudo determinar el número de chunks: {e}")
            
            # Si hay muchos chunks, usar búsqueda por lotes
            if total_chunks > 10000 and not filters:  # Solo usar fast_search si no hay filtros específicos
                logger.info(f"Usando búsqueda vectorial por lotes para {total_chunks} chunks")
                results = self.fast_vector_search(
                    query_embedding=query_embedding_np.tolist(),
                    n_results=n_results
                )
                
                # Si se solicitan vecinos, agregarlos
                if include_neighbors and results:
                    results = self._add_neighbors_to_results(results)
                    
                return results
            else:
                # Usar búsqueda estándar
                results = self._vector_search_manual(
                    query_embedding=query_embedding_np.tolist(),
                    filters=filters,
                    n_results=n_results,
                    include_neighbors=include_neighbors
                )
                
                return results
                
        except Exception as e:
            logger.error(f"Error en vector_search: {e}")
            # En caso de error, intentar con el método manual
            try:
                return self._vector_search_manual(
                    query_embedding=query_embedding,
                    filters=filters,
                    n_results=n_results,
                    include_neighbors=include_neighbors
                )
            except Exception as e2:
                logger.error(f"Error en búsqueda vectorial de respaldo: {e2}")
                return []
    
    def _vector_search_manual(self, query_embedding: List[float], filters=None, n_results=5, include_neighbors=False):
        """
        Implementación manual de búsqueda vectorial cuando no hay extensiones especializadas.
        
        Args:
            query_embedding: Vector de embedding de la consulta
            filters: Filtros para la búsqueda
            n_results: Número máximo de resultados
            include_neighbors: Si se incluyen chunks vecinos en los resultados
        
        Returns:
            Lista de chunks ordenados por similitud
        """
        results = []
        
        try:
            # Serializar el embedding de consulta para comparación
            query_embedding_blob = self.serialize_vector(query_embedding)
            
            # Construir la consulta base
            base_query = """
                SELECT c.id, c.document_id, c.text, c.header, c.page, c.embedding,
                       d.title, d.url, d.file_path
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
            """
            
            # Añadir condiciones de filtro si existen
            params = []
            if filters:
                for key, value in filters.items():
                    if key == 'document_id':
                        base_query += " AND c.document_id = ?"
                        params.append(value)
                    elif key == 'header':
                        base_query += " AND c.header LIKE ?"
                        params.append(f"%{value}%")
                    # Añadir más filtros según sea necesario
            
            # Ejecutar la consulta
            self._cursor.execute(base_query, params)
            all_chunks = self._cursor.fetchall()
            
            # Calcular similitud con cada chunk
            chunk_similarities = []
            for chunk in all_chunks:
                # Extraer los campos del chunk
                chunk_id = chunk[0]
                document_id = chunk[1]
                text = chunk[2]
                header = chunk[3]
                page = chunk[4]
                embedding_blob = chunk[5]
                title = chunk[6]
                url = chunk[7]
                file_path = chunk[8]
                
                # Deserializar el embedding
                if embedding_blob:
                    embedding = self.deserialize_vector(embedding_blob)
                    
                    # Calcular similitud por coseno
                    similarity = self._cosine_similarity(
                        np.array(query_embedding), 
                        np.array(embedding)
                    )
                    
                    # Filtrar por umbral de similitud
                    if similarity >= self._similarity_threshold:
                        chunk_similarities.append({
                            'id': chunk_id,
                            'document_id': document_id,
                            'text': text,
                            'header': header,
                            'page': page,
                            'title': title,
                            'url': url,
                            'file_path': file_path,
                            'similarity': float(similarity)
                        })
            
            # Ordenar por similitud descendente
            chunk_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Limitar resultados
            results = chunk_similarities[:n_results]
            
            # Incluir chunks vecinos si se solicita
            if include_neighbors and results:
                for i, result in enumerate(results.copy()):
                    neighbors = self._get_adjacent_chunks(result['document_id'], result['id'])
                    
                    # Añadir vecinos a los resultados, marcándolos como vecinos
                    for neighbor in neighbors:
                        neighbor['is_neighbor'] = True
                        neighbor['neighbor_of'] = result['id']
                        # Asegurar que no estemos duplicando chunks
                        if not any(r['id'] == neighbor['id'] for r in results):
                            results.append(neighbor)
            
        except Exception as e:
            logger.error(f"Error en búsqueda vectorial: {e}")
        
        return results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calcula la similitud de coseno entre dos vectores.
        
        Args:
            vec1: Primer vector
            vec2: Segundo vector
            
        Returns:
            Similitud de coseno (float entre -1 y 1)
        """
        # Manejar vectores vacíos
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
            
        # Calcular normas
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Evitar división por cero
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # Calcular similitud de coseno
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _get_adjacent_chunks(self, document_id: int, chunk_id: int) -> List[Dict[str, Any]]:
        """
        Obtiene los chunks adyacentes (anterior y siguiente) a un chunk dado.
        
        Args:
            document_id: ID del documento
            chunk_id: ID del chunk
            
        Returns:
            Lista de chunks adyacentes
        """
        if not self._conn:
            return []
            
        try:
            # Obtener el chunk anterior
            self._cursor.execute("""
                SELECT c.id, c.document_id, c.text, c.header, c.page, d.title, d.url, d.file_path
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = ? AND c.id < ?
                ORDER BY c.id DESC
                LIMIT 1
            """, (document_id, chunk_id))
            prev_chunk = self._cursor.fetchone()
            
            # Obtener el chunk siguiente
            self._cursor.execute("""
                SELECT c.id, c.document_id, c.text, c.header, c.page, d.title, d.url, d.file_path
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = ? AND c.id > ?
                ORDER BY c.id ASC
                LIMIT 1
            """, (document_id, chunk_id))
            next_chunk = self._cursor.fetchone()
            
            # Procesar los resultados
            adjacent_chunks = []
            
            if prev_chunk:
                adjacent_chunks.append({
                    'id': prev_chunk[0],
                    'document_id': prev_chunk[1],
                    'text': prev_chunk[2],
                    'header': prev_chunk[3],
                    'page': prev_chunk[4],
                    'title': prev_chunk[5],
                    'url': prev_chunk[6],
                    'file_path': prev_chunk[7],
                    'position': 'previous'
                })
                
            if next_chunk:
                adjacent_chunks.append({
                    'id': next_chunk[0],
                    'document_id': next_chunk[1],
                    'text': next_chunk[2],
                    'header': next_chunk[3],
                    'page': next_chunk[4],
                    'title': next_chunk[5],
                    'url': next_chunk[6],
                    'file_path': next_chunk[7],
                    'position': 'next'
                })
                
            return adjacent_chunks
            
        except Exception as e:
            logger.error(f"Error al obtener chunks adyacentes: {e}")
            return []
    
    def get_chunks_by_document(self, document_id: int, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtiene los chunks asociados a un documento específico a partir de su ID.
        
        Args:
            document_id: ID del documento del cual se desean obtener los chunks.
            offset: Desfase para la paginación (por defecto 0).
            limit: Número máximo de chunks a retornar (por defecto 100).
            
        Returns:
            Lista de diccionarios, cada uno representando un chunk.
        """
        if not self._conn:
            logger.error("No hay conexión establecida en DuckDBVectorialDatabase")
            return []
        
        try:
            # Obtener chunks con la información del documento
            query = """
                SELECT c.id, c.document_id, c.text, c.header, c.page, c.embedding, 
                       d.title, d.url, d.file_path, c.created_at
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = ?
                ORDER BY c.id ASC
                LIMIT ? OFFSET ?
            """
            self._cursor.execute(query, (document_id, limit, offset))
            rows = self._cursor.fetchall()
            
            chunks = []
            for row in rows:
                chunk_id, doc_id, text, header, page, embedding_blob, title, url, file_path, created_at = row
                
                # Construir el diccionario del chunk
                chunk = {
                    "id": chunk_id,
                    "document_id": doc_id,
                    "text": text,
                    "header": header,
                    "page": page,
                    "title": title,
                    "url": url,
                    "file_path": file_path,
                    "created_at": created_at
                }
                
                # Si hay embedding, deserializarlo
                if embedding_blob is not None:
                    chunk["embedding"] = self.deserialize_vector(embedding_blob)
                
                chunks.append(chunk)
            
            logger.info(f"Recuperados {len(chunks)} chunks para el documento ID: {document_id} (offset: {offset}, limit: {limit})")
            return chunks
            
        except Exception as e:
            logger.error(f"Error en get_chunks_by_document: {e}")
            return []
    
    def batch_insert_chunks(self, chunks: List[Dict[str, Any]], document_id: int) -> bool:
        """
        Inserta múltiples chunks en la base de datos de forma optimizada.
        
        Args:
            chunks: Lista de diccionarios con información de chunks
            document_id: ID del documento al que pertenecen estos chunks
            
        Returns:
            True si la inserción fue exitosa, False en caso contrario
        """
        if not self._conn or not self._schema_created:
            logger.error("No hay conexión a la base de datos o el esquema no está creado")
            return False
            
        if not chunks:
            logger.warning("No hay chunks para insertar")
            return True
            
        try:
            # Comenzar transacción para optimizar rendimiento
            self.begin_transaction()
            
            # Preparar parámetros para inserción masiva
            chunk_params = []
            
            for chunk in chunks:
                # Procesar el embedding
                embedding = chunk.get('embedding')
                serialized_embedding = None
                
                if embedding is not None:
                    serialized_embedding = self.serialize_vector(embedding)
                
                # Preparar parámetros para este chunk
                chunk_params.append((
                    document_id,
                    chunk.get('text', ''),
                    chunk.get('header', None),
                    chunk.get('page', None),
                    serialized_embedding,
                    self._embedding_dim if serialized_embedding else None
                ))
            
            # Realizar inserción masiva
            if chunk_params:
                self._conn.executemany("""
                    INSERT INTO chunks (document_id, text, header, page, embedding, embedding_dim)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, chunk_params)
            
            # Confirmar transacción
            self.commit_transaction()
            
            logger.info(f"Insertados {len(chunks)} chunks en lote para el documento {document_id}")
            return True
        except Exception as e:
            # Revertir cambios en caso de error
            self.rollback_transaction()
            logger.error(f"Error al insertar chunks en lote: {e}")
            return False
            
    def convert_embedding_dimension(self, embedding: List[float], target_dim: int) -> np.ndarray:
        """
        Adapta la dimensión de un embedding al tamaño deseado.
        
        Args:
            embedding: Vector original
            target_dim: Dimensión objetivo
            
        Returns:
            Vector redimensionado como numpy array
        """
        embedding_np = np.array(embedding, dtype=np.float32)
        current_dim = len(embedding_np)
        
        if current_dim == target_dim:
            return embedding_np
            
        logger.debug(f"Adaptando dimensión de embedding de {current_dim} a {target_dim}")
        
        if current_dim > target_dim:
            # Truncar si es más grande
            return embedding_np[:target_dim]
        else:
            # Rellenar con ceros si es más pequeño
            padding = np.zeros(target_dim - current_dim, dtype=np.float32)
            return np.concatenate([embedding_np, padding])
            
    def fast_vector_search(self, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Versión optimizada de búsqueda vectorial para grandes volúmenes de datos.
        
        Utiliza técnicas de procesamiento por lotes para mejorar el rendimiento.
        
        Args:
            query_embedding: Vector de embedding de la consulta
            n_results: Número máximo de resultados a retornar
            
        Returns:
            Lista de chunks ordenados por similitud
        """
        if not self._conn:
            logger.error("No hay conexión a la base de datos")
            return []
        
        try:    
            # Normalizar y adaptar el embedding de consulta
            query_embedding_np = np.array(query_embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query_embedding_np)
            if query_norm > 0:
                query_embedding_np = query_embedding_np / query_norm
                
            if len(query_embedding_np) != self._embedding_dim:
                query_embedding_np = self.convert_embedding_dimension(query_embedding, self._embedding_dim)
                
            # Serializar embedding para comparación
            query_embedding_blob = self.serialize_vector(query_embedding_np.tolist())
            
            # Obtener todos los IDs de chunks con embeddings en un solo paso
            try:
                self._cursor.execute("""
                    SELECT id FROM chunks WHERE embedding IS NOT NULL
                """)
                chunk_ids = [row[0] for row in self._cursor.fetchall()]
                
                # Si no hay chunks, retornar lista vacía
                if not chunk_ids:
                    logger.warning("No se encontraron chunks con embeddings en la base de datos")
                    return []
            except Exception as db_error:
                logger.error(f"Error al obtener IDs de chunks: {db_error}")
                return []
                
            # Procesar en lotes para evitar sobrecarga de memoria
            batch_size = 1000
            all_similarities = []
            total_processed = 0
            
            for i in range(0, len(chunk_ids), batch_size):
                batch_ids = chunk_ids[i:i+batch_size]
                placeholders = ', '.join(['?' for _ in batch_ids])
                
                try:
                    # Obtener embeddings para este lote
                    self._cursor.execute(f"""
                        SELECT c.id, c.document_id, c.text, c.header, c.page, c.embedding,
                               d.title, d.url, d.file_path
                        FROM chunks c
                        JOIN documents d ON c.document_id = d.id
                        WHERE c.id IN ({placeholders})
                    """, batch_ids)
                    
                    batch_results = self._cursor.fetchall()
                    
                    # Calcular similitudes para cada chunk en el lote
                    for row in batch_results:
                        if row is None or len(row) < 6:
                            continue
                            
                        chunk_id, doc_id, text, header, page, embedding_blob = row[:6]
                        title, url, file_path = row[6:9] if len(row) >= 9 else (None, None, None)
                        
                        if embedding_blob:
                            try:
                                # Deserializar embedding
                                embedding = self.deserialize_vector(embedding_blob)
                                
                                # Calcular similitud
                                similarity = self._cosine_similarity(
                                    query_embedding_np, 
                                    np.array(embedding, dtype=np.float32)
                                )
                                
                                # Agregar a resultados si supera el umbral
                                if similarity >= self._similarity_threshold:
                                    all_similarities.append({
                                        'id': chunk_id,
                                        'document_id': doc_id,
                                        'text': text,
                                        'header': header,
                                        'page': page,
                                        'title': title,
                                        'url': url,
                                        'file_path': file_path,
                                        'similarity': float(similarity)
                                    })
                            except Exception as vec_error:
                                logger.debug(f"Error procesando vector para chunk {chunk_id}: {vec_error}")
                                continue
                    
                    total_processed += len(batch_results)
                    
                except Exception as batch_error:
                    logger.error(f"Error procesando lote de chunks {i}-{i+batch_size}: {batch_error}")
                    # Continuar con el siguiente lote
            
            # Log informativo sobre el procesamiento
            logger.info(f"Procesados {total_processed} chunks en búsqueda vectorial. Encontrados {len(all_similarities)} resultados relevantes.")
            
            # Ordenar por similitud y limitar resultados
            all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return all_similarities[:n_results]
                
        except Exception as e:
            logger.error(f"Error en búsqueda vectorial rápida: {e}")
            return []
    
    def store_metadata(self, key: str, value: Any) -> bool:
        """
        Almacena un metadato en la base de datos.
        
        Args:
            key: Clave del metadato
            value: Valor del metadato
            
        Returns:
            True si se almacenó correctamente, False en caso contrario
        """
        if not self._conn:
            logger.warning(f"No se puede almacenar metadato '{key}': no hay conexión a la base de datos")
            
            # Si tenemos una ruta de base de datos guardada, intentar reconectar
            if self._db_path:
                logger.info(f"Intentando reconectar a la base de datos {self._db_path} para guardar metadatos")
                if self.connect(self._db_path):
                    logger.info("Reconexión exitosa, continuando con el almacenamiento de metadatos")
                else:
                    logger.error(f"No se pudo reconectar a la base de datos {self._db_path}")
                    return False
            else:
                return False
            
        try:
            # Verificar si la tabla existe
            try:
                self._conn.execute("SELECT 1 FROM db_metadata LIMIT 1")
            except Exception:
                # Si hay error, crear la tabla
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS db_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            
            # Convertir el valor a string si es necesario
            if not isinstance(value, str):
                if isinstance(value, (int, float, bool, type(None))):
                    value = str(value)
                else:
                    value = json.dumps(value)
            
            # Insertar o actualizar el metadato
            self._conn.execute("""
                INSERT OR REPLACE INTO db_metadata (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP);
            """, (key, value))
            
            return True
        except Exception as e:
            logger.error(f"Error al almacenar metadato '{key}': {e}")
            return False
    
    def _add_neighbors_to_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Añade chunks vecinos a los resultados de búsqueda.
        
        Args:
            results: Lista de resultados de búsqueda
            
        Returns:
            Lista aumentada con chunks vecinos
        """
        if not results:
            return results
            
        final_results = results.copy()
        
        for result in results:
            neighbors = self._get_adjacent_chunks(result['document_id'], result['id'])
            
            # Añadir vecinos a los resultados, marcándolos como vecinos
            for neighbor in neighbors:
                neighbor['is_neighbor'] = True
                neighbor['neighbor_of'] = result['id']
                # Asegurar que no estemos duplicando chunks
                if not any(r['id'] == neighbor['id'] for r in final_results):
                    final_results.append(neighbor)
        
        return final_results
    
    def begin_transaction(self) -> bool:
        """
        Inicia una transacción en DuckDB.
        
        Returns:
            bool: True si se inició correctamente, False en caso contrario
        """
        try:
            if not self._conn:
                logger.error("No hay conexión a la base de datos para iniciar transacción")
                return False
                
            # Iniciar transacción en DuckDB
            self._conn.execute("BEGIN TRANSACTION")
            logger.debug("Transacción iniciada en DuckDB")
            return True
        except Exception as e:
            logger.error(f"Error al iniciar transacción en DuckDB: {e}")
            return False
    
    def commit_transaction(self) -> bool:
        """
        Confirma una transacción activa en DuckDB.
        
        Returns:
            bool: True si se confirmó correctamente, False en caso contrario
        """
        try:
            if not self._conn:
                logger.error("No hay conexión a la base de datos para confirmar transacción")
                return False
                
            # Confirmar transacción en DuckDB
            self._conn.execute("COMMIT")
            logger.debug("Transacción confirmada en DuckDB")
            return True
        except Exception as e:
            logger.error(f"Error al confirmar transacción en DuckDB: {e}")
            # Intentar hacer rollback en caso de error
            try:
                self._conn.execute("ROLLBACK")
            except:
                pass
            return False
    
    def rollback_transaction(self) -> bool:
        """
        Revierte una transacción activa en DuckDB.
        
        Returns:
            bool: True si se revirtió correctamente, False en caso contrario
        """
        try:
            if not self._conn:
                logger.error("No hay conexión a la base de datos para revertir transacción")
                return False
                
            # Revertir transacción en DuckDB
            self._conn.execute("ROLLBACK")
            logger.debug("Transacción revertida en DuckDB")
            return True
        except Exception as e:
            logger.error(f"Error al revertir transacción en DuckDB: {e}")
            return False