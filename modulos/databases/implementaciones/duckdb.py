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
            
            # En DuckDB, la conexión en memoria se especifica con ':memory:'
            if db_path == ':memory:':
                self._conn = duckdb.connect(database=':memory:')
            else:
                # Asegurarse de que el directorio existe
                os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
                self._conn = duckdb.connect(database=db_path)
            
            # Inicializar el cursor para compatibilidad con la clase abstracta
            self._cursor = self._conn
            
            logger.info(f"Conexión exitosa a DuckDB: {db_path}")
            
            # Crear el esquema tras conectar exitosamente
            self.create_schema()
            
            return True
        except Exception as e:
            logger.error(f"Error al conectar a DuckDB: {e}")
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
            # Crear secuencia para IDs de documentos
            self._conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS doc_id_seq;
            """)
            
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
            self._conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS chunk_id_seq;
            """)
            
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
            
            # Crear índice para búsquedas rápidas por document_id
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
            
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
            
        try:
            # Intentar cargar la extensión httpfs para acceder a datos remotos
            self._conn.execute("INSTALL httpfs;")
            self._conn.execute("LOAD httpfs;")
            
            self._ext_loaded = True
            logger.info("Extensiones DuckDB (httpfs) cargadas correctamente")
            return True
        except Exception as e:
            logger.warning(f"No se pudieron cargar extensiones DuckDB: {e}")
            return False
    
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
        
        # Verificar si el documento ya existe, usando el file_path
        file_path = document.get("file_path", "")
        if file_path and self.document_exists(file_path):
            try:
                # Obtener el ID del documento existente
                result = self._conn.execute("SELECT id FROM documents WHERE file_path = ?", (file_path,)).fetchone()
                if result:
                    doc_id = result[0]
                    logger.info(f"El documento ya existe con ID: {doc_id}, se omite la inserción")
                    return doc_id
            except Exception as e:
                logger.warning(f"Error al verificar documento existente: {e}")
                # Continuamos con la inserción normal en caso de error
        
        try:
            # Comenzar transacción
            self._conn.execute("BEGIN TRANSACTION")
            
            # Insertar documento sin especificar ID (se usará la secuencia doc_id_seq)
            self._conn.execute("""
                INSERT INTO documents (title, url, file_path)
                VALUES (?, ?, ?)
            """, (
                document.get("title", "Sin título"),
                document.get("url", f"local://{file_path if file_path else 'unknown'}"),
                file_path
            ))
            
            # Obtener el ID generado para el documento
            result = self._conn.execute("SELECT last_insert_rowid()").fetchone()
            doc_id = result[0]
            
            # Preparar para inserción masiva de chunks
            chunk_params = []
            for chunk in chunks:
                embedding = chunk.get("embedding")
                embedding_blob = None
                
                # Procesar y serializar el embedding con la dimensión fija
                if embedding is not None:
                    if isinstance(embedding, list):
                        embedding_blob = self.serialize_vector(embedding)
                    elif isinstance(embedding, np.ndarray):
                        embedding_blob = self.serialize_vector(embedding.tolist())
                
                # Añadir parámetros para este chunk (sin incluir el ID, que será generado por la secuencia)
                chunk_params.append((
                    doc_id,
                    chunk.get("text", ""),
                    chunk.get("header", None),
                    chunk.get("page", None),
                    embedding_blob,
                    self._embedding_dim if embedding_blob else None
                ))
            
            # Insertar todos los chunks de una vez, sin especificar IDs
            if chunk_params:
                self._conn.executemany("""
                    INSERT INTO chunks (document_id, text, header, page, embedding, embedding_dim)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, chunk_params)
            
            # Confirmar transacción
            self._conn.execute("COMMIT")
            
            logger.info(f"Documento insertado correctamente con ID: {doc_id} y {len(chunks)} chunks")
            return doc_id
        except Exception as e:
            # Revertir transacción en caso de error
            self._conn.execute("ROLLBACK")
            logger.error(f"Error al insertar documento: {e}")
            raise
    
    def document_exists(self, file_path: str) -> bool:
        """
        Verifica si un documento ya existe en la base de datos.
        
        Args:
            file_path: Ruta al archivo del documento
            
        Returns:
            True si el documento existe, False en caso contrario
        """
        if not self._conn:
            logger.error("No hay conexión a la base de datos")
            return False
        
        try:
            result = self._conn.execute(
                "SELECT COUNT(*) FROM documents WHERE file_path = ?",
                [file_path]
            ).fetchone()
            
            return result[0] > 0 if result else False
        except Exception as e:
            logger.error(f"Error al verificar existencia del documento: {e}")
            return False
    
    def get_chunks_by_document(self, document_id: int, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtiene todos los chunks asociados a un documento.
        
        Args:
            document_id: ID del documento
            offset: Desplazamiento para paginación
            limit: Límite de resultados para paginación
            
        Returns:
            Lista de chunks asociados al documento
        """
        if not self._conn:
            logger.error("No hay conexión a la base de datos")
            return []
        
        try:
            # Obtener todos los chunks del documento
            result = self._conn.execute("""
                SELECT c.id, c.document_id, c.text, c.embedding, c.embedding_dim, c.header, c.page,
                       d.title, d.url, d.file_path, c.created_at
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = ?
                ORDER BY c.id
                LIMIT ? OFFSET ?
            """, [document_id, limit, offset]).fetchall()
            
            chunks = []
            for row in result:
                chunk_id, doc_id, text, embedding_blob, embedding_dim, header, page, title, url, file_path, created_at = row
                
                chunks.append({
                    "id": chunk_id,
                    "document_id": doc_id,
                    "text": text,
                    "header": header,
                    "page": page,
                    "title": title,
                    "url": url,
                    "file_path": file_path,
                    "created_at": created_at
                })
            
            logger.info(f"Recuperados {len(chunks)} chunks para el documento ID: {document_id} (offset: {offset}, limit: {limit})")
            return chunks
        except Exception as e:
            logger.error(f"Error al obtener chunks del documento {document_id}: {e}")
            return []

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calcula la similitud de coseno entre dos vectores.
        
        Args:
            vec1: Primer vector
            vec2: Segundo vector
            
        Returns:
            Valor de similitud de coseno entre 0 y 1
        """
        # Asegurar que los vectores tengan la misma dimensión
        min_dim = min(len(vec1), len(vec2))
        vec1 = vec1[:min_dim]
        vec2 = vec2[:min_dim]
        
        # Calcular similitud de coseno: cos(θ) = (a·b)/(|a|·|b|)
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        # Evitar división por cero
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def vector_search(self, 
                      query_embedding: List[float], 
                      n_results: int = 5, 
                      include_neighbors: bool = False) -> List[Dict[str, Any]]:
        """
        Realiza una búsqueda vectorial por similitud de coseno.
        
        El embedding de consulta se adapta automáticamente a la dimensión fija configurada.
        
        Args:
            query_embedding: Vector de consulta
            n_results: Número máximo de resultados a devolver
            include_neighbors: Si es True, incluye chunks vecinos (mismo documento)
            
        Returns:
            Lista de chunks más similares al vector de consulta
        """
        if not self._conn:
            logger.error("No hay conexión a la base de datos")
            return []
        
        try:
            # Adaptar el embedding a la dimensión fija
            if len(query_embedding) != self._embedding_dim:
                if len(query_embedding) > self._embedding_dim:
                    query_embedding = query_embedding[:self._embedding_dim]
                    logger.debug(f"Embedding de consulta truncado a {self._embedding_dim} dimensiones")
                else:
                    query_embedding = query_embedding + [0.0] * (self._embedding_dim - len(query_embedding))
                    logger.debug(f"Embedding de consulta rellenado a {self._embedding_dim} dimensiones")
            
            # Convertir embedding de consulta a array numpy
            query_vector = np.array(query_embedding, dtype=np.float32)
            
            # Normalizar el vector de consulta para similitud por coseno
            query_norm = np.linalg.norm(query_vector)
            if query_norm > 0:
                query_vector = query_vector / query_norm
            
            # Obtener todos los chunks con sus embeddings
            result = self._conn.execute("""
                SELECT c.id, c.document_id, c.text, c.embedding, c.embedding_dim, 
                       c.header, c.page, d.file_path, d.title, d.url
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                ORDER BY c.id
            """).fetchall()
            
            # Calcular similitud de coseno manualmente
            similarities = []
            for row in result:
                chunk_id, doc_id, text, embedding_blob, _, header, page, file_path, title, url = row
                
                if not embedding_blob:
                    continue
                
                # Deserializar embedding - siempre usar la dimensión fija de la clase
                try:
                    chunk_vector = np.array(self.deserialize_vector(embedding_blob), dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Error al deserializar vector del chunk {chunk_id}: {e}")
                    continue
                
                # Normalizar el vector del chunk para similitud por coseno
                chunk_norm = np.linalg.norm(chunk_vector)
                if chunk_norm > 0:
                    chunk_vector = chunk_vector / chunk_norm
                
                # Calcular similitud de coseno
                similarity = np.dot(query_vector, chunk_vector)
                
                # Solo incluir resultados por encima del umbral de similitud
                if similarity >= self._similarity_threshold:
                    similarities.append({
                        "id": chunk_id,
                        "document_id": doc_id,
                        "text": text,
                        "header": header,
                        "page": page,
                        "file_path": file_path,
                        "title": title,
                        "url": url,
                        "similarity": float(similarity)
                    })
            
            # Ordenar por similitud descendente
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Tomar los primeros n_results
            top_results = similarities[:n_results]
            
            # Si se solicita incluir vecinos, añadir chunks del mismo documento
            if include_neighbors and top_results:
                best_match = top_results[0]
                neighbors = self._get_adjacent_chunks(best_match["document_id"], best_match["id"])
                
                if neighbors:
                    # Agregar los vecinos al principio de los resultados
                    results = neighbors + [best_match] + top_results[1:]
                    return results
            
            logger.info(f"Búsqueda vectorial completada. Encontrados {len(top_results)} resultados.")
            return top_results
        except Exception as e:
            logger.error(f"Error en búsqueda vectorial: {e}")
            return []
    
    def _get_adjacent_chunks(self, document_id: int, chunk_id: int) -> List[Dict[str, Any]]:
        """
        Obtiene los chunks adyacentes (anterior y siguiente) a un chunk específico.
        
        Args:
            document_id: ID del documento
            chunk_id: ID del chunk de referencia
            
        Returns:
            Lista con los chunks adyacentes (anterior y siguiente)
        """
        adjacent_chunks = []
        
        try:
            # Obtener el chunk anterior (ID menor más cercano)
            prev_chunk = self._conn.execute("""
                SELECT c.id, c.document_id, c.text, c.header, c.page, d.title, d.url, d.file_path
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = ? AND c.id < ?
                ORDER BY c.id DESC
                LIMIT 1
            """, [document_id, chunk_id]).fetchone()
            
            if prev_chunk:
                chunk_id, doc_id, text, header, page, title, url, file_path = prev_chunk
                adjacent_chunks.append({
                    "id": chunk_id,
                    "document_id": doc_id,
                    "text": text,
                    "header": header,
                    "page": page,
                    "title": title,
                    "url": url,
                    "file_path": file_path,
                    "similarity": 0.0  # Marcar como vecino con similitud 0
                })
            
            # Obtener el chunk siguiente (ID mayor más cercano)
            next_chunk = self._conn.execute("""
                SELECT c.id, c.document_id, c.text, c.header, c.page, d.title, d.url, d.file_path
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = ? AND c.id > ?
                ORDER BY c.id ASC
                LIMIT 1
            """, [document_id, chunk_id]).fetchone()
            
            if next_chunk:
                chunk_id, doc_id, text, header, page, title, url, file_path = next_chunk
                adjacent_chunks.append({
                    "id": chunk_id,
                    "document_id": doc_id,
                    "text": text,
                    "header": header,
                    "page": page,
                    "title": title,
                    "url": url,
                    "file_path": file_path,
                    "similarity": 0.0  # Marcar como vecino con similitud 0
                })
            
            return adjacent_chunks
        except Exception as e:
            logger.error(f"Error al obtener chunks adyacentes: {e}")
            return []
    
    def create_vector_index(self, force_rebuild: bool = False) -> bool:
        """
        Crea o reconstruye un índice vectorial para búsquedas rápidas.
        
        Args:
            force_rebuild: Si es True, fuerza la reconstrucción del índice aunque ya exista
            
        Returns:
            True si se creó correctamente, False en caso contrario
        """
        # DuckDB actualmente no soporta índices vectoriales nativos
        # Esta función es un placeholder para compatibilidad con la interfaz
        logger.info("Índices vectoriales no soportados nativamente en DuckDB")
        return True
    
    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Inserta múltiples chunks en la base de datos.
        
        Los embeddings se adaptan automáticamente a la dimensión fija configurada.
        
        Args:
            chunks: Lista de diccionarios con los datos de los chunks
                Cada chunk debe contener al menos 'text' y 'embedding'
        
        Returns:
            bool: True si la inserción fue exitosa, False en caso contrario.
        """
        if not self._conn:
            logger.error("No hay conexión a la base de datos")
            return False
            
        try:
            # Iniciar transacción
            self._conn.execute("BEGIN TRANSACTION")
            
            # Preparar parámetros para inserción masiva
            chunk_params = []
            
            for chunk in chunks:
                # Serializar el embedding usando la dimensión fija
                embedding = chunk.get('embedding')
                embedding_blob = None
                
                if embedding is not None:
                    if isinstance(embedding, list):
                        embedding_blob = self.serialize_vector(embedding)
                    elif isinstance(embedding, np.ndarray):
                        embedding_blob = self.serialize_vector(embedding.tolist())
                
                # Extraer otros campos
                document_id = chunk.get('document_id')
                text = chunk.get('text', '')
                header = chunk.get('header')
                page = chunk.get('page')
                
                # Añadir a la lista de parámetros (sin incluir el ID, que será generado por la secuencia)
                chunk_params.append((
                    document_id,
                    text,
                    header,
                    page,
                    embedding_blob,
                    self._embedding_dim if embedding_blob else None
                ))
            
            # Insertar todos los chunks de una vez, sin especificar IDs
            if chunk_params:
                self._conn.executemany("""
                    INSERT INTO chunks (document_id, text, header, page, embedding, embedding_dim)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, chunk_params)
            
            # Confirmar transacción
            self._conn.execute("COMMIT")
            
            logger.info(f"Se insertaron {len(chunks)} chunks en la base de datos DuckDB")
            return True
            
        except Exception as e:
            self._conn.execute("ROLLBACK")
            logger.error(f"Error al insertar chunks en DuckDB: {e}")
            return False
    
    def insert_chunk(self, text: str, embedding: List[float], metadata: Dict[str, Any] = None) -> str:
        """
        Inserta un único chunk en la base de datos.
        
        Args:
            text: Texto del chunk
            embedding: Vector de embedding
            metadata: Metadatos asociados al chunk
            
        Returns:
            ID del chunk insertado como string, None si falla
        """
        if not self._conn:
            logger.error("No hay conexión a la base de datos")
            return None
            
        try:
            # Preparar el embedding
            embedding_blob = self.serialize_vector(embedding) if embedding else None
            
            # Extraer campos específicos de metadata si existen
            document_id = None
            header = None
            page = None
            
            if metadata:
                document_id = metadata.get('document_id')
                header = metadata.get('header')
                page = metadata.get('page')
            
            # Insertar el chunk sin especificar ID (se usará la secuencia chunk_id_seq)
            self._conn.execute("""
                INSERT INTO chunks (document_id, text, header, page, embedding, embedding_dim)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                document_id,
                text,
                header,
                page,
                embedding_blob,
                self._embedding_dim if embedding_blob else None
            ])
            
            # Obtener el ID generado
            result = self._conn.execute("SELECT last_insert_rowid()").fetchone()
            chunk_id = result[0]
            
            return str(chunk_id)
            
        except Exception as e:
            logger.error(f"Error al insertar chunk en DuckDB: {e}")
            return None
            
    def optimize_database(self) -> bool:
        """
        Optimiza la base de datos realizando operaciones de limpieza y compactación.
        
        Returns:
            True si se optimizó correctamente, False en caso contrario
        """
        if not self._conn:
            logger.error("No hay conexión a la base de datos")
            return False
        
        if self._db_path == ':memory:':
            logger.info("Base de datos en memoria, no se requiere optimización")
            return True
            
        try:
            self._conn.execute("VACUUM;")
            logger.info("Base de datos DuckDB optimizada correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al optimizar la base de datos DuckDB: {e}")
            return False
    
    def vector_search(self, 
                     query_embedding: List[float], 
                     filters: Optional[Dict[str, Any]] = None, 
                     n_results: int = 5, 
                     include_neighbors: bool = False) -> List[Dict[str, Any]]:
        """
        Realiza una búsqueda vectorial por similitud de coseno.
        
        El embedding de consulta se adapta automáticamente a la dimensión fija configurada.
        
        Args:
            query_embedding: Vector de consulta
            filters: Filtros para la búsqueda (por ejemplo, document_id, min_similarity)
            n_results: Número máximo de resultados a devolver
            include_neighbors: Si es True, incluye chunks vecinos (mismo documento)
            
        Returns:
            Lista de chunks más similares al vector de consulta
        """
        if not self._conn:
            logger.error("No hay conexión a la base de datos")
            return []
        
        # DuckDB no tiene una implementación nativa de búsqueda vectorial
        # así que usamos la implementación manual de la clase base
        return self._vector_search_manual(
            query_embedding=query_embedding,
            filters=filters,
            n_results=n_results,
            include_neighbors=include_neighbors
        )
    
    def _get_adjacent_chunks(self, document_id: int, chunk_id: int) -> List[Dict[str, Any]]:
        """
        Obtiene los chunks adyacentes (anterior y siguiente) a un chunk específico.
        
        Args:
            document_id: ID del documento
            chunk_id: ID del chunk de referencia
            
        Returns:
            Lista con los chunks adyacentes (anterior y siguiente)
        """
        # Implementación específica para DuckDB
        adjacent_chunks = []
        
        try:
            # Obtener el chunk anterior (ID menor más cercano)
            prev_chunk = self._conn.execute("""
                SELECT c.id, c.document_id, c.text, c.header, c.page, d.title, d.url, d.file_path
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = ? AND c.id < ?
                ORDER BY c.id DESC
                LIMIT 1
            """, [document_id, chunk_id]).fetchone()
            
            if prev_chunk:
                chunk_id, doc_id, text, header, page, title, url, file_path = prev_chunk
                adjacent_chunks.append({
                    "id": chunk_id,
                    "document_id": doc_id,
                    "text": text,
                    "header": header,
                    "page": page,
                    "title": title,
                    "url": url,
                    "file_path": file_path,
                    "similarity": 0.0  # Marcar como vecino con similitud 0
                })
            
            # Obtener el chunk siguiente (ID mayor más cercano)
            next_chunk = self._conn.execute("""
                SELECT c.id, c.document_id, c.text, c.header, c.page, d.title, d.url, d.file_path
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = ? AND c.id > ?
                ORDER BY c.id ASC
                LIMIT 1
            """, [document_id, chunk_id]).fetchone()
            
            if next_chunk:
                chunk_id, doc_id, text, header, page, title, url, file_path = next_chunk
                adjacent_chunks.append({
                    "id": chunk_id,
                    "document_id": doc_id,
                    "text": text,
                    "header": header,
                    "page": page,
                    "title": title,
                    "url": url,
                    "file_path": file_path,
                    "similarity": 0.0  # Marcar como vecino con similitud 0
                })
            
            return adjacent_chunks
            
        except Exception as e:
            logger.error(f"Error al obtener chunks adyacentes en DuckDB: {e}")
            return []