from abc import ABC, abstractmethod
import logging
import json
import struct
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

class VectorialDatabase(ABC):
    """
    Clase abstracta que define la interfaz para las bases de datos vectoriales.
    
    Define métodos para:
    - Conectar/desconectar a la base de datos
    - Crear el esquema
    - Insertar documentos y chunks
    - Buscar documentos por embedding
    - Etc.
    
    Las clases que implementen esta interfaz deben proporcionar una implementación
    concreta para estos métodos.
    """
    
    def __init__(self):
        """Inicializa el logger y otros atributos comunes."""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._conn = None
        self._cursor = None
        self._db_path = None
        self._metadata = {}  # Caché local de metadatos
    
    @abstractmethod
    def connect(self, db_path):
        """
        Conecta a la base de datos.
        
        Args:
            db_path (str): Ruta a la base de datos
            
        Returns:
            bool: True si la conexión fue exitosa, False en caso contrario
        """
        pass
    
    @abstractmethod
    def close_connection(self):
        """
        Cierra la conexión a la base de datos.
        
        Returns:
            bool: True si el cierre fue exitoso, False en caso contrario
        """
        pass
    
    def close(self):
        """
        Alias para close_connection() por compatibilidad.
        
        Returns:
            bool: True si el cierre fue exitoso, False en caso contrario
        """
        return self.close_connection()
    
    @abstractmethod
    def create_schema(self):
        """
        Crea el esquema de la base de datos si no existe.
        
        Returns:
            bool: True si la creación fue exitosa, False en caso contrario
        """
        pass
    
    @abstractmethod
    def insert_document(self, document, chunks):
        """
        Inserta un documento y sus chunks en la base de datos.
        
        Args:
            document (dict): Diccionario con los datos del documento
            chunks (list): Lista de diccionarios con los datos de los chunks
            
        Returns:
            int: ID del documento insertado, None si falla
        """
        pass
    
    @abstractmethod
    def get_chunks_by_document(self, document_id, offset=0, limit=100):
        """
        Obtiene los chunks de un documento.
        
        Args:
            document_id (int): ID del documento
            offset (int, optional): Offset para paginación
            limit (int, optional): Límite para paginación
            
        Returns:
            list: Lista de chunks
        """
        pass
    
    @abstractmethod
    def vector_search(self, query_embedding, filters=None, n_results=5, include_neighbors=False):
        """
        Realiza una búsqueda por similitud vectorial.
        
        Args:
            query_embedding (list): Vector de embedding de la consulta
            filters (dict, optional): Filtros para la búsqueda
            n_results (int, optional): Número máximo de resultados
            include_neighbors (bool, optional): Si se incluyen los chunks vecinos en los resultados
            
        Returns:
            list: Lista de chunks ordenados por similitud
        """
        pass
    
    def document_exists(self, file_path):
        """
        Verifica si un documento ya existe en la base de datos.
        
        Args:
            file_path (str): Ruta del archivo del documento
            
        Returns:
            bool: True si el documento existe, False en caso contrario
        """
        try:
            self._cursor.execute(
                "SELECT COUNT(*) FROM documents WHERE file_path = ?", 
                (file_path,)
            )
            count = self._cursor.fetchone()[0]
            return count > 0
        except Exception as e:
            self._logger.error(f"Error al verificar existencia del documento: {str(e)}")
            return False
    
    def serialize_vector(self, vector):
        """
        Serializa un vector a bytes para almacenamiento.
        
        Args:
            vector (list): Vector de embeddings
            
        Returns:
            bytes: Vector serializado
        """
        return struct.pack(f'{len(vector)}f', *vector)
    
    def deserialize_vector(self, serialized, vector_dim=None):
        """
        Deserializa un vector desde bytes.
        
        Args:
            serialized (bytes): Vector serializado
            vector_dim (int, optional): Dimensión del vector
            
        Returns:
            list: Vector deserializado
        """
        if vector_dim is None:
            vector_dim = len(serialized) // 4  # 4 bytes por float
        
        return list(struct.unpack(f'{vector_dim}f', serialized))
    
    def insert_chunks(self, chunks):
        """
        Inserta múltiples chunks en la base de datos.
        
        Args:
            chunks (list): Lista de diccionarios con los datos de los chunks.
                Cada chunk debe contener al menos 'text', 'embedding', 'metadata'
        
        Returns:
            bool: True si la inserción fue exitosa, False en caso contrario.
        """
        # Implementación por defecto que debe ser sobreescrita
        return False
    
    def insert_chunk(self, text, embedding, metadata=None):
        """
        Inserta un único chunk en la base de datos.
        
        Args:
            text (str): Texto del chunk
            embedding (list): Vector de embedding
            metadata (dict, optional): Metadatos asociados al chunk
            
        Returns:
            str: ID del chunk insertado, None si falla
        """
        # Implementación por defecto que debe ser sobreescrita
        return None

    @abstractmethod
    def insert_document_metadata(self, document):
        """
        Inserta solo los metadatos de un documento en la base de datos, sin los chunks.
        Útil para el procesamiento en streaming de documentos grandes.
        
        Args:
            document (dict): Diccionario con los datos del documento
            
        Returns:
            int: ID del documento insertado, None si falla
        """
        pass
    
    @abstractmethod
    def insert_single_chunk(self, document_id, chunk_data):
        """
        Inserta un solo chunk asociado a un documento específico.
        
        Args:
            document_id (int): ID del documento al que pertenece el chunk
            chunk_data (dict): Diccionario con los datos del chunk:
                - text (str): Texto del chunk
                - header (str, opcional): Encabezado del chunk
                - page (str, opcional): Número o identificador de página
                - embedding (list): Vector de embedding del chunk
                - embedding_dim (int): Dimensión del embedding
                
        Returns:
            int: ID del chunk insertado, None si falla
        """
        pass

    def get_db_path(self) -> str:
        """
        Devuelve la ruta de la base de datos.
        
        Returns:
            La ruta de la base de datos o None si no se ha conectado todavía.
        """
        return self._db_path
    
    # --- NUEVOS MÉTODOS PARA GESTIÓN DE METADATOS Y OPTIMIZACIÓN ---
    
    def store_metadata(self, key: str, value: Any) -> bool:
        """
        Almacena un metadato en la base de datos.
        
        Args:
            key: Clave del metadato
            value: Valor del metadato (debe ser serializable a JSON)
            
        Returns:
            bool: True si se almacenó correctamente
        """
        try:
            # Crear tabla de metadatos si no existe
            self._cursor.execute("""
                CREATE TABLE IF NOT EXISTS db_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Serializar el valor a JSON si es necesario
            if not isinstance(value, (str, int, float, bool, type(None))):
                value = json.dumps(value)
                
            # Insertar o actualizar el metadato
            self._cursor.execute("""
                INSERT OR REPLACE INTO db_metadata (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP);
            """, (key, value))
            
            self._conn.commit()
            self._metadata[key] = value  # Actualizar caché local
            return True
            
        except Exception as e:
            self._logger.error(f"Error al almacenar metadato {key}: {e}")
            return False
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Recupera un metadato de la base de datos.
        
        Args:
            key: Clave del metadato
            default: Valor por defecto si la clave no existe
            
        Returns:
            El valor del metadato, o el valor por defecto si no existe
        """
        # Primero intentar obtener de la caché en memoria
        if key in self._metadata:
            return self._metadata[key]
            
        try:
            # Verificar si existe la tabla
            self._cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='db_metadata';
            """)
            if not self._cursor.fetchone():
                return default
                
            # Obtener el metadato
            self._cursor.execute("SELECT value FROM db_metadata WHERE key = ?;", (key,))
            result = self._cursor.fetchone()
            
            if result:
                try:
                    # Intentar deserializar JSON por si acaso
                    value = json.loads(result[0])
                    self._metadata[key] = value
                    return value
                except (json.JSONDecodeError, TypeError):
                    # Si no es JSON, devolver el valor tal cual
                    self._metadata[key] = result[0]
                    return result[0]
            
            return default
            
        except Exception as e:
            self._logger.error(f"Error al recuperar metadato {key}: {e}")
            return default
    
    def list_metadata(self) -> Dict[str, Any]:
        """
        Lista todos los metadatos almacenados en la base de datos.
        
        Returns:
            Dict[str, Any]: Diccionario con todos los metadatos
        """
        try:
            # Verificar si existe la tabla
            self._cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='db_metadata';
            """)
            if not self._cursor.fetchone():
                return {}
                
            # Obtener todos los metadatos
            self._cursor.execute("SELECT key, value FROM db_metadata;")
            results = self._cursor.fetchall()
            
            metadata = {}
            for key, value in results:
                try:
                    # Intentar deserializar JSON
                    metadata[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # Si no es JSON, usar el valor tal cual
                    metadata[key] = value
            
            # Actualizar caché en memoria
            self._metadata.update(metadata)
            return metadata
            
        except Exception as e:
            self._logger.error(f"Error al listar metadatos: {e}")
            return {}
    
    @abstractmethod
    def optimize_database(self) -> bool:
        """
        Optimiza la base de datos (compactación, recreación de índices, etc.)
        
        Returns:
            bool: True si la optimización fue exitosa, False en caso contrario
        """
        pass
    
    # --- MÉTODOS PARA MANEJO DE TRANSACCIONES ---
    
    def begin_transaction(self) -> bool:
        """
        Inicia una transacción manual para inserción masiva.
        Útil para mejorar rendimiento con muchas inserciones.
        
        Returns:
            bool: True si se inició correctamente la transacción
        """
        try:
            if hasattr(self, "_conn") and self._conn:
                self._conn.execute("BEGIN TRANSACTION;")
                self._logger.info("Transacción iniciada")
                return True
            return False
        except Exception as e:
            self._logger.error(f"Error al iniciar transacción: {e}")
            return False
    
    def commit_transaction(self) -> bool:
        """
        Confirma una transacción en curso.
        
        Returns:
            bool: True si se confirmó correctamente la transacción
        """
        try:
            if hasattr(self, "_conn") and self._conn:
                self._conn.commit()
                self._logger.info("Transacción confirmada")
                return True
            return False
        except Exception as e:
            self._logger.error(f"Error al confirmar transacción: {e}")
            return False
    
    def rollback_transaction(self) -> bool:
        """
        Revierte una transacción en curso.
        
        Returns:
            bool: True si se revirtió correctamente la transacción
        """
        try:
            if hasattr(self, "_conn") and self._conn:
                self._conn.rollback()
                self._logger.info("Transacción revertida")
                return True
            return False
        except Exception as e:
            self._logger.error(f"Error al revertir transacción: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la base de datos (documentos, chunks, etc.)
        
        Returns:
            Dict[str, Any]: Estadísticas de la base de datos
        """
        stats = {}
        
        try:
            # Total de documentos
            self._cursor.execute("SELECT COUNT(*) FROM documents;")
            stats["total_documents"] = self._cursor.fetchone()[0]
            
            # Total de chunks
            self._cursor.execute("SELECT COUNT(*) FROM chunks;")
            stats["total_chunks"] = self._cursor.fetchone()[0]
            
            # Documento más reciente
            self._cursor.execute("""
                SELECT id, title, created_at FROM documents
                ORDER BY created_at DESC LIMIT 1;
            """)
            doc = self._cursor.fetchone()
            if doc:
                stats["latest_document"] = {
                    "id": doc[0],
                    "title": doc[1],
                    "created_at": doc[2]
                }
            
            # Fecha de creación de la base de datos
            stats["db_created"] = self.get_metadata("db_created", "unknown")
            
            # Tamaño de la base de datos
            try:
                if self._db_path and self._db_path != ":memory:":
                    import os
                    stats["db_size_mb"] = os.path.getsize(self._db_path) / (1024 * 1024)
            except Exception:
                pass
            
            return stats
            
        except Exception as e:
            self._logger.error(f"Error al obtener estadísticas: {e}")
            return {"error": str(e)}
            
    def convert_embedding_dimension(self, embedding: List[float], target_dim: int) -> List[float]:
        """
        Convierte un embedding a una dimensión específica (truncando o rellenando).
        Útil para compatibilidad entre diferentes modelos.
        
        Args:
            embedding: Vector de embedding original
            target_dim: Dimensión objetivo
            
        Returns:
            Vector de embedding convertido
        """
        if len(embedding) == target_dim:
            return embedding
            
        if len(embedding) > target_dim:
            # Truncar si es más grande
            return embedding[:target_dim]
        else:
            # Rellenar con ceros
            return embedding + [0.0] * (target_dim - len(embedding))
    
    def _vector_search_manual(self, query_embedding: List[float], filters=None, n_results=5, include_neighbors=False):
        """
        Implementación manual de búsqueda vectorial cuando no hay soporte nativo.
        Esta implementación carga todos los embeddings y calcula similitud manualmente.
        
        Args:
            query_embedding: Vector de consulta
            filters: Filtros adicionales para la búsqueda
            n_results: Número máximo de resultados
            include_neighbors: Si se incluyen chunks vecinos
            
        Returns:
            Lista de chunks ordenados por similitud
        """
        try:
            # Convertir embedding de consulta a array numpy para cálculos eficientes
            query_vector = np.array(query_embedding, dtype=np.float32)
            
            # Normalizar el vector de consulta para similitud por coseno
            query_norm = np.linalg.norm(query_vector)
            if query_norm > 0:
                query_vector = query_vector / query_norm
            
            # Obtener todos los chunks con sus embeddings
            self._cursor.execute("""
                SELECT c.id, c.document_id, c.text, c.embedding, c.embedding_dim, 
                       c.header, c.page, d.title, d.url, d.file_path
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
            """)
            
            rows = self._cursor.fetchall()
            
            # Calcular similitud para cada chunk
            similarities = []
            
            for row in rows:
                # Extraer campos según el orden de la consulta
                chunk_id = row[0]
                doc_id = row[1]
                text = row[2]
                embedding_blob = row[3]
                embedding_dim = row[4] if row[4] else self._embedding_dim
                header = row[5]
                page = row[6]
                title = row[7]
                url = row[8]
                file_path = row[9]
                
                if not embedding_blob:
                    continue
                
                # Deserializar embedding
                try:
                    chunk_vector = np.array(self.deserialize_vector(embedding_blob, embedding_dim), dtype=np.float32)
                except Exception as e:
                    self._logger.warning(f"Error al deserializar vector del chunk {chunk_id}: {str(e)}")
                    continue
                
                # Normalizar el vector del chunk
                chunk_norm = np.linalg.norm(chunk_vector)
                if chunk_norm > 0:
                    chunk_vector = chunk_vector / chunk_norm
                
                # Calcular similitud de coseno
                similarity = np.dot(query_vector, chunk_vector)
                
                # Aplicar filtros si existen
                if filters:
                    if 'document_id' in filters and doc_id != filters['document_id']:
                        continue
                    if 'min_similarity' in filters and similarity < filters['min_similarity']:
                        continue
                
                # Solo incluir resultados por encima del umbral
                similarity_threshold = filters.get('min_similarity', self._similarity_threshold) if filters else self._similarity_threshold
                if similarity >= similarity_threshold:
                    similarities.append({
                        "id": chunk_id,
                        "document_id": doc_id,
                        "text": text,
                        "header": header,
                        "page": page,
                        "title": title,
                        "url": url,
                        "file_path": file_path,
                        "similarity": float(similarity)
                    })
            
            # Ordenar por similitud descendente
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Limitar a n_results
            top_results = similarities[:n_results]
            
            # Incluir chunks vecinos si se solicita
            if include_neighbors and top_results:
                best_match = top_results[0]
                neighbors = self._get_adjacent_chunks(best_match["document_id"], best_match["id"])
                
                if neighbors:
                    # Agregar vecinos al principio de los resultados
                    return neighbors + top_results
            
            return top_results
            
        except Exception as e:
            self._logger.error(f"Error en búsqueda vectorial manual: {str(e)}")
            return []
            
    def _get_adjacent_chunks(self, document_id, chunk_id):
        """
        Obtiene chunks adyacentes al chunk especificado.
        Implementación básica que debe ser personalizada en clases concretas si es necesario.
        
        Args:
            document_id: ID del documento
            chunk_id: ID del chunk
            
        Returns:
            Lista de chunks adyacentes
        """
        adjacent_chunks = []
        
        try:
            # Obtener el chunk anterior (ID menor más cercano)
            self._cursor.execute("""
                SELECT c.id, c.document_id, c.text, c.header, c.page, d.title, d.url, d.file_path
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = ? AND c.id < ?
                ORDER BY c.id DESC
                LIMIT 1
            """, [document_id, chunk_id])
            
            prev_chunk = self._cursor.fetchone()
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
            self._cursor.execute("""
                SELECT c.id, c.document_id, c.text, c.header, c.page, d.title, d.url, d.file_path
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = ? AND c.id > ?
                ORDER BY c.id ASC
                LIMIT 1
            """, [document_id, chunk_id])
            
            next_chunk = self._cursor.fetchone()
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
            self._logger.error(f"Error al obtener chunks adyacentes: {str(e)}")
            return []

    def insert_chunks_batch(self, document_id, chunks_data):
        """
        Inserta un lote de chunks asociados a un documento específico.
        
        Este método permite la inserción eficiente de múltiples chunks en una sola operación,
        lo que puede mejorar sustancialmente el rendimiento al reducir el número de
        operaciones individuales y aprovechar transacciones.
        
        Las implementaciones concretas pueden optimizar este proceso utilizando
        características específicas del motor de base de datos subyacente.
        
        Args:
            document_id (int): ID del documento al que pertenecen los chunks
            chunks_data (List[dict]): Lista de diccionarios con los datos de cada chunk:
                - text (str): Texto del chunk
                - header (str, opcional): Encabezado del chunk
                - page (str, opcional): Número o identificador de página
                - embedding (list): Vector de embedding del chunk
                - embedding_dim (int): Dimensión del embedding
                
        Returns:
            List[int]: Lista de IDs de los chunks insertados, o None si falla
        """
        # Implementación por defecto que inserta cada chunk individualmente
        # Las clases derivadas deberían sobrescribir esto con una implementación más eficiente
        self._logger.debug(f"Insertando batch de {len(chunks_data)} chunks usando método por defecto")
        chunk_ids = []
        try:
            # Asegurarse de que estamos en una transacción
            in_transaction = getattr(self, '_in_transaction', False)
            if not in_transaction:
                self.begin_transaction()
                transaction_started = True
            else:
                transaction_started = False
                
            # Procesar cada chunk
            for chunk_data in chunks_data:
                chunk_id = self.insert_single_chunk(document_id, chunk_data)
                if chunk_id:
                    chunk_ids.append(chunk_id)
                else:
                    self._logger.warning("Fallo al insertar chunk individual en batch")
            
            # Commit solo si iniciamos la transacción
            if transaction_started:
                self.commit_transaction()
                
            return chunk_ids
        except Exception as e:
            self._logger.error(f"Error al insertar batch de chunks: {e}")
            # Rollback solo si iniciamos la transacción
            if 'transaction_started' in locals() and transaction_started:
                self.rollback_transaction()
            return None