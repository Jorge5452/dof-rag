"""
Aplicación principal del sistema RAG.

Esta clase encapsula toda la funcionalidad principal del sistema RAG en un objeto
reutilizable, facilitando su integración en diferentes contextos (CLI, API, chatbot).
"""

import logging
import time
import os
import json
import gc
from typing import List, Dict, Any, Optional, Union, Generator, Tuple
import sqlite3
import glob
import weakref
import uuid

from config import Config
from modulos.clientes.FactoryClient import ClientFactory
from modulos.databases.FactoryDatabase import DatabaseFactory
from modulos.embeddings.embeddings_factory import EmbeddingFactory
from colorama import Fore, Style
from modulos.session_manager.session_manager import SessionManager

logger = logging.getLogger(__name__)
config = Config()

# Mantener un registro de todas las instancias RagApp activas
_active_instances = weakref.WeakSet()

class RagApp:
    """
    Clase principal que encapsula la funcionalidad del sistema RAG.
    
    Esta clase integra todos los componentes del sistema RAG (base de datos,
    embeddings, cliente de IA) y ofrece métodos para procesar consultas.
    """
    
    def __init__(
        self,
        database_name: str,
        ai_client: Optional[str] = None,
        streaming: bool = False,
        session_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        **kwargs
    ):
        """
        Inicializa la aplicación RAG.
        
        Args:
            database_name: Nombre de la base de datos a utilizar
            ai_client: Tipo de cliente de IA a utilizar (openai, gemini, ollama)
            streaming: Si se deben generar respuestas en streaming
            session_id: ID de sesión para tracking y persistencia
            collection_name: Nombre de la colección para bases de datos que soportan múltiples colecciones
            **kwargs: Argumentos adicionales para configuración
        """
        self.database_name = database_name
        self.streaming = streaming
        self.session_id = session_id or str(uuid.uuid4())
        self.collection_name = collection_name
        self.closed = False
        self.last_activity = time.time()
        self.creation_time = time.time()
        self.query_count = 0
        
        # Capturar el ID del proceso actual para debugging
        self.process_id = os.getpid()
        
        # Inicializar base de datos
        self._initialize_database(database_name, collection_name)
        
        # Inicializar cliente de IA
        self._initialize_ai_client(ai_client, **kwargs)
        
        # Inicializar modelo de embeddings
        self._initialize_embedding_model()
        
        # Registrar esta instancia
        _active_instances.add(self)
        
        # Registrar en el gestor de sesiones si está disponible
        try:
            session_manager = SessionManager()
            session_manager.update_session_metadata(self.session_id, {
                "database_name": database_name,
                "streaming": streaming,
                "created_at": self.creation_time,
                "process_id": self.process_id,
                "collection_name": collection_name
            })
        except Exception as e:
            logger.warning(f"No se pudo registrar la sesión en el gestor: {e}")
        
        logger.info(f"RagApp inicializada para {self.database_name} (session_id={self.session_id})")
    
    def __del__(self):
        """Destructor para limpiar recursos automáticamente"""
        try:
            self.close()
        except:
            pass
    
    def close(self):
        """Cierra y libera todos los recursos utilizados por esta instancia"""
        if self.closed:
            return
            
        try:
            # Cerrar conexión a la base de datos
            if self.db is not None:
                try:
                    self.db.close()
                    logger.debug(f"Conexión a base de datos cerrada: {self.database_name}")
                except Exception as e:
                    logger.warning(f"Error al cerrar conexión a base de datos: {e}")
                finally:
                    self.db = None
            
            # Liberar referencia al modelo de embeddings
            if self.embedding_manager is not None:
                try:
                    EmbeddingFactory.release_embedding_manager(self.embedding_manager)
                    logger.debug("Referencia a modelo de embeddings liberada")
                except Exception as e:
                    logger.warning(f"Error al liberar referencia a modelo de embeddings: {e}")
                finally:
                    self.embedding_manager = None
            
            # Liberar cliente IA
            if self.ai_client is not None:
                self.ai_client = None
                logger.debug("Referencia a cliente IA liberada")
                
            self.closed = True
            logger.info("Recursos de RagApp liberados correctamente")
            
        except Exception as e:
            logger.error(f"Error al cerrar recursos de RagApp: {e}")
    
    def _initialize_database(self, database_name: str, collection_name: Optional[str] = None) -> None:
        """
        Inicializa la conexión a la base de datos.
        
        Args:
            database_name: Nombre de la base de datos
            collection_name: Nombre de la colección (opcional)
        """
        try:
            factory = DatabaseFactory()
            self.db = factory.get_database_instance(
                custom_name=database_name,
                session_id=self.session_id,
                embedding_dim=768  # Valor predeterminado, se actualizará al cargar el modelo
            )
            
            # Obtener información sobre la base de datos
            if hasattr(self.db, 'get_type'):
                self.db_type = self.db.get_type()
            else:
                self.db_type = "unknown"
                
            if hasattr(self.db, 'get_path'):
                self.db_path = self.db.get_path()
            else:
                self.db_path = "unknown"
                
            # Obtener metadatos de sesión si están disponibles
            if hasattr(self.db, 'get_metadata'):
                # Intentar obtener todos los metadatos o manejar el caso donde get_metadata requiere un key
                try:
                    # Primero intentar sin argumentos para el caso de que get_metadata devuelva todos los metadatos
                    self.session = self.db.get_metadata() or {}
                except TypeError:
                    # Si get_metadata requiere un argumento key, construir un diccionario con metadatos clave
                    self.session = {}
                    for key in ['embedding_model', 'chunking_method', 'created_at', 'session_id']:
                        try:
                            value = self.db.get_metadata(key)
                            if value:
                                self.session[key] = value
                        except Exception:
                            # Ignorar errores al obtener metadatos individuales
                            pass
            else:
                self.session = {}
                
            logger.info(f"Base de datos inicializada: {database_name}, tipo: {self.db_type}")
        except Exception as e:
            logger.error(f"Error al inicializar base de datos: {e}")
            self.db = None
            raise ValueError(f"Error al inicializar base de datos {database_name}: {e}")
    
    def _initialize_ai_client(self, ai_client_name: Optional[str] = None, **kwargs) -> None:
        """
        Inicializa el cliente de IA.
        
        Args:
            ai_client_name: Nombre del cliente de IA (opcional)
            **kwargs: Configuración adicional para el cliente
        """
        try:
            factory = ClientFactory()
            
            # Si no se especifica un cliente, usar el predeterminado de la configuración
            if not ai_client_name:
                ai_client_config = config.get_ai_client_config()
                ai_client_name = ai_client_config.get("type", "openai")
                logger.debug(f"Usando cliente de IA predeterminado: {ai_client_name}")
            
            # Configurar streaming si aplica
            kwargs["stream"] = self.streaming
            
            # Crear el cliente
            self.ai_client = factory.get_client(ai_client_name, **kwargs)
            self.ai_client_type = ai_client_name
            logger.info(f"Cliente IA inicializado: {ai_client_name}, streaming={self.streaming}")
        except Exception as e:
            logger.error(f"Error al inicializar cliente IA: {e}")
            self.ai_client = None
            self.ai_client_type = "none"
            
    def _initialize_embedding_model(self) -> None:
        """
        Inicializa el modelo de embeddings basado en la configuración de la base de datos.
        """
        try:
            # Obtener el modelo de embedding desde los metadatos de la sesión
            embedding_model = self.session.get("embedding_model", "modernbert")
            self.embedding_manager = EmbeddingFactory.get_embedding_manager(embedding_model)
            self.embedding_manager.load_model()
            logger.info(f"Modelo de embeddings inicializado: {embedding_model}")
        except Exception as e:
            logger.error(f"Error al inicializar modelo de embeddings: {e}")
            self.embedding_manager = None
            # No lanzamos excepción para permitir operaciones sin vectores
    
    def _check_closed(self) -> bool:
        """
        Verifica si la instancia ha sido cerrada.
        
        Returns:
            True si la instancia está cerrada, False en caso contrario
        """
        if self.closed:
            logger.warning(f"Intento de uso de RagApp cerrada (session_id={self.session_id})")
            return True
        return False
    
    def query(self, query_text: str, return_context: bool = False) -> Union[
            str, Tuple[str, List[Dict[str, Any]]], Generator[str, None, None]
        ]:
        """
        Procesa una consulta y devuelve la respuesta.
        
        Args:
            query_text: Texto de la consulta
            return_context: Si debe devolver también el contexto usado
            
        Returns:
            Respuesta generada o tupla (respuesta, contexto) si return_context=True,
            o un generador para streaming
        """
        if self._check_closed():
            return "Esta sesión ha sido cerrada. Por favor inicie una nueva sesión."
        
        # Actualizar timestamp de última actividad
        self.last_activity = time.time()
        self.query_count += 1
        
        # Generar un ID único para este mensaje
        message_id = str(uuid.uuid4())
        
        try:
            start_time = time.time()
            logger.info(f"Iniciando consulta: '{query_text}' (session_id={self.session_id}, message_id={message_id})")
            
            # Verificar que la base de datos esté inicializada
            if not self.db:
                logger.error("Base de datos no inicializada")
                return "Error: Base de datos no inicializada."
            
            # Obtener embedding de la consulta
            if self.embedding_manager:
                query_embedding = self.embedding_manager.get_query_embedding(query_text)
            else:
                logger.error("Modelo de embeddings no disponible")
                return "Error: Modelo de embeddings no disponible."
            
            # Realizar búsqueda de documentos similares
            context_data = self.db.vector_search(query_embedding, 
                                                n_results=config.get_processing_config().get("max_chunks_to_retrieve", 5))
            
            if not context_data:
                logger.warning("No se encontró contexto relevante para la consulta")
                context_data = []
            else:
                logger.debug(f"Contexto recuperado: {len(context_data)} fragmentos")
            
            # Si no hay cliente IA, devolver solo el contexto
            if not self.ai_client:
                logger.warning("Cliente IA no disponible, devolviendo solo contexto")
                response = "No se puede generar respuesta: Cliente IA no disponible."
                
                if return_context:
                    return response, context_data
                return response
            
            # Generar respuesta usando el cliente de IA
            if self.streaming and not return_context:
                # Crear un generador para streaming solo si no se solicita el contexto
                def response_generator():
                    try:
                        # Iniciar streaming
                        for chunk in self.ai_client.generate_response(query_text, context_data):
                            yield chunk
                        
                        # Al finalizar, almacenar el contexto en el gestor de sesiones
                        try:
                            session_manager = SessionManager()
                            session_manager.store_message_context(self.session_id, message_id, context_data)
                        except Exception as e:
                            logger.error(f"Error al almacenar contexto: {e}")
                        
                        # Enviar el ID del mensaje como último chunk
                        yield f"\n<hidden_message_id>{message_id}</hidden_message_id>"
                        
                    except Exception as e:
                        logger.error(f"Error en streaming de respuesta: {e}")
                        yield f"\nError durante la generación: {str(e)}"
                        yield f"\n<hidden_message_id>{message_id}</hidden_message_id>"
                
                return response_generator()
            else:
                # Si se solicita el contexto o no está en modo streaming, generar respuesta completa
                if self.streaming:
                    # Simulamos una respuesta para devolver con el contexto
                    # ya que no podemos devolver un generador y el contexto a la vez
                    response = "Procesando consulta en modo streaming..."
                else:
                    # Generar respuesta completa normalmente
                    response = self.ai_client.generate_response(query_text, context_data)
                
                # Almacenar contexto en el gestor de sesiones
                try:
                    session_manager = SessionManager()
                    session_manager.store_message_context(self.session_id, message_id, context_data)
                except Exception as e:
                    logger.error(f"Error al almacenar contexto: {e}")
                
                # Registrar tiempo de procesamiento
                process_time = time.time() - start_time
                logger.info(f"Consulta completada en {process_time:.2f}s (session_id={self.session_id}, tokens={len(response.split())})")
                
                # Añadir el ID del mensaje a la respuesta no streaming de forma más discreta
                if not self.streaming:
                    response = f"{response}\n<hidden_message_id>{message_id}</hidden_message_id>"
                
                if return_context:
                    return response, context_data
                return response
                
        except Exception as e:
            logger.error(f"Error al procesar consulta: {e}", exc_info=True)
            error_msg = f"Error al procesar la consulta: {str(e)}"
            
            if return_context:
                return error_msg, []
            return error_msg
    
    @staticmethod
    def list_available_databases() -> List[Dict[str, Any]]:
        """
        Lista todas las bases de datos disponibles directamente leyendo los archivos.
        
        Returns:
            Lista de diccionarios con información de las bases de datos
        """
        # Obtener la configuración de base de datos
        database_config = config.get_database_config()
        
        # Determinar el directorio de bases de datos
        db_dir = database_config.get("sqlite", {}).get("db_dir", "modulos/databases/db")
        
        # Asegurar que la ruta es absoluta
        if not os.path.isabs(db_dir):
            db_dir = os.path.join(os.path.abspath(os.getcwd()), db_dir)
        
        # Verificar que el directorio existe
        if not os.path.exists(db_dir):
            logger.warning(f"El directorio de bases de datos no existe: {db_dir}")
            return []
        
        databases = []
        
        # Buscar archivos SQLite
        sqlite_files = glob.glob(os.path.join(db_dir, "*.sqlite"))
        for file_path in sqlite_files:
            db_name = os.path.basename(file_path).replace(".sqlite", "")
            
            # Intentar leer metadatos
            meta_path = f"{file_path}.meta.json"
            metadata = {}
            
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                except:
                    pass
            
            db_info = {
                "id": db_name,
                "name": db_name,
                "type": "sqlite",
                "path": file_path,
                "size": os.path.getsize(file_path),
                "created_at": metadata.get("created_at", os.path.getctime(file_path)),
                "metadata": metadata
            }
            
            databases.append(db_info)
        
        # Buscar archivos DuckDB
        duckdb_files = glob.glob(os.path.join(db_dir, "*.duckdb"))
        for file_path in duckdb_files:
            db_name = os.path.basename(file_path).replace(".duckdb", "")
            
            # Intentar leer metadatos
            meta_path = f"{file_path}.meta.json"
            metadata = {}
            
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                except:
                    pass
            
            db_info = {
                "id": db_name,
                "name": db_name,
                "type": "duckdb",
                "path": file_path,
                "size": os.path.getsize(file_path),
                "created_at": metadata.get("created_at", os.path.getctime(file_path)),
                "metadata": metadata
            }
            
            databases.append(db_info)
        
        # Ordenar por fecha de creación (más reciente primero)
        databases.sort(key=lambda x: x["created_at"], reverse=True)
        
        return databases
    
    @staticmethod
    def cleanup_instances():
        """
        Limpia y libera los recursos de todas las instancias activas.
        Útil para llamar durante el cierre del programa.
        """
        instances_count = len(_active_instances)
        
        if instances_count > 0:
            logger.info(f"Limpiando {instances_count} instancias RagApp activas")
            
            for instance in list(_active_instances):
                try:
                    instance.close()
                except:
                    pass
            
            # Forzar recolección de basura
            _active_instances.clear()
            gc.collect()
            
            logger.info("Limpieza de instancias RagApp completada")
        else:
            logger.debug("No hay instancias RagApp activas para limpiar") 