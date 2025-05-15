"""
Gestión de sesiones para el sistema RAG.

Este módulo permite gestionar sesiones de RAG, manteniendo un registro de las
configuraciones utilizadas y facilitando el acceso a bases de datos existentes.
"""

import os
import json
import logging
import time
import uuid
import re
import threading
import psutil
import gc
import glob
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import weakref

from config import Config
from modulos.databases.FactoryDatabase import DatabaseFactory
from modulos.embeddings.embeddings_factory import EmbeddingFactory
# Evitar importación directa que causa dependencia circular
# from modulos.resource_management.resource_manager import ResourceManager

logger = logging.getLogger(__name__)

# Instanciar la clase Config correctamente
config = Config()

# Método get_value personalizado para mantener compatibilidad
def get_value(section, key, default=None):
    """
    Obtiene un valor de configuración específico.
    
    Args:
        section: Sección de la configuración
        key: Clave del valor
        default: Valor por defecto si no se encuentra
        
    Returns:
        Valor de configuración o valor por defecto
    """
    try:
        # Intentar obtener el método getter correspondiente
        getter_method = getattr(config, f"get_{section}_config", None)
        if getter_method is None:
            # Si el método no existe, buscar en la configuración general
            general_config = config.get_general_config() or {}
            if section in general_config:
                return general_config[section].get(key, default)
            return default
            
        # Obtener la configuración de la sección
        section_config = getter_method()
        # Verificar que section_config no sea None antes de llamar a get()
        if section_config is None:
            return default
            
        return section_config.get(key, default)
    except (AttributeError, KeyError, Exception) as e:
        logger.warning(f"Error al obtener valor de configuración {section}.{key}: {e}")
        return default

# Configuración de límites y timeouts para sesiones
try:
    # Intentar obtener configuración de sesiones con manejo de errores
    MAX_SESSIONS = get_value("sessions", "max_sessions", 50)  
    SESSION_TIMEOUT = get_value("sessions", "timeout", 3600)  # 1 hora por defecto
    CLEANUP_INTERVAL = get_value("sessions", "cleanup_interval", 300)  # 5 minutos por defecto
    MAX_CONTEXTS_PER_SESSION = get_value("sessions", "max_contexts", 50)  # Máximo número de contextos por sesión
except Exception as e:
    # Si falla, usar valores predeterminados y registrar el error
    logger.debug(f"Usando valores predeterminados para configuración de sesiones: {e}")
    MAX_SESSIONS = 50
    SESSION_TIMEOUT = 3600
    CLEANUP_INTERVAL = 300
    MAX_CONTEXTS_PER_SESSION = 50

class SessionManager:
    """
    Gestiona las sesiones de usuarios y sus datos asociados.
    
    Implementa un patrón singleton para asegurar una única instancia global.
    Incluye funcionalidades para:
    - Crear y mantener sesiones
    - Almacenar datos de contexto por mensaje
    - Limpiar sesiones inactivas
    - Monitorear uso de recursos
    """
    
    _instance = None
    _lock = threading.RLock()
    
    # Valores predeterminados de configuración
    MAX_SESSIONS = 100            # Límite máximo de sesiones activas
    SESSION_TIMEOUT = 3600 * 24 * 7  # 1 semana por defecto, podría ser configurable
    CLEANUP_INTERVAL = 60         # Intervalo (en segundos) entre limpiezas automáticas
    MAX_CONTEXTS_PER_SESSION = 30 # Número máximo de contextos almacenados por sesión
    
    def __new__(cls):
        """
        Implementación del patrón singleton.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SessionManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """
        Inicializa el gestor de sesiones.
        """
        # Evitar reinicialización del singleton
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized: # Doble check por si otro hilo inicializó
                return
            
            self.config = Config()
            self.general_config = self.config.get_general_config()
            self.sessions_dir = Path(self.general_config.get("sessions_dir", "sessions"))
            self.sessions_dir.mkdir(parents=True, exist_ok=True)

            self.sessions_file = self.sessions_dir / "sessions.json"
            self.db_metadata_file = self.sessions_dir / "db_metadata.json"

            self.sessions: Dict[str, Any] = self._load_from_file(self.sessions_file)
            self.db_metadata: Dict[str, Any] = self._load_from_file(self.db_metadata_file)
            
            # Validar y limpiar metadatos de BD al inicio (eliminar entradas sin archivo físico)
            self._validate_db_metadata()

            # Inicializar espacio para contextos
            self.contexts = {}

            # Inicializar referencia a ResourceManager (inicialización perezosa)
            self._resource_manager = None

            self._initialized = True
            logger.info("SessionManager inicializado. La limpieza ahora es coordinada por ResourceManager.")
    
    def _load_from_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Carga datos desde un archivo JSON.
        
        Args:
            file_path: Ruta al archivo JSON
            
        Returns:
            Diccionario con los datos cargados
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error al cargar datos desde {file_path}: {e}")
            return {}
    
    def _validate_db_metadata(self):
        """
        Valida y limpia metadatos de base de datos al inicio.
        """
        try:
            # Obtener la lista de bases de datos disponibles
            databases = self.list_available_databases()
            
            # Filtrar metadatos de bases de datos que no tienen archivo físico
            self.db_metadata = {db_name: metadata for db_name, metadata in databases.items() if metadata.get('db_path')}
            
            # Guardar metadatos actualizados
            self._save_to_file(self.db_metadata_file, self.db_metadata)
            
            logger.info("Metadatos de base de datos actualizados y guardados correctamente")
        except Exception as e:
            logger.error(f"Error al validar y limpiar metadatos de base de datos: {e}")
    
    def _save_to_file(self, file_path: Path, data: Dict[str, Any]):
        """
        Guarda datos en un archivo JSON.
        
        Args:
            file_path: Ruta al archivo JSON
            data: Datos a guardar
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Error al guardar datos en {file_path}: {e}")
    
    def create_session(self, session_id: Optional[str] = None, 
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Crea una nueva sesión.
        
        Args:
            session_id: ID de sesión personalizado (opcional)
            metadata: Datos adicionales para la sesión
        
        Returns:
            ID de la sesión creada
        
        Raises:
            ValueError: Si se alcanza el límite máximo de sesiones
        """
        with self._lock:
            # Verificar límite de sesiones
            if len(self.sessions) >= self.MAX_SESSIONS:
                # Intentar limpiar primero
                self.clean_expired_sessions(aggressive=True)
                
                # Si aún hay demasiadas sesiones, rechazar la creación
                if len(self.sessions) >= self.MAX_SESSIONS:
                    logger.error(f"Límite de sesiones alcanzado: {self.MAX_SESSIONS}")
                    raise ValueError(f"Se alcanzó el límite máximo de sesiones ({self.MAX_SESSIONS})")
            
            # Generar ID si no se proporciona
            if not session_id:
                # UUID simple basado en timestamp
                session_id = f"session_{int(time.time() * 1000)}"
            
            # Verificar que no exista ya
            if session_id in self.sessions:
                logger.warning(f"La sesión {session_id} ya existe, actualizando")
                return self.update_session_metadata(session_id, metadata)
            
            # Crear la sesión
            session_data = {
                "id": session_id,
                "created_at": time.time(),
                "last_activity": time.time(),
                "files": [],  # Lista vacía para almacenar archivos procesados
                "processing_stats": {
                    "total_files_processed": 0,
                    "total_chunks_generated": 0,
                    "creation_date": datetime.now().isoformat()
                }
            }
            
            # Añadir metadata si se proporciona
            if metadata:
                session_data.update(metadata)
            
            # Almacenar sesión
            self.sessions[session_id] = session_data
            
            # Inicializar espacio para contextos
            self.contexts[session_id] = {}
            
            # Persistir inmediatamente
            self._save_sessions()
            
            logger.info(f"Sesión {session_id} creada correctamente")
            return session_id
    
    def update_session_metadata(self, session_id: str, 
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Actualiza los metadatos de una sesión existente.
        
        Args:
            session_id: ID de la sesión
            metadata: Datos actualizados para la sesión
        
        Returns:
            ID de la sesión actualizada
        
        Raises:
            KeyError: Si la sesión no existe
        """
        with self._lock:
            # Verificar que exista la sesión
            if session_id not in self.sessions:
                logger.warning(f"Intento de actualizar sesión inexistente: {session_id}, creando nueva")
                return self.create_session(session_id, metadata)
            
            # Actualizar timestamp de actividad
            self.sessions[session_id]["last_activity"] = time.time()
            
            # Añadir o actualizar metadata si se proporciona
            if metadata:
                self.sessions[session_id].update(metadata)
            
            # Persistir los cambios inmediatamente
            try:
                self._save_to_file(self.sessions_file, self.sessions)
                logger.debug(f"Sesiones guardadas después de actualizar {session_id}")
            except Exception as e:
                logger.error(f"Error al guardar sesiones después de actualizar {session_id}: {e}")
            
            return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene los datos de una sesión.
        
        Args:
            session_id: ID de la sesión
        
        Returns:
            Datos de la sesión o None si no existe
        """
        with self._lock:
            if session_id not in self.sessions:
                return None
            
            # Actualizar timestamp de actividad al acceder
            self.sessions[session_id]["last_activity"] = time.time()
            
            # Devolver copia para evitar modificaciones no controladas
            return dict(self.sessions[session_id])
    
    def delete_session(self, session_id: str) -> bool:
        """
        Elimina una sesión y sus datos asociados.
        
        Args:
            session_id: ID de la sesión a eliminar
        
        Returns:
            True si se eliminó correctamente, False si no existía
        """
        with self._lock:
            if session_id not in self.sessions:
                return False
            
            # Eliminar contextos asociados
            if session_id in self.contexts:
                del self.contexts[session_id]
            
            # Registrar metadata antes de eliminar
            created_at = self.sessions[session_id].get("created_at", 0)
            last_activity = self.sessions[session_id].get("last_activity", 0)
            duration = int(time.time() - created_at)
            
            # Eliminar sesión
            del self.sessions[session_id]
            
            # Guardar los cambios en el archivo para mantener persistencia
            try:
                self._save_to_file(self.sessions_file, self.sessions)
                logger.debug(f"Sesiones guardadas después de eliminar {session_id}")
            except Exception as e:
                logger.error(f"Error al guardar sesiones después de eliminar {session_id}: {e}")
            
            logger.info(f"Sesión eliminada: {session_id} (duración: {duration}s, "
                      f"inactividad: {int(time.time() - last_activity)}s)")
            return True
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene todas las sesiones activas.
        
        Returns:
            Diccionario con todas las sesiones {session_id: session_data}
        """
        with self._lock:
            # Devolver copia para evitar modificaciones no controladas
            return {sid: dict(data) for sid, data in self.sessions.items()}
    
    def store_message_context(self, session_id: str, message_id: str, 
                            context_data: List[Dict[str, Any]]) -> bool:
        """
        Almacena el contexto utilizado para generar una respuesta.
        
        Args:
            session_id: ID de la sesión
            message_id: ID del mensaje
            context_data: Datos de contexto a almacenar
        
        Returns:
            True si se almacenó correctamente, False en caso contrario
        """
        with self._lock:
            # Verificar que exista la sesión
            if session_id not in self.sessions:
                logger.warning(f"Intento de almacenar contexto en sesión inexistente: {session_id}")
                return False
            
            # Actualizar timestamp de actividad
            self.sessions[session_id]["last_activity"] = time.time()
            
            # Inicializar diccionario de contextos si no existe
            if session_id not in self.contexts:
                self.contexts[session_id] = {}
            
            # Verificar límite de contextos por sesión
            contexts = self.contexts[session_id]
            if len(contexts) >= self.MAX_CONTEXTS_PER_SESSION:
                # Eliminar el contexto más antiguo
                oldest_message_id = min(contexts.keys())
                del contexts[oldest_message_id]
                logger.debug(f"Límite de contextos alcanzado para sesión {session_id}, eliminando el más antiguo")
            
            # Almacenar contexto
            self.contexts[session_id][message_id] = context_data
            
            logger.debug(f"Contexto almacenado: sesión={session_id}, mensaje={message_id}, "
                       f"fragmentos={len(context_data)}")
            return True
    
    def get_message_context(self, session_id: str, message_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Obtiene el contexto utilizado para un mensaje específico.
        
        Args:
            session_id: ID de la sesión
            message_id: ID del mensaje
        
        Returns:
            Datos de contexto o None si no existe
        """
        with self._lock:
            # Verificar que exista la sesión
            if session_id not in self.sessions or session_id not in self.contexts:
                return None
            
            # Actualizar timestamp de actividad al acceder
            self.sessions[session_id]["last_activity"] = time.time()
            
            # Obtener contexto
            if message_id not in self.contexts[session_id]:
                return None
            
            # Devolver copia para evitar modificaciones no controladas
            return list(self.contexts[session_id][message_id])
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el uso de recursos.
        
        Returns:
            Diccionario con métricas de recursos
        """
        with self._lock:
            # Actualizar información antes de devolver
            self._update_resource_usage()
            
            # Devolver copia para evitar modificaciones no controladas
            return dict(self.resource_usage)
    
    def list_available_databases(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Lista todas las bases de datos disponibles con sus metadatos.
        
        Args:
            force_refresh: Si es True, fuerza una nueva búsqueda en disco
            
        Returns:
            Diccionario de bases de datos {db_name: metadata}
        """
        # Usar caché si está disponible y no se fuerza actualización
        if hasattr(self, '_db_cache') and not force_refresh:
            # Verificar si el caché no es muy viejo (5 minutos)
            if hasattr(self, '_db_cache_time') and time.time() - self._db_cache_time < 300:
                return self._db_cache
        
        # Obtener la configuración de base de datos
        database_config = config.get_database_config()
        
        # Lista para todas las bases de datos encontradas
        databases = {}
        
        # Buscar en todos los tipos de bases de datos compatibles
        db_types = ['sqlite', 'duckdb']
        
        for db_type in db_types:
            # Determinar el directorio de bases de datos para este tipo
            db_dir = database_config.get(db_type, {}).get("db_dir", f"modulos/databases/db/{db_type}")
            
            # Asegurar que la ruta es absoluta
            if not os.path.isabs(db_dir):
                db_dir = os.path.join(os.path.abspath(os.getcwd()), db_dir)
            
            # Verificar que el directorio existe
            if not os.path.exists(db_dir):
                logger.debug(f"El directorio de bases de datos {db_type} no existe: {db_dir}")
                continue
            
            # Buscar archivos de base de datos
            extensions = ['.sqlite', '.db'] if db_type == 'sqlite' else ['.duckdb']
            
            for extension in extensions:
                db_files = glob.glob(os.path.join(db_dir, f"*{extension}"))
                
                for file_path in db_files:
                    db_name = os.path.basename(file_path).replace(extension, "")
                    
                    # Buscar metadatos en múltiples ubicaciones
                    metadata = self._find_database_metadata(file_path, db_name)
                    
                    if not metadata:
                        # Crear metadatos básicos si no se encuentran
                        metadata = {
                            "id": db_name,
                            "name": db_name,
                            "db_type": db_type,
                            "db_path": file_path,
                            "size": os.path.getsize(file_path),
                            "created_at": os.path.getctime(file_path),
                            "last_used": os.path.getmtime(file_path)
                        }
                    else:
                        # Asegurar que hay campos básicos
                        metadata.update({
                            "id": db_name,
                            "name": db_name,
                            "db_type": metadata.get("db_type", db_type),
                            "db_path": file_path,
                            "size": os.path.getsize(file_path)
                        })
                    
                    # Añadir a la lista de bases de datos
                    databases[db_name] = metadata
        
        # Guardar en caché
        self._db_cache = databases
        self._db_cache_time = time.time()
        
        return databases
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        Devuelve una lista de todas las sesiones disponibles con información resumida.
        
        Returns:
            Lista de diccionarios con información resumida de sesiones
        """
        with self._lock:
            results = []
            for session_id, session_data in self.sessions.items():
                # Información básica
                session_info = {
                    "id": session_id,
                    "created_at": session_data.get("created_at", 0),
                    "last_activity": session_data.get("last_activity", 0)
                }
                
                # Conteo de archivos
                files = session_data.get("files", [])
                session_info["file_count"] = len(files)
                
                # Añadir información resumida de archivos si hay archivos
                if files:
                    # Calcular tamaño total (si está disponible)
                    total_size = 0
                    total_chunks = 0
                    file_list = []
                    
                    for file_item in files:
                        if isinstance(file_item, dict):
                            file_name = file_item.get("name", os.path.basename(file_item.get("path", "unknown")))
                            file_info = {"name": file_name}
                            
                            # Añadir tamaño si está disponible
                            if "size" in file_item:
                                file_info["size"] = file_item["size"]
                                total_size += file_item["size"]
                                
                            # Añadir conteo de chunks si está disponible
                            if "chunks" in file_item:
                                file_info["chunks"] = file_item["chunks"]
                                total_chunks += file_item["chunks"]
                                
                            file_list.append(file_info)
                        elif isinstance(file_item, str):
                            file_list.append({"name": os.path.basename(file_item)})
                    
                    session_info["files"] = file_list[:5]  # Mostrar solo los primeros 5 archivos
                    if len(file_list) > 5:
                        session_info["files"].append({"name": f"...y {len(file_list) - 5} más"})
                    
                    if total_size > 0:
                        session_info["total_size"] = total_size
                    if total_chunks > 0:
                        session_info["total_chunks"] = total_chunks
                
                # Información de base de datos
                if "db_name" in session_data:
                    session_info["db_name"] = session_data["db_name"]
                    
                results.append(session_info)
            
            return results
    
    def get_database_by_index(self, index: int, session_id: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Obtiene una instancia de base de datos y sus metadatos por índice.
        
        Args:
            index: Índice de la base de datos en la lista ordenada
            session_id: ID de la sesión (opcional)
            
        Returns:
            Tupla (instancia de base de datos, metadatos)
            
        Raises:
            IndexError: Si el índice está fuera de rango
            ValueError: Si no se puede cargar la base de datos
        """
        # Obtener lista de bases de datos
        databases = {}
        
        if session_id:
            # Si hay sesión, filtrar bases de datos de esa sesión
            session_dbs = self.get_session_databases(session_id)
            for db_info in session_dbs:
                db_name = db_info['name']
                # Buscar metadatos completos
                all_dbs = self.list_available_databases()
                if db_name in all_dbs:
                    databases[db_name] = all_dbs[db_name]
        else:
            # Sin sesión, obtener todas las bases de datos
            databases = self.list_available_databases()
        
        # Convertir a lista y ordenar por último uso
        sorted_dbs = []
        for name, metadata in databases.items():
            sorted_dbs.append((name, metadata))
        
        sorted_dbs.sort(key=lambda x: x[1].get('last_used', 0), reverse=True)
        
        # Verificar índice
        if not sorted_dbs:
            raise ValueError("No hay bases de datos disponibles")
            
        if index < 0 or index >= len(sorted_dbs):
            raise IndexError(f"Índice de base de datos fuera de rango: {index}")
        
        # Obtener nombre y metadatos
        db_name, metadata = sorted_dbs[index]
        db_path = metadata.get('db_path')
        
        # Verificar campos necesarios y completar si faltan
        required_fields = ['db_type', 'embedding_model', 'embedding_dim', 'chunking_method']
        missing_fields = [field for field in required_fields if field not in metadata]
        
        if missing_fields:
            logger.warning(f"Campos faltantes en metadatos: {missing_fields}. Intentando recuperar...")
            
            # Completar campos faltantes
            if 'db_type' not in metadata and db_path:
                if db_path.endswith('.sqlite') or db_path.endswith('.db'):
                    metadata['db_type'] = 'sqlite'
                elif db_path.endswith('.duckdb'):
                    metadata['db_type'] = 'duckdb'
            
            if 'embedding_dim' not in metadata:
                # Usar una dimensión estándar como valor por defecto
                metadata['embedding_dim'] = 768
                logger.warning("Usando dimensión de embedding por defecto: 768")
                
            if 'embedding_model' not in metadata:
                embedding_config = config.get_embedding_config()
                metadata['embedding_model'] = embedding_config.get('model', 'modernbert')
                
            if 'chunking_method' not in metadata:
                chunks_config = config.get_chunks_config()
                metadata['chunking_method'] = chunks_config.get('method', 'character')
        
        # Cargar la base de datos con todos los parámetros necesarios
        try:
            from modulos.databases.FactoryDatabase import DatabaseFactory
            db = DatabaseFactory.get_database_instance(
                db_type=metadata.get('db_type', 'sqlite'),
                embedding_dim=metadata.get('embedding_dim', 768),
                embedding_model=metadata.get('embedding_model', 'modernbert'),
                chunking_method=metadata.get('chunking_method', 'character'),
                load_existing=True,
                db_path=db_path
            )
            
            # Actualizar timestamp de último uso
            metadata['last_used'] = time.time()
            self.register_database(db_name, metadata)
            
            # Si hay una sesión, asociar la base de datos
            if session_id:
                self.associate_database_with_session(session_id, db_name, metadata)
            
            return db, metadata
        except Exception as e:
            logger.error(f"Error al cargar base de datos {db_name}: {e}")
            raise ValueError(f"No se pudo cargar la base de datos: {e}")
    
    def register_database(self, db_name: str, metadata: Dict[str, Any]) -> bool:
        """
        Registra una nueva base de datos en el sistema con sus metadatos asociados.
        
        Args:
            db_name: Nombre de la base de datos (debe ser igual al session_id)
            metadata: Metadatos de la base de datos (tipo, modelo, dimensiones, etc.)
            
        Returns:
            True si se registró correctamente, False en caso contrario
        """
        try:
            # Asegurar que existe la ruta del archivo de metadatos
            db_path = metadata.get('db_path')
            if not db_path:
                logger.error(f"No se pudo registrar la base de datos {db_name}: falta la ruta del archivo")
                return False
                
            # Verificar que todos los campos requeridos estén presentes
            required_fields = ['db_type', 'embedding_model', 'embedding_dim', 'chunking_method']
            missing_fields = [field for field in required_fields if field not in metadata]
            
            if missing_fields:
                logger.warning(f"Campos faltantes en metadatos: {missing_fields}. Intentando completar...")
                
                # Intentar completar campos faltantes
                if 'db_type' in missing_fields and 'db_path' in metadata:
                    if metadata['db_path'].endswith('.sqlite'):
                        metadata['db_type'] = 'sqlite'
                    elif metadata['db_path'].endswith('.duckdb'):
                        metadata['db_type'] = 'duckdb'
                
                if 'embedding_model' in missing_fields:
                    from config import Config
                    embedding_config = Config().get_embedding_config()
                    metadata['embedding_model'] = embedding_config.get('model', 'modernbert')
                    
                if 'chunking_method' in missing_fields:
                    from config import Config
                    chunks_config = Config().get_chunks_config()
                    metadata['chunking_method'] = chunks_config.get('method', 'character')
                    
                # Verificar nuevamente los campos requeridos
                missing_fields = [field for field in required_fields if field not in metadata]
                if missing_fields:
                    logger.error(f"No se pudo completar los campos requeridos: {missing_fields}")
                    return False
            
            # Asegurar que id y name sean consistentes (mismo valor que db_name)
            metadata['id'] = db_name
            metadata['name'] = db_name
            
            # Asegurar que session_id existe y es igual a db_name (unificación de IDs)
            metadata['session_id'] = db_name
            
            # Añadir timestamps si no existen
            current_time = time.time()
            if 'created_at' not in metadata:
                metadata['created_at'] = current_time
                
            # Actualizar timestamp de último uso
            metadata["last_used"] = current_time
            
            # Guardar en la estructura interna
            self.db_metadata[db_name] = metadata
            self._save_db_metadata()
            
            # Guardar metadatos en un archivo JSON junto a la base de datos
            meta_path = f"{db_path}.meta.json"
            
            # Guardar metadatos en formato JSON
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
            # También guardar una copia en la carpeta de sesiones si está asociada a una sesión
            session_dir = os.path.join(
                self.general_config.get('sessions_dir', 'sessions'),
                db_name  # Usar db_name como session_id (ahora son iguales)
            )
            os.makedirs(session_dir, exist_ok=True)
            session_meta_path = os.path.join(session_dir, f"{db_name}.meta.json")
            
            with open(session_meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
            # Intentar guardar los metadatos dentro de la base de datos también
            try:
                from modulos.databases.FactoryDatabase import DatabaseFactory
                db = DatabaseFactory.get_database_instance(
                    db_type=metadata.get('db_type', 'sqlite'),
                    embedding_dim=metadata.get('embedding_dim', 768),
                    load_existing=True,
                    db_path=db_path
                )
                
                # Guardar todos los metadatos dentro de la base de datos
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        db.store_metadata(key, value)
                    else:
                        # Convertir a JSON para tipos complejos
                        db.store_metadata(key, json.dumps(value))
            except Exception as e:
                logger.warning(f"No se pudieron guardar metadatos dentro de la base de datos: {e}")
                
            logger.info(f"Base de datos registrada correctamente: {db_name}")
            return True
        except Exception as e:
            logger.error(f"Error al registrar base de datos {db_name}: {e}")
            return False

    def _find_database_metadata(self, file_path: str, db_name: str) -> Dict[str, Any]:
        """
        Busca metadatos de base de datos en múltiples ubicaciones.
        
        Args:
            file_path: Ruta al archivo de base de datos
            db_name: Nombre de la base de datos
            
        Returns:
            Metadatos encontrados o diccionario vacío
        """
        metadata = {}
        
        # Lista de posibles ubicaciones para metadatos
        meta_locations = [
            f"{file_path}.meta.json",  # Junto a la base de datos
        ]
        
        # Buscar en carpeta de sesiones
        sessions_dir = config.get_general_config().get('sessions_dir', 'sessions')
        if os.path.exists(sessions_dir):
            session_folders = [f for f in os.listdir(sessions_dir) 
                              if os.path.isdir(os.path.join(sessions_dir, f))]
            
            for session_id in session_folders:
                session_meta_path = os.path.join(sessions_dir, session_id, f"{db_name}.meta.json")
                meta_locations.append(session_meta_path)
        
        # Buscar metadatos en cada ubicación
        for meta_path in meta_locations:
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        
                        # Si encontramos metadatos, actualizar y salir
                        metadata.update(meta)
                        logger.debug(f"Metadatos encontrados en {meta_path}")
                        break
                except Exception as e:
                    logger.warning(f"Error al leer metadatos de {meta_path}: {e}")
        
        # Si no encontramos metadatos en archivos, intentar leerlos de la base de datos
        if not metadata:
            try:
                from modulos.databases.FactoryDatabase import DatabaseFactory
                
                # Intentar determinar tipo de base de datos por extensión
                db_type = 'sqlite'
                if file_path.endswith('.duckdb'):
                    db_type = 'duckdb'
                    
                # Intentar cargar la base de datos
                db = DatabaseFactory.get_database_instance(
                    db_type=db_type,
                    embedding_dim=384,  # Dimensión por defecto
                    load_existing=True,
                    db_path=file_path
                )
                
                # Leer todos los metadatos disponibles
                db_metadata = db.list_metadata() if hasattr(db, 'list_metadata') else {}
                metadata.update(db_metadata)
                
                logger.debug(f"Metadatos recuperados desde la base de datos: {len(db_metadata)} campos")
            except Exception as e:
                logger.debug(f"No se pudieron leer metadatos desde la base de datos: {e}")
        
        return metadata

    def associate_database_with_session(self, session_id: str, db_name: str, db_metadata: Dict[str, Any]) -> bool:
        """
        Asocia una base de datos con una sesión específica.
        
        NOTA: Este método se mantiene por compatibilidad pero ya no es necesario en el nuevo enfoque
        donde las sesiones y las bases de datos se crean juntas con el mismo ID.
        
        Args:
            session_id: ID de la sesión
            db_name: Nombre de la base de datos
            db_metadata: Metadatos de la base de datos
            
        Returns:
            True si se asoció correctamente, False en caso contrario
        """
        logger.warning("El método associate_database_with_session está obsoleto. Use create_unified_session en su lugar.")
        
        try:
            # Verificar que la sesión existe
            if session_id not in self.sessions:
                logger.warning(f"Intento de asociar base de datos a sesión inexistente: {session_id}")
                return False
            
            # En el nuevo enfoque, la base de datos y la sesión deben tener el mismo ID
            if session_id != db_name:
                logger.warning(f"IDs de sesión ({session_id}) y base de datos ({db_name}) no coinciden. " 
                              f"Se recomienda usar create_unified_session en su lugar.")
            
            # Actualizar metadatos con la sesión
            db_metadata['session_id'] = session_id
            db_metadata['id'] = db_name  # Mantener ID de base de datos
            db_metadata['name'] = db_name  # Mantener nombre de base de datos
            
            # Registrar la base de datos
            success = self.register_database(db_name, db_metadata)
            
            if not success:
                return False
            
            # Actualizar información de la sesión
            with self._lock:
                # Inicializar lista de bases de datos si no existe
                if 'databases' not in self.sessions[session_id]:
                    self.sessions[session_id]['databases'] = []
                
                # Añadir referencia a la base de datos
                db_info = {
                    'name': db_name,
                    'path': db_metadata.get('db_path', ''),
                    'model': db_metadata.get('embedding_model', 'unknown'),
                    'chunking': db_metadata.get('chunking_method', 'unknown'),
                    'associated_at': time.time()
                }
                
                # Verificar si ya existe
                exists = False
                for i, existing_db in enumerate(self.sessions[session_id]['databases']):
                    if existing_db['name'] == db_name:
                        # Actualizar en lugar de añadir
                        self.sessions[session_id]['databases'][i] = db_info
                        exists = True
                        break
                
                # Añadir si no existe
                if not exists:
                    self.sessions[session_id]['databases'].append(db_info)
                
                # Actualizar timestamp de actividad
                self.sessions[session_id]['last_activity'] = time.time()
                
                # Guardar cambios en sesiones
                self._save_sessions()
                
                return True
        except Exception as e:
            logger.error(f"Error al asociar base de datos {db_name} con sesión {session_id}: {e}")
            return False

    def get_session_databases(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Obtiene las bases de datos asociadas a una sesión.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Lista de bases de datos asociadas a la sesión
        """
        with self._lock:
            if session_id not in self.sessions:
                return []
            
            # Devolver bases de datos asociadas o lista vacía
            return self.sessions[session_id].get('databases', [])

    def store_session_config(self, session_id: str, config_data: Dict[str, Any]) -> bool:
        """
        Almacena datos de configuración específicos de una sesión.
        
        Args:
            session_id: ID de la sesión
            config_data: Datos de configuración a almacenar
            
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            with self._lock:
                # Verificar que la sesión existe
                if session_id not in self.sessions:
                    logger.warning(f"Intento de guardar configuración en sesión inexistente: {session_id}")
                    return False
                
                # Obtener la configuración actual o inicializar
                if 'config' not in self.sessions[session_id]:
                    self.sessions[session_id]['config'] = {}
                
                # Actualizar con los nuevos datos
                self.sessions[session_id]['config'].update(config_data)
                
                # Actualizar timestamp de actividad
                self.sessions[session_id]['last_activity'] = time.time()
                
                # Guardar en disco si es posible
                try:
                    sessions_dir = config.get_general_config().get('sessions_dir', 'sessions')
                    session_path = os.path.join(sessions_dir, session_id)
                    os.makedirs(session_path, exist_ok=True)
                    
                    config_path = os.path.join(session_path, 'config.json')
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(self.sessions[session_id]['config'], f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"No se pudo guardar configuración en disco: {e}")
                
                return True
        except Exception as e:
            logger.error(f"Error al guardar configuración en sesión {session_id}: {e}")
            return False

    def get_session_config(self, session_id: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Recupera datos de configuración de una sesión.
        
        Args:
            session_id: ID de la sesión
            key: Clave específica a recuperar (opcional)
            default: Valor por defecto si no se encuentra
            
        Returns:
            Datos de configuración completos o valor específico
        """
        with self._lock:
            # Verificar que la sesión existe
            if session_id not in self.sessions:
                logger.warning(f"Intento de recuperar configuración de sesión inexistente: {session_id}")
                return default
            
            # Si no hay configuración guardada
            if 'config' not in self.sessions[session_id]:
                return default
            
            # Si se solicita una clave específica
            if key is not None:
                return self.sessions[session_id]['config'].get(key, default)
            
            # Devolver toda la configuración
            return dict(self.sessions[session_id]['config'])

    def _save_sessions(self) -> None:
        """
        Guarda el estado actual de las sesiones en el archivo de sesiones.
        
        Este método es llamado después de modificar sesiones, como al crear
        una nueva sesión, eliminar una existente o después de una limpieza masiva.
        """
        try:
            with self._lock:
                # Guardar sesiones actuales en el archivo
                self._save_to_file(self.sessions_file, self.sessions)
            logger.debug(f"Sesiones guardadas en {self.sessions_file}")
        except Exception as e:
            logger.error(f"Error al guardar sesiones: {e}")

    def _save_db_metadata(self) -> None:
        """
        Guarda los metadatos de bases de datos en su archivo correspondiente.
        """
        try:
            with self._lock:
                # Guardar metadatos de bases de datos
                self._save_to_file(self.db_metadata_file, self.db_metadata)
            logger.debug(f"Metadatos de BD guardados en {self.db_metadata_file}")
        except Exception as e:
            logger.error(f"Error al guardar metadatos de bases de datos: {e}")

    def clean_expired_sessions(self, aggressive: bool = False, cleanup_reason: str = "routine") -> Dict[str, Any]:
        """
        Limpia sesiones expiradas y recursos asociados.

        Esta función ahora es típicamente invocada por `ResourceManager.request_cleanup`.
        La decisión de si la limpieza es `aggressive` la toma `ResourceManager`
        basándose en los umbrales globales de recursos.

        Args:
            aggressive (bool): Indica si se debe realizar una limpieza más agresiva
                               (eliminar sesiones más antiguas si aún hay demasiadas
                               después de eliminar las expiradas).
                               Defaults to False.
            cleanup_reason (str): Motivo por el cual se está ejecutando la limpieza.
                                  Defaults to "routine".

        Returns:
            Dict[str, Any]: Resultados de la limpieza (sesiones eliminadas, errores).
        """
        results = {
            "status": "success", 
            "timeout_removed": 0,
            "aggressive_removed": 0,
            "remaining_sessions": len(self.sessions)
        }
        logger.info(f"Iniciando clean_expired_sessions. Invocación agresiva: {aggressive}, Razón: {cleanup_reason}")
        with self._lock:
            try:
                current_time = time.time()
                sessions_to_remove_by_timeout = []
                
                # Identificar sesiones expiradas por timeout
                for session_id, session_data in list(self.sessions.items()): # Iterar sobre una copia para modificar el original
                    last_activity = session_data.get("last_activity", 0)
                    if current_time - last_activity > self.SESSION_TIMEOUT:
                        sessions_to_remove_by_timeout.append(session_id)
                
                for session_id in sessions_to_remove_by_timeout:
                    if session_id in self.sessions: # Doble check por si acaso
                        self.delete_session(session_id)
                        results["timeout_removed"] += 1
                
                if results["timeout_removed"] > 0:
                    logger.info(f"{results['timeout_removed']} sesiones eliminadas por timeout.")

                # Si la limpieza es agresiva y aún hay demasiadas sesiones
                # (MAX_SESSIONS es un límite superior, aquí actuamos si estamos cerca)
                if aggressive and len(self.sessions) > self.MAX_SESSIONS * 0.8: 
                    logger.warning(f"Limpieza agresiva de sesiones activada debido a alta presión y {len(self.sessions)}/{self.MAX_SESSIONS} sesiones.")
                    # Ordenar por actividad (más antiguas primero), excluyendo las que se actualizaron muy recientemente
                    # para evitar eliminar sesiones que acaban de interactuar.
                    min_inactive_time_for_aggressive = 60 # No eliminar si se usó en el último minuto, por ejemplo
                    
                    eligible_for_aggressive_removal = []
                    for session_id, session_data in self.sessions.items():
                        if current_time - session_data.get("last_activity", 0) > min_inactive_time_for_aggressive:
                            eligible_for_aggressive_removal.append((session_id, session_data.get("last_activity", 0)))
                    
                    # Ordenar las elegibles por más antiguas primero
                    eligible_for_aggressive_removal.sort(key=lambda x: x[1])
                    
                    num_to_remove_aggressively = len(self.sessions) - int(self.MAX_SESSIONS * 0.7) # Reducir al 70%
                    num_to_remove_aggressively = max(0, num_to_remove_aggressively)

                    if num_to_remove_aggressively > 0 and eligible_for_aggressive_removal:
                        actual_removed_aggressively = 0
                        for i in range(min(num_to_remove_aggressively, len(eligible_for_aggressive_removal))):
                            session_id_to_remove = eligible_for_aggressive_removal[i][0]
                            if session_id_to_remove in self.sessions:
                                logger.warning(f"Eliminando sesión {session_id_to_remove} agresivamente por restricción de recursos.")
                                self.delete_session(session_id_to_remove)
                                results["aggressive_removed"] += 1
                                actual_removed_aggressively += 1
                        if actual_removed_aggressively > 0:
                            logger.info(f"{actual_removed_aggressively} sesiones eliminadas agresivamente.")

            except Exception as e:
                logger.error(f"Error durante clean_expired_sessions: {e}", exc_info=True)
                results["status"] = "error"
                results["error"] = str(e)
            
            # Guardar estado de sesiones después de la limpieza
            try:
                self._save_sessions()
            except Exception as e:
                logger.error(f"Error al guardar sesiones después de limpieza: {e}")
                results["sessions_save_error"] = str(e)

        results["remaining_sessions"] = len(self.sessions)
        logger.info(f"clean_expired_sessions completado. Resultados: {results}")
        return results

    def get_active_sessions_count(self) -> int:
        """Devuelve el número actual de sesiones activas."""
        with self._lock:
            return len(self.sessions)

    # Propiedad para inicialización perezosa de ResourceManager
    @property
    def resource_manager(self):
        """Obtiene la instancia de ResourceManager con inicialización perezosa y prevención de ciclos."""
        if self._resource_manager is None:
            # Detectar ciclo de inicialización
            if getattr(self, '_initializing_resource_manager', False):
                logger.warning("Ciclo de inicialización detectado entre SessionManager y ResourceManager")
                return None
            
            self._initializing_resource_manager = True
            try:
                # Importación dentro del método para evitar dependencia circular
                from modulos.resource_management.resource_manager import ResourceManager
                self._resource_manager = ResourceManager()
                logger.debug("ResourceManager recuperado para SessionManager.")
            except Exception as e:
                logger.error(f"Error al acceder a ResourceManager: {e}")
            finally:
                self._initializing_resource_manager = False
        return self._resource_manager

    def add_context(self, session_id: str, query: str, response: str, sources: Optional[List[Dict]] = None) -> None:
        """
        Añade un contexto de pregunta-respuesta a una sesión.
        
        Args:
            session_id: ID de la sesión
            query: Consulta del usuario
            response: Respuesta generada
            sources: Fuentes utilizadas para la respuesta
        """
        if session_id not in self.sessions:
            logger.warning(f"Intento de añadir contexto a una sesión inexistente: {session_id}")
            return
        
        if session_id not in self.contexts:
            self.contexts[session_id] = []
            
        # Añadir contexto
        context_entry = {
            "query": query,
            "response": response,
            "timestamp": time.time(),
            "sources": sources or []
        }
        
        # Actualizar último uso
        self.update_session_metadata(session_id, {"last_activity": time.time()})
        
        # Añadir al contexto y limitar tamaño
        self.contexts[session_id].append(context_entry)
        if len(self.contexts[session_id]) > self.MAX_CONTEXT_ENTRIES:
            # Remover los más antiguos manteniendo el máximo
            self.contexts[session_id] = self.contexts[session_id][-self.MAX_CONTEXT_ENTRIES:]

    def update_session_file_list(self, session_id: str, new_files: List[str], 
                           file_metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> bool:
        """
        Actualiza la lista de archivos en los metadatos de una sesión con una lista simple de rutas.
        
        Esta versión simplificada almacena solo las rutas de los archivos, sin metadatos adicionales.
        
        Args:
            session_id: ID de la sesión a actualizar
            new_files: Lista de rutas (str) de archivos a añadir
            file_metadata: Diccionario con metadatos adicionales (ignorado en esta implementación)
                          
        Returns:
            bool: True si la actualización fue exitosa, False en caso contrario
        """
        with self._lock:
            try:
                # Verificar si la sesión existe
                if session_id not in self.sessions:
                    logger.error(f"No se puede actualizar lista de archivos: la sesión {session_id} no existe")
                    return False
                    
                # Obtener la sesión actual
                session = self.sessions[session_id]
                
                # Inicializar o recuperar la lista de archivos
                files = session.get("files", [])
                
                # Añadir nuevas rutas, evitando duplicados
                for file_path in new_files:
                    abs_path = os.path.abspath(file_path)
                    if abs_path not in files:
                        files.append(abs_path)
                
                # Actualizar la lista en la sesión
                session["files"] = files
                
                # Guardar los cambios
                self._save_sessions()
                
                logger.info(f"Lista de archivos actualizada para la sesión {session_id}: " 
                           f"{len(files)} archivos")
                return True
                
            except Exception as e:
                logger.error(f"Error al actualizar la lista de archivos para la sesión {session_id}: {e}")
                return False

    def get_processed_files(self, session_id: str, 
                        directory_filter: Optional[str] = None,
                        pattern_filter: Optional[str] = None) -> List[str]:
        """
        Obtiene la lista de archivos procesados para una sesión con opciones de filtrado.
        
        Args:
            session_id: ID de la sesión
            directory_filter: Directorio para filtrar (opcional)
            pattern_filter: Patrón de nombre para filtrar (opcional, soporta comodines * y ?)
            
        Returns:
            Lista de rutas de archivos procesados
        """
        with self._lock:
            try:
                # Verificar si la sesión existe
                if session_id not in self.sessions:
                    logger.warning(f"No se pueden obtener archivos procesados: la sesión {session_id} no existe")
                    return []
                
                # Obtener la sesión actual
                session = self.sessions[session_id]
                
                # Comprobar si hay archivos registrados
                if "files" not in session or not session["files"]:
                    return []
                
                # Obtener lista de archivos
                files_list = session["files"]
                result_files = []
                
                # Aplicar filtros
                for file_path in files_list:
                    # Verificar que sea una ruta de archivo válida (string)
                    if not isinstance(file_path, str):
                        continue
                    
                    # Filtrar por directorio si se especifica
                    if directory_filter and not file_path.startswith(os.path.abspath(directory_filter)):
                        continue
                    
                    # Filtrar por patrón si se especifica
                    if pattern_filter:
                        import fnmatch
                        file_name = os.path.basename(file_path)
                        if not fnmatch.fnmatch(file_name, pattern_filter):
                            continue
                    
                    # Añadir a la lista de resultados
                    result_files.append(file_path)
                
                return result_files
                
            except Exception as e:
                logger.error(f"Error al obtener archivos procesados para la sesión {session_id}: {e}")
                return []

    def get_session_files_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Obtiene un resumen de los archivos procesados en una sesión.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Diccionario con estadísticas sobre los archivos procesados:
            - total_files: Número total de archivos
            - file_types: Diccionario con conteo por tipo de archivo
        """
        with self._lock:
            try:
                # Obtener todos los archivos procesados
                files = self.get_processed_files(session_id)
                
                if not files:
                    return {
                        "total_files": 0,
                        "file_types": {}
                    }
                
                # Inicializar contadores
                total_files = len(files)
                file_types = {}
                
                # Analizar cada archivo
                for file_path in files:
                    # Determinar el tipo de archivo basado en la extensión
                    if isinstance(file_path, str):
                        file_ext = os.path.splitext(file_path)[1].lower()
                        file_type = file_ext[1:] if file_ext.startswith('.') else file_ext
                        file_types[file_type] = file_types.get(file_type, 0) + 1
                
                return {
                    "total_files": total_files,
                    "file_types": file_types
                }
                
            except Exception as e:
                logger.error(f"Error al obtener resumen de archivos para la sesión {session_id}: {e}")
                return {
                    "error": str(e),
                    "total_files": 0
                }

    def create_unified_session(self, 
                          database_metadata: Dict[str, Any],
                          files_list: Optional[List[str]] = None) -> str:
        """
        Crea una sesión unificada que integra metadatos de sesión y base de datos con un ID común.
        
        Este método reemplaza los anteriores métodos separados para crear sesiones y registrar bases de datos,
        asegurando que todo se cree de una sola vez al final del procesamiento.
        
        Args:
            database_metadata: Metadatos completos de la base de datos
            files_list: Lista opcional de rutas de archivos procesados
            
        Returns:
            ID de la sesión/base de datos unificada
        """
        with self._lock:
            try:
                # Verificar límite de sesiones
                if len(self.sessions) >= self.MAX_SESSIONS:
                    # Intentar limpiar primero
                    self.clean_expired_sessions(aggressive=True)
                    
                    # Si aún hay demasiadas sesiones, rechazar la creación
                    if len(self.sessions) >= self.MAX_SESSIONS:
                        logger.error(f"Límite de sesiones alcanzado: {self.MAX_SESSIONS}")
                        raise ValueError(f"Se alcanzó el límite máximo de sesiones ({self.MAX_SESSIONS})")
                
                # Obtener o generar el ID único para sesión/base de datos
                db_name = database_metadata.get("name", "")
                session_id = db_name
                
                # Si no hay nombre, generarlo a partir de los metadatos
                if not session_id:
                    # Generar un nombre basado en configuración (ej: rag_modernbert_page_sqlite_xxx)
                    db_type = database_metadata.get("db_type", "sqlite")
                    embedding_model = database_metadata.get("embedding_model", "default")
                    chunking_method = database_metadata.get("chunking_method", "default")
                    
                    # Generar un hash corto único
                    import hashlib
                    import time
                    hash_input = f"{embedding_model}_{chunking_method}_{time.time()}"
                    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
                    
                    # Construir nombre con formato consistente
                    session_id = f"rag_{embedding_model}_{chunking_method}_{db_type}_{short_hash}"
                    database_metadata["name"] = session_id
                
                # Asegurarnos que ID y nombre sean iguales
                database_metadata["id"] = session_id
                
                # Crear la sesión con timestamp actual
                current_time = time.time()
                session_data = {
                    "id": session_id,
                    "created_at": current_time,
                    "last_activity": current_time,
                    "files": files_list or [],  # Lista simplificada de archivos
                    "processing_stats": {
                        "total_files_processed": len(files_list) if files_list else 0,
                        "creation_date": datetime.now().isoformat()
                    }
                }
                
                # Unificar los IDs entre sesión y base de datos
                database_metadata["session_id"] = session_id
                
                # Verificar que estén todos los campos requeridos
                required_fields = ["db_type", "db_path", "embedding_dim", "embedding_model", 
                                  "chunking_method", "session_id", "created_at", "last_used"]
                
                # Completar campos faltantes 
                for field in required_fields:
                    if field not in database_metadata:
                        if field == "created_at" or field == "last_used":
                            database_metadata[field] = current_time
                
                # Almacenar la sesión
                self.sessions[session_id] = session_data
                
                # Inicializar espacio para contextos
                self.contexts[session_id] = {}
                
                # Registrar la base de datos con los mismos metadatos
                success = self.register_database(session_id, database_metadata)
                if not success:
                    logger.error(f"Error al registrar base de datos: {session_id}")
                    # Limpiar la sesión creada si falla el registro de la BD
                    if session_id in self.sessions:
                        del self.sessions[session_id]
                    if session_id in self.contexts:
                        del self.contexts[session_id]
                    raise ValueError("Error al registrar la base de datos")
                
                # Persistir sesiones inmediatamente
                self._save_sessions()
                
                logger.info(f"Sesión unificada {session_id} creada correctamente")
                return session_id
                
            except Exception as e:
                logger.error(f"Error al crear sesión unificada: {e}")
                raise
