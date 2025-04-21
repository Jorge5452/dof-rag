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
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import weakref

from config import Config
from modulos.databases.FactoryDatabase import DatabaseFactory
from modulos.embeddings.embeddings_factory import EmbeddingFactory

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
    SESSION_TIMEOUT = 3600        # Tiempo de inactividad (en segundos) antes de eliminar sesión
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
            # Sesiones y contextos
            self.sessions = {}              # {session_id: session_data}
            self.contexts = {}              # {session_id: {message_id: context_data}}
            
            # Monitoreo de recursos
            self.resource_usage = {
                "memory_percent": 0.0,      # Porcentaje de memoria utilizada
                "cpu_percent": 0.0,         # Porcentaje de CPU utilizada
                "active_sessions": 0,       # Número de sesiones activas
                "last_check": time.time(),  # Timestamp de la última verificación
                "cleanup_count": 0          # Contador de limpiezas realizadas
            }
            
            # Cargar configuración
            self._load_config()
            
            # Iniciar hilo de limpieza
            self._start_cleanup_thread()
            
            logger.info(f"SessionManager inicializado: max_sessions={self.MAX_SESSIONS}, "
                      f"timeout={self.SESSION_TIMEOUT}s, cleanup_interval={self.CLEANUP_INTERVAL}s")
            
            self._initialized = True
    
    def _load_config(self):
        """
        Carga los valores de configuración desde config.py
        """
        try:
            # Cargar configuración de sesiones
            session_config = get_value("sessions", {})
            
            # Establecer valores desde configuración o usar predeterminados
            self.MAX_SESSIONS = session_config.get("max_sessions", self.MAX_SESSIONS)
            self.SESSION_TIMEOUT = session_config.get("timeout", self.SESSION_TIMEOUT)
            self.CLEANUP_INTERVAL = session_config.get("cleanup_interval", self.CLEANUP_INTERVAL)
            self.MAX_CONTEXTS_PER_SESSION = session_config.get("max_contexts", self.MAX_CONTEXTS_PER_SESSION)
            
            # Validar valores
            if self.MAX_SESSIONS < 1:
                logger.warning("max_sessions debe ser al menos 1, usando valor 10")
                self.MAX_SESSIONS = 10
                
            if self.SESSION_TIMEOUT < 60:
                logger.warning("timeout debe ser al menos 60 segundos, usando valor 3600")
                self.SESSION_TIMEOUT = 3600
            
            logger.debug(f"Configuración cargada: max_sessions={self.MAX_SESSIONS}, "
                       f"timeout={self.SESSION_TIMEOUT}s")
                       
        except Exception as e:
            # Cambiar nivel de log de error a debug ya que manejamos el error correctamente
            logger.debug(f"Usando valores predeterminados para configuración: {e}")
            # Valores predeterminados ya definidos en la clase
    
    def _start_cleanup_thread(self):
        """
        Inicia el hilo de limpieza automática de sesiones.
        """
        def cleanup_thread():
            try:
                while True:
                    # Dormir el intervalo configurado
                    time.sleep(self.CLEANUP_INTERVAL)
                    
                    # Actualizar uso de recursos
                    self._update_resource_usage()
                    
                    # Verificar si es necesario realizar limpieza
                    current_time = time.time()
                    memory_high = self.resource_usage["memory_percent"] > 80
                    cpu_high = self.resource_usage["cpu_percent"] > 70
                    many_sessions = len(self.sessions) > self.MAX_SESSIONS * 0.8
                    
                    # Realizar limpieza normal o agresiva según uso de recursos
                    if memory_high or cpu_high or many_sessions:
                        logger.info(f"Limpieza agresiva - Memoria: {self.resource_usage['memory_percent']:.1f}%, "
                                  f"CPU: {self.resource_usage['cpu_percent']:.1f}%, "
                                  f"Sesiones: {len(self.sessions)}/{self.MAX_SESSIONS}")
                        self.clean_expired_sessions(aggressive=True)
                    else:
                        # Limpieza normal
                        self.clean_expired_sessions()
            
            except Exception as e:
                logger.error(f"Error en hilo de limpieza: {e}")
        
        # Iniciar hilo de limpieza como daemon
        cleanup_thread = threading.Thread(target=cleanup_thread, daemon=True)
        cleanup_thread.name = "SessionCleanupThread"
        cleanup_thread.start()
        logger.debug("Hilo de limpieza de sesiones iniciado")
    
    def _update_resource_usage(self):
        """
        Actualiza información sobre uso de recursos del sistema.
        """
        try:
            # Obtener proceso actual
            process = psutil.Process(os.getpid())
            
            # Actualizar métricas
            self.resource_usage["memory_percent"] = process.memory_percent()
            self.resource_usage["cpu_percent"] = process.cpu_percent(interval=0.1)
            self.resource_usage["active_sessions"] = len(self.sessions)
            self.resource_usage["last_check"] = time.time()
            
            # Log informativo periódico (cada 10 limpiezas)
            if self.resource_usage["cleanup_count"] % 10 == 0:
                logger.info(f"Estado del sistema - Memoria: {self.resource_usage['memory_percent']:.1f}%, "
                          f"CPU: {self.resource_usage['cpu_percent']:.1f}%, "
                          f"Sesiones activas: {len(self.sessions)}")
            
            self.resource_usage["cleanup_count"] += 1
            
        except Exception as e:
            logger.error(f"Error al actualizar uso de recursos: {e}")
    
    def clean_expired_sessions(self, aggressive: bool = False):
        """
        Elimina sesiones inactivas basadas en su último tiempo de actividad.
        
        Args:
            aggressive: Si es True, usa un timeout más corto para limpieza agresiva
        """
        with self._lock:
            try:
                # Determinar el timeout a utilizar
                timeout = self.SESSION_TIMEOUT
                if aggressive:
                    # En modo agresivo, reducir el timeout a la mitad o menos
                    timeout = min(self.SESSION_TIMEOUT // 2, 600)  # Máx 10 minutos en modo agresivo
                
                current_time = time.time()
                sessions_to_remove = []
                
                # Identificar sesiones expiradas
                for session_id, session_data in self.sessions.items():
                    last_activity = session_data.get("last_activity", 0)
                    if current_time - last_activity > timeout:
                        sessions_to_remove.append(session_id)
                
                # Eliminar sesiones expiradas
                for session_id in sessions_to_remove:
                    self.delete_session(session_id)
                
                # Si hay demasiadas sesiones incluso después de limpiar las expiradas,
                # eliminar las más antiguas (excepto las activas recientemente)
                if aggressive and len(self.sessions) > self.MAX_SESSIONS * 0.9:
                    # Ordenar por actividad (más antiguas primero)
                    sorted_sessions = sorted(
                        self.sessions.items(),
                        key=lambda x: x[1].get("last_activity", 0)
                    )
                    
                    # Determinar cuántas sesiones eliminar (hasta un 25% de las más antiguas)
                    sessions_to_force_remove = sorted_sessions[:max(1, len(sorted_sessions) // 4)]
                    
                    for session_id, _ in sessions_to_force_remove:
                        logger.warning(f"Eliminando sesión {session_id} por restricción de recursos")
                        self.delete_session(session_id)
                
                # Forzar recolección de basura si la limpieza fue agresiva
                if aggressive:
                    gc.collect()
                
                if sessions_to_remove:
                    logger.info(f"Limpieza completada: {len(sessions_to_remove)} sesiones eliminadas, "
                              f"{len(self.sessions)} activas")
                
            except Exception as e:
                logger.error(f"Error durante limpieza de sesiones: {e}")
    
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
                "last_activity": time.time()
            }
            
            # Añadir metadata si se proporciona
            if metadata:
                session_data.update(metadata)
            
            # Almacenar sesión
            self.sessions[session_id] = session_data
            
            # Inicializar espacio para contextos
            self.contexts[session_id] = {}
            
            logger.info(f"Sesión creada: {session_id} ({len(self.sessions)} sesiones activas)")
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
        Lista todas las sesiones activas.
        
        Returns:
            Lista de sesiones con sus metadatos
        """
        with self._lock:
            # Crear una lista con todas las sesiones
            sessions_list = []
            for session_id, session_data in self.sessions.items():
                # Crear una copia para evitar modificaciones no controladas
                session_copy = dict(session_data)
                session_copy["id"] = session_id
                sessions_list.append(session_copy)
            
            # Ordenar por último uso (más reciente primero)
            sessions_list.sort(key=lambda x: x.get("last_activity", 0), reverse=True)
            
            return sessions_list
    
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
            db_name: Nombre de la base de datos
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
                logger.warning(f"Campos faltantes en metadatos: {missing_fields}. Intentando recuperar...")
                
                # Intentar completar campos faltantes
                if 'db_type' in missing_fields and 'db_path' in metadata:
                    if metadata['db_path'].endswith('.sqlite'):
                        metadata['db_type'] = 'sqlite'
                    elif metadata['db_path'].endswith('.duckdb'):
                        metadata['db_type'] = 'duckdb'
                
                if 'embedding_model' in missing_fields:
                    embedding_config = config.get_embedding_config()
                    metadata['embedding_model'] = embedding_config.get('model', 'modernbert')
                    
                if 'chunking_method' in missing_fields:
                    chunks_config = config.get_chunks_config()
                    metadata['chunking_method'] = chunks_config.get('method', 'character')
                    
                # Verificar nuevamente los campos requeridos
                missing_fields = [field for field in required_fields if field not in metadata]
                if missing_fields:
                    logger.error(f"No se pudo completar los campos requeridos: {missing_fields}")
                    return False
                    
            # Añadir metadatos extras de configuración que pueden ser útiles
            # para reconstruir el entorno completo
            if 'additional_config' not in metadata:
                metadata['additional_config'] = {}
            
            # Añadir timestmaps si no existen
            if 'created_at' not in metadata:
                metadata['created_at'] = time.time()
                
            # Actualizar timestamp de último uso
            metadata["last_used"] = time.time()
            
            # Guardar metadatos en un archivo JSON junto a la base de datos
            meta_path = f"{db_path}.meta.json"
            
            # Guardar metadatos en formato JSON
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
            # También guardar una copia en la carpeta de sesiones si está asociada a una sesión
            if 'session_id' in metadata and metadata['session_id']:
                session_dir = os.path.join(
                    config.get_general_config().get('sessions_dir', 'sessions'),
                    metadata['session_id']
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
        
        Args:
            session_id: ID de la sesión
            db_name: Nombre de la base de datos
            db_metadata: Metadatos de la base de datos
            
        Returns:
            True si se asoció correctamente, False en caso contrario
        """
        try:
            # Verificar que la sesión existe
            if session_id not in self.sessions:
                logger.warning(f"Intento de asociar base de datos a sesión inexistente: {session_id}")
                return False
            
            # Actualizar metadatos con la sesión
            db_metadata['session_id'] = session_id
            
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
