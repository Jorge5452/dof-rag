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
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from config import config
from modulos.databases.FactoryDatabase import DatabaseFactory
from modulos.embeddings.embeddings_factory import EmbeddingFactory

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Gestor de sesiones para el sistema RAG.
    
    Permite:
    - Crear nuevas sesiones
    - Recuperar sesiones anteriores
    - Listar bases de datos disponibles
    - Asegurar compatibilidad entre ingestión y consulta
    """
    
    _instance = None
    
    def __new__(cls):
        """Implementa el patrón Singleton."""
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Inicializa el gestor de sesiones."""
        if self._initialized:
            return
            
        self._initialized = True
        self._sessions = {}
        self._current_session = None
        self._sessions_file = Path("sessions.json")
        self._load_sessions()
        
        # Directorio donde se almacenan las sesiones
        self.sessions_dir = Path(config.get_general_config().get("sessions_dir", "sessions"))
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # Verificar si el directorio existe y crearlo si es necesario
        if not self.sessions_dir.exists():
            logger.info(f"Creando directorio de sesiones en: {self.sessions_dir}")
            try:
                self.sessions_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directorio de sesiones creado exitosamente: {self.sessions_dir}")
            except Exception as e:
                logger.error(f"Error al crear directorio de sesiones {self.sessions_dir}: {e}")
                raise RuntimeError(f"No se pudo crear el directorio de sesiones: {e}")
        else:
            logger.debug(f"Directorio de sesiones ya existe: {self.sessions_dir}")
        
        # Verificar permisos de escritura en el directorio
        if not os.access(self.sessions_dir, os.W_OK):
            logger.warning(f"¡ADVERTENCIA! El directorio de sesiones no tiene permisos de escritura: {self.sessions_dir}")
            logger.warning("Esto puede causar errores al guardar sesiones o metadatos")
        else:
            logger.debug(f"Directorio de sesiones tiene permisos de escritura: {self.sessions_dir}")
        
        # Archivo que mapea las bases de datos a sus metadatos
        self.db_metadata_file = self.sessions_dir / "db_metadata.json"
        
        # Cargar o crear el archivo de metadatos si no existe
        self.db_metadata = self._load_db_metadata()
    
    def _load_sessions(self) -> None:
        """Carga las sesiones guardadas desde el archivo."""
        if self._sessions_file.exists():
            try:
                with open(self._sessions_file, 'r') as f:
                    self._sessions = json.load(f)
                logger.info(f"Sesiones cargadas: {len(self._sessions)}")
            except Exception as e:
                logger.error(f"Error al cargar sesiones: {e}")
                self._sessions = {}
    
    def _save_sessions(self) -> None:
        """Guarda las sesiones actuales en el archivo."""
        try:
            with open(self._sessions_file, 'w') as f:
                json.dump(self._sessions, f, indent=2)
            logger.debug("Sesiones guardadas correctamente")
        except Exception as e:
            logger.error(f"Error al guardar sesiones: {e}")
    
    def _load_db_metadata(self) -> Dict[str, Any]:
        """
        Carga los metadatos de las bases de datos desde el archivo y los archivos .meta.json.
        
        Returns:
            Dict con los metadatos de las bases de datos
        """
        metadata = {}
        
        # 1. Cargar metadatos del archivo central (si existe)
        if self.db_metadata_file.exists():
            try:
                with open(self.db_metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Metadatos centrales cargados: encontradas {len(metadata)} bases de datos registradas")
            except json.JSONDecodeError:
                logger.warning("Error al cargar metadatos centrales. Archivo posiblemente corrupto.")
                metadata = {}
        
        # 2. Buscar archivos de base de datos y sus metadatos individuales
        db_config = config.get_database_config()
        db_dir = Path(db_config.get("sqlite", {}).get("db_dir", "modulos/databases/db"))
        
        # Buscar todas las bases de datos físicas
        db_files = list(db_dir.glob("*.db"))
        
        # Buscar sus archivos .meta.json correspondientes
        for db_file in db_files:
            db_name = db_file.stem
            meta_file = Path(str(db_file) + ".meta.json")
            
            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        db_meta = json.load(f)
                    
                    # Añadir o actualizar metadatos
                    if db_name not in metadata or metadata[db_name].get("last_used", 0) < db_meta.get("last_used", 0):
                        # El meta.json tiene información más reciente
                        metadata[db_name] = db_meta
                        # Asegurar que la ruta es correcta
                        metadata[db_name]["db_path"] = str(db_file)
                        logger.debug(f"Metadatos actualizados desde {meta_file}")
                except Exception as e:
                    logger.error(f"Error al cargar metadatos desde {meta_file}: {e}")
        
        return metadata
    
    def _save_db_metadata(self) -> None:
        """Guarda los metadatos de las bases de datos en el archivo central y en los archivos individuales."""
        # 1. Guardar en archivo central
        try:
            with open(self.db_metadata_file, 'w') as f:
                json.dump(self.db_metadata, f, indent=2)
            logger.debug(f"Metadatos centrales guardados: {len(self.db_metadata)} bases de datos")
        except Exception as e:
            logger.error(f"Error al guardar metadatos centrales: {e}")
        
        # 2. Guardar metadatos individuales para cada base de datos
        for db_name, metadata in self.db_metadata.items():
            db_path = metadata.get("db_path")
            if db_path and os.path.exists(db_path) and db_path.endswith((".db", ".sqlite", ".duckdb")):
                metadata_path = f"{db_path}.meta.json"
                try:
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    logger.debug(f"Metadatos guardados para {db_name} en {metadata_path}")
                except Exception as e:
                    logger.error(f"Error al guardar metadatos para {db_name}: {e}")
    
    def register_database(self, db_name: str, metadata: Dict[str, Any]) -> None:
        """
        Registra una base de datos con sus metadatos.
        
        Args:
            db_name: Nombre de la base de datos
            metadata: Diccionario con metadatos (embedding_model, chunking_method, etc.)
        """
        # Asegurar que tengamos todos los metadatos críticos
        required_fields = ["embedding_model", "embedding_dim", "chunking_method", "db_type"]
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Falta el campo obligatorio '{field}' en los metadatos de la base de datos")
        
        # Agregar timestamp de creación si no existe
        if "created_at" not in metadata:
            metadata["created_at"] = time.time()
        
        # Actualizar metadatos
        self.db_metadata[db_name] = metadata
        self._save_db_metadata()
    
    def update_database_metadata(self, db_name: str, new_metadata: Dict[str, Any]) -> None:
        """
        Actualiza los metadatos de una base de datos existente.
        
        Args:
            db_name: Nombre de la base de datos
            new_metadata: Nuevos metadatos a incorporar
        """
        if db_name not in self.db_metadata:
            raise ValueError(f"Base de datos '{db_name}' no encontrada")
            
        # Actualizar metadatos manteniendo timestamp original
        self.db_metadata[db_name].update(new_metadata)
        self.db_metadata[db_name]["last_used"] = time.time()
        self._save_db_metadata()
    
    def get_database_metadata(self, db_name: str) -> Dict[str, Any]:
        """
        Obtiene los metadatos de una base de datos.
        
        Args:
            db_name: Nombre de la base de datos
            
        Returns:
            Diccionario con los metadatos
        """
        if db_name not in self.db_metadata:
            return {}
        return self.db_metadata[db_name]
    
    def create_session(self, 
                      embedding_model: Optional[str] = None,
                      chunking_method: Optional[str] = None,
                      db_type: Optional[str] = None,
                      custom_name: Optional[str] = None) -> str:
        """
        Crea una nueva sesión con la configuración especificada.
        Si ya existe una sesión con la misma configuración, la reutiliza.
        
        Args:
            embedding_model: Modelo de embeddings a utilizar
            chunking_method: Método de chunking a utilizar
            db_type: Tipo de base de datos a utilizar
            custom_name: Nombre personalizado para la sesión
            
        Returns:
            ID de la sesión creada o reutilizada
        """
        # Obtener valores de configuración si no se proporcionan
        if embedding_model is None:
            embedding_config = config.get_embedding_config()
            embedding_model = embedding_config.get("model", "modernbert")
        
        if chunking_method is None:
            chunks_config = config.get_chunks_config()
            chunking_method = chunks_config.get("method", "context")
        
        if db_type is None:
            db_config = config.get_database_config()
            db_type = db_config.get("type", "sqlite")
        
        # Cargar información del modelo de embeddings para obtener dimensión
        embedding_manager = EmbeddingFactory.get_embedding_manager(embedding_model)
        embedding_dim = None
        
        try:
            # Intentar cargar el modelo para obtener su dimensión real
            embedding_manager.load_model()
            embedding_dim = embedding_manager.embedding_dim
        except Exception as e:
            logger.warning(f"No se pudo determinar dimensión del embedding: {e}")
            # Usar valores por defecto según el modelo si falla la carga
            if "modernbert" in embedding_model.lower():
                embedding_dim = 768
            elif "e5" in embedding_model.lower():
                embedding_dim = 1024
            else:
                embedding_dim = 384  # Dimensión por defecto
        
        # Obtener configuraciones específicas de cada componente
        embedding_config = config.get_embedding_config()
        chunks_config = config.get_chunks_config()
        db_config = config.get_database_config()
        
        # Verificar si ya existe una sesión con esta misma configuración
        for session_id, session_info in self._sessions.items():
            if (session_info.get('embedding_model') == embedding_model and
                session_info.get('chunking_method') == chunking_method and
                session_info.get('db_type') == db_type):
                
                logger.info(f"Reutilizando sesión existente '{session_id}' con configuración idéntica")
                
                # Actualizar timestamp de último uso
                self._sessions[session_id]["last_used"] = time.time()
                
                # Asegurar que tiene información de dimensión si no existía anteriormente
                if "embedding_dim" not in self._sessions[session_id] and embedding_dim:
                    self._sessions[session_id]["embedding_dim"] = embedding_dim
                
                self._current_session = session_id
                self._save_sessions()
                
                return session_id
        
        # Si no existe una sesión con esta configuración, crear una nueva
        session_id = custom_name or str(uuid.uuid4())[:8]
        
        # Generar nombre de base de datos basado en la configuración
        db_name = self._generate_db_name(embedding_model, chunking_method, db_type)
        db_dir = db_config.get(db_type, {}).get("db_dir", "modulos/databases/db")
        if not os.path.isabs(db_dir):
            db_dir = os.path.abspath(db_dir)
        
        # Determinar extensión correcta para el tipo de base de datos
        extension = ".db"
        if db_type == "sqlite":
            extension = ".sqlite"
        elif db_type == "duckdb":
            extension = ".duckdb"
            
        db_path = os.path.join(db_dir, f"{db_name}{extension}")
        
        # Crear registro de sesión con información detallada
        session_info = {
            "id": session_id,
            "created_at": time.time(),
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim,
            "chunking_method": chunking_method,
            "db_type": db_type,
            "db_path": db_path,
            "db_name": db_name,
            "custom_name": custom_name,
            "last_used": time.time(),
            # Configuraciones específicas de componentes
            "embedding_config": embedding_config,
            "chunks_config": chunks_config,
            "db_config": {k: v for k, v in db_config.items() if k != "password"}  # Eliminar datos sensibles
        }
        
        # Guardar sesión
        self._sessions[session_id] = session_info
        self._current_session = session_id
        self._save_sessions()
        
        logger.info(f"Creada nueva sesión '{session_id}' con modelo {embedding_model} ({embedding_dim}d), "
                   f"chunking {chunking_method}, DB {db_type}")
        return session_id
    
    def _generate_db_name(self, embedding_model: str, chunking_method: str, db_type: str) -> str:
        """
        Genera un nombre para la base de datos basado en la configuración.
        
        Args:
            embedding_model: Modelo de embedding utilizado
            chunking_method: Método de chunking utilizado
            db_type: Tipo de base de datos
            
        Returns:
            Nombre seguro para la base de datos
        """
        # Extraer nombre corto de modelos
        if '/' in embedding_model:
            embedding_model = embedding_model.split('/')[1]
        
        # Crear nombre seguro
        safe_name = f"rag_{embedding_model}_{chunking_method}_{db_type}"
        
        # Reemplazar caracteres no permitidos en nombres de archivo
        safe_name = safe_name.replace('-', '_').replace('.', '_').lower()
        
        return safe_name
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Obtiene información de una sesión específica.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Información de la sesión
            
        Raises:
            ValueError: Si la sesión no existe
        """
        if session_id not in self._sessions:
            raise ValueError(f"No existe sesión con ID: {session_id}")
        
        # Actualizar timestamp de último uso
        self._sessions[session_id]["last_used"] = time.time()
        self._save_sessions()
        
        return self._sessions[session_id]
    
    def get_latest_session(self) -> Dict[str, Any]:
        """
        Obtiene la sesión más reciente.
        
        Returns:
            Información de la sesión más reciente
            
        Raises:
            ValueError: Si no hay sesiones disponibles
        """
        if not self._sessions:
            raise ValueError("No hay sesiones disponibles")
        
        # Encontrar la sesión más reciente por fecha de último uso
        latest_session_id = max(self._sessions, key=lambda x: self._sessions[x].get("last_used", 0))
        return self.get_session(latest_session_id)
    
    def get_session_database(self, session_id: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Obtiene la base de datos asociada a una sesión.
        
        Args:
            session_id: ID de la sesión. Si es None, se usa la sesión más reciente.
            
        Returns:
            Tupla con (instancia_db, metadata_sesión)
            
        Raises:
            ValueError: Si no se encuentra la sesión o hay incompatibilidad
        """
        try:
            # Obtener información de sesión
            if session_id is None:
                session = self.get_latest_session()
            else:
                session = self.get_session(session_id)
            
            # Obtener modelo de embeddings para determinar dimensión
            embedding_model = session["embedding_model"]
            chunking_method = session["chunking_method"]
            db_type = session["db_type"]
            
            # Usar dimensión almacenada en la sesión si está disponible
            embedding_dim = session.get("embedding_dim")
            
            # Si no está disponible, cargar el modelo para determinar la dimensión
            if embedding_dim is None:
                # Cargar el mismo modelo de embeddings usado en la sesión
                embedding_manager = EmbeddingFactory.get_embedding_manager(embedding_model)
                embedding_manager.load_model()
                embedding_dim = embedding_manager.embedding_dim
                
                # Actualizar la sesión con esta información
                session["embedding_dim"] = embedding_dim
                self._sessions[session_id] = session
                self._save_sessions()
            
            # Verificar si existe una ruta específica para la base de datos
            db_path = session.get("db_path")
            
            # Obtener instancia de base de datos con la configuración exacta de la sesión
            db = DatabaseFactory().get_database_instance(
                db_type=db_type,
                embedding_dim=embedding_dim,
                embedding_model=embedding_model,
                chunking_method=chunking_method,
                session_id=session_id,
                custom_name=session.get("custom_name")
            )
            
            # Si tenemos una ruta específica y el archivo existe, asegurar que conecta a esa ruta
            if db_path and os.path.exists(db_path):
                logger.info(f"Conectando a base de datos existente: {db_path}")
                db.connect(db_path)
            
            return db, session
            
        except Exception as e:
            logger.error(f"Error al obtener base de datos de sesión: {e}")
            raise
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        Lista todas las sesiones disponibles.
        
        Returns:
            Lista de información de sesiones
        """
        # Devolver lista ordenada por último uso (más reciente primero)
        sessions_list = list(self._sessions.values())
        sessions_list.sort(key=lambda x: x.get("last_used", 0), reverse=True)
        return sessions_list
    
    def list_available_databases(self) -> Dict[str, Dict[str, Any]]:
        """
        Lista todas las bases de datos disponibles.
        
        Returns:
            Diccionario con nombres de bases de datos y sus metadatos
        """
        # Actualizar información de la ruta del archivo
        db_config = config.get_database_config()
        db_dir = Path(db_config.get("sqlite", {}).get("db_dir", "modulos/databases/db"))
        
        # Asegurar que la ruta es absoluta
        if not db_dir.is_absolute():
            db_dir = Path(os.path.abspath(db_dir))
            
        # Verificar que el directorio existe
        if not db_dir.exists():
            logger.warning(f"El directorio de bases de datos no existe: {db_dir}")
            return {}
            
        logger.info(f"Buscando bases de datos en: {db_dir}")
        
        # Buscar todas las bases de datos físicas con diferentes extensiones
        db_files = []
        for ext in [".db", ".sqlite", ".duckdb"]:
            ext_files = list(db_dir.glob(f"*{ext}"))
            db_files.extend(ext_files)
            logger.debug(f"Encontrados {len(ext_files)} archivos con extensión {ext}")
        
        logger.info(f"Total de archivos de base de datos encontrados: {len(db_files)}")
        
        # Actualizar la estructura de metadatos
        databases = {}
        
        # Recorrer los archivos existentes
        for db_file in db_files:
            db_name = db_file.stem
            extension = db_file.suffix
            
            # Buscar archivo de metadatos asociado
            metadata_file = Path(f"{db_file}.meta.json")
            
            # Incluir metadatos existentes o crear nuevos
            if metadata_file.exists():
                # Cargar metadatos desde el archivo .meta.json
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    # Asegurar que la ruta es correcta
                    metadata["db_path"] = str(db_file)
                    # Determinar el tipo de base de datos basado en la extensión
                    if extension == ".sqlite":
                        metadata["db_type"] = "sqlite"
                    elif extension == ".duckdb":
                        metadata["db_type"] = "duckdb"
                    # Actualizar metadatos centrales si es necesario
                    self.db_metadata[db_name] = metadata
                    logger.debug(f"Metadatos cargados desde {metadata_file}")
                except json.JSONDecodeError as e:
                    logger.error(f"Error al cargar metadatos desde {metadata_file}: {e}")
                    if db_name in self.db_metadata:
                        metadata = self.db_metadata[db_name]
                        metadata["db_path"] = str(db_file)
                    else:
                        metadata = self._create_default_metadata(db_file)
            elif db_name in self.db_metadata:
                # Usar metadatos existentes en memoria
                metadata = self.db_metadata[db_name]
                metadata["db_path"] = str(db_file)
                # Actualizar tipo si no coincide con la extensión
                if extension == ".sqlite" and metadata.get("db_type") != "sqlite":
                    metadata["db_type"] = "sqlite"
                    logger.debug(f"Tipo actualizado a sqlite para {db_name} basado en extensión")
                elif extension == ".duckdb" and metadata.get("db_type") != "duckdb":
                    metadata["db_type"] = "duckdb"
                    logger.debug(f"Tipo actualizado a duckdb para {db_name} basado en extensión")
            else:
                # Crear metadatos mínimos para bases de datos no registradas
                metadata = self._create_default_metadata(db_file)
                # Inferir tipo basado en extensión
                if extension == ".sqlite":
                    metadata["db_type"] = "sqlite"
                elif extension == ".duckdb":
                    metadata["db_type"] = "duckdb"
                logger.debug(f"Creados metadatos por defecto para {db_name} ({extension})")
                
            databases[db_name] = metadata
        
        # Guardar cambios en caso de haber actualizado algún metadato
        self._save_db_metadata()
        
        # Mostrar información resumida
        db_types = {}
        for name, meta in databases.items():
            db_type = meta.get("db_type", "unknown")
            db_types[db_type] = db_types.get(db_type, 0) + 1
        
        for db_type, count in db_types.items():
            logger.info(f"- Bases de datos de tipo {db_type}: {count}")
        
        return databases
    
    def _create_default_metadata(self, db_file: Path) -> Dict[str, Any]:
        """
        Crea metadatos por defecto para una base de datos.
        
        Args:
            db_file: Ruta al archivo de base de datos
            
        Returns:
            Dict con metadatos por defecto
        """
        # Determinar el tipo de base de datos basado en la extensión
        extension = db_file.suffix.lower()
        db_type = "sqlite"  # Valor por defecto
        
        if extension == ".sqlite":
            db_type = "sqlite"
        elif extension == ".duckdb":
            db_type = "duckdb"
        elif extension == ".db":
            # Para archivos .db, intentar inferir por el nombre
            if "duckdb" in db_file.stem.lower():
                db_type = "duckdb"
        
        logger.debug(f"Inferido tipo de base de datos {db_type} para {db_file} basado en extensión {extension}")
        
        metadata = {
            "db_path": str(db_file),
            "db_type": db_type,
            "embedding_model": "desconocido",
            "embedding_dim": 0,
            "chunking_method": "desconocido",
            "created_at": os.path.getctime(db_file),
            "note": f"Metadatos inferidos automáticamente basados en extensión {extension}"
        }
        
        # Registrar para futuras consultas
        self.db_metadata[db_file.stem] = metadata
        return metadata
    
    def get_database_by_index(self, index: int) -> Tuple[Any, Dict[str, Any]]:
        """
        Obtiene una base de datos por su índice en la lista de bases de datos disponibles.
        
        Args:
            index: Índice de la base de datos (0 es la más reciente)
            
        Returns:
            Tupla con (instancia_db, metadata)
            
        Raises:
            ValueError: Si el índice está fuera de rango
        """
        # Obtener las bases de datos disponibles
        databases = self.list_available_databases()
        
        # Crear una lista ordenada de las bases de datos
        sorted_dbs = []
        for name, metadata in databases.items():
            sorted_dbs.append((name, metadata))
        
        # Ordenar por último uso si está disponible
        sorted_dbs.sort(key=lambda x: x[1].get('last_used', 0), reverse=True)
        
        # Verificar que el índice es válido
        if index < 0 or index >= len(sorted_dbs):
            raise ValueError(f"Índice de base de datos fuera de rango: {index}. Rango válido: 0-{len(sorted_dbs)-1}")
        
        # Obtener la base de datos seleccionada
        name, metadata = sorted_dbs[index]
        
        # Obtener información necesaria para cargar la base de datos
        embedding_model = metadata.get("embedding_model", "modernbert")
        chunking_method = metadata.get("chunking_method", "context") 
        db_type = metadata.get("db_type", "sqlite")
        session_id = metadata.get("session_id")
        
        # Cargar el modelo de embeddings para determinar la dimensión
        embedding_manager = EmbeddingFactory.get_embedding_manager(embedding_model)
        embedding_manager.load_model()
        embedding_dim = embedding_manager.embedding_dim
        
        # Obtener instancia de base de datos
        db = DatabaseFactory().get_database_instance(
            db_type=db_type,
            embedding_dim=embedding_dim,
            embedding_model=embedding_model,
            chunking_method=chunking_method,
            session_id=session_id
        )
        
        # Conectar a la base de datos
        db_path = metadata.get('db_path')
        if db_path:
            db.connect(db_path)
        
        # Actualizar last_used en los metadatos
        metadata["last_used"] = time.time()
        self._store_db_metadata(name, metadata)
        
        return db, metadata

    def _store_db_metadata(self, name: str, metadata: Dict[str, Any]) -> None:
        """
        Almacena los metadatos actualizados de una base de datos.
        
        Args:
            name: Nombre de la base de datos
            metadata: Metadatos actualizados
        """
        # 1. Actualizar el repositorio central de metadatos
        self.db_metadata[name] = metadata
        
        # 2. Guardar en el archivo individual
        db_path = metadata.get('db_path')
        if db_path and db_path.endswith((".db", ".sqlite", ".duckdb")):
            metadata_path = f"{db_path}.meta.json"
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.debug(f"Metadatos individuales actualizados para {name} en {metadata_path}")
            except Exception as e:
                logger.error(f"Error al guardar metadatos individuales para {name}: {e}")
        
        # 3. Guardar todos los metadatos centrales también
        self._save_db_metadata()
