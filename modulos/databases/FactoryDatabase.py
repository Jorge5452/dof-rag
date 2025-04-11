import os
import logging
from typing import Dict, Any, Optional
import json
import time
from pathlib import Path

from modulos.databases.VectorialDatabase import VectorialDatabase
from config import config

logger = logging.getLogger(__name__)

class DatabaseFactory:
    """
    Factory para crear instancias de bases de datos vectoriales.
    Implementa el patrón Singleton para instancias de base de datos.
    
    Esta clase ahora incorpora un sistema de nombrado inteligente para evitar
    conflictos cuando se utilizan diferentes configuraciones de embedding y chunking.
    """
    
    # Diccionario para almacenar instancias creadas (Singleton)
    _instances = {}
    
    @classmethod
    def get_database_instance(cls, 
                             db_type: Optional[str] = None, 
                             embedding_dim: Optional[int] = None,
                             embedding_model: Optional[str] = None,
                             chunking_method: Optional[str] = None,
                             session_id: Optional[str] = None,
                             custom_name: Optional[str] = None) -> VectorialDatabase:
        """
        Obtiene una instancia de base de datos del tipo especificado con configuración específica.
        
        Args:
            db_type: Tipo de base de datos ('sqlite', 'duckdb', etc.). Si es None, se usa el valor de config.
            embedding_dim: Dimensión del embedding. Este valor debe venir del modelo de embeddings.
            embedding_model: Nombre del modelo de embedding utilizado (para tracking).
            chunking_method: Método de chunking utilizado (para tracking).
            session_id: Identificador de sesión único (si se proporciona, se usa para identificar la base de datos).
            custom_name: Nombre personalizado para la base de datos (tiene prioridad sobre el generado).
            
        Returns:
            Instancia de VectorialDatabase.
            
        Raises:
            ValueError: Si el tipo de base de datos no es válido o si no se proporciona embedding_dim.
        """
        # Validar que tenemos una dimensión de embedding
        if embedding_dim is None:
            raise ValueError("Se requiere la dimensión del embedding para inicializar la base de datos vectorial")
        
        # Cargar configuración si no se proporcionan argumentos
        if db_type is None:
            db_config = cls._load_config()
            db_type = db_config.get("type", "sqlite")
            
        # Obtener información de configuración actual si no se proporciona
        if embedding_model is None:
            embedding_config = config.get_embedding_config()
            embedding_model = embedding_config.get("model", "modernbert")
            
        if chunking_method is None:
            chunks_config = config.get_chunks_config()
            chunking_method = chunks_config.get("method", "context")
        
        # Normalizar el tipo de base de datos
        db_type = db_type.lower()
        
        # Verificar que la implementación existe
        db_class = cls._get_db_class(db_type)
        
        # Generar un nombre único para la base de datos basado en la configuración
        if custom_name:
            db_name = custom_name
        elif session_id:
            db_name = f"rag_db_{session_id}"
        else:
            db_name = cls._generate_db_name(embedding_model, chunking_method, db_type)
        
        # Determinar la ruta a la base de datos
        db_path = cls._get_db_path(db_type, db_name)
        
        # Crear clave única que incluye la dimensión
        instance_key = f"{db_type}:{db_path}:{embedding_dim}"
        
        # Comprobar si ya existe una instancia para esta clave
        if instance_key not in cls._instances:
            # Crear nueva instancia con la dimensión específica
            logger.info(
                f"Creando nueva instancia de base de datos {db_type} "
                f"en {db_path} con dimensión {embedding_dim}, "
                f"modelo {embedding_model}, chunking {chunking_method}"
            )
            db_instance = db_class(embedding_dim=embedding_dim)
            
            # Añadir atributos para mayor trazabilidad
            db_instance._embedding_model = embedding_model
            db_instance._chunking_method = chunking_method
            db_instance._session_id = session_id
            
            # Guardar los metadatos de configuración (útil para session_manager)
            metadata = {
                "db_type": db_type,
                "db_path": db_path,
                "embedding_dim": embedding_dim,
                "embedding_model": embedding_model,
                "chunking_method": chunking_method,
                "session_id": session_id,
                "custom_name": custom_name,
                "created_at": time.time(),
                "last_used": time.time()
            }
            
            # Almacenar metadatos para recuperación posterior
            cls._store_db_metadata(db_path, metadata)
            
            # Intentar conectar a la base de datos
            try:
                db_instance.connect(db_path)
                # Si es una nueva base de datos, guardar metadatos dentro de ella también
                if hasattr(db_instance, "store_metadata"):
                    for key, value in metadata.items():
                        db_instance.store_metadata(key, value)
            except Exception as e:
                logger.error(f"Error al conectar a la base de datos: {e}")
                
            cls._instances[instance_key] = db_instance
        else:
            logger.debug(f"Reutilizando instancia existente de base de datos {db_type} con dimensión {embedding_dim}")
            
            # Actualizar timestamp de último uso en metadatos
            try:
                metadata = cls._load_db_metadata(db_path)
                if metadata:
                    metadata["last_used"] = time.time()
                    cls._store_db_metadata(db_path, metadata)
            except Exception as e:
                logger.debug(f"No se pudo actualizar timestamp de uso: {e}")
        
        return cls._instances[instance_key]
    
    @classmethod
    def _get_db_class(cls, db_type: str):
        """
        Obtiene la clase de base de datos apropiada según el tipo.
        
        Args:
            db_type: Tipo de base de datos ('sqlite', 'duckdb', etc.)
            
        Returns:
            Clase de base de datos.
            
        Raises:
            ValueError: Si el tipo de base de datos no es válido.
        """
        if db_type == "sqlite":
            from modulos.databases.implementaciones.sqlite import SQLiteVectorialDatabase
            return SQLiteVectorialDatabase
        elif db_type == "duckdb":
            try:
                from modulos.databases.implementaciones.duckdb import DuckDBVectorialDatabase
                return DuckDBVectorialDatabase
            except ImportError as e:
                logger.error(f"No se pudo importar DuckDBVectorialDatabase: {e}")
                raise ValueError(f"DuckDB no está disponible: {e}")
        elif db_type == "postgresql":
            try:
                # from modulos.databases.implementaciones.postgresql import PostgreSQLVectorialDatabase
                # return PostgreSQLVectorialDatabase
                pass
            except ImportError as e:
                logger.error(f"No se pudo importar PostgreSQLVectorialDatabase: {e}")
                raise ValueError(f"PostgreSQL no está disponible: {e}")
        else:
            raise ValueError(f"Tipo de base de datos no válido: {db_type}")
    
    @classmethod
    def _generate_db_name(cls, embedding_model: str, chunking_method: str, db_type: str) -> str:
        """
        Genera un nombre único para la base de datos basado en la configuración.
        
        Args:
            embedding_model: Modelo de embedding utilizado.
            chunking_method: Método de chunking utilizado.
            db_type: Tipo de base de datos.
            
        Returns:
            Nombre para la base de datos.
        """
        # Extraer nombre corto de modelos
        if '/' in embedding_model:
            embedding_model = embedding_model.split('/')[1]
        
        # Crear nombre seguro
        safe_name = f"rag_{embedding_model}_{chunking_method}_{db_type}"
        
        # Reemplazar caracteres no permitidos en nombres de archivo
        safe_name = safe_name.replace('-', '_').replace('.', '_').lower()
        
        return safe_name
    
    @classmethod
    def _get_db_path(cls, db_type: str, db_name: str) -> str:
        """
        Obtiene la ruta de la base de datos para el tipo y nombre específicos.
        
        Args:
            db_type: Tipo de base de datos.
            db_name: Nombre de la base de datos.
            
        Returns:
            Ruta a la base de datos.
        """
        # Para bases de datos de archivo (SQLite, DuckDB)
        if db_type in ["sqlite", "duckdb"]:
            db_config = cls._load_config()
            db_type_config = db_config.get(db_type, {})
            
            # Obtener directorio base y asegurar que es absoluto
            db_dir = db_type_config.get("db_dir", "modulos/databases/db")
            
            # Asegurar que la ruta es absoluta
            if not os.path.isabs(db_dir):
                db_dir = os.path.abspath(db_dir)
            
            # Si el nombre está vacío, generar un nombre basado en timestamp
            if not db_name:
                import time
                timestamp = int(time.time())
                db_name = f"rag_db_{timestamp}"
            
            # Asegurar extensión correcta
            extension = ".db"
            if db_type == "sqlite":
                extension = ".sqlite"  # Cambio importante: .db -> .sqlite para SQLite
            elif db_type == "duckdb":
                extension = ".duckdb"
                
            if not db_name.endswith(extension):
                db_name = f"{db_name}{extension}"
            
            # Crear el directorio si no existe - IMPORTANTE: ¡Esto debe ejecutarse siempre!
            os.makedirs(db_dir, exist_ok=True)
            
            # Construir la ruta completa
            db_path = os.path.join(db_dir, db_name)
            
            logger.info(f"Ruta de base de datos generada: {db_path}")
            return db_path
        
        # Para bases de datos de servidor (PostgreSQL, etc.)
        else:
            # Devolvemos una cadena de conexión o identificador
            return db_name
    
    @classmethod
    def _store_db_metadata(cls, db_path: str, metadata: Dict[str, Any]) -> None:
        """
        Almacena los metadatos de configuración de la base de datos.
        
        Args:
            db_path: Ruta al archivo de la base de datos.
            metadata: Metadatos a almacenar.
        """
        # Para bases de datos de archivo, guardar un archivo de metadatos junto a la DB
        try:
            if isinstance(db_path, str) and os.path.exists(db_path):
                # Determinar la extensión correcta para el metadato
                extension = os.path.splitext(db_path)[1]
                metadata_path = f"{db_path}.meta.json"
                
                # Asegurarse de que la ruta del archivo está en los metadatos
                metadata["db_path"] = db_path
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.debug(f"Metadatos guardados en {metadata_path}")
            else:
                logger.warning(f"No se pudo guardar metadatos, la ruta no existe: {db_path}")
        except Exception as e:
            logger.warning(f"No se pudieron guardar los metadatos de la base de datos: {e}")
    
    @classmethod
    def get_available_databases(cls) -> Dict[str, Dict[str, Any]]:
        """
        Escanea y devuelve todas las bases de datos disponibles con sus metadatos.
        
        Returns:
            Diccionario con nombres de bases de datos y sus metadatos.
        """
        available_dbs = {}
        
        # Buscar en el directorio de bases de datos
        db_config = cls._load_config()
        
        # Cargar directamente desde la configuración el directorio principal
        db_dir = Path(db_config.get("sqlite", {}).get("db_dir", "modulos/databases/db"))
        
        # Asegurar que es una ruta absoluta
        if not db_dir.is_absolute():
            db_dir = Path(os.path.abspath(db_dir))
        
        logger.info(f"Buscando bases de datos en: {db_dir}")
        
        # Verificar que el directorio existe
        if not db_dir.exists():
            logger.warning(f"El directorio de bases de datos no existe: {db_dir}")
            return available_dbs
        
        # Buscar archivos con extensiones comunes de bases de datos
        db_files_by_ext = {}
        total_files = 0
        
        for extension in [".db", ".sqlite", ".duckdb"]:
            db_files = list(db_dir.glob(f"*{extension}"))
            db_files_by_ext[extension] = db_files
            total_files += len(db_files)
            logger.debug(f"Encontrados {len(db_files)} archivos con extensión {extension} en {db_dir}")
        
        if total_files == 0:
            logger.warning(f"No se encontraron archivos de bases de datos en {db_dir}")
            return available_dbs
            
        logger.info(f"Total de bases de datos encontradas: {total_files}")
        
        # Procesar cada base de datos por extensión
        for extension, db_files in db_files_by_ext.items():
            for db_file in db_files:
                # Usar nombre de archivo sin extensión como clave
                key = db_file.stem
                
                # Comprobar si hay un archivo de metadatos
                meta_path = Path(f"{db_file}.meta.json")
                
                if meta_path.exists():
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            # Añadir la ruta absoluta al archivo
                            metadata["db_path"] = str(db_file)
                            
                            # Verificar y corregir el tipo de base de datos en los metadatos
                            db_type = metadata.get("db_type", "unknown")
                            if extension == ".sqlite" and db_type != "sqlite":
                                logger.warning(f"Corrigiendo tipo de base de datos en metadatos de {key}: {db_type} -> sqlite")
                                metadata["db_type"] = "sqlite"
                            elif extension == ".duckdb" and db_type != "duckdb":
                                logger.warning(f"Corrigiendo tipo de base de datos en metadatos de {key}: {db_type} -> duckdb")
                                metadata["db_type"] = "duckdb"
                                
                            available_dbs[key] = metadata
                            logger.debug(f"Encontrada base de datos con metadatos: {key} (tipo: {metadata['db_type']})")
                    except Exception as e:
                        logger.warning(f"Error al cargar metadatos de {meta_path}: {e}")
                        # Crear metadatos básicos en caso de error
                        available_dbs[key] = cls._create_default_metadata_for_file(db_file, extension)
                else:
                    # Sin metadatos, crear información básica
                    available_dbs[key] = cls._create_default_metadata_for_file(db_file, extension)
                    logger.debug(f"Encontrada base de datos sin metadatos: {key} (tipo inferido: {available_dbs[key]['db_type']})")
        
        # Ordenar por fecha de creación/uso
        for key in available_dbs:
            if "last_used" not in available_dbs[key] and "created_at" in available_dbs[key]:
                available_dbs[key]["last_used"] = available_dbs[key]["created_at"]
        
        # Mostrar resumen por tipo
        db_types = {}
        for key, metadata in available_dbs.items():
            db_type = metadata.get("db_type", "unknown")
            db_types[db_type] = db_types.get(db_type, 0) + 1
            
        for db_type, count in db_types.items():
            logger.info(f"Bases de datos de tipo {db_type}: {count}")
            
        logger.info(f"Total de bases de datos disponibles: {len(available_dbs)}")
        return available_dbs
    
    @classmethod
    def _create_default_metadata_for_file(cls, db_file: Path, extension: str) -> Dict[str, Any]:
        """
        Crea metadatos por defecto para un archivo de base de datos.
        
        Args:
            db_file: Ruta al archivo de base de datos
            extension: Extensión del archivo
            
        Returns:
            Dict con metadatos básicos
        """
        # Inferir el tipo de base de datos según la extensión
        db_type = "sqlite"  # valor por defecto
        if extension == ".sqlite":
            db_type = "sqlite"
        elif extension == ".duckdb":
            db_type = "duckdb"
        elif extension == ".db":
            # Para archivos .db, verificar si el nombre sugiere otro tipo
            if "duckdb" in db_file.stem.lower():
                db_type = "duckdb"
        
        metadata = {
            "db_type": db_type,
            "db_path": str(db_file),
            "embedding_model": "desconocido",
            "chunking_method": "desconocido",
            "created_at": os.path.getctime(db_file),
            "note": "Metadatos inferidos automáticamente"
        }
        
        return metadata
    
    @classmethod
    def _load_config(cls) -> Dict[str, Any]:
        """
        Carga la configuración de la base de datos.
        
        Returns:
            Configuración de la base de datos.
        """
        return config.get_database_config()
    
    @classmethod
    def close_all_instances(cls) -> None:
        """
        Cierra todas las instancias de base de datos abiertas.
        """
        for key, instance in list(cls._instances.items()):
            try:
                logger.info(f"Cerrando conexión a base de datos: {key}")
                instance.close_connection()
            except Exception as e:
                logger.error(f"Error al cerrar la base de datos {key}: {e}")
        
        # Limpiar el diccionario de instancias
        cls._instances.clear()
    
    @classmethod
    def _load_db_metadata(cls, db_path: str) -> Dict[str, Any]:
        """
        Carga los metadatos de una base de datos.
        
        Args:
            db_path: Ruta al archivo de la base de datos
            
        Returns:
            Dict[str, Any]: Metadatos de la base de datos
        """
        metadata = {}
        
        try:
            if isinstance(db_path, str) and os.path.exists(db_path):
                # Determinar la extensión correcta para el archivo de metadatos
                metadata_path = f"{db_path}.meta.json"
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        logger.debug(f"Metadatos cargados desde {metadata_path}")
                else:
                    # Intentar inferir información básica
                    filename = os.path.basename(db_path)
                    extension = os.path.splitext(db_path)[1].lower()
                    
                    # Determinar el tipo de base de datos por la extensión
                    db_type = "unknown"
                    if extension == ".sqlite":
                        db_type = "sqlite"
                    elif extension == ".duckdb":
                        db_type = "duckdb"
                    
                    # Crear metadatos básicos
                    metadata = {
                        "db_type": db_type,
                        "db_path": db_path,
                        "created_at": os.path.getctime(db_path),
                        "note": "Metadatos inferidos automáticamente"
                    }
                    
                    # Intentar extraer información del nombre
                    parts = filename.replace('.', '_').split('_')
                    if len(parts) >= 3 and parts[0] == "rag":
                        metadata["embedding_model"] = parts[1] if len(parts) > 1 else "unknown"
                        metadata["chunking_method"] = parts[2] if len(parts) > 2 else "unknown"
                    
                    logger.debug(f"Metadatos inferidos para {db_path}")
        except Exception as e:
            logger.warning(f"Error al cargar metadatos de {db_path}: {e}")
            
        return metadata
    
    @classmethod
    def get_db_statistics(cls) -> Dict[str, Any]:
        """
        Obtiene estadísticas de todas las bases de datos disponibles.
        
        Returns:
            Dict[str, Any]: Estadísticas agregadas de todas las bases de datos
        """
        stats = {
            "total_databases": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "databases": {}
        }
        
        # Obtener todas las bases de datos disponibles
        databases = cls.get_available_databases()
        
        stats["total_databases"] = len(databases)
        
        # Intentar obtener estadísticas detalladas de cada base de datos
        for name, metadata in databases.items():
            db_path = metadata.get("db_path")
            if not db_path or not os.path.exists(db_path):
                continue
                
            # Abrir la base de datos temporalmente para obtener estadísticas
            try:
                db_type = metadata.get("db_type", "sqlite")
                embedding_dim = metadata.get("embedding_dim", 384)  # valor por defecto
                
                db_class = cls._get_db_class(db_type)
                db_instance = db_class(embedding_dim=embedding_dim)
                db_instance.connect(db_path)
                
                # Obtener estadísticas
                db_stats = db_instance.get_statistics()
                
                # Actualizar contadores globales
                stats["total_documents"] += db_stats.get("total_documents", 0)
                stats["total_chunks"] += db_stats.get("total_chunks", 0)
                
                # Almacenar estadísticas específicas
                stats["databases"][name] = db_stats
                
                # Cerrar la conexión
                db_instance.close_connection()
                
            except Exception as e:
                logger.warning(f"Error al obtener estadísticas de {name}: {e}")
                stats["databases"][name] = {"error": str(e)}
        
        return stats
    
    @classmethod
    def optimize_all_databases(cls) -> Dict[str, bool]:
        """
        Optimiza todas las bases de datos disponibles.
        
        Returns:
            Dict[str, bool]: Resultados de optimización por base de datos
        """
        results = {}
        
        # Obtener todas las bases de datos disponibles
        databases = cls.get_available_databases()
        
        for name, metadata in databases.items():
            db_path = metadata.get("db_path")
            if not db_path or not os.path.exists(db_path):
                results[name] = False
                continue
                
            # Abrir la base de datos temporalmente para optimización
            try:
                db_type = metadata.get("db_type", "sqlite")
                embedding_dim = metadata.get("embedding_dim", 384)  # valor por defecto
                
                db_class = cls._get_db_class(db_type)
                db_instance = db_class(embedding_dim=embedding_dim)
                db_instance.connect(db_path)
                
                # Ejecutar optimización
                success = db_instance.optimize_database()
                results[name] = success
                
                # Cerrar la conexión
                db_instance.close_connection()
                
            except Exception as e:
                logger.warning(f"Error al optimizar {name}: {e}")
                results[name] = False
        
        return results
    
    @staticmethod
    def create_database(db_type: str, db_name: str, embedding_dim: int) -> Any:
        """
        Crea una instancia de base de datos según el tipo especificado.
        
        Args:
            db_type: Tipo de base de datos ('sqlite', 'duckdb', etc.)
            db_name: Nombre de la base de datos
            embedding_dim: Dimensión de los embeddings a almacenar
            
        Returns:
            Instancia de la base de datos
        """
        if db_type == "sqlite":
            from modulos.databases.implementaciones.sqlite import SQLiteVectorialDatabase
            return SQLiteVectorialDatabase(embedding_dim=embedding_dim)
        elif db_type == "duckdb":
            from modulos.databases.implementaciones.duckdb import DuckDBVectorialDatabase
            return DuckDBVectorialDatabase(embedding_dim=embedding_dim)
        else:
            raise ValueError(f"Tipo de base de datos no soportado: {db_type}")
    
    @staticmethod
    def load_database(db_name: str, metadata: Dict[str, Any]) -> Any:
        """
        Carga una base de datos existente usando sus metadatos.
        
        Args:
            db_name: Nombre de la base de datos
            metadata: Metadatos de la base de datos
            
        Returns:
            Instancia de la base de datos cargada
        """
        # Extraer metadatos necesarios
        db_type = metadata.get("db_type", "sqlite")
        embedding_dim = metadata.get("embedding_dim", 0)
        
        if embedding_dim <= 0:
            raise ValueError(f"Dimensión de embedding inválida en metadatos: {embedding_dim}")
        
        return DatabaseFactory.create_database(db_type, db_name, embedding_dim)