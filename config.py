import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """
    Clase para gestionar la configuración centralizada del sistema RAG.
    Carga la configuración desde un archivo YAML y proporciona métodos
    para acceder a las diferentes secciones de configuración.
    """
    _instance = None
    
    def __new__(cls, config_path: str = "config.yaml"):
        """
        Implementa el patrón Singleton para asegurar una única instancia
        de configuración en toda la aplicación.
        """
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa la configuración desde un archivo YAML.
        
        Args:
            config_path: Ruta al archivo de configuración YAML.
        """
        # Evita reinicializar si ya está inicializado (parte del patrón Singleton)
        if self.initialized:
            return
            
        # Cargar variables de entorno desde .env si existe
        load_dotenv()
        
        self.config_path = config_path
        self.config = self._load_config()
        self.initialized = True
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Carga la configuración desde el archivo YAML.
        
        Returns:
            Diccionario con la configuración.
            
        Raises:
            FileNotFoundError: Si no se encuentra el archivo de configuración.
        """
        config_path = Path(self.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {self.config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def _process_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa variables de entorno en la configuración.
        
        Args:
            config: Configuración como diccionario.
            
        Returns:
            Configuración con variables de entorno procesadas.
        """
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, (dict, list)):
                    config[key] = self._process_env_vars(value)
                elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    config[key] = os.environ.get(env_var, "")
        elif isinstance(config, list):
            for i, item in enumerate(config):
                config[i] = self._process_env_vars(item)
        
        return config
    
    def get_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración completa.
        
        Returns:
            Configuración como diccionario.
        """
        # Procesar variables de entorno antes de devolver
        return self._process_env_vars(self.config.copy())
    
    def get_general_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración general del sistema.
        
        Returns:
            Diccionario con la configuración general.
        """
        general_config = self.config.get("general", {})
        return self._process_env_vars(general_config)
    
    def get_chunks_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración de chunking de documentos.
        
        Returns:
            Diccionario con la configuración de chunks.
        """
        chunks_config = self.config.get("chunks", {})
        return self._process_env_vars(chunks_config)
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración del modelo de embeddings.
        
        Returns:
            Diccionario con la configuración de embeddings.
        """
        embedding_config = self.config.get("embeddings", {})
        return self._process_env_vars(embedding_config)
    
    def get_database_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración de la base de datos.
        
        Returns:
            Diccionario con la configuración de la base de datos.
        """
        db_config = self.config.get("database", {})
        db_config = self._update_db_paths(db_config)
        return self._process_env_vars(db_config)
    
    def _update_db_paths(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actualiza las rutas de la base de datos para asegurar que sean correctas
        y que los directorios existan.
        
        Args:
            db_config: Configuración original de la base de datos.
            
        Returns:
            Configuración actualizada con rutas absolutas.
        """
        db_type = db_config.get("type", "sqlite")
        
        # Obtener la configuración específica del tipo de base de datos
        type_config = db_config.get(db_type, {})
        
        if "db_dir" in type_config and "db_name" in type_config:
            db_dir = Path(type_config["db_dir"])
            db_name = type_config["db_name"]
            
            # Asegurar que el directorio exista
            os.makedirs(db_dir, exist_ok=True)
            
            # Calcular y agregar la ruta completa
            db_path = db_dir / db_name
            type_config["db_path"] = str(db_path)
            db_config[db_type] = type_config
        
        return db_config
    
    def get_ai_client_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración del cliente de IA.
        
        Returns:
            Diccionario con la configuración del cliente de IA.
        """
        ai_client_config = self.config.get("ai_client", {})
        return self._process_env_vars(ai_client_config)
    
    def get_processing_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración de procesamiento de documentos.
        
        Returns:
            Diccionario con la configuración de procesamiento.
        """
        processing_config = self.config.get("processing", {})
        return self._process_env_vars(processing_config)
    
    def get_specific_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Obtiene la configuración específica de un modelo de embedding.
        
        Args:
            model_type: Tipo de modelo (ej. 'modernbert', 'cde-small', 'e5-small')
        
        Returns:
            Configuración específica del modelo.
        """
        embedding_config = self.get_embedding_config()
        model_config = embedding_config.get(model_type, {})
        return model_config
    
    def get_specific_ai_config(self, ai_type: str) -> Dict[str, Any]:
        """
        Obtiene la configuración específica de un cliente de IA.
        
        Args:
            ai_type: Tipo de cliente (ej. 'openai', 'gemini', 'ollama')
        
        Returns:
            Configuración específica del cliente de IA.
        """
        ai_config = self.get_ai_client_config()
        specific_config = ai_config.get(ai_type, {})
        
        # Combinar con los parámetros generales si existen
        general_params = ai_config.get("general", {})
        
        # Los parámetros específicos tienen prioridad sobre los generales
        combined_config = {**general_params, **specific_config}
        return combined_config
    
    def get_chunker_method_config(self, method: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene la configuración de un método específico de chunking.
        
        Args:
            method: Método de chunking ('character', 'token', 'context'). 
                   Si es None, se utiliza el método configurado por defecto.
        
        Returns:
            Configuración del método de chunking.
        """
        chunks_config = self.get_chunks_config()
        if method is None:
            method = chunks_config.get("method", "context")
        
        method_config = chunks_config.get(method, {})
        return method_config
    
    def save_config(self) -> None:
        """
        Guarda la configuración actual en el archivo YAML.
        """
        with open(self.config_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.config, file, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """
        Actualiza un valor específico en la configuración.
        
        Args:
            section: Sección de la configuración (ej. 'database', 'embeddings')
            key: Clave a actualizar
            value: Nuevo valor
        """
        if section in self.config:
            if key in self.config[section]:
                self.config[section][key] = value
            else:
                # Si la clave no existe, la añadimos
                self.config[section][key] = value
        else:
            # Si la sección no existe, la creamos
            self.config[section] = {key: value}
    
    def get_database_instance_config(self, db_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene la configuración específica para inicializar una instancia de base de datos.
        
        Args:
            db_type: Tipo de base de datos. Si es None, se usa el configurado.
            
        Returns:
            Configuración para inicializar la base de datos.
        """
        db_config = self.get_database_config()
        
        if db_type is None:
            db_type = db_config.get("type", "sqlite")
        
        type_config = db_config.get(db_type, {})
        
        # Añadir el tipo a la configuración
        instance_config = {"type": db_type, **type_config}
        
        return instance_config

# Instancia global para uso como singleton
config = Config()