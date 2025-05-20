"""
Gestor de embeddings para el sistema RAG.

Este módulo se encarga de cargar y gestionar los modelos de embeddings,
utilizando sentence-transformers para manejar automáticamente los modelos.
"""
import logging
from typing import List, Optional
import numpy as np

# Configurar logging
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Gestor para modelos de embeddings.
    Maneja la carga, gestión y generación de embeddings usando
    diversos modelos como ModernBERT, CDE, E5, etc.
    """
    
    def __init__(self, model_type: str = None):
        """
        Inicializa el gestor de embeddings.
        
        Args:
            model_type: Tipo de modelo a utilizar (modernbert, cde-small, e5-small).
                       Si es None, se usa el configurado en config.yaml.
        """
        # Importar aquí para evitar la carga prematura
        from config import config
        
        # Cargar configuración de embeddings
        self.embedding_config = config.get_embedding_config()
        
        # Determinar el modelo a utilizar
        if model_type is None:
            model_type = self.embedding_config.get("model", "modernbert")
        
        self.model_type = model_type
        self.model_config = config.get_specific_model_config(model_type)
        self.model_name = self.model_config.get("model_name", "")
        
        # Inicializar atributos para el modelo cargado (lazy loading)
        self._model = None
        self._embedding_dim = None
        
        logger.info(f"EmbeddingManager inicializado, modelo seleccionado: {model_type} ({self.model_name})")
    
    def load_model(self) -> None:
        """
        Carga el modelo de embeddings según la configuración.
        
        Sentence-transformers se encarga de gestionar automáticamente la caché de modelos.
        """
        # Importar aquí para permitir lazy loading y mejorar rendimiento
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Cargando modelo de embeddings: {self.model_name}")
        
        # Obtener parámetros de configuración
        trust_remote_code = self.embedding_config.get("trust_remote_code", False)
        device = self.model_config.get("device", "cpu")
        
        try:
            # Cargar el modelo - sentence-transformers lo buscará en caché local o lo descargará
            self._model = SentenceTransformer(
                self.model_name, 
                device=device,
                trust_remote_code=trust_remote_code
            )
            
            # Obtener dimensión del embedding
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Modelo cargado correctamente. Dimensión del embedding: {self._embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error al cargar modelo de embeddings: {e}")
            raise
    
    @property
    def model(self):
        """
        Acceso al modelo con carga automática si no está cargado.
        
        Returns:
            El modelo de embeddings cargado
        """
        if self._model is None:
            self.load_model()
        return self._model
    
    @property
    def embedding_dim(self) -> int:
        """
        Dimensión del embedding con carga automática si es necesario.
        
        Returns:
            La dimensión del embedding
        """
        if self._embedding_dim is None:
            if self._model is None:
                self.load_model()
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
        return self._embedding_dim
    
    def get_document_embedding(self, header: Optional[str], text: str) -> List[float]:
        """
        Genera embeddings para documentos o chunks combinando encabezado y texto.
        
        Args:
            header: Encabezado del documento o chunk (puede ser None)
            text: Texto principal del documento o chunk
            
        Returns:
            Lista de valores float representando el embedding
        """
        # Combinar encabezado y texto para generar el embedding
        if header and header.strip():
            full_text = f"{header} - {text}"
        else:
            full_text = text
        
        # Generar embedding
        embedding = self.model.encode(full_text)
        
        # Convertir a lista si es un array numpy
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        
        return embedding
    
    def get_query_embedding(self, query: str) -> List[float]:
        """
        Genera embeddings para consultas.
        
        Args:
            query: Texto de la consulta
            
        Returns:
            Lista de valores float representando el embedding
        """
        # Algunos modelos requieren formateo especial para queries (e.g., E5)
        if self.model_type == "e5-small" and self.model_config.get("prefix_queries", False):
            query = f"query: {query}"
        
        # Generar embedding
        embedding = self.model.encode(query)
        
        # Convertir a lista si es un array numpy
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        
        return embedding
    
    def get_dimensions(self) -> int:
        """
        Obtiene la dimensión del embedding.
        
        Returns:
            Dimensión del embedding (int)
        """
        return self.embedding_dim
    
    def batch_encode(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para múltiples textos procesando uno por uno.
        Esta implementación evita el procesamiento por lotes real para prevenir problemas de memoria.
        
        Args:
            texts: Lista de textos a codificar
            
        Returns:
            Lista de embeddings (cada uno como lista de floats)
        """
        # Procesar uno por uno en vez de por lotes para evitar problemas de memoria
        embeddings = []
        for text in texts:
            # Usar encode con un solo texto
            embedding = self.model.encode(text)
            
            # Convertir a lista si es un array numpy
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            embeddings.append(embedding)
            
        return embeddings

    def get_batch_embeddings(self, headers: List[Optional[str]], texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para múltiples pares de encabezado-texto.
        Esta implementación evita el procesamiento por lotes real y usa get_document_embedding
        uno por uno para evitar problemas de memoria.
        
        Args:
            headers: Lista de encabezados (puede contener None o strings vacíos)
            texts: Lista de textos principales
            
        Returns:
            Lista de embeddings (cada uno como lista de floats)
        """
        # Procesamos uno por uno para evitar problemas de memoria
        embeddings = []
        for header, text in zip(headers, texts):
            embedding = self.get_document_embedding(header, text)
            embeddings.append(embedding)
                
        return embeddings
