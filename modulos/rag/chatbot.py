"""
Implementación de chatbot RAG con respuestas en tiempo real.

Este módulo proporciona una clase especializada para integrar el sistema RAG 
en aplicaciones de chatbot, con soporte para respuestas en tiempo real (streaming),
manejo de sesiones de usuario y optimizaciones para entornos de alta concurrencia.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Generator, Tuple
from datetime import datetime
import uuid

from modulos.rag.app import RagApp

logger = logging.getLogger(__name__)

class RagChatbot:
    """
    Clase que implementa un chatbot RAG con respuestas en tiempo real.
    
    Esta clase extiende la funcionalidad de RagApp para ofrecer características
    específicas para chatbots, como manejo de sesiones de usuario, historial
    de conversaciones y optimizaciones para respuestas rápidas.
    """
    
    def __init__(
        self,
        database_name: Optional[str] = None,
        database_index: Optional[int] = None,
        ai_client: Optional[str] = None,
        streaming: bool = True,
        max_chunks: int = 5,
        max_history_length: int = 10,
        session_timeout: int = 3600  # 1 hora
    ):
        """
        Inicializa el chatbot RAG.
        
        Args:
            database_name: Nombre específico de la base de datos a utilizar
            database_index: Índice de la base de datos (0 es la más reciente)
            ai_client: Tipo de cliente de IA a utilizar (openai, gemini, ollama)
            streaming: Si se deben generar respuestas en streaming (habilitado por defecto)
            max_chunks: Número máximo de chunks a recuperar para cada consulta
            max_history_length: Número máximo de mensajes a conservar en el historial
            session_timeout: Tiempo en segundos después del cual una sesión se considera caducada
        """
        # Inicializar la aplicación RAG base
        self.rag_app = RagApp(
            database_name=database_name,
            database_index=database_index,
            ai_client=ai_client,
            streaming=streaming,
            max_chunks=max_chunks
        )
        
        # Configuración específica del chatbot
        self.max_history_length = max_history_length
        self.session_timeout = session_timeout
        
        # Almacenamiento de sesiones de usuarios
        # {session_id: {created_at: timestamp, last_used: timestamp, history: [...]}}
        self.user_sessions = {}
        
        logger.info("RagChatbot inicializado")
    
    def create_session(self) -> str:
        """
        Crea una nueva sesión de usuario.
        
        Returns:
            ID de la sesión creada
        """
        session_id = str(uuid.uuid4())
        timestamp = time.time()
        
        self.user_sessions[session_id] = {
            "created_at": timestamp,
            "last_used": timestamp,
            "history": []
        }
        
        logger.info(f"Nueva sesión de usuario creada: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene una sesión de usuario por su ID.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Información de la sesión o None si no existe
        """
        session = self.user_sessions.get(session_id)
        
        if not session:
            return None
            
        # Verificar si la sesión ha caducado
        if time.time() - session["last_used"] > self.session_timeout:
            logger.info(f"Sesión caducada: {session_id}")
            return None
            
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """
        Elimina una sesión de usuario.
        
        Args:
            session_id: ID de la sesión a eliminar
            
        Returns:
            True si la sesión fue eliminada, False si no existía
        """
        if session_id in self.user_sessions:
            del self.user_sessions[session_id]
            logger.info(f"Sesión eliminada: {session_id}")
            return True
        
        return False
    
    def _update_session_history(
        self, 
        session_id: str, 
        query: str, 
        response: str
    ) -> None:
        """
        Actualiza el historial de una sesión con una nueva interacción.
        
        Args:
            session_id: ID de la sesión
            query: Consulta del usuario
            response: Respuesta del sistema
        """
        session = self.get_session(session_id)
        
        if not session:
            logger.warning(f"Intento de actualizar historial de sesión inexistente: {session_id}")
            return
        
        # Actualizar timestamp de último uso
        session["last_used"] = time.time()
        
        # Añadir la interacción al historial
        interaction = {
            "timestamp": time.time(),
            "query": query,
            "response": response
        }
        
        session["history"].append(interaction)
        
        # Limitar el tamaño del historial
        if len(session["history"]) > self.max_history_length:
            session["history"] = session["history"][-self.max_history_length:]
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de interacciones de una sesión.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Lista de interacciones o lista vacía si la sesión no existe
        """
        session = self.get_session(session_id)
        
        if not session:
            return []
            
        return session["history"]
    
    def _process_streaming_response(
        self, 
        response: Generator[str, None, None]
    ) -> Tuple[str, Generator[str, None, None]]:
        """
        Procesa una respuesta en streaming para devolver tanto el generador como capturar el texto completo.
        
        Args:
            response: Generador de respuesta en streaming
            
        Returns:
            Tupla con (texto completo acumulado, nuevo generador)
        """
        # Creamos una lista para acumular los fragmentos
        accumulated_chunks = []
        
        # Creamos un nuevo generador que acumula y reenvía los fragmentos
        def stream_and_accumulate():
            for chunk in response:
                accumulated_chunks.append(chunk)
                yield chunk
                
        # Devolvemos una función que permite obtener el texto completo después de consumir el generador
        def get_full_text():
            return ''.join(accumulated_chunks)
            
        return get_full_text, stream_and_accumulate()
    
    def process_query(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        stream: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Procesa una consulta de usuario en el contexto de una sesión de chat.
        
        Args:
            query: Texto de la consulta
            session_id: ID de sesión (si es None, se crea una nueva)
            stream: Si se debe devolver la respuesta en streaming
            
        Returns:
            Diccionario con la información de la respuesta:
            - response: Texto de respuesta o generador de streaming
            - session_id: ID de la sesión utilizada
            - is_streaming: True si la respuesta es streaming
            - timestamp: Timestamp de la consulta
        """
        # Obtener o crear sesión
        if session_id:
            session = self.get_session(session_id)
            if not session:
                # Si la sesión no existe o ha caducado, crear una nueva
                session_id = self.create_session()
        else:
            # Crear una nueva sesión
            session_id = self.create_session()
        
        # Procesar la consulta usando la aplicación RAG
        start_time = time.time()
        
        try:
            response = self.rag_app.process_query(query, stream=stream)
            
            # Si es una respuesta en streaming, necesitamos manejarla especialmente
            is_streaming = hasattr(response, '__iter__') and not isinstance(response, str)
            
            if is_streaming:
                # Procesar la respuesta en streaming para capturar el texto completo
                get_full_text, streaming_generator = self._process_streaming_response(response)
                
                # Crear una función que permita actualizar el historial después de consumir el streaming
                def update_history_after_streaming():
                    # Obtener el texto completo una vez consumido el generador
                    full_text = get_full_text()
                    
                    # Actualizar el historial con el texto completo
                    self._update_session_history(session_id, query, full_text)
                
                # Devolver un objeto con la respuesta en streaming y metadatos
                return {
                    "response": streaming_generator,
                    "session_id": session_id,
                    "is_streaming": True,
                    "timestamp": start_time,
                    "update_history": update_history_after_streaming
                }
            else:
                # Respuesta normal (no streaming)
                # Actualizar el historial de la sesión
                self._update_session_history(session_id, query, response)
                
                # Devolver objeto con la respuesta y metadatos
                return {
                    "response": response,
                    "session_id": session_id,
                    "is_streaming": False,
                    "timestamp": start_time
                }
        
        except Exception as e:
            error_msg = f"Error al procesar consulta en chatbot: {str(e)}"
            logger.error(error_msg)
            
            # Actualizar el historial con el error
            self._update_session_history(session_id, query, error_msg)
            
            # Devolver objeto con el error
            return {
                "response": error_msg,
                "session_id": session_id,
                "is_streaming": False,
                "error": True,
                "timestamp": start_time
            }
    
    def clean_expired_sessions(self) -> int:
        """
        Limpia las sesiones caducadas.
        
        Returns:
            Número de sesiones eliminadas
        """
        current_time = time.time()
        sessions_to_delete = []
        
        # Identificar sesiones caducadas
        for session_id, session in self.user_sessions.items():
            if current_time - session["last_used"] > self.session_timeout:
                sessions_to_delete.append(session_id)
        
        # Eliminar sesiones caducadas
        for session_id in sessions_to_delete:
            del self.user_sessions[session_id]
        
        count = len(sessions_to_delete)
        if count > 0:
            logger.info(f"Se eliminaron {count} sesiones caducadas")
        
        return count
    
    def get_active_sessions_count(self) -> int:
        """
        Obtiene el número de sesiones activas.
        
        Returns:
            Número de sesiones activas
        """
        self.clean_expired_sessions()  # Limpiar sesiones caducadas primero
        return len(self.user_sessions)
    
    def extract_context_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extrae el contexto utilizado de una respuesta completa.
        
        Args:
            response: Respuesta completa con formato
            
        Returns:
            Diccionario con el texto de respuesta y el contexto utilizado
        """
        result = {
            "response_text": response,
            "context_text": None
        }
        
        # Extraer respuesta y contexto si existen los separadores
        if "=======================  RESPUESTA  =======================" in response:
            parts = response.split("=======================  RESPUESTA  =======================")
            
            if len(parts) > 1:
                # Extraer la parte de respuesta
                response_text = parts[1].split("=======================  CONTEXTO  =======================")[0].strip()
                result["response_text"] = response_text
                
                # Extraer contexto si existe
                context_parts = response.split("=======================  CONTEXTO  =======================")
                if len(context_parts) > 1:
                    result["context_text"] = context_parts[1].strip()
        
        return result 