"""
Implementación de chatbot RAG con respuestas en tiempo real.

Este módulo proporciona una clase especializada para integrar el sistema RAG 
en aplicaciones de chatbot, con soporte para respuestas en tiempo real (streaming),
manejo de sesiones de usuario y optimizaciones para entornos de alta concurrencia.
"""

import logging
import time
import json
import threading
from typing import Dict, List, Any, Optional, Union, Generator, Tuple
from datetime import datetime
import uuid

from modulos.rag.app import RagApp
from modulos.session_manager.session_manager import SessionManager

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
        ai_client: Optional[str] = None,
        streaming: bool = True,
        max_chunks: int = 5,
        max_history_length: int = 10,
        session_id: Optional[str] = None
    ):
        """
        Inicializa el chatbot RAG.
        
        Args:
            database_name: Nombre específico de la base de datos a utilizar
            ai_client: Tipo de cliente de IA a utilizar (openai, gemini, ollama)
            streaming: Si se deben generar respuestas en streaming (habilitado por defecto)
            max_chunks: Número máximo de chunks a recuperar para cada consulta
            max_history_length: Número máximo de mensajes a conservar en el historial
            session_id: ID de sesión existente a utilizar (opcional)
        """
        # Inicializar Session Manager
        self.session_manager = SessionManager()
        
        # Crear una sesión o usar la proporcionada
        if session_id:
            self.session_id = session_id
            if not self.session_manager.get_session(session_id):
                # Si la sesión no existe, crear una nueva
                self.session_id = self.session_manager.create_session(session_id)
        else:
            self.session_id = self.session_manager.create_session()
        
        # Inicializar la aplicación RAG base con la sesión
        self.rag_app = RagApp(
            database_name=database_name,
            ai_client=ai_client,
            streaming=streaming,
            session_id=self.session_id
        )
        
        # Configuración específica del chatbot
        self.max_history_length = max_history_length
        
        # Guardar configuración en el gestor de sesiones
        self.session_manager.store_session_config(self.session_id, {
            "chatbot": {
                "max_history_length": max_history_length,
                "streaming": streaming,
                "max_chunks": max_chunks
            }
        })
        
        # Actualizar metadatos de la sesión
        self.session_manager.update_session_metadata(self.session_id, {
            "app_type": "chatbot",
            "database_name": database_name,
            "ai_client": ai_client,
            "streaming": streaming,
            "created_at": time.time()
        })
        
        logger.info(f"RagChatbot inicializado con session_id={self.session_id}")
    
    def create_session(self) -> str:
        """
        Crea una nueva sesión de usuario.
        
        Returns:
            ID de la sesión creada
        """
        # Crear nueva sesión usando el SessionManager
        session_id = self.session_manager.create_session(metadata={
            "app_type": "chatbot",
            "database_name": self.rag_app.database_name,
            "ai_client": getattr(self.rag_app, "ai_client_type", None),
            "streaming": self.rag_app.streaming,
            "created_at": time.time()
        })
        
        # Guardar configuración en la sesión
        self.session_manager.store_session_config(session_id, {
            "chatbot": {
                "max_history_length": self.max_history_length,
                "streaming": self.rag_app.streaming,
                "history": []
            }
        })
        
        logger.info(f"Nueva sesión de chatbot creada: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene una sesión de usuario por su ID.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Información de la sesión o None si no existe
        """
        # Obtener sesión del SessionManager
        session_data = self.session_manager.get_session(session_id)
        
        if not session_data:
            logger.warning(f"Sesión no encontrada: {session_id}")
            return None
        
        # Obtener configuración específica de chatbot
        chatbot_config = self.session_manager.get_session_config(session_id, "chatbot", {})
        
        # Combinar información para mantener compatibilidad con código existente
        combined_info = {
            **session_data,
            "history": chatbot_config.get("history", [])
        }
        
        return combined_info
    
    def delete_session(self, session_id: str) -> bool:
        """
        Elimina una sesión de usuario.
        
        Args:
            session_id: ID de la sesión a eliminar
            
        Returns:
            True si la sesión fue eliminada, False si no existía
        """
        # Eliminar sesión usando el SessionManager
        result = self.session_manager.delete_session(session_id)
        
        if result:
            logger.info(f"Sesión eliminada: {session_id}")
        else:
            logger.warning(f"Intento de eliminar sesión inexistente: {session_id}")
        
        return result
    
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
        # Obtener historial actual
        chatbot_config = self.session_manager.get_session_config(session_id, "chatbot", {})
        history = chatbot_config.get("history", [])
        
        # Añadir la interacción al historial
        interaction = {
            "timestamp": time.time(),
            "query": query,
            "response": response
        }
        
        history.append(interaction)
        
        # Limitar el tamaño del historial
        if len(history) > self.max_history_length:
            history = history[-self.max_history_length:]
        
        # Actualizar configuración de la sesión con el nuevo historial
        self.session_manager.store_session_config(session_id, {
            "chatbot": {
                **chatbot_config,
                "history": history,
                "last_interaction": time.time()
            }
        })
        
        # Actualizar metadata de la sesión
        self.session_manager.update_session_metadata(session_id, {
            "last_activity": time.time(),
            "last_query": query[:50] + "..." if len(query) > 50 else query,
        })
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de interacciones de una sesión.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Lista de interacciones o lista vacía si la sesión no existe
        """
        # Obtener historial de la configuración de la sesión
        chatbot_config = self.session_manager.get_session_config(session_id, "chatbot", {})
        return chatbot_config.get("history", [])
    
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
        
        # Función para obtener el texto completo cuando se ha consumido el generador
        def get_full_text():
            return "".join(accumulated_chunks)
        
        return get_full_text, stream_and_accumulate()
    
    def process_query(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        stream: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Procesa una consulta del usuario y genera una respuesta.
        
        Args:
            query: Consulta del usuario
            session_id: ID de sesión (opcional, usa la sesión del constructor si no se proporciona)
            stream: Forzar o desactivar streaming (opcional, usa el valor predeterminado si no se proporciona)
            
        Returns:
            Diccionario con la respuesta y metadatos
        """
        # Usar la sesión proporcionada o la del constructor
        current_session_id = session_id or self.session_id
        
        # Comprobar que la sesión existe
        session_data = self.session_manager.get_session(current_session_id)
        if not session_data:
            logger.error(f"Sesión no encontrada: {current_session_id}")
            return {
                "status": "error",
                "message": "Sesión no válida o caducada",
                "session_id": current_session_id
            }
        
        # Configurar streaming
        use_streaming = stream if stream is not None else self.rag_app.streaming
        
        try:
            start_time = time.time()
            
            # Procesar la consulta
            message_id = str(uuid.uuid4())
            
            if use_streaming:
                # Para respuestas en streaming, necesitamos capturar el texto completo
                generator = self.rag_app.query(query)
                get_full_text, stream_gen = self._process_streaming_response(generator)
                
                # Crear un hilo para actualizar el historial cuando se consuma el streaming
                def update_history_after_streaming():
                    # Obtener el texto completo una vez consumido el generador
                    try:
                        full_response = get_full_text()
                        self._update_session_history(current_session_id, query, full_response)
                    except Exception as e:
                        logger.error(f"Error al actualizar historial después de streaming: {e}")
                
                # Preparar respuesta con el generador de streaming
                return {
                    "status": "success",
                    "response": stream_gen,
                    "streaming": True,
                    "session_id": current_session_id,
                    "message_id": message_id,
                    "start_time": start_time,
                    "update_history": update_history_after_streaming
                }
            else:
                # Para respuestas no streaming, obtenemos la respuesta completa
                response = self.rag_app.query(query)
                
                # Actualizar historial inmediatamente
                self._update_session_history(current_session_id, query, response)
                
                # Calcular tiempo de respuesta
                response_time = time.time() - start_time
                
                return {
                    "status": "success",
                    "response": response,
                    "streaming": False,
                    "session_id": current_session_id,
                    "message_id": message_id,
                    "response_time": response_time
                }
        
        except Exception as e:
            logger.error(f"Error al procesar consulta: {e}")
            return {
                "status": "error",
                "message": str(e),
                "session_id": current_session_id
            }
    
    def clean_expired_sessions(self) -> int:
        """
        Limpia sesiones caducadas.
        
        Returns:
            Número de sesiones eliminadas
        """
        # Delegar la limpieza al SessionManager
        result = self.session_manager.clean_expired_sessions()
        logger.info(f"Limpieza de sesiones completada: {result} sesiones eliminadas")
        return result
    
    def get_active_sessions_count(self) -> int:
        """
        Obtiene el número de sesiones activas.
        
        Returns:
            Número de sesiones activas
        """
        # Obtener todas las sesiones del tipo chatbot
        all_sessions = self.session_manager.list_sessions()
        chatbot_sessions = [s for s in all_sessions if s.get("app_type") == "chatbot"]
        return len(chatbot_sessions)
    
    def extract_context_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extrae información de contexto de una respuesta, si está presente.
        
        Args:
            response: Respuesta completa del sistema
            
        Returns:
            Diccionario con el contexto extraído o vacío si no se encuentra
        """
        # Verificar si hay un ID de mensaje oculto
        message_id = None
        if "<hidden_message_id>" in response:
            import re
            match = re.search(r'<hidden_message_id>(.*?)</hidden_message_id>', response)
            if match:
                message_id = match.group(1)
                # Eliminar el ID oculto de la respuesta
                response = response.replace(match.group(0), "")
        
        # Si tenemos un message_id, intentar obtener su contexto
        context_data = []
        if message_id:
            try:
                context_data = self.session_manager.get_message_context(self.session_id, message_id) or []
            except Exception as e:
                logger.error(f"Error al obtener contexto para message_id={message_id}: {e}")
        
        return {
            "message_id": message_id,
            "context": context_data,
            "clean_response": response
        }
    
    def close(self):
        """
        Cierra el chatbot y libera sus recursos.
        """
        if hasattr(self, 'rag_app') and self.rag_app:
            self.rag_app.close()
        logger.info(f"RagChatbot cerrado (session_id={self.session_id})") 