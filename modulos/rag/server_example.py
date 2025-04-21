"""
Ejemplo de servidor web para el chatbot RAG con respuestas en tiempo real.

Este módulo proporciona un ejemplo práctico de cómo integrar el chatbot RAG
en un servidor web usando Flask y respuestas en streaming.
"""

import logging
import time
import json
import os
from typing import Dict, Any, List

from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
import threading

from modulos.rag.chatbot import RagChatbot
from modulos.session_manager.session_manager import SessionManager
from modulos.rag.app import RagApp
from config import Config

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear la aplicación Flask
app = Flask(__name__)
config = Config()

# Chatbot global para sesiones compartidas
chatbot = None

# Diccionario para almacenar chatbots específicos por sesión
active_chatbots = {}

# Instancia del SessionManager
session_manager = SessionManager()

def initialize_chatbot(database_name=None):
    """
    Inicializa el chatbot RAG con la configuración por defecto o específica.
    
    Args:
        database_name: Nombre de la base de datos a utilizar (opcional)
    """
    global chatbot
    
    try:
        # Inicializar el chatbot
        chatbot = RagChatbot(
            database_name=database_name,
            streaming=True,
            max_chunks=5
        )
        logger.info(f"Chatbot RAG inicializado correctamente con database_name={database_name}")
        return True
    except Exception as e:
        logger.error(f"Error al inicializar el chatbot RAG: {e}")
        return False

# Middleware para CORS (necesario para aplicaciones web)
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Ruta principal para servir la interfaz web estática
@app.route('/')
def index():
    """Sirve la página principal"""
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    return send_from_directory(static_dir, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Sirve archivos estáticos"""
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    return send_from_directory(static_dir, filename)

# Ruta para verificar el estado del servidor
@app.route('/health', methods=['GET'])
def health_check():
    """Verifica el estado del servidor."""
    resource_usage = session_manager.get_resource_usage()
    
    # Estadísticas del servidor
    stats = {
        "status": "ok",
        "timestamp": time.time(),
        "sessions": {
            "active": len(session_manager.get_all_sessions()),
            "active_chatbots": len(active_chatbots)
        },
        "resources": {
            "memory_percent": resource_usage.get("memory_percent", 0),
            "cpu_percent": resource_usage.get("cpu_percent", 0)
        }
    }
    
    return jsonify(stats)

# Ruta para listar bases de datos disponibles
@app.route('/databases', methods=['GET'])
def list_databases():
    """Lista las bases de datos disponibles."""
    try:
        # Usar SessionManager para listar bases de datos
        db_dict = session_manager.list_available_databases()
        
        # Formatear resultados
        databases = []
        for db_name, metadata in db_dict.items():
            size_mb = metadata.get("size", 0) / (1024 * 1024) if metadata.get("size") else 0
            
            databases.append({
                "id": db_name,
                "name": metadata.get("name", db_name),
                "description": metadata.get("description", ""),
                "size": metadata.get("size", 0),
                "size_formatted": f"{size_mb:.2f} MB",
                "type": metadata.get("db_type", "unknown"),
                "created_at": metadata.get("created_at", 0)
            })
        
        # Ordenar por fecha de creación (más reciente primero)
        databases.sort(key=lambda x: x["created_at"], reverse=True)
        
        return jsonify({
            "status": "success",
            "count": len(databases),
            "databases": databases
        })
    except Exception as e:
        logger.error(f"Error al listar bases de datos: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Ruta para crear una nueva sesión
@app.route('/sessions', methods=['POST'])
def create_session():
    """Crea una nueva sesión de chat."""
    try:
        # Obtener parámetros de la petición
        data = request.get_json() or {}
        database_name = data.get('database_name')
        ai_client = data.get('ai_client')
        
        # Crear un chatbot específico para esta sesión
        try:
            session_chatbot = RagChatbot(
                database_name=database_name,
                ai_client=ai_client,
                streaming=True
            )
            
            # El constructor de RagChatbot ya crea una sesión
            session_id = session_chatbot.session_id
            
            # Almacenar el chatbot específico para esta sesión
            active_chatbots[session_id] = session_chatbot
            
            # Obtener metadatos de la sesión
            session_data = session_manager.get_session(session_id)
            
            return jsonify({
                "status": "success",
                "session_id": session_id,
                "message": "Sesión creada correctamente",
                "database": database_name,
                "ai_client": ai_client or config.get_ai_client_config().get("type", "openai"),
                "created_at": session_data.get("created_at", time.time())
            })
        except Exception as e:
            logger.error(f"Error al crear chatbot con base de datos {database_name}: {e}")
            return jsonify({
                "status": "error",
                "message": f"Error al inicializar con la base de datos seleccionada: {str(e)}"
            }), 400
    
    except Exception as e:
        logger.error(f"Error al crear sesión: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Ruta para procesar consultas (respuesta completa)
@app.route('/query', methods=['POST'])
def process_query():
    """Procesa una consulta y devuelve una respuesta completa."""
    try:
        # Obtener parámetros de la petición
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "status": "error",
                "message": "Parámetro 'query' requerido"
            }), 400
        
        query = data['query']
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                "status": "error",
                "message": "Parámetro 'session_id' requerido"
            }), 400
        
        # Verificar que la sesión existe en SessionManager
        if not session_manager.get_session(session_id):
            return jsonify({
                "status": "error",
                "message": "Sesión no válida o expirada"
            }), 404
        
        # Seleccionar el chatbot apropiado o crear uno nuevo si es necesario
        if session_id in active_chatbots:
            bot = active_chatbots[session_id]
        else:
            # Obtener datos de la sesión
            session_data = session_manager.get_session(session_id)
            database_name = session_data.get("database_name")
            ai_client = session_data.get("ai_client")
            
            # Crear un nuevo chatbot para esta sesión
            try:
                bot = RagChatbot(
                    database_name=database_name,
                    ai_client=ai_client,
                    streaming=False,
                    session_id=session_id
                )
                active_chatbots[session_id] = bot
            except Exception as e:
                logger.error(f"Error al recrear chatbot para sesión {session_id}: {e}")
                return jsonify({
                    "status": "error",
                    "message": f"Error al procesar la consulta: {str(e)}"
                }), 500
        
        # Procesar la consulta (sin streaming)
        result = bot.process_query(query, session_id=session_id, stream=False)
        
        # Verificar resultado
        if result.get("status") == "error":
            return jsonify({
                "status": "error",
                "message": result.get("message", "Error desconocido al procesar la consulta")
            }), 500
        
        # Extraer respuesta y contexto
        response_text = result.get("response", "")
        context_info = bot.extract_context_from_response(response_text)
        
        return jsonify({
            "status": "success",
            "response": context_info.get("clean_response", response_text),
            "message_id": context_info.get("message_id"),
            "context": context_info.get("context", []),
            "session_id": session_id,
            "timestamp": time.time(),
            "response_time": result.get("response_time", 0)
        })
    
    except Exception as e:
        logger.error(f"Error al procesar consulta: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Ruta para procesar consultas con streaming
@app.route('/query/stream', methods=['POST'])
def process_query_stream():
    """Procesa una consulta y devuelve una respuesta en streaming."""
    try:
        # Obtener parámetros de la petición
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "status": "error",
                "message": "Parámetro 'query' requerido"
            }), 400
        
        query = data['query']
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                "status": "error",
                "message": "Parámetro 'session_id' requerido"
            }), 400
        
        # Verificar que la sesión existe en SessionManager
        if not session_manager.get_session(session_id):
            return jsonify({
                "status": "error",
                "message": "Sesión no válida o expirada"
            }), 404
        
        # Seleccionar el chatbot apropiado o crear uno nuevo
        if session_id in active_chatbots:
            bot = active_chatbots[session_id]
        else:
            # Obtener datos de la sesión
            session_data = session_manager.get_session(session_id)
            database_name = session_data.get("database_name")
            ai_client = session_data.get("ai_client")
            
            # Crear un nuevo chatbot para esta sesión
            try:
                bot = RagChatbot(
                    database_name=database_name,
                    ai_client=ai_client,
                    streaming=True,
                    session_id=session_id
                )
                active_chatbots[session_id] = bot
            except Exception as e:
                logger.error(f"Error al recrear chatbot para sesión {session_id}: {e}")
                return jsonify({
                    "status": "error",
                    "message": f"Error al procesar la consulta: {str(e)}"
                }), 500
        
        # Procesar la consulta con streaming habilitado
        result = bot.process_query(query, session_id=session_id, stream=True)
        
        # Verificar si hay un error en el resultado
        if result.get("status") == "error":
            return jsonify({
                "status": "error",
                "message": result.get("message", "Error desconocido al procesar la consulta")
            }), 500
        
        # Obtener el generador de streaming
        streaming_generator = result.get("response")
        update_history_callback = result.get("update_history")
        
        # Configurar generador para el cliente
        def generate():
            try:
                # Enviar cada chunk al cliente en formato SSE (Server-Sent Events)
                for chunk in streaming_generator:
                    # Solo enviar chunks no vacíos
                    if chunk and isinstance(chunk, str):
                        # Verificar si contiene un message_id oculto y eliminarlo
                        if "<hidden_message_id>" in chunk:
                            import re
                            match = re.search(r'<hidden_message_id>(.*?)</hidden_message_id>', chunk)
                            if match:
                                chunk = chunk.replace(match.group(0), "")
                        
                        # Solo enviar si hay contenido después de limpiar
                        if chunk.strip():
                            yield f"data: {chunk.strip()}\n\n"
                
                # Señalizar el final del streaming
                yield "data: [DONE]\n\n"
                
                # Actualizar el historial después de completar el streaming
                if update_history_callback:
                    try:
                        update_history_callback()
                    except Exception as e:
                        logger.error(f"Error al actualizar historial después de streaming: {e}")
            
            except Exception as e:
                error_msg = f"Error en streaming: {str(e)}"
                logger.error(error_msg)
                yield f"data: {error_msg}\n\n"
                yield "data: [DONE]\n\n"
        
        # Configurar respuesta SSE
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
    
    except Exception as e:
        logger.error(f"Error al procesar consulta streaming: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Ruta para obtener el historial de una sesión
@app.route('/sessions/<session_id>/history', methods=['GET'])
def get_session_history(session_id):
    """Obtiene el historial de interacciones de una sesión."""
    try:
        # Verificar que la sesión existe
        if not session_manager.get_session(session_id):
            return jsonify({
                "status": "error",
                "message": "Sesión no encontrada"
            }), 404
        
        # Seleccionar el chatbot apropiado o crear uno temporal para acceder al historial
        if session_id in active_chatbots:
            bot = active_chatbots[session_id]
        else:
            # Obtener datos de la sesión
            session_data = session_manager.get_session(session_id)
            
            # Crear un chatbot temporal con la sesión existente
            bot = RagChatbot(session_id=session_id)
        
        # Obtener historial
        history = bot.get_session_history(session_id)
        
        # Formatear historial para la respuesta
        formatted_history = []
        for item in history:
            formatted_history.append({
                "timestamp": item.get("timestamp", 0),
                "query": item.get("query", ""),
                "response": item.get("response", "")
            })
        
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "history_count": len(formatted_history),
            "history": formatted_history
        })
    
    except Exception as e:
        logger.error(f"Error al obtener historial: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Ruta para eliminar una sesión
@app.route('/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Elimina una sesión."""
    try:
        # Verificar que la sesión existe
        if not session_manager.get_session(session_id):
            return jsonify({
                "status": "error",
                "message": "Sesión no encontrada"
            }), 404
        
        # Cerrar y eliminar el chatbot específico si existe
        if session_id in active_chatbots:
            try:
                active_chatbots[session_id].close()
            except Exception as e:
                logger.warning(f"Error al cerrar chatbot de sesión {session_id}: {e}")
            
            del active_chatbots[session_id]
        
        # Eliminar la sesión del SessionManager
        session_manager.delete_session(session_id)
        
        return jsonify({
            "status": "success",
            "message": "Sesión eliminada correctamente"
        })
    
    except Exception as e:
        logger.error(f"Error al eliminar sesión: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def start_server(host='0.0.0.0', port=5000, debug=False):
    """
    Inicia el servidor web.
    
    Args:
        host: Dirección IP donde escuchar (por defecto: todas)
        port: Puerto donde escuchar
        debug: Activar modo de depuración
    """
    # Inicializar el chatbot global por defecto
    # Esto ya no es necesario, ya que cada sesión utilizará su propio chatbot
    # initialize_chatbot()
    
    logger.info(f"Iniciando servidor en {host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)

# Si se ejecuta como script independiente
if __name__ == '__main__':
    start_server(debug=True) 