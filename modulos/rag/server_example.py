"""
Ejemplo de servidor web para el chatbot RAG con respuestas en tiempo real.

Este módulo proporciona un ejemplo práctico de cómo integrar el chatbot RAG
en un servidor web usando Flask y respuestas en streaming.
"""

import logging
import time
import json
from typing import Dict, Any, List

from flask import Flask, request, jsonify, Response, stream_with_context
import threading

from modulos.rag.chatbot import RagChatbot
from modulos.utils.logging_utils import setup_logging

# Configurar logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear la aplicación Flask
app = Flask(__name__)

# Chatbot global - En producción se recomienda una inicialización más controlada
chatbot = None

# Diccionario para almacenar sesiones activas (para el ejemplo)
active_chatbots = {}

def initialize_chatbot(database_index=None):
    """
    Inicializa el chatbot RAG con la configuración por defecto o específica.
    
    Args:
        database_index: Índice de la base de datos a utilizar (opcional)
    """
    global chatbot
    
    try:
        # Inicializar el chatbot con streaming habilitado
        chatbot = RagChatbot(
            database_index=database_index,
            streaming=True,
            max_chunks=5
        )
        logger.info("Chatbot RAG inicializado correctamente")
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

# Ruta para verificar el estado del servidor
@app.route('/health', methods=['GET'])
def health_check():
    """Verifica el estado del servidor."""
    global chatbot
    
    if chatbot is None:
        return jsonify({
            "status": "error",
            "message": "Chatbot no inicializado"
        }), 500
    
    # Estadísticas básicas
    stats = {
        "status": "ok",
        "active_sessions": chatbot.get_active_sessions_count(),
        "timestamp": time.time()
    }
    
    return jsonify(stats)

# Ruta para listar bases de datos disponibles
@app.route('/databases', methods=['GET'])
def list_databases():
    """Lista las bases de datos disponibles."""
    global chatbot
    
    if chatbot is None:
        if not initialize_chatbot():
            return jsonify({
                "status": "error",
                "message": "No se pudo inicializar el chatbot"
            }), 500
    
    try:
        databases = chatbot.rag_app.get_available_databases()
        return jsonify({
            "status": "ok",
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
    global chatbot
    
    if chatbot is None:
        if not initialize_chatbot():
            return jsonify({
                "status": "error",
                "message": "No se pudo inicializar el chatbot"
            }), 500
    
    try:
        # Obtener parámetros de la petición
        data = request.get_json() or {}
        database_index = data.get('database_index')
        
        # Si se especifica un índice de base de datos diferente,
        # crear un chatbot específico para esta sesión
        if database_index is not None:
            try:
                session_chatbot = RagChatbot(
                    database_index=database_index,
                    streaming=True
                )
                session_id = session_chatbot.create_session()
                active_chatbots[session_id] = session_chatbot
            except Exception as e:
                logger.error(f"Error al crear chatbot con base de datos {database_index}: {e}")
                return jsonify({
                    "status": "error",
                    "message": f"Error al inicializar con la base de datos seleccionada: {str(e)}"
                }), 400
        else:
            # Usar el chatbot global
            session_id = chatbot.create_session()
        
        # Devolver el ID de sesión
        return jsonify({
            "status": "ok",
            "session_id": session_id,
            "message": "Sesión creada correctamente",
            "custom_database": database_index is not None
        })
        
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
    global chatbot
    
    if chatbot is None:
        if not initialize_chatbot():
            return jsonify({
                "status": "error",
                "message": "No se pudo inicializar el chatbot"
            }), 500
    
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
        
        # Seleccionar el chatbot apropiado
        bot = active_chatbots.get(session_id, chatbot) if session_id else chatbot
        
        # Procesar la consulta (sin streaming)
        result = bot.process_query(query, session_id=session_id, stream=False)
        
        # Extraer respuesta y contexto
        if isinstance(result['response'], str):
            context_info = bot.extract_context_from_response(result['response'])
            
            return jsonify({
                "status": "ok",
                "response": context_info['response_text'],
                "context": context_info['context_text'],
                "session_id": result['session_id'],
                "timestamp": result['timestamp']
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Error inesperado en la respuesta"
            }), 500
        
    except Exception as e:
        logger.error(f"Error al procesar consulta: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Ruta para procesar consultas en streaming
@app.route('/query/stream', methods=['POST'])
def process_query_stream():
    """Procesa una consulta y devuelve una respuesta en streaming."""
    global chatbot
    
    if chatbot is None:
        if not initialize_chatbot():
            return jsonify({
                "status": "error",
                "message": "No se pudo inicializar el chatbot"
            }), 500
    
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
        
        # Seleccionar el chatbot apropiado
        bot = active_chatbots.get(session_id, chatbot) if session_id else chatbot
        
        # Procesar la consulta (con streaming)
        result = bot.process_query(query, session_id=session_id, stream=True)
        
        if not result.get('is_streaming', False):
            # Si por alguna razón no obtuvimos streaming, devolver respuesta normal
            return jsonify({
                "status": "ok",
                "response": result['response'],
                "session_id": result['session_id'],
                "timestamp": result['timestamp']
            })
        
        # Función para generar la respuesta en streaming
        def generate():
            try:
                # Primero enviamos un objeto JSON con los metadatos
                metadata = json.dumps({
                    "status": "ok",
                    "is_streaming": True,
                    "session_id": result['session_id'],
                    "timestamp": result['timestamp']
                })
                yield f"data: {metadata}\n\n"
                
                # Luego enviamos los fragmentos de la respuesta
                for chunk in result['response']:
                    if chunk:
                        # Formato Server-Sent Events (SSE)
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                # Al finalizar, actualizamos el historial
                result['update_history']()
                
                # Señalizar el final del streaming
                yield f"data: {json.dumps({'end': True})}\n\n"
                
            except Exception as e:
                logger.error(f"Error durante streaming: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        # Crear respuesta en streaming con formato SSE
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
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
    """Obtiene el historial de una sesión."""
    global chatbot
    
    if chatbot is None:
        if not initialize_chatbot():
            return jsonify({
                "status": "error",
                "message": "No se pudo inicializar el chatbot"
            }), 500
    
    try:
        # Seleccionar el chatbot apropiado
        bot = active_chatbots.get(session_id, chatbot)
        
        # Obtener historial
        history = bot.get_session_history(session_id)
        
        return jsonify({
            "status": "ok",
            "session_id": session_id,
            "history": history
        })
        
    except Exception as e:
        logger.error(f"Error al obtener historial: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Inicialización del servidor
def start_server(host='0.0.0.0', port=5000, debug=False):
    """
    Inicializa y arranca el servidor.
    
    Args:
        host: Host para escuchar
        port: Puerto para escuchar
        debug: Modo debug para Flask
    """
    # Inicializar el chatbot
    if not initialize_chatbot():
        logger.error("No se pudo inicializar el chatbot al arrancar el servidor")
    
    # Iniciar el servidor
    logger.info(f"Iniciando servidor en {host}:{port}")
    app.run(host=host, port=port, debug=debug)

# Ejemplo de uso directo
if __name__ == '__main__':
    start_server(port=5000, debug=True) 