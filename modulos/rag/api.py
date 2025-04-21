from flask import Flask, request, jsonify, Response, send_from_directory
import os
import uuid
import json
import gc
import psutil
import threading
import time
from typing import Dict, List, Optional, Any, Generator, Deque
import sys
from datetime import datetime
from collections import deque, defaultdict

# Importaciones de módulos propios
from modulos.rag.app import RagApp
from modulos.session_manager.session_manager import SessionManager
from config import Config

# Obtener la ruta absoluta del directorio estático
static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app = Flask(__name__, static_folder=static_folder, static_url_path='')
config = Config()

# Instancia del gestor de sesiones
session_manager = SessionManager()

# Ya no necesitamos mantener un seguimiento manual de las sesiones
# ni definir parámetros de configuración que ya están en SessionManager
# Las siguientes variables se conservan para compatibilidad en caso de que haya
# referencias en otras partes del código
SESSION_TIMEOUT = session_manager.SESSION_TIMEOUT
MAX_SESSIONS = session_manager.MAX_SESSIONS
MAX_CONTEXTS_PER_SESSION = session_manager.MAX_CONTEXTS_PER_SESSION
CLEANUP_INTERVAL = session_manager.CLEANUP_INTERVAL

# Diccionario para almacenar instancias RagApp temporales por sesión
# Este diccionario no almacena datos de sesión, solo las instancias RagApp
# Key: session_id, Value: RagApp instance
ragapp_instances: Dict[str, RagApp] = {}

# Lock para operaciones de acceso a las instancias RagApp
instances_lock = threading.RLock()

@app.route('/')
def index():
    """Sirve la página principal"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/app.js')
def serve_js():
    """Sirve el archivo JavaScript principal"""
    return send_from_directory(app.static_folder, 'app.js')

@app.route('/style.css')
def serve_css():
    """Sirve el archivo CSS principal"""
    return send_from_directory(app.static_folder, 'style.css')

@app.route('/api/databases', methods=['GET'])
def list_databases():
    """Lista las bases de datos disponibles utilizando el SessionManager"""
    try:
        # Utilizar el SessionManager para listar bases de datos
        db_dict = session_manager.list_available_databases()
        
        # Transformar el diccionario de bases de datos al formato esperado por el cliente
        database_list = []
        for db_name, metadata in db_dict.items():
            # Formatear tamaño
            size = metadata.get("size", 0)
            size_mb = size / (1024 * 1024) if size else 0
            size_formatted = f"{size_mb:.2f} MB"
            
            # Formatear fecha de creación
            created_at = metadata.get("created_at", 0)
            if created_at > 0:
                created_formatted = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
            else:
                created_formatted = "Desconocida"
            
            # Crear entrada con información detallada
            db_entry = {
                "id": db_name,
                "name": metadata.get("name", db_name),
                "display_name": f"{metadata.get('name', db_name)} ({metadata.get('db_type', 'unknown')}, {size_formatted})",
                "type": metadata.get("db_type", "unknown"),
                "size": size,
                "size_formatted": size_formatted,
                "created_at": created_at,
                "created_formatted": created_formatted,
                "has_metadata": bool(metadata.get("metadata", {}))
            }
            
            database_list.append(db_entry)
        
        app.logger.info(f"Bases de datos encontradas: {len(database_list)}")
        
        # Verificar si hay bases de datos
        if not database_list:
            app.logger.warning("No se encontraron bases de datos")
            return jsonify({
                "status": "success",
                "message": "No hay bases de datos disponibles",
                "databases": []
            })
        
        # Devolver sólo los nombres para mantener compatibilidad con el cliente actual
        database_names = [db.get("id") for db in database_list]
        
        return jsonify({
            "status": "success",
            "databases": database_names,
            "database_details": database_list
        })
    except Exception as e:
        app.logger.error(f"Error al listar bases de datos: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "databases": []
        }), 500

@app.route('/api/sessions', methods=['POST'])
def create_session():
    """Crea una nueva sesión para la base de datos especificada"""
    try:
        # Verificar si hay demasiadas sesiones activas
        if len(session_manager.get_all_sessions()) >= MAX_SESSIONS:
            # Intentar limpiar sesiones inactivas primero
            session_manager.clean_expired_sessions()
            
            # Si todavía hay demasiadas sesiones después de la limpieza
            if len(session_manager.get_all_sessions()) >= MAX_SESSIONS:
                return jsonify({
                    'status': 'error',
                    'message': f'Se ha alcanzado el límite máximo de sesiones activas ({MAX_SESSIONS}). Por favor, intente más tarde.'
                }), 429  # Too Many Requests
        
        data = request.json
        database_name = data.get('database_name')
        
        if not database_name:
            return jsonify({
                'status': 'error',
                'message': 'No se especificó nombre de base de datos'
            }), 400
        
        # Parámetros para la creación de la sesión
        ai_client = data.get('ai_client', config.get_ai_client_config().get('type', 'openai'))
        streaming = data.get('streaming', True)
        
        app.logger.info(f"Creando sesión con base de datos: {database_name}")
        
        # Crear sesión en el SessionManager
        session_id = session_manager.create_session(metadata={
            "database_name": database_name,
            "ai_client": ai_client,
            "streaming": streaming,
            "created_at": time.time(),
            "api_created": True
        })
        
        # Inicializar la aplicación RAG y guardar la instancia
        try:
            with instances_lock:
                rag_app = RagApp(
                    database_name=database_name,
                    ai_client=ai_client,
                    streaming=streaming,
                    session_id=session_id
                )
                
                # Almacenar la instancia RagApp
                ragapp_instances[session_id] = rag_app
            
            # Obtener información de la sesión del SessionManager
            session_data = session_manager.get_session(session_id)
            
            # Obtener información detallada de la base de datos
            db_info = {
                "id": database_name,
                "type": rag_app.db_type,
                "path": rag_app.db_path,
                "session": rag_app.session
            }
            
            app.logger.info(f"Sesión creada correctamente: {session_id} con base de datos {database_name}")
            
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'database_name': database_name,
                'db_type': rag_app.db_type,
                'db_info': db_info,
                'session_data': session_data
            })
        except Exception as e:
            # Si falla, eliminar la sesión creada
            session_manager.delete_session(session_id)
            app.logger.error(f"Error al inicializar RagApp: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f"Error al inicializar la sesión: {str(e)}"
            }), 500
    
    except Exception as e:
        app.logger.error(f"Error al crear sesión: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/query', methods=['POST'])
def process_query():
    """Procesa una consulta y devuelve la respuesta (con streaming opcional)"""
    try:
        data = request.json
        session_id = data.get('session_id')
        query = data.get('query')
        stream = data.get('stream', True)  # Por defecto, usar streaming
        
        if not session_id or not query:
            return jsonify({
                'status': 'error',
                'message': 'Faltan parámetros requeridos (session_id, query)'
            }), 400
        
        # Verificar si la sesión existe
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return jsonify({
                'status': 'error',
                'message': 'Sesión no válida o expirada'
            }), 404
        
        # Obtener la instancia RagApp para esta sesión
        rag_app = None
        with instances_lock:
            # Si ya tenemos una instancia, usarla
            if session_id in ragapp_instances:
                rag_app = ragapp_instances[session_id]
                
                # Verificar si la instancia está cerrada o inválida
                if getattr(rag_app, 'closed', True):
                    # Eliminar la instancia cerrada
                    del ragapp_instances[session_id]
                    rag_app = None
        
        # Si no tenemos una instancia válida, crear una nueva
        if rag_app is None:
            # Obtener datos de la sesión para inicializar RagApp
            database_name = session_data.get('database_name')
            ai_client = session_data.get('ai_client')
            streaming = session_data.get('streaming', True)
            
            if not database_name:
                return jsonify({
                    'status': 'error',
                    'message': 'Datos de sesión incompletos (falta database_name)'
                }), 400
            
            try:
                # Crear una nueva instancia con los datos de la sesión
                rag_app = RagApp(
                    database_name=database_name,
                    ai_client=ai_client,
                    streaming=streaming,
                    session_id=session_id
                )
                
                # Guardar la nueva instancia
                with instances_lock:
                    ragapp_instances[session_id] = rag_app
                    
                app.logger.info(f"Nueva instancia RagApp creada para sesión: {session_id}")
            except Exception as e:
                app.logger.error(f"Error al recrear instancia RagApp: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f"Error al procesar consulta: {str(e)}"
                }), 500
        
        # Procesar la consulta con la instancia RagApp
        if stream:
            # Modo streaming: devolver un generador de respuesta
            def generate_streaming_response():
                try:
                    # Iniciar timestamp para medir tiempo de respuesta
                    start_time = time.time()
                    
                    # Usar el método de consulta de RagApp
                    generator = rag_app.query(query)
                    if not hasattr(generator, '__next__') and not hasattr(generator, 'send'):
                        # No es un generador, devolver como texto simple
                        yield f"data: {generator}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    
                    # Procesar flujo de respuesta
                    message_id = None
                    for chunk in generator:
                        # Extraer message_id si está presente al final del chunk
                        if "<hidden_message_id>" in chunk:
                            # Extraer ID y eliminar del chunk
                            import re
                            match = re.search(r'<hidden_message_id>(.*?)</hidden_message_id>', chunk)
                            if match:
                                message_id = match.group(1)
                                chunk = chunk.replace(match.group(0), "")
                        
                        # Enviar chunk al cliente
                        if chunk:
                            yield f"data: {chunk}\n\n"
                    
                    # Enviar señal de finalización
                    yield "data: [DONE]\n\n"
                    
                    # Registrar tiempo de respuesta
                    response_time = time.time() - start_time
                    app.logger.info(f"Consulta streaming completada en {response_time:.2f}s (session_id={session_id})")
                    
                    # Si tenemos un message_id, actualizar metadatos de la sesión
                    if message_id:
                        session_manager.update_session_metadata(session_id, {
                            "last_query_time": start_time,
                            "last_response_time": response_time,
                            "last_message_id": message_id
                        })
                
                except Exception as e:
                    # En caso de error, enviar mensaje de error y cerrar streaming
                    error_msg = f"Error durante el procesamiento: {str(e)}"
                    app.logger.error(error_msg)
                    yield f"data: {error_msg}\n\n"
                    yield "data: [DONE]\n\n"
            
            # Configurar respuesta streaming
            return Response(
                generate_streaming_response(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            # Modo no streaming: devolver respuesta completa
            try:
                start_time = time.time()
                
                # Obtener respuesta completa y contexto
                response, context = rag_app.query(query, return_context=True)
                
                # Extraer message_id si está presente
                message_id = None
                if "<hidden_message_id>" in response:
                    import re
                    match = re.search(r'<hidden_message_id>(.*?)</hidden_message_id>', response)
                    if match:
                        message_id = match.group(1)
                        response = response.replace(match.group(0), "")
                
                # Registrar tiempo de respuesta
                response_time = time.time() - start_time
                app.logger.info(f"Consulta completada en {response_time:.2f}s (session_id={session_id})")
                
                # Actualizar metadatos de la sesión
                if message_id:
                    session_manager.update_session_metadata(session_id, {
                        "last_query_time": start_time,
                        "last_response_time": response_time,
                        "last_message_id": message_id
                    })
                
                # Preparar respuesta con contexto
                context_data = []
                for item in context:
                    # Limpiar el contexto para la respuesta (eliminar binarios grandes)
                    cleaned_item = {k: v for k, v in item.items() if k != 'embedding'}
                    context_data.append(cleaned_item)
                
                return jsonify({
                    'status': 'success',
                    'response': response,
                    'context': context_data,
                    'message_id': message_id,
                    'response_time': response_time
                })
            
            except Exception as e:
                app.logger.error(f"Error al procesar consulta: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f"Error al procesar consulta: {str(e)}"
                }), 500
    
    except Exception as e:
        app.logger.error(f"Error general en proceso de consulta: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/context/<session_id>/latest', methods=['GET'])
def get_latest_context(session_id):
    """Obtiene el contexto más reciente para una sesión"""
    try:
        # Verificar que la sesión exista
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return jsonify({
                'status': 'error',
                'message': 'Sesión no encontrada'
            }), 404
        
        # Obtener el último message_id de la sesión
        message_id = session_data.get("last_message_id")
        if not message_id:
            return jsonify({
                'status': 'success',
                'message': 'No hay consultas recientes en esta sesión',
                'context': []
            })
        
        # Obtener el contexto para este mensaje
        context_data = session_manager.get_message_context(session_id, message_id)
        if not context_data:
            return jsonify({
                'status': 'success',
                'message': 'No se encontró contexto para la última consulta',
                'context': []
            })
        
        # Limpiar datos binarios del contexto
        cleaned_context = []
        for item in context_data:
            # Eliminar campos binarios grandes
            cleaned_item = {k: v for k, v in item.items() if k != 'embedding'}
            cleaned_context.append(cleaned_item)
        
        return jsonify({
            'status': 'success',
            'message_id': message_id,
            'context': cleaned_context
        })
    
    except Exception as e:
        app.logger.error(f"Error al obtener contexto: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Elimina una sesión y libera sus recursos"""
    try:
        # Verificar que la sesión exista
        if not session_manager.get_session(session_id):
            return jsonify({
                'status': 'error',
                'message': 'Sesión no encontrada'
            }), 404
        
        # Cerrar la instancia RagApp si existe
        with instances_lock:
            if session_id in ragapp_instances:
                try:
                    rag_app = ragapp_instances[session_id]
                    rag_app.close()
                except Exception as e:
                    app.logger.warning(f"Error al cerrar instancia RagApp: {str(e)}")
                
                # Eliminar la instancia
                del ragapp_instances[session_id]
        
        # Eliminar la sesión del SessionManager
        session_manager.delete_session(session_id)
        
        # Forzar recolección de basura
        gc.collect()
        
        app.logger.info(f"Sesión eliminada correctamente: {session_id}")
        
        return jsonify({
            'status': 'success',
            'message': 'Sesión eliminada correctamente'
        })
    
    except Exception as e:
        app.logger.error(f"Error al eliminar sesión: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/diagnostics', methods=['GET'])
def diagnostics():
    """Obtiene información de diagnóstico del sistema"""
    try:
        # Obtener información de recursos del SessionManager
        resource_usage = session_manager.get_resource_usage()
        
        # Obtener lista de sesiones
        sessions = session_manager.list_sessions()
        
        # Crear respuesta con información detallada
        response = {
            'status': 'success',
            'timestamp': time.time(),
            'memory': {
                'total': resource_usage.get('memory_total', 0),
                'available': resource_usage.get('memory_available', 0),
                'percent': resource_usage.get('memory_percent', 0),
                'process': resource_usage.get('memory_process', 0),
                'process_percent': resource_usage.get('memory_process_percent', 0)
            },
            'cpu': {
                'percent': resource_usage.get('cpu_percent', 0),
                'cores': resource_usage.get('cpu_count', 0)
            },
            'sessions': {
                'active': len(sessions),
                'limit': MAX_SESSIONS,
                'ragapp_instances': len(ragapp_instances)
            },
            'system': {
                'platform': sys.platform,
                'python_version': sys.version.split()[0],
                'pid': os.getpid(),
                'uptime': resource_usage.get('uptime', 0)
            },
            'session_list': sessions
        }
        
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f"Error al obtener diagnósticos: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/message-context/<message_id>', methods=['GET'])
def get_message_context(message_id):
    """Obtiene el contexto utilizado para un mensaje específico"""
    try:
        app.logger.info(f"Solicitando contexto para mensaje: {message_id}")
        
        session_id = request.args.get('session_id')
        if not session_id:
            # Intentar buscar el message_id en todas las sesiones activas
            sessions = session_manager.get_all_sessions()
            found_session = None
            
            app.logger.info(f"No se proporcionó session_id, buscando en {len(sessions)} sesiones activas")
            
            for sid, session_data in sessions.items():
                # Comprobar si este message_id está en la sesión
                if session_data.get('last_message_id') == message_id:
                    found_session = sid
                    app.logger.info(f"Mensaje encontrado en sesión: {sid}")
                    break
            
            if not found_session:
                app.logger.warning(f"No se encontró sesión para el mensaje: {message_id}")
                return jsonify({
                    'status': 'error',
                    'message': 'Se requiere session_id o no se encontró el mensaje en ninguna sesión activa'
                }), 400
            
            session_id = found_session
        
        app.logger.info(f"Obteniendo contexto para mensaje {message_id} en sesión {session_id}")
        
        # Obtener contexto del mensaje
        context_data = session_manager.get_message_context(session_id, message_id)
        
        if not context_data:
            app.logger.warning(f"No se encontró contexto para el mensaje {message_id}")
            return jsonify({
                'status': 'error',
                'message': 'No se encontró contexto para el mensaje especificado'
            }), 404
        
        app.logger.info(f"Contexto encontrado para mensaje {message_id}: {len(context_data)} fragmentos")
        
        # Limpiar datos binarios del contexto
        cleaned_context = []
        for item in context_data:
            # Eliminar campos binarios grandes
            cleaned_item = {k: v for k, v in item.items() if k != 'embedding'}
            # Añadir campos necesarios si faltan
            if 'relevance' not in cleaned_item:
                cleaned_item['relevance'] = 100
            if 'header' not in cleaned_item and 'title' in cleaned_item:
                cleaned_item['header'] = cleaned_item['title']
            cleaned_context.append(cleaned_item)
        
        # Obtener datos de la sesión
        session_data = session_manager.get_session(session_id)
        session_config = session_manager.get_session_config(session_id)
        
        # Obtener información de la consulta si está disponible
        query_info = session_config.get('last_query', {}) if session_config else {}
        
        return jsonify({
            'status': 'success',
            'message_id': message_id,
            'session_id': session_id,
            'query': query_info.get('text', 'Consulta no disponible'),
            'timestamp': query_info.get('timestamp', 0),
            'context_count': len(cleaned_context),
            'session_info': {
                'database_name': session_data.get('database_name', 'desconocida'),
                'created_at': session_data.get('created_at', 0)
            },
            'context': cleaned_context
        })
    
    except Exception as e:
        app.logger.error(f"Error al obtener contexto de mensaje {message_id}: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/cleanup', methods=['POST'])
def trigger_cleanup():
    """Desencadena una limpieza manual de recursos"""
    try:
        # Forzar limpieza en SessionManager
        aggressive = request.json.get('aggressive', False) if request.is_json else False
        result = session_manager.clean_expired_sessions(aggressive=aggressive)
        
        # Limpiar instancias RagApp huérfanas
        with instances_lock:
            sessions = session_manager.get_all_sessions()
            orphaned = []
            
            # Identificar instancias sin sesión válida
            for session_id in list(ragapp_instances.keys()):
                if session_id not in sessions:
                    orphaned.append(session_id)
            
            # Cerrar y eliminar instancias huérfanas
            for session_id in orphaned:
                try:
                    rag_app = ragapp_instances[session_id]
                    rag_app.close()
                except Exception:
                    pass
                
                del ragapp_instances[session_id]
        
        # Forzar recolección de basura
        gc.collect()
        
        return jsonify({
            'status': 'success',
            'message': f'Limpieza completada: {result}',
            'orphaned_instances': len(orphaned) if 'orphaned' in locals() else 0,
            'remaining_instances': len(ragapp_instances)
        })
    
    except Exception as e:
        app.logger.error(f"Error durante la limpieza manual: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def run_api(host='0.0.0.0', port=5000, debug=False):
    """Inicia el servidor API"""
    app.logger.info(f"Iniciando API RAG en {host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)

# Si se ejecuta directamente este archivo
if __name__ == '__main__':
    # Configurar logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Iniciar API
    run_api(debug=True) 