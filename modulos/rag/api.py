from flask import Flask, request, jsonify, Response, send_from_directory
import os
import uuid
import json
import glob
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

# Configuración de límites y timeouts
SESSION_TIMEOUT = 3600  # 1 hora de inactividad antes de limpiar una sesión
MAX_SESSIONS = 50  # Número máximo de sesiones activas simultáneas
MAX_CONTEXTS_PER_SESSION = 20  # Máximo número de contextos a guardar por sesión
CLEANUP_INTERVAL = 300  # Intervalo de limpieza automática (5 minutos)

# Diccionario para almacenar las sesiones activas
# Key: session_id, Value: tupla (RagApp instance, database_name, last_activity_timestamp)
active_sessions: Dict[str, tuple] = {}

# Últimos contextos por sesión para mostrar en la interfaz, usando deque para limitar automáticamente
last_contexts: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=MAX_CONTEXTS_PER_SESSION))

# Lock para operaciones de limpieza y acceso a las sesiones
sessions_lock = threading.RLock()

# Estado de limpieza en progreso
cleanup_in_progress = False

def cleanup_inactive_sessions():
    """Limpia las sesiones inactivas basado en el tiempo de inactividad"""
    global cleanup_in_progress
    
    if cleanup_in_progress:
        return
    
    cleanup_in_progress = True
    try:
        current_time = time.time()
        sessions_to_remove = []
        
        with sessions_lock:
            for session_id, (rag_app, db_name, last_activity) in active_sessions.items():
                if current_time - last_activity > SESSION_TIMEOUT:
                    sessions_to_remove.append(session_id)
            
            # Eliminar las sesiones inactivas
            for session_id in sessions_to_remove:
                # Limpiar recursos de RagApp (cierra conexiones a bases de datos)
                try:
                    rag_app = active_sessions[session_id][0]
                    if hasattr(rag_app, 'db') and rag_app.db:
                        rag_app.db.close()
                    
                    # Limpiar referencia al cliente AI y otros objetos grandes
                    rag_app.ai_client = None
                    rag_app.embedding_manager = None
                except Exception as e:
                    app.logger.error(f"Error al limpiar recursos de sesión {session_id}: {str(e)}")
                
                # Eliminar la sesión
                del active_sessions[session_id]
                
                # También eliminar los contextos asociados
                if session_id in last_contexts:
                    del last_contexts[session_id]
            
            if sessions_to_remove:
                app.logger.info(f"Limpieza automática: {len(sessions_to_remove)} sesiones inactivas eliminadas")
                # Forzar recolección de basura después de limpiar sesiones
                gc.collect()
    finally:
        cleanup_in_progress = False

def update_session_activity(session_id):
    """Actualiza el timestamp de última actividad de una sesión"""
    with sessions_lock:
        if session_id in active_sessions:
            rag_app, db_name, _ = active_sessions[session_id]
            active_sessions[session_id] = (rag_app, db_name, time.time())

# Iniciar un hilo para limpieza periódica
def start_cleanup_thread():
    def cleanup_worker():
        while True:
            try:
                time.sleep(CLEANUP_INTERVAL)
                cleanup_inactive_sessions()
            except Exception as e:
                app.logger.error(f"Error en hilo de limpieza: {str(e)}")
    
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    app.logger.info(f"Hilo de limpieza automática iniciado (intervalo: {CLEANUP_INTERVAL}s)")

# Iniciar el hilo de limpieza cuando se inicia la aplicación
start_cleanup_thread()

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
    """Lista las bases de datos disponibles buscando archivos en el directorio configurado"""
    try:
        # Utilizar el método estático de RagApp para listar bases de datos
        databases = RagApp.list_available_databases()
        
        # Transformar la lista de bases de datos al formato esperado por el cliente
        database_list = []
        for db in databases:
            # Formatear tamaño
            size_mb = db.get("size", 0) / (1024 * 1024)
            size_formatted = f"{size_mb:.2f} MB"
            
            # Formatear fecha de creación
            created_at = db.get("created_at", 0)
            if created_at > 0:
                from datetime import datetime
                created_formatted = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
            else:
                created_formatted = "Desconocida"
            
            # Crear entrada con información detallada
            db_entry = {
                "id": db.get("id"),
                "name": db.get("name"),
                "display_name": f"{db.get('name')} ({db.get('type')}, {size_formatted})",
                "type": db.get("type"),
                "size": db.get("size"),
                "size_formatted": size_formatted,
                "created_at": created_at,
                "created_formatted": created_formatted,
                "has_metadata": bool(db.get("metadata"))
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
        # Verificar si ya hay demasiadas sesiones activas
        with sessions_lock:
            if len(active_sessions) >= MAX_SESSIONS:
                # Intentar limpiar sesiones inactivas primero
                cleanup_inactive_sessions()
                
                # Si todavía hay demasiadas sesiones después de la limpieza
                if len(active_sessions) >= MAX_SESSIONS:
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
        
        # Inicializar la aplicación RAG con los parámetros dados
        try:
            rag_app = RagApp(
                database_name=database_name,
                ai_client=ai_client,
                streaming=streaming
            )
            
            # Generar un ID de sesión único
            session_id = str(uuid.uuid4())
            
            # Almacenar la sesión con timestamp de creación
            current_time = time.time()
            with sessions_lock:
                active_sessions[session_id] = (rag_app, database_name, current_time)
            
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
                'db_info': db_info
            })
        except Exception as e:
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
        
        # Verificar si la sesión existe y actualizar su actividad
        with sessions_lock:
            if session_id not in active_sessions:
                return jsonify({
                    'status': 'error',
                    'message': 'Sesión no válida o expirada'
                }), 404
            
            # Actualizar timestamp de actividad
            update_session_activity(session_id)
            
            # Obtener la instancia de RagApp para esta sesión
            rag_app, database_name, _ = active_sessions[session_id]
        
        # Generar un ID único para este mensaje para referencia futura
        message_id = str(uuid.uuid4())
        
        # Procesar la consulta y obtener respuesta con contexto
        if stream:
            # Para streaming, manejar de forma especial
            app.logger.info(f"Procesando consulta en modo streaming: {query[:30]}...")
            
            # Obtener el contexto por separado para almacenarlo
            try:
                # Obtener respuesta y contexto usando la nueva implementación
                dummy_response, context_data = rag_app.query(query, return_context=True)
                
                # Verificar si el contexto es válido
                if context_data and isinstance(context_data, list):
                    # Almacenar el contexto para esta consulta usando deque (limitado automáticamente)
                    with sessions_lock:
                        last_contexts[session_id].append({
                            'message_id': message_id,
                            'query': query,
                            'context': context_data,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    app.logger.info(f"Contexto almacenado para mensaje {message_id} de la sesión {session_id}: {len(context_data)} fragmentos")
                else:
                    app.logger.warning(f"Contexto vacío o inválido para mensaje {message_id}")
                    # Almacenar un contexto vacío para evitar errores al solicitar fuentes
                    with sessions_lock:
                        last_contexts[session_id].append({
                            'message_id': message_id,
                            'query': query,
                            'context': [],
                            'timestamp': datetime.now().isoformat()
                        })
            except Exception as context_err:
                app.logger.error(f"Error al obtener contexto: {str(context_err)}")
                # Almacenar un contexto vacío para evitar errores al solicitar fuentes
                with sessions_lock:
                    last_contexts[session_id].append({
                        'message_id': message_id,
                        'query': query,
                        'context': [],
                        'timestamp': datetime.now().isoformat()
                    })
            
            def generate_streaming_response():
                try:
                    # Llamar al método query solo una vez y verificar si devuelve un generador
                    result = rag_app.query(query, return_context=False)
                    
                    # Si el resultado es un generador (streaming)
                    if hasattr(result, '__iter__') and not isinstance(result, str):
                        for chunk in result:
                            if chunk:
                                # Enviar texto en texto plano sin formato JSON
                                yield chunk
                    else:
                        # Si no es un generador, enviar la respuesta completa de una vez
                        yield result
                        
                    # Al final del streaming, enviar el ID del mensaje
                    # Usamos un formato especial que el cliente puede detectar
                    yield f"\n\n<hidden_message_id>{message_id}</hidden_message_id>"
                        
                except Exception as e:
                    app.logger.error(f"Error en streaming: {str(e)}")
                    yield f"Error al procesar la consulta: {str(e)}"
                finally:
                    # Actualizar timestamp de actividad al finalizar
                    update_session_activity(session_id)
            
            # Devolver una respuesta de streaming con texto plano
            return Response(generate_streaming_response(), mimetype='text/plain')
        else:
            # Para respuestas no streaming, devolver respuesta completa con contexto
            result, context_data = rag_app.query(query, return_context=True)
            
            # Almacenar el contexto para esta consulta
            with sessions_lock:
                last_contexts[session_id].append({
                    'message_id': message_id,
                    'query': query,
                    'context': context_data,
                    'timestamp': datetime.now().isoformat()
                })
            
            return jsonify({
                'status': 'success',
                'message_id': message_id,
                'response': result,
                'context': context_data,
                'database': {
                    'name': database_name,
                    'type': rag_app.db_type,
                    'path': rag_app.db_path
                }
            })
    
    except Exception as e:
        app.logger.error(f"Error al procesar consulta: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/context/<session_id>/latest', methods=['GET'])
def get_latest_context(session_id):
    """Obtiene el contexto de la última consulta para una sesión específica"""
    # Actualizar timestamp de actividad
    update_session_activity(session_id)
    
    if session_id not in last_contexts or not last_contexts[session_id]:
        return jsonify({
            'status': 'error',
            'message': 'No hay contexto disponible para esta sesión'
        }), 404
    
    # Devolver el contexto más reciente
    latest_context = last_contexts[session_id][-1]
    
    return jsonify({
        'status': 'success',
        'query': latest_context['query'],
        'context': latest_context['context']
    })

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Elimina una sesión y libera sus recursos"""
    with sessions_lock:
        if session_id in active_sessions:
            try:
                # Liberar recursos de forma explícita
                rag_app, _, _ = active_sessions[session_id]
                
                # Cerrar conexión a la base de datos si existe
                if hasattr(rag_app, 'db') and rag_app.db is not None:
                    rag_app.db.close()
                
                # Eliminar referencias a objetos grandes
                rag_app.ai_client = None
                rag_app.embedding_manager = None
                rag_app.db = None
                
                # Eliminar la sesión del diccionario
                del active_sessions[session_id]
                
                # Eliminar contextos asociados
                if session_id in last_contexts:
                    del last_contexts[session_id]
                
                # Forzar recolección de basura
                gc.collect()
                
                return jsonify({
                    'status': 'success',
                    'message': 'Sesión eliminada correctamente'
                })
            except Exception as e:
                app.logger.error(f"Error al limpiar recursos de sesión {session_id}: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f'Error al eliminar la sesión: {str(e)}'
                }), 500
    
    return jsonify({
        'status': 'error',
        'message': 'Sesión no encontrada'
    }), 404

@app.route('/api/diagnostics', methods=['GET'])
def diagnostics():
    """Proporciona información de diagnóstico sobre el sistema y uso de recursos"""
    try:
        # Obtener información de la configuración
        database_config = config.get_database_config()
        db_type = database_config.get('type', 'sqlite')
        db_dir = database_config.get(db_type, {}).get('db_dir', 'modulos/databases/db')
        
        # Asegurar que la ruta es absoluta
        if not os.path.isabs(db_dir):
            db_dir = os.path.join(os.path.abspath(os.getcwd()), db_dir)
        
        # Verificar que el directorio existe
        dir_exists = os.path.exists(db_dir)
        
        # Listar archivos en el directorio si existe
        files_in_dir = []
        if dir_exists:
            for file in os.listdir(db_dir):
                file_path = os.path.join(db_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    files_in_dir.append({
                        'name': file,
                        'size': size,
                        'size_human': f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"
                    })
        
        # Contar bases de datos por tipo
        sqlite_count = len([f for f in files_in_dir if f['name'].endswith('.sqlite')])
        duckdb_count = len([f for f in files_in_dir if f['name'].endswith('.duckdb')])
        
        # Número de sesiones activas
        active_sessions_count = len(active_sessions)
        
        # Información sobre el uso de memoria del proceso
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage = {
            'rss': memory_info.rss,  # Resident Set Size
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms': memory_info.vms,  # Virtual Memory Size
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent(),
            'cpu_percent': process.cpu_percent(interval=0.1)
        }
        
        # Estadísticas del sistema
        system_stats = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'available_memory_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
        
        # Información de sesiones más detallada
        session_details = []
        with sessions_lock:
            for session_id, (rag_app, db_name, last_activity) in active_sessions.items():
                session_details.append({
                    'id': session_id,
                    'database': db_name,
                    'db_type': rag_app.db_type if hasattr(rag_app, 'db_type') else "unknown",
                    'last_activity': last_activity,
                    'inactive_seconds': time.time() - last_activity,
                    'contexts_stored': len(last_contexts.get(session_id, [])),
                    'ai_client_type': rag_app.ai_client_type if hasattr(rag_app, 'ai_client_type') else "unknown"
                })
        
        # Estado de la recolección de basura
        gc_stats = {
            'objects_tracked': len(gc.get_objects()),
            'garbage_count': len(gc.garbage),
            'collections': gc.get_count()
        }
        
        return jsonify({
            'status': 'success',
            'environment': {
                'python_version': sys.version,
                'working_directory': os.getcwd(),
                'timestamp': datetime.now().isoformat()
            },
            'database': {
                'configured_type': db_type,
                'directory': db_dir,
                'directory_exists': dir_exists,
                'sqlite_count': sqlite_count,
                'duckdb_count': duckdb_count,
                'total_files': len(files_in_dir),
                'files': files_in_dir
            },
            'sessions': {
                'active_count': active_sessions_count,
                'session_ids': list(active_sessions.keys()),
                'max_sessions': MAX_SESSIONS,
                'session_timeout': SESSION_TIMEOUT,
                'cleanup_interval': CLEANUP_INTERVAL,
                'max_contexts_per_session': MAX_CONTEXTS_PER_SESSION,
                'session_details': session_details
            },
            'resources': {
                'memory': memory_usage,
                'system': system_stats,
                'garbage_collection': gc_stats
            }
        })
    except Exception as e:
        app.logger.error(f"Error en diagnóstico: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/message-context/<message_id>', methods=['GET'])
def get_message_context(message_id):
    """Obtiene el contexto usado para generar una respuesta específica"""
    try:
        session_id = request.args.get('session_id')
        
        if not session_id:
            app.logger.warning(f"Solicitud de contexto sin session_id para message_id={message_id}")
            return jsonify({
                'status': 'error',
                'message': 'Se requiere el parámetro session_id'
            }), 400
        
        app.logger.info(f"Solicitando contexto para message_id={message_id}, session_id={session_id}")
        
        # Actualizar timestamp de actividad si la sesión existe
        with sessions_lock:
            if session_id in active_sessions:
                update_session_activity(session_id)
            else:
                app.logger.warning(f"Sesión {session_id} no encontrada, pero continuando con la búsqueda de contexto")
                # No devolvemos error aquí para permitir recuperar contexto de sesiones ya cerradas
        
        # Primera estrategia: buscar en last_contexts
        with sessions_lock:
            if session_id in last_contexts:
                # Buscar el mensaje específico en la sesión
                for message_data in last_contexts[session_id]:
                    if message_data.get('message_id') == message_id:
                        # Formatear el contexto para una visualización más amigable
                        formatted_context = []
                        
                        for idx, chunk in enumerate(message_data.get('context', [])):
                            # Solo incluir chunks que sean documentos reales, no instrucciones
                            if not any(exclude in chunk.get('header', '') for exclude in ['Instrucciones', 'Tono de conversación']):
                                # Calcular el porcentaje de relevancia basado en la similitud
                                similarity = chunk.get('similarity', 0)
                                relevance_percentage = int(similarity * 100) if similarity else 100
                                
                                # Obtener metadatos del documento
                                doc_name = chunk.get('document', '')
                                doc_display = doc_name
                                if '/' in doc_name:
                                    doc_display = doc_name.split('/')[-1]
                                
                                formatted_chunk = {
                                    'id': idx + 1,
                                    'text': chunk.get('text', '').strip(),
                                    'header': chunk.get('header', 'Sin título').strip(),
                                    'relevance': relevance_percentage,
                                    'page': chunk.get('page', 'N/A'),
                                    'document': doc_display,
                                    'document_full': doc_name
                                }
                                
                                formatted_context.append(formatted_chunk)
                        
                        # Ordenar por relevancia
                        formatted_context.sort(key=lambda x: x['relevance'], reverse=True)
                        
                        # Si no hay contexto después del filtrado, indicarlo amablemente
                        if not formatted_context:
                            app.logger.warning(f"No hay fragmentos válidos después del filtrado para message_id={message_id}")
                            return jsonify({
                                'status': 'success',
                                'message_id': message_id,
                                'query': message_data.get('query', ''),
                                'context': [],
                                'message': 'No hay fuentes disponibles para esta respuesta.',
                                'timestamp': message_data.get('timestamp')
                            })
                        
                        app.logger.info(f"Contexto recuperado de last_contexts para message_id={message_id}: {len(formatted_context)} fragmentos")
                        return jsonify({
                            'status': 'success',
                            'message_id': message_id,
                            'query': message_data.get('query', ''),
                            'context': formatted_context,
                            'timestamp': message_data.get('timestamp')
                        })
        
        # Segunda estrategia: intentar obtener el contexto del SessionManager
        try:
            session_manager = SessionManager()
            context_data = session_manager.get_message_context(session_id, message_id)
            
            if context_data:
                app.logger.info(f"Contexto recuperado del SessionManager para message_id={message_id}: {len(context_data)} fragmentos")
                
                # Formatear el contexto para una visualización más amigable
                formatted_context = []
                
                for idx, chunk in enumerate(context_data):
                    # Solo incluir chunks que sean documentos reales, no instrucciones
                    if not any(exclude in chunk.get('header', '') for exclude in ['Instrucciones', 'Tono de conversación']):
                        # Calcular el porcentaje de relevancia basado en la similitud
                        similarity = chunk.get('similarity', 0)
                        relevance_percentage = int(similarity * 100) if similarity else 100
                        
                        # Obtener metadatos del documento
                        doc_name = chunk.get('document', '')
                        doc_display = doc_name
                        if '/' in doc_name:
                            doc_display = doc_name.split('/')[-1]
                        
                        formatted_chunk = {
                            'id': idx + 1,
                            'text': chunk.get('text', '').strip(),
                            'header': chunk.get('header', 'Sin título').strip(),
                            'relevance': relevance_percentage,
                            'page': chunk.get('page', 'N/A'),
                            'document': doc_display,
                            'document_full': doc_name
                        }
                        
                        formatted_context.append(formatted_chunk)
                
                # Ordenar por relevancia
                formatted_context.sort(key=lambda x: x['relevance'], reverse=True)
                
                # Si no hay contexto después del filtrado, indicarlo amablemente
                if not formatted_context:
                    app.logger.warning(f"No hay fragmentos válidos después del filtrado para message_id={message_id}")
                    return jsonify({
                        'status': 'success',
                        'message_id': message_id,
                        'query': 'No disponible',
                        'context': [],
                        'message': 'No hay fuentes disponibles para esta respuesta.'
                    })
                
                return jsonify({
                    'status': 'success',
                    'message_id': message_id,
                    'query': 'No disponible',
                    'context': formatted_context
                })
        except Exception as e:
            app.logger.error(f"Error al recuperar contexto desde SessionManager: {str(e)}")
            # Continuamos al fallback final
        
        # Si llegamos aquí, no se encontró el contexto
        app.logger.warning(f"No se encontró contexto para message_id={message_id}, session_id={session_id}")
        return jsonify({
            'status': 'not_found',
            'message': 'No se encontraron fuentes para esta respuesta.',
            'context': []
        }), 404
    
    except Exception as e:
        app.logger.error(f"Error general al obtener contexto: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Error al procesar la solicitud: {str(e)}',
            'context': []
        }), 500

@app.route('/api/cleanup', methods=['POST'])
def trigger_cleanup():
    """Endpoint para forzar la limpieza de sesiones inactivas"""
    try:
        # Ejecutar limpieza
        with sessions_lock:
            old_count = len(active_sessions)
            cleanup_inactive_sessions()
            new_count = len(active_sessions)
            
        # Forzar recolección de basura
        gc.collect()
        
        return jsonify({
            'status': 'success',
            'message': f'Limpieza completada. Sesiones eliminadas: {old_count - new_count}',
            'sessions_before': old_count,
            'sessions_after': new_count
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error durante la limpieza: {str(e)}'
        }), 500

def run_api(host='0.0.0.0', port=5000, debug=False):
    """Ejecuta el servidor API Flask"""
    # Configurar opciones avanzadas
    app.config['JSON_SORT_KEYS'] = False  # Mantener orden de claves en JSON
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # No formatear JSON para ahorrar ancho de banda
    
    # Iniciar la aplicación
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_api(debug=True) 