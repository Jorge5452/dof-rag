# 🤖 Chat API - Documentación del Chatbot RAG

## Tabla de Contenidos

- [🚀 Características Principales](#-características-principales)
- [🏗️ Arquitectura del Chatbot](#️-arquitectura-del-chatbot)
- [⚙️ Configuración e Instalación](#️-configuración-e-instalación)
- [🔌 API REST - Endpoints](#-api-rest---endpoints)
- [🌐 Interfaz Web](#-interfaz-web)
- [⚡ Streaming y Sesiones](#-streaming-y-sesiones)
- [📝 Ejemplos de Uso](#-ejemplos-de-uso)
- [🔧 Personalización y Configuración Avanzada](#-personalización-y-configuración-avanzada)
- [🛠️ Solución de Problemas](#️-solución-de-problemas)

---

## 🚀 Características Principales

El sistema RAG incluye una **interfaz de chatbot completa** que combina una API REST robusta con una interfaz web moderna para proporcionar una experiencia de usuario fluida y profesional.

### **Funcionalidades Clave**
- ✅ **Chat en tiempo real** con respuestas streaming (Server-Sent Events)
- ✅ **Gestión de sesiones** de usuario con historial persistente
- ✅ **Selección dinámica** de bases de datos desde la interfaz
- ✅ **Múltiples temas visuales** (oscuro, claro, verde agua)
- ✅ **Exportación de conversaciones** en formato texto
- ✅ **API REST completa** para integración externa
- ✅ **Responsive design** para dispositivos móviles y desktop
- ✅ **Configuración en tiempo real** desde la interfaz web
- ✅ **Manejo automático de errores** y reconexión
- ✅ **Soporte multilingüe** (español por defecto)

---

## 🏗️ Arquitectura del Chatbot

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Browser   │    │   Flask Server  │    │   RAG System    │
│   -----------   │    │   -----------   │    │   -----------   │
│ - HTML/CSS/JS   │◄──►│ - API Endpoints │◄──►│ - RagChatbot    │
│ - EventSource   │    │ - CORS Support  │    │ - SessionMgr    │
│ - Dynamic UI    │    │ - Static Files  │    │ - Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                        ▲                        ▲
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │  Configuration  │
                        │  -----------    │
                        │ - config.yaml   │
                        │ - .env          │
                        │ - Sessions DB   │
                        └─────────────────┘
```

### **Componentes Principales**

#### **RagChatbot (modulos/rag/chatbot.py)**
- Clase principal que extiende `RagApp`
- Gestión de sesiones de usuario
- Procesamiento de respuestas streaming y no-streaming
- Manejo de historial de conversaciones
- Integración con `SessionManager`

#### **Flask API (modulos/rag/api.py)**
- Servidor web con endpoints REST
- Manejo de Server-Sent Events para streaming
- Gestión de instancias `RagChatbot` por sesión
- Servicio de archivos estáticos (HTML/CSS/JS)

#### **Interfaz Web (modulos/rag/static/)**
- `index.html`: Estructura principal de la aplicación
- `style.css`: Estilos y temas visuales
- `app.js`: Lógica JavaScript para interacción

---

## ⚙️ Configuración e Instalación

### **Prerequisitos**
- Python 3.8+
- Sistema RAG configurado y funcionando
- Al menos una base de datos con documentos procesados

### **Instalación Rápida**

1. **Verificar configuración del sistema RAG:**
```bash
python run.py --list-sessions
```

2. **Iniciar el servidor del chatbot:**
```bash
cd modulos/rag
python api.py
```

3. **Acceder a la interfaz web:**
```
http://localhost:5000
```

### **Configuración en config.yaml**

```yaml
# Configuración específica para el chatbot
ai_client:
  type: "gemini"  # o "openai", "ollama"
  general:
    stream: true  # Habilitar streaming
    temperature: 0.7
    max_tokens: 2048

# Configuración de sesiones
sessions:
  max_sessions: 50
  timeout: 604800  # 7 días
  cleanup_interval: 300  # 5 minutos
```

---

## 🔌 API REST - Endpoints

### **GET /databases**
Lista todas las bases de datos disponibles.

**Respuesta:**
```json
{
  "databases": [
    {
      "index": 0,
      "id": "session_20241201_120000",
      "name": "Documentos Técnicos",
      "created_at": "2024-12-01T12:00:00",
      "embedding_model": "modernbert",
      "db_type": "duckdb",
      "file_count": 15,
      "total_chunks": 1250
    }
  ]
}
```

### **POST /query**
Procesa una consulta del usuario.

**Parámetros:**
```json
{
  "query": "¿Qué es RAG?",
  "database_index": 0,
  "stream": true,
  "session_id": "user_session_123"
}
```

**Respuesta (no-streaming):**
```json
{
  "status": "success",
  "response": "RAG significa Recuperación Aumentada de Generación...",
  "session_id": "user_session_123",
  "message_id": "msg_1733072400",
  "response_time": 2.34
}
```

**Respuesta (streaming):**
```
Content-Type: text/event-stream

data: {"type": "start", "session_id": "user_session_123"}

data: {"type": "content", "content": "RAG significa"}

data: {"type": "content", "content": " Recuperación"}

data: {"type": "end", "session_id": "user_session_123"}
```

### **GET /health**
Verifica el estado del servidor.

**Respuesta:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600,
  "active_sessions": 5
}
```

---

## 🌐 Interfaz Web

### **Selección de Bases de Datos**
La interfaz permite seleccionar entre bases de datos disponibles:
- **Lista automática** de bases de datos al cargar la página
- **Información detallada** de cada base de datos (modelo, chunks, archivos)
- **Cambio dinámico** de base de datos sin recargar la página

### **Sistema de Temas**
Tres temas visuales disponibles:

#### **Tema Oscuro (por defecto)**
- Fondo negro/gris oscuro
- Texto blanco/gris claro
- Acentos en azul y verde

#### **Tema Claro**
- Fondo blanco/gris claro
- Texto negro/gris oscuro
- Acentos en azul y verde

#### **Tema Verde Agua**
- Fondo en tonos aqua/turquesa
- Contraste optimizado
- Estilo único y relajante

### **Funcionalidades de Chat**
- **Área de mensajes** con scroll automático
- **Campo de entrada** con contador de caracteres
- **Botones de acción** (enviar, limpiar, configuración, exportar)
- **Indicadores de estado** (escribiendo, cargando, error)
- **Panel de fuentes** expandible para mostrar contexto

---

## ⚡ Streaming y Sesiones

### **Respuestas Streaming**
El sistema utiliza **Server-Sent Events (SSE)** para proporcionar respuestas en tiempo real:

```javascript
// Código JavaScript para recibir streaming
const eventSource = new EventSource('/query_stream');

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  
  if (data.type === 'content') {
    appendToMessage(data.content);
  } else if (data.type === 'end') {
    finalizeMessage();
  }
};
```

### **Gestión de Sesiones**
- **Identificación única** por navegador/usuario
- **Historial persistente** de conversaciones
- **Limpieza automática** de sesiones expiradas
- **Metadata enriquecida** (timestamp, contexto usado, etc.)

**Estructura de sesión:**
```json
{
  "session_id": "chatbot_20241201_user123",
  "created_at": 1733072400,
  "last_activity": 1733072450,
  "history": [
    {
      "timestamp": 1733072420,
      "query": "¿Qué es RAG?",
      "response": "RAG significa..."
    }
  ],
  "database_used": "session_20241201_120000",
  "settings": {
    "theme": "dark",
    "show_sources": true
  }
}
```

---

## 📝 Ejemplos de Uso

### **Integración JavaScript**
```javascript
// Inicializar chatbot
class RagChatbot {
  constructor(baseUrl = 'http://localhost:5000') {
    this.baseUrl = baseUrl;
    this.sessionId = this.generateSessionId();
  }

  async sendQuery(query, databaseIndex = 0) {
    const response = await fetch(`${this.baseUrl}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: query,
        database_index: databaseIndex,
        session_id: this.sessionId,
        stream: false
      })
    });

    return await response.json();
  }

  setupStreaming(callback) {
    const eventSource = new EventSource(`${this.baseUrl}/query_stream`);
    
    eventSource.onmessage = function(event) {
      const data = JSON.parse(event.data);
      callback(data);
    };

    return eventSource;
  }
}

// Uso
const chatbot = new RagChatbot();
const response = await chatbot.sendQuery("¿Qué es RAG?");
console.log(response);
```

### **Integración Python**
```python
import requests
import json

class RagChatbotClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session_id = f"python_client_{int(time.time())}"
    
    def get_databases(self):
        """Obtener lista de bases de datos disponibles"""
        response = requests.get(f"{self.base_url}/databases")
        return response.json()
    
    def send_query(self, query, database_index=0, stream=False):
        """Enviar consulta al chatbot"""
        payload = {
            "query": query,
            "database_index": database_index,
            "session_id": self.session_id,
            "stream": stream
        }
        
        response = requests.post(
            f"{self.base_url}/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        return response.json()

# Uso
client = RagChatbotClient()
databases = client.get_databases()
print(f"Bases de datos disponibles: {len(databases['databases'])}")

response = client.send_query("¿Cuál es el tema principal?")
print(f"Respuesta: {response['response']}")
```

### **Uso con cURL**
```bash
# Listar bases de datos
curl -X GET http://localhost:5000/databases

# Enviar consulta
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Qué es RAG?",
    "database_index": 0,
    "session_id": "curl_session_123",
    "stream": false
  }'

# Verificar estado del servidor
curl -X GET http://localhost:5000/health
```

---

## 🔧 Personalización y Configuración Avanzada

### **Configuración de Modelos de IA**

**OpenAI:**
```yaml
ai_client:
  type: "openai"
  openai:
    model: "gpt-3.5-turbo"
    api_key_env: "OPENAI_API_KEY"
    temperature: 0.7
    max_tokens: 2048
```

**Gemini:**
```yaml
ai_client:
  type: "gemini"
  gemini:
    model: "gemini-2.0-flash"
    api_key_env: "GEMINI_API_KEY"
    temperature: 0.7
```

**Ollama (local):**
```yaml
ai_client:
  type: "ollama"
  ollama:
    model: "llama2"
    api_url: "http://localhost:11434"
    timeout: 60
```

### **Personalización de la Interfaz**

**Modificar temas CSS:**
```css
/* Tema personalizado en style.css */
:root[data-theme="custom"] {
  --bg-primary: #1a1a2e;
  --bg-secondary: #16213e;
  --text-primary: #eee;
  --accent-color: #0f4c75;
  --success-color: #3282b8;
}
```

**Configuración JavaScript:**
```javascript
// Personalizar configuración en app.js
const CONFIG = {
  themes: ['dark', 'light', 'aqua', 'custom'],
  autoSave: true,
  maxHistoryLength: 100,
  streamingEnabled: true,
  showTimestamps: true
};
```

### **Variables de Entorno**
```env
# .env file
FLASK_PORT=5000
FLASK_DEBUG=false
CORS_ORIGINS=http://localhost:3000,https://mydomain.com
MAX_CONTENT_LENGTH=16777216  # 16MB
SESSION_TIMEOUT=604800       # 7 días
```

---

## 🛠️ Solución de Problemas

### **Problemas Comunes**

#### **Error: "No databases available"**
```bash
# Verificar que hay bases de datos
python run.py --list-sessions

# Si no hay, ingestar documentos
python run.py --ingest --files documents/
```

#### **Error de conexión al servidor**
```bash
# Verificar que el servidor está ejecutándose
curl http://localhost:5000/health

# Iniciar servidor si no está activo
cd modulos/rag
python api.py
```

#### **Streaming no funciona**
1. Verificar que el navegador soporta EventSource
2. Comprobar configuración CORS
3. Revisar logs del servidor Flask

#### **Sesiones no se guardan**
1. Verificar permisos de escritura en directorio `sessions/`
2. Comprobar configuración de timeout en `config.yaml`
3. Revisar logs de SessionManager

### **Logs y Debugging**

**Habilitar debug mode:**
```yaml
# config.yaml
general:
  debug: true
  log_level: "DEBUG"
```

**Monitorear logs:**
```bash
# Logs del sistema RAG
tail -f logs/rag_system.log

# Logs del servidor Flask
cd modulos/rag
python api.py --debug
```

### **Optimización de Rendimiento**

**Para muchas sesiones concurrentes:**
```yaml
sessions:
  max_sessions: 100
  cleanup_interval: 60  # Limpieza más frecuente

resource_management:
  concurrency:
    io_workers: 8
    max_total_workers: 16
```

**Para respuestas más rápidas:**
```yaml
ai_client:
  general:
    max_tokens: 1024  # Respuestas más cortas
    temperature: 0.3  # Más determinística

processing:
  max_chunks_to_retrieve: 3  # Menos contexto
```

---

## 📋 API Reference Completa

| Endpoint | Método | Descripción | Parámetros |
|----------|--------|-------------|------------|
| `/` | GET | Interfaz web principal | - |
| `/databases` | GET | Lista bases de datos | - |
| `/query` | POST | Procesa consulta | `query`, `database_index`, `session_id`, `stream` |
| `/health` | GET | Estado del servidor | - |
| `/static/<path>` | GET | Archivos estáticos | `path` |

**Estados de respuesta HTTP:**
- `200`: Éxito
- `400`: Error en parámetros
- `404`: Recurso no encontrado
- `500`: Error interno del servidor

---

## 📞 Soporte

Para soporte adicional:
1. Revisar logs en `logs/rag_system.log`
2. Verificar configuración en `config.yaml`
3. Consultar documentación principal en `README.md`
4. Verificar estado de recursos con `python run.py --resource-status`

---

*Documentación actualizada para la versión 1.0 del sistema RAG Chat API*
