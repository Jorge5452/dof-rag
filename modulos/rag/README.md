# RAG Module

This module implements a Retrieval Augmented Generation (RAG) system with a modular architecture designed for flexibility and extensibility. The RAG module is now fully integrated with the SessionManager for improved session handling, resource management, and context persistence.

## Core Components

- **app.py**: Main RAG application class that integrates databases, embeddings, and AI clients
- **api.py**: REST API implementation for accessing RAG functionality over HTTP
- **chatbot.py**: Specialized implementation for chatbot-style applications
- **server_example.py**: Example web server with streaming support

## Integration with SessionManager

The RAG module and SessionManager are tightly integrated to provide:

1. **Centralized Session Management**: All sessions are managed through the SessionManager singleton
2. **Database Registry**: Databases are registered and associated with sessions
3. **Context Storage**: Query contexts are stored and can be retrieved by message ID
4. **Resource Optimization**: Improved memory usage and automatic cleanup of inactive sessions
5. **Configuration Storage**: Session-specific configurations are persisted

## Usage

### Basic RAG Usage

```python
from modulos.rag.app import RagApp

# Create a RAG instance with a specific database
rag = RagApp(database_name="my_database", streaming=True)

# Process a query
response = rag.query("What is RAG?")
print(response)

# Close and release resources
rag.close()
```

### Chatbot Usage

```python
from modulos.rag.chatbot import RagChatbot

# Create a chatbot with a specific database
chatbot = RagChatbot(database_name="my_database", streaming=True)

# Process a query
result = chatbot.process_query("What is RAG?")

# If streaming is enabled
if result.get("streaming"):
    response_generator = result.get("response")
    for chunk in response_generator:
        print(chunk, end="", flush=True)
else:
    print(result.get("response"))

# Get the last message context
message_id = result.get("message_id")
context = chatbot.extract_context_from_response(result.get("response"))

# Close the chatbot
chatbot.close()
```

### API Server

To run the API server:

```bash
python -m modulos.rag.api
```

Or the example server:

```bash
python -m modulos.rag.server_example
```

## API Endpoints

The API server provides these main endpoints:

- `GET /api/databases`: Lists available databases
- `POST /api/sessions`: Creates a new session
- `POST /api/query`: Processes a query and returns a response
- `GET /api/context/<session_id>/latest`: Gets the context for the latest query
- `DELETE /api/sessions/<session_id>`: Deletes a session
- `GET /api/diagnostics`: Gets system diagnostics information

## Session Management

Sessions are managed through the SessionManager which:

1. Creates and tracks sessions
2. Stores query contexts by message ID
3. Cleans up expired sessions
4. Monitors resource usage
5. Provides a centralized configuration store

## Database Management

Databases are discovered and managed by the SessionManager which:

1. Scans for available databases
2. Reads database metadata
3. Associates databases with sessions
4. Provides database registry

## Extending the RAG Module

The module can be extended by:

1. Adding new database implementations in `modulos/databases/implementaciones/`
2. Adding new embedding models in `modulos/embeddings/implementaciones/`
3. Adding new AI clients in `modulos/clientes/implementaciones/`
4. Creating specialized UI implementations using the API

## Configuration

The RAG module uses the central configuration from `config.py` which loads settings from `config.yaml`. Key configuration sections include:

- `general`: General settings
- `databases`: Database configuration
- `embeddings`: Embedding model settings
- `ai_client`: AI client configuration (OpenAI, Gemini, Ollama, etc.)
- `sessions`: Session management settings 