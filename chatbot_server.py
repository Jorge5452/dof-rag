#!/usr/bin/env python
"""
Script para ejecutar el servidor de chatbot RAG.

Este script inicia un servidor web que proporciona una API REST para interactuar
con el sistema RAG a través de un chatbot con soporte para respuestas en streaming.

Uso:
    python chatbot_server.py [--port PUERTO] [--host HOST] [--debug]
"""

import argparse
import logging
import os
import sys
from modulos.rag.api import run_api
from modulos.utils.logging_utils import setup_logging, silence_verbose_loggers
from config import Config

config = Config()

def main():
    """Función principal que procesa los argumentos e inicia el servidor."""
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(
        description="Servidor de chatbot RAG con respuestas en tiempo real",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Argumentos para la configuración del servidor
    parser.add_argument("--port", type=int, default=5000, 
                        help="Puerto en el que escuchará el servidor")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="Host en el que escuchará el servidor")
    parser.add_argument("--debug", action="store_true", 
                        help="Activar modo de depuración")
    parser.add_argument("--no-streaming", action="store_true",
                        help="Desactivar streaming (respuestas en tiempo real)")
    
    # Procesar argumentos
    args = parser.parse_args()
    
    # Configurar logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    
    # Silenciar loggers ruidosos si no estamos en modo debug
    if not args.debug:
        silence_verbose_loggers()
    
    # Forzar modo streaming en la configuración
    ai_config = config.get_ai_client_config()
    
    # Sobrescribir la configuración para siempre usar streaming a menos que se especifique --no-streaming
    if not args.no_streaming:
        # Activar streaming si no se especifica lo contrario
        if isinstance(ai_config, dict) and "parameters" in ai_config:
            ai_config["parameters"]["stream"] = True
            config.update_config(["ai_client", "parameters", "stream"], True)
            logging.info("Modo streaming activado para el chatbot")
    
    # Obtener la ruta del directorio base del proyecto
    project_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(project_dir, "modulos", "rag", "static")
    
    # Verificar que existen los archivos necesarios
    if os.path.exists(static_dir):
        required_files = ["index.html", "app.js", "style.css"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(static_dir, f))]
        
        if missing_files:
            print(f"ADVERTENCIA: Faltan archivos en el directorio estático: {', '.join(missing_files)}")
            for f in missing_files:
                print(f"  - No se encuentra: {os.path.join(static_dir, f)}")
    else:
        print(f"ADVERTENCIA: No se encuentra el directorio estático en {static_dir}")
        print("Esto puede causar problemas al servir la interfaz web.")
    
    # Verificar disponibilidad de bases de datos
    try:
        from modulos.rag.app import RagApp
        databases = RagApp.list_available_databases()
        if databases:
            print(f"Bases de datos disponibles: {len(databases)}")
            for i, db in enumerate(databases[:5]):  # Mostrar solo las primeras 5 para no saturar la consola
                print(f"  - {db['name']} ({db['type']}, {os.path.getsize(db['path']) / 1024 / 1024:.2f} MB)")
            if len(databases) > 5:
                print(f"  ... y {len(databases) - 5} más")
        else:
            print("ADVERTENCIA: No se encontraron bases de datos disponibles.")
            print("Primero debe ingestar documentos usando el comando de ingesta.")
    except Exception as e:
        print(f"Error al verificar bases de datos: {e}")
    
    # Imprimir información de arranque
    print(f"Iniciando servidor de chatbot RAG en {args.host}:{args.port}")
    print(f"Modo debug: {'activado' if args.debug else 'desactivado'}")
    print(f"Modo streaming: {'desactivado' if args.no_streaming else 'activado'}")
    print(f"Interfaz web disponible en: http://localhost:{args.port}/")
    print("Presiona Ctrl+C para detener el servidor")
    
    # Iniciar el servidor
    try:
        run_api(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nServidor detenido")
    except Exception as e:
        print(f"Error al iniciar el servidor: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 