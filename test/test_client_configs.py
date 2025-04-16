#!/usr/bin/env python
"""
Script de pruebas para verificar la correcta configuración de clientes IA.
"""

import os
import sys
from pathlib import Path

# Añadir directorio raíz al path para poder importar los módulos
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from modulos.clientes.FactoryClient import ClientFactory

def test_client_factory():
    """Prueba la fábrica de clientes y la gestión de configuraciones."""
    config = Config()
    
    # Prueba los tipos de cliente disponibles
    clients = ['openai', 'gemini', 'ollama']
    
    for client_type in clients:
        print(f"\n=== Prueba de configuración para cliente: {client_type} ===")
        
        # Obtener la configuración filtrada para este cliente
        filtered_config = config.get_specific_ai_config(client_type)
        
        print(f"Parámetros configurados para {client_type}:")
        for key, value in filtered_config.items():
            print(f"  - {key}: {value}")
        
        # Crear cliente con configuración por defecto
        try:
            client = ClientFactory.get_client(client_type)
            print(f"✓ Cliente {client_type} creado correctamente")
            
            # Verificar atributos clave
            attrs_to_check = ['model_name', 'temperature', 'top_p']
            
            for attr in attrs_to_check:
                if hasattr(client, attr):
                    print(f"  ✓ Atributo '{attr}': {getattr(client, attr)}")
                else:
                    print(f"  ✗ Falta el atributo '{attr}'")
            
        except Exception as e:
            print(f"✗ Error al crear cliente {client_type}: {str(e)}")

def test_client_override():
    """Prueba la capacidad de anular parámetros de configuración."""
    print("\n=== Prueba de anulación de parámetros ===")
    
    # Probar con el cliente predeterminado
    client_type = Config().get_ai_client_config().get("type", "openai")
    
    # Parámetros de prueba
    test_params = {
        "temperature": 0.1,
        "system_prompt": "Esto es un prompt de prueba",
        "max_tokens": 500
    }
    
    try:
        client = ClientFactory.get_client(client_type, **test_params)
        print(f"✓ Cliente {client_type} creado con parámetros personalizados")
        
        # Verificar los parámetros anulados
        for param, value in test_params.items():
            attr_name = param
            # Manejar nombres de atributos específicos
            if param == "max_tokens" and client_type == "gemini":
                attr_name = "max_tokens"  # En el objeto se llama max_tokens aunque internamente se use max_output_tokens
                
            if hasattr(client, attr_name):
                actual_value = getattr(client, attr_name)
                if actual_value == value:
                    print(f"  ✓ Parámetro '{param}' correctamente anulado: {actual_value}")
                else:
                    print(f"  ✗ Parámetro '{param}' no anulado correctamente. Esperado: {value}, Actual: {actual_value}")
            else:
                print(f"  ✗ Cliente no tiene el atributo '{attr_name}'")
                
    except Exception as e:
        print(f"✗ Error en prueba de anulación de parámetros: {str(e)}")

if __name__ == "__main__":
    print("=== PRUEBAS DE CONFIGURACIÓN DE CLIENTES IA ===")
    test_client_factory()
    test_client_override()
    print("\nPruebas completadas.") 