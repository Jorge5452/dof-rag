import os
from config import Config
from modulos.clientes.FactoryClient import ClientFactory
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Verificar variables de entorno
print("=== Variables de entorno ===")
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    masked_key = gemini_key[:4] + "..." + gemini_key[-4:] if len(gemini_key) > 8 else "***"
    print(f"GEMINI_API_KEY encontrada: {masked_key}")
else:
    print("GEMINI_API_KEY no encontrada")

# Verificar configuración
print("\n=== Configuración desde config.yaml ===")
config = Config()
ai_config = config.get_ai_client_config()
client_type = ai_config.get("type")
print(f"Tipo de cliente por defecto: {client_type}")

# Obtener configuración específica para Gemini
gemini_config = config.get_specific_ai_config("gemini")
print("Configuración de Gemini:")
for key, value in gemini_config.items():
    if key == "api_key" and value:
        masked_value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
        print(f"  - {key}: {masked_value}")
    else:
        print(f"  - {key}: {value}")

# Intentar crear un cliente Gemini
print("\n=== Intentando crear un cliente Gemini ===")
try:
    client = ClientFactory.get_client("gemini")
    print("Cliente Gemini creado exitosamente")
except Exception as e:
    print(f"Error al crear cliente Gemini: {str(e)}") 