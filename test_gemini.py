"""
Script de prueba simple para verificar la conexión con la API de Gemini.

Uso:
    python test_gemini.py
"""

import os
from dotenv import load_dotenv
from google import genai
# Eliminar la importación de types que no se usa correctamente
import google.generativeai as genai

def test_gemini_api():
    """Prueba simple de conexión a la API de Gemini"""
    load_dotenv()
    
    # Obtener API key de las variables de entorno
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("ERROR: No se encontró la API key GEMINI_API_KEY en variables de entorno")
        return
    
    # Eliminar comillas si están presentes
    if api_key.startswith('"') and api_key.endswith('"'):
        api_key = api_key[1:-1]
        print("Comillas dobles eliminadas de la API key")
    elif api_key.startswith("'") and api_key.endswith("'"):
        api_key = api_key[1:-1]
        print("Comillas simples eliminadas de la API key")
    
    print(f"Inicializando cliente Gemini...")
    
    try:
        # Inicializar el cliente
        genai.configure(api_key=api_key)
        
        # Probar diferentes modelos
        test_models = ["gemini-2.0-flash"]
        successful_model = None
        
        for model in test_models:
            try:
                print(f"\nProbando modelo: {model}")
                
                # Prueba básica con generación de contenido
                prompt = "Genera una respuesta corta en español a esta pregunta simple: ¿Cuánto es 2+2?"
                
                # Configuración de generación como un diccionario simple
                generation_config = {
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "top_k": 20,
                    "max_output_tokens": 100,
                }
                
                # Intentar generar contenido
                model_obj = genai.GenerativeModel(model_name=model)
                response = model_obj.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                # Verificar si la respuesta tiene texto
                if hasattr(response, 'text'):
                    print(f"✓ Respuesta exitosa del modelo {model}:")
                    print(response.text)
                    successful_model = model
                    break
                else:
                    print(f"✗ El modelo {model} no devolvió texto en la respuesta")
                    print(f"Respuesta: {response}")
            except Exception as model_err:
                print(f"✗ Error con modelo {model}: {str(model_err)}")
        
        if successful_model:
            print(f"\n✅ Prueba exitosa con modelo: {successful_model}")
            
            # Prueba con instrucciones más detalladas (similar al formato RAG)
            print("\nProbando formato de instrucciones RAG...")
            
            system_prompt = "Eres un asistente preciso especializado en responder preguntas."
            user_query = "¿Cuál es la capital de Francia?"
            
            formatted_prompt = f"""
{system_prompt}

INSTRUCCIONES IMPORTANTES:
1. Responde EXCLUSIVAMENTE basándote en la información proporcionada.
2. Proporciona una respuesta completa, clara y siempre en español.

PREGUNTA DEL USUARIO:
{user_query}

RESPUESTA (en español):
"""
            
            # Intentar generar contenido con formato RAG
            model_obj = genai.GenerativeModel(model_name=successful_model)
            response = model_obj.generate_content(
                formatted_prompt,
                generation_config=generation_config
            )
            
            print("\nRespuesta RAG:")
            if hasattr(response, 'text'):
                print(response.text)
            else:
                print(f"✗ Sin texto en respuesta. Respuesta: {response}")
        else:
            print("\n❌ No se encontró ningún modelo Gemini compatible")
        
    except Exception as e:
        print(f"ERROR GENERAL: {str(e)}")
        import traceback
        print(f"Detalles: {traceback.format_exc()}")

if __name__ == "__main__":
    test_gemini_api() 