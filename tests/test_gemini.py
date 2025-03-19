# /// script
# dependencies = [
#   "python-dotenv>=0.9.9",
#   "google>=0.3.0",
#   "google-genai>=1.3.0",
#   "pillow"
# ]
# ///

from PIL import Image
import time
import datetime
import os
import glob
import threading
import concurrent.futures
from google import genai
from google.genai import types
import re
import argparse
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Creamos un lock global para sincronizar el acceso a la API
api_lock = threading.Lock()


class GeminiConfig:
    """
    Clase para manejar la configuración de la API de Gemini.
    """

    def __init__(
        self,
        model="gemini-2.0-flash",
        max_tokens=256,
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        response_mime_type="text/plain",
    ):
        # Obtener la API key desde variables de entorno
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No se encontró la variable de entorno GEMINI_API_KEY. Por favor, configúrela."
            )

        self.model = model
        self.max_tokens = max_tokens

        # Parámetros de generación
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.response_mime_type = response_mime_type

    def get_generate_config(self):
        """Retorna la configuración para la generación de contenido"""
        return types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_tokens,
            response_mime_type=self.response_mime_type,
        )

    def get_client(self):
        """Retorna un cliente inicializado con la API key configurada"""
        return genai.Client(api_key=self.api_key)

    def get_output_filename(self):
        """Genera el nombre del archivo de salida basado en el modelo y tokens"""

        # Usar el nombre del modelo completo, reemplazando caracteres problemáticos para nombres de archivo
        model_name = self.model.replace(".", "-").replace(":", "_")
        return f"./tests/results/{model_name}_tokens_{self.max_tokens}.txt"


# Variable para el archivo de resultados - ahora será generada automáticamente
def initialize_config(token_limit=256):
    """Inicializa la configuración con el límite de tokens especificado"""
    return GeminiConfig(max_tokens=token_limit)


# Inicializar la configuración por defecto
GEMINI_CONFIG = initialize_config()
GEMINI_MODEL = GEMINI_CONFIG.model
MAX_TOKENS = GEMINI_CONFIG.max_tokens
archivo = GEMINI_CONFIG.get_output_filename()


def estimar_tokens(texto):
    """
    Estima el número de tokens en un texto.
    Esta es una estimación aproximada, ya que la tokenización exacta
    depende del tokenizador específico que usa Gemini.
    """
    if not texto:
        return 0

    # Método simple basado en palabras (aproximado)
    palabras = re.findall(r"\w+|[^\w\s]", texto)
    num_palabras = len(palabras)

    # Algunos tokens son subpalabras, otros son múltiples palabras
    # Factor de ajuste aproximado: 1.3 tokens por palabra
    return round(num_palabras * 1.3)


def initialize_gemini_client():
    """
    Inicializa y retorna un cliente de Gemini.
    """
    return GEMINI_CONFIG.get_client()


def process_image_with_gemini(client, image_path, question, stream=False):
    """
    Procesa una única imagen usando Gemini.
    Se protege cada llamada a la API con un lock para evitar problemas de concurrencia.
    Retorna un diccionario con los tiempos y las respuestas, o un error en caso de fallo.

    Si stream=True, usa el modo streaming para la generación de respuestas.
    """
    result = {"image_path": image_path}
    start = time.time()

    try:
        image = Image.open(image_path)
    except Exception as e:
        result["error"] = f"Error abriendo imagen: {str(e)}"
        return result

    try:
        with api_lock:
            # Generar descripción de la imagen
            caption_start = time.time()

            if not stream:
                caption_response = client.models.generate_content(
                    model=GEMINI_CONFIG.model,
                    contents=[
                        """Describe detalladamente la imagen en español, priorizando la información clave:  
            - **Texto:** Si contiene texto (como en diagramas, infografías o documentos oficiales), transcribe el contenido principal.  
            - **Mapas:** Menciona lugares, nombres geográficos y símbolos relevantes.  
            - **Esquemas o diagramas:** Explica las relaciones o procesos principales.  
            - **Logos:** Describe el diseño, colores y, si es reconocible, a qué entidad pertenece.  
            - **Datos visuales:** Si hay gráficos o estadísticas, resume los valores más importantes.  
            Sé claro y preciso, priorizando la información más relevante.""",
                        image,
                    ],
                    config=GEMINI_CONFIG.get_generate_config(),
                )
                caption = (
                    caption_response.text
                    if hasattr(caption_response, "text")
                    else str(caption_response)
                )
            else:
                # Modo streaming para la descripción
                caption = ""
                for chunk in client.models.generate_content_stream(
                    model=GEMINI_CONFIG.model,
                    contents=[
                        """Describe detalladamente la imagen en español, priorizando la información clave:  
            - **Texto:** Si contiene texto (como en diagramas, infografías o documentos oficiales), transcribe el contenido principal.  
            - **Mapas:** Menciona lugares, nombres geográficos y símbolos relevantes.  
            - **Esquemas o diagramas:** Explica las relaciones o procesos principales.  
            - **Logos:** Describe el diseño, colores y, si es reconocible, a qué entidad pertenece.  
            - **Datos visuales:** Si hay gráficos o estadísticas, resume los valores más importantes.  
            Sé claro y preciso, priorizando la información más relevante.""",
                        image,
                    ],
                    config=GEMINI_CONFIG.get_generate_config(),
                ):
                    caption += chunk.text if hasattr(chunk, "text") else ""

            caption_time = time.time() - caption_start
            caption_tokens = estimar_tokens(caption)

            # Responder a la pregunta sobre la imagen
            query_start = time.time()

            if not stream:
                answer_response = client.models.generate_content(
                    model=GEMINI_CONFIG.model,
                    contents=[
                        question
                        + " Sé muy breve y conciso. Responde COMPLETAMENTE en español. Máximo 100 palabras.",
                        image,
                    ],
                    config=GEMINI_CONFIG.get_generate_config(),
                )
                answer = (
                    answer_response.text
                    if hasattr(answer_response, "text")
                    else str(answer_response)
                )
            else:
                # Modo streaming para la respuesta
                answer = ""
                for chunk in client.models.generate_content_stream(
                    model=GEMINI_CONFIG.model,
                    contents=[
                        question
                        + " Sé muy breve y conciso. Responde COMPLETAMENTE en español. Máximo 100 palabras.",
                        image,
                    ],
                    config=GEMINI_CONFIG.get_generate_config(),
                ):
                    answer += chunk.text if hasattr(chunk, "text") else ""

            query_time = time.time() - query_start
            answer_tokens = estimar_tokens(answer)

        total_time = time.time() - start
        result["caption_time"] = caption_time
        result["query_time"] = query_time
        result["total_time"] = total_time
        result["caption"] = caption
        result["answer"] = answer
        result["caption_tokens"] = caption_tokens
        result["answer_tokens"] = answer_tokens
        result["total_tokens"] = caption_tokens + answer_tokens
        result["streaming"] = stream
    except Exception as e:
        result["error"] = f"Error procesando imagen con Gemini: {str(e)}"

    return result


def process_text_with_gemini(client, prompt, stream=False):
    """
    Procesa un texto usando Gemini.
    """
    result = {"prompt": prompt}
    start = time.time()

    try:
        with api_lock:
            if not stream:
                response = client.models.generate_content(
                    model=GEMINI_CONFIG.model,
                    contents=prompt,
                    config=GEMINI_CONFIG.get_generate_config(),
                )
                text_response = (
                    response.text if hasattr(response, "text") else str(response)
                )
            else:
                # Modo streaming
                text_response = ""
                for chunk in client.models.generate_content_stream(
                    model=GEMINI_CONFIG.model,
                    contents=prompt,
                    config=GEMINI_CONFIG.get_generate_config(),
                ):
                    text_response += chunk.text if hasattr(chunk, "text") else ""
                    # Si se desea ver la respuesta en tiempo real, descomentar:
                    # print(chunk.text, end="", flush=True)

        process_time = time.time() - start
        tokens = estimar_tokens(text_response)

        result["response"] = text_response
        result["process_time"] = process_time
        result["tokens"] = tokens
        result["streaming"] = stream
    except Exception as e:
        result["error"] = f"Error procesando texto con Gemini: {str(e)}"

    return result


def guardar_resultado_imagen(img_res, log_file=None):
    """
    Guarda el resultado de una imagen en el archivo de resultados (modo append).
    """
    if log_file is None:
        log_file = GEMINI_CONFIG.get_output_filename()

    separator = "\n" + "=" * 80 + "\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(separator)
        f.write("RESULTADOS DE IMAGEN\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Modelo: {GEMINI_MODEL}, Tokens máx: {MAX_TOKENS}\n")
        f.write(f"Imagen: {os.path.basename(img_res.get('image_path', 'N/A'))}\n")

        if "error" in img_res:
            f.write("Error: " + img_res["error"] + "\n")
        else:
            f.write("\nMÉTRICAS:\n")
            f.write(f"- Descripción: {img_res.get('caption_time', 0):.4f}s | ")
            f.write(f"Query: {img_res.get('query_time', 0):.4f}s | ")
            f.write(f"Total: {img_res.get('total_time', 0):.4f}s | ")
            f.write(f"Tokens: {img_res.get('total_tokens', 0)}\n")

            f.write("\nRESULTADOS:\n")
            f.write("Pregunta: ¿Qué se observa en esta imagen?\n\n")
            f.write(f"Descripción: {img_res.get('caption', '')}\n\n")
            f.write(f"Respuesta: {img_res.get('answer', '')}\n")
        f.write(separator + "\n")


def guardar_resumen_final(results, log_file=None):
    """
    Guarda un resumen final de todos los resultados al final del archivo.
    """
    if log_file is None:
        log_file = GEMINI_CONFIG.get_output_filename()

    separator = "\n" + "=" * 80 + "\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(separator)
        f.write("RESUMEN FINAL DE PROCESAMIENTO\n")
        f.write(f"Timestamp finalización: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Modelo: {GEMINI_MODEL}, Tokens máx: {MAX_TOKENS}\n")

        # Estadísticas básicas
        total_images = len(results.get("images", []))
        successful_images = sum(
            1 for img in results.get("images", []) if "error" not in img
        )
        failed_images = total_images - successful_images

        f.write(f"Total imágenes procesadas: {total_images}\n")
        f.write(f"Exitosas: {successful_images}, Errores: {failed_images}\n")

        # Tiempo total
        total_time = results.get("total_time", 0)
        f.write(f"Tiempo total de procesamiento: {total_time:.2f} segundos\n")

        if successful_images > 0:
            # Calcular promedios
            avg_caption_time = (
                sum(
                    img.get("caption_time", 0)
                    for img in results.get("images", [])
                    if "error" not in img
                )
                / successful_images
            )
            avg_query_time = (
                sum(
                    img.get("query_time", 0)
                    for img in results.get("images", [])
                    if "error" not in img
                )
                / successful_images
            )
            avg_total_time = (
                sum(
                    img.get("total_time", 0)
                    for img in results.get("images", [])
                    if "error" not in img
                )
                / successful_images
            )
            avg_tokens = (
                sum(
                    img.get("total_tokens", 0)
                    for img in results.get("images", [])
                    if "error" not in img
                )
                / successful_images
            )

            f.write("\nTiempos promedio por imagen:\n")
            f.write(f"- Descripción: {avg_caption_time:.4f}s | ")
            f.write(f"Query: {avg_query_time:.4f}s | ")
            f.write(f"Total: {avg_total_time:.4f}s | ")
            f.write(f"Tokens: {avg_tokens:.1f}\n")

        f.write(separator)


def process_images_with_threads(
    image_files, question, max_workers=None, use_streaming=False
):
    """
    Procesa las imágenes en hilos usando ThreadPoolExecutor y la API de Gemini.

    Parámetros:
    - image_files: Lista de rutas de imágenes a procesar
    - question: Pregunta a realizar sobre cada imagen
    - max_workers: Número máximo de workers en el ThreadPool
    - use_streaming: Si es True, usa el modo streaming para las respuestas
    """
    results = {}
    results["model"] = GEMINI_CONFIG.model
    results["max_tokens"] = GEMINI_CONFIG.max_tokens
    start_total = time.time()

    try:
        init_start = time.time()
        client = initialize_gemini_client()
        init_time = time.time() - init_start
        results["init_time"] = init_time
    except Exception as e:
        results["error"] = f"Error inicializando el cliente de Gemini: {str(e)}"
        return results

    results["images"] = []
    total_images = len(image_files)
    completed = 0

    # Calcular el 70% de los núcleos disponibles
    if max_workers is None:
        num_cpu = os.cpu_count() or 1
        max_workers = max(1, int(num_cpu * 0.7))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_image_with_gemini, client, img_path, question, use_streaming
            ): img_path
            for img_path in image_files
        }
        for future in concurrent.futures.as_completed(futures):
            image_result = future.result()
            results["images"].append(image_result)
            completed += 1

            # Guardar inmediatamente el resultado de la imagen en el archivo
            guardar_resultado_imagen(image_result)

            elapsed = time.time() - start_total
            avg_time = elapsed / completed
            remaining = total_images - completed
            estimated_remaining = remaining * avg_time
            print(
                f"[Gemini] Imagen {completed}/{total_images} procesada en {image_result.get('total_time', 0):.2f} s. "
                f"Tiempo total: {elapsed:.2f} s. Estimado restante: {estimated_remaining:.2f} s",
                flush=True,
            )

    results["total_time"] = time.time() - start_total

    # Guardar resumen final
    guardar_resumen_final(results)

    return results


if __name__ == "__main__":
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Procesar imágenes con Gemini API")
    parser.add_argument(
        "--tokens",
        type=int,
        default=256,
        choices=[256, 512, 1024, 2048, 4096, 8192],
        help="Número máximo de tokens en la respuesta (default: 256)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Usar modo streaming para las respuestas",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./imagenes_prueba",
        help="Directorio con las imágenes de prueba (default: ./imagenes_prueba)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="¿Qué se observa en esta imagen?",
        help="Pregunta a realizar sobre cada imagen",
    )

    args = parser.parse_args()

    # Inicializar la configuración con los tokens especificados
    GEMINI_CONFIG = initialize_config(args.tokens)
    GEMINI_MODEL = GEMINI_CONFIG.model
    MAX_TOKENS = GEMINI_CONFIG.max_tokens
    archivo = GEMINI_CONFIG.get_output_filename()

    # Directorio con las imágenes de prueba
    image_directory = args.dir
    image_files = glob.glob(os.path.join(image_directory, "*.*"))
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")
    image_files = [img for img in image_files if img.lower().endswith(valid_extensions)]

    if not image_files:
        print("No se encontraron imágenes en el directorio:", image_directory)
        exit(1)

    # Pregunta en español (Gemini es multilingüe, por lo que no necesitamos traducción)
    question = args.question

    # Limpiar el archivo de resultados al inicio
    with open(archivo, "w", encoding="utf-8") as f:
        f.write(
            f"Resultados de predicciones con Gemini ({GEMINI_MODEL}) - max_tokens: {MAX_TOKENS}\n"
        )

    print("\n" + "=" * 80)
    print(
        f"Iniciando predicciones con Gemini ({GEMINI_MODEL}) - max_tokens: {MAX_TOKENS}"
    )
    print(f"Modo streaming: {'Activado' if args.streaming else 'Desactivado'}")
    print(f"Archivo de resultados: {archivo}")
    results = process_images_with_threads(
        image_files, question, use_streaming=args.streaming
    )
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print(f"Procesamiento finalizado en {results.get('total_time', 0):.2f} s")
        print(f"Resumen guardado en: {archivo}")
    print("=" * 80 + "\n")
