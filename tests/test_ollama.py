# /// script
# dependencies = [
#   "llm-ollama>=0.9.1",
#   "llm>=0.23",
#   "imagehash>=4.3.2",
#   "Pillow",
#   "python-dotenv",
#   "tqdm"
# ]
# ///

#!/usr/bin/env python3
"""
Script para procesar imágenes usando un LLM local (Ollama) con enfoque en descripción de imágenes.
Utiliza hilos para procesamiento en paralelo y asegura el acceso al modelo mediante threading.Lock().
"""

import os
import glob
import time
import datetime
import re
import argparse
import logging
import base64
import mimetypes
import llm
from llm import UnknownModelError
from PIL import Image
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuración básica de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalImageAttachment:
    """
    Clase para envolver una imagen local y proporcionar su contenido en base64,
    además de resolver su tipo MIME.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def base64_content(self):
        with open(self.file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def resolve_type(self):
        mime_type, _ = mimetypes.guess_type(self.file_path)
        return mime_type or "application/octet-stream"


class OllamaImageProcessor:
    """
    Clase para procesar imágenes usando un modelo local de Ollama.
    Provee métodos para obtener información básica de la imagen, generar un prompt
    y procesar la imagen con el modelo en un entorno multihilo.
    """
    def __init__(self, model="gemma3:4b", temperature=0.5, max_tokens=4096,
                top_p=0.5, num_ctx=8192, top_k=20, output_file=None, mode="single"):
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.num_ctx = num_ctx
        self.top_k = top_k  
        self.mode = mode
        
        # Gestión del archivo de salida
        if output_file is None:
            model_clean = model.replace(":", "_")
            mode_suffix = "single" if mode == "single" else "batch"
            self.output_file = f"./tests/results/ollama__{model_clean}_{mode_suffix}_tokens_{max_tokens}.txt"
        else:
            self.output_file = output_file
            
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
        self.token_factor = 1.3  # Factor para estimar tokens

        # Intentar cargar el modelo una sola vez
        try:
            self.model = llm.get_model(model)
        except UnknownModelError as e:
            logger.error(f"Error al obtener el modelo {model}: {e}")
            raise

        # Lock para proteger el acceso al modelo en entornos multihilo
        self.model_lock = threading.Lock()

    def extract_basic_image_info(self, image_path):
        """
        Extrae información básica de una imagen: nombre, formato, modo y dimensiones.
        """
        try:
            info = {
                "filename": os.path.basename(image_path),
                "path": image_path
            }
            with Image.open(image_path) as img:
                info["format"] = img.format
                info["mode"] = img.mode
                info["size"] = img.size
            return info
        except Exception as e:
            logger.warning(f"Error extrayendo información de {image_path}: {e}")
            return {"filename": os.path.basename(image_path),
                    "path": image_path,
                    "error": str(e)}

    def create_description_prompt(self, image_info):
        """
        Crea un prompt más efectivo para solicitar una descripción detallada de la imagen.
        """
        prompt = f"""Necesito que analices esta imagen y proporciones un resumen conciso de su contenido principal.
            
            La imagen es un archivo {image_info.get('format', 'desconocido')} de 
            {image_info.get('size', (0, 0))[0]}x{image_info.get('size', (0, 0))[1]} píxeles.

            Tu tarea:
            1. Resumir brevemente qué contiene la imagen en 2-3 frases.
            2. Si hay texto visible, proporcionar un resumen de su contenido principal (no transcripción completa).
            3. Si es una gráfica o diagrama, explicar qué tipo de datos o información representa.
            4. Si es un mapa, identificar qué región o lugar muestra y cualquier elemento destacado.
            5. Identificar el propósito o contexto probable de la imagen (educativo, informativo, promocional, etc).

            Tu respuesta debe ser clara y directa, entre 100-200 palabras, en español.
            Enfócate en IDENTIFICAR y RESUMIR, no en describir detalles visuales menores.
            """
        return prompt

    def estimate_tokens(self, text):
        """
        Estima el número de tokens en un texto (aproximación basada en palabras y signos).
        """
        if not text:
            return 0
        words = re.findall(r'\w+|[^\w\s]', text)
        return int(len(words) * self.token_factor)

    def process_image(self, image_path, max_retries=3):
        """
        Procesa una imagen y obtiene su descripción usando el modelo local.
        Se usa un threading.Lock() para proteger el acceso al modelo. En caso de respuestas
        muy cortas, se reintenta agregando instrucciones al prompt.
        """
        start_time = time.time()
        image_info = self.extract_basic_image_info(image_path)
        base_prompt = self.create_description_prompt(image_info)
        prompt = base_prompt

        # Preparar la imagen como attachment
        attachment = LocalImageAttachment(os.path.abspath(image_path))

        retries = 0
        while retries <= max_retries:
            try:
                with self.model_lock:
                    base_params = {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_ctx": self.num_ctx,
                        "top_k": self.top_k,
                        "num_predict": self.max_tokens,
                        "stop": ["###"]
                    }
                    response = self.model.prompt(prompt, attachments=[attachment], **base_params)
                response_text = response.text()

                # Si la respuesta es muy corta (menos de 100 caracteres), reintentamos
                if len(response_text.strip()) < 100:
                    if retries < max_retries:
                        retries += 1
                        logger.warning(f"Respuesta demasiado corta para {image_path}, reintento {retries}/{max_retries}")
                        # Prompt mejorado para reintento que enfatiza la necesidad de más detalles
                        prompt = base_prompt + """

                            NOTA IMPORTANTE: Tu respuesta anterior fue demasiado breve o incompleta.

                            Por favor, proporciona una respuesta más completa siguiendo estas instrucciones específicas:

                            1. IDENTIFICA CLARAMENTE el tipo de imagen: fotografía, gráfico, documento, mapa, etc.

                            2. DESCRIBE DE MANERA ESPECÍFICA:
                            - Si contiene texto: resume su contenido principal y menciona títulos o encabezados importantes
                            - Si es un gráfico/tabla: indica qué tipo de datos muestra y conclusiones principales
                            - Si es un mapa: especifica la región/lugar y elementos geográficos destacados
                            - Si es una fotografía: menciona objetos, personas, colores predominantes y composición general

                            3. CONTEXTUALIZA la imagen: indica su propósito probable (educativo, informativo, comercial, etc.)

                            Tu respuesta debe tener entre 150-200 palabras y ser ESTRUCTURADA según los puntos anteriores.
                            No te extiendas en detalles visuales menores, pero asegúrate de IDENTIFICAR todos los elementos importantes.
                            """
                        continue

                duration = time.time() - start_time
                tokens = self.estimate_tokens(response_text)
                logger.info(f"Procesada {image_path} en {duration:.2f}s, tokens estimados: {tokens}")

                return {
                    "image_path": image_path,
                    "response": response_text.strip(),
                    "processing_time": duration,
                    "token_count": tokens,
                    "prompt_used": prompt
                }

            except Exception as e:
                if retries == max_retries:
                    duration = time.time() - start_time
                    logger.error(f"Error procesando {image_path} después de {max_retries} intentos: {e}")
                    return {"image_path": image_path,
                            "error": str(e),
                            "processing_time": duration}
                retries += 1
                logger.warning(f"Error en intento {retries}/{max_retries} para {image_path}: {e}")
                time.sleep(1)  # Pausa entre reintentos

        duration = time.time() - start_time
        return {"image_path": image_path,
                "error": "No se pudo obtener una respuesta válida después de múltiples intentos",
                "processing_time": duration}

    def process_batch(self, image_files, max_workers=None):
        """
        Procesa una lista de imágenes en paralelo utilizando ThreadPoolExecutor.
        """
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_image, img): img for img in image_files}
            for future in as_completed(futures):
                img_path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completado: {img_path}")
                except Exception as e:
                    logger.error(f"Error procesando {img_path}: {e}")
                    results.append({"image_path": img_path,
                                    "error": str(e),
                                    "processing_time": 0})
        return results

    def save_results(self, results):
        """
        Guarda los resultados de procesamiento en un archivo de texto con formato estandarizado.
        Usa append para acumular resultados en múltiples ejecuciones.
        """
        # Verificar si el archivo existe para decidir si añadir encabezado
        file_exists = os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0
        
        with open(self.output_file, "a", encoding="utf-8") as f:
            # Solo añadir encabezado si el archivo es nuevo
            if not file_exists:
                f.write(f"Resultados de descripciones de imágenes con Ollama - Modelo: {self.model_name} - max_tokens: {self.max_tokens}\n\n")
            
            # Añadir marca de nueva ejecución
            separator = "\n" + "=" * 80 + "\n"
            f.write(separator)
            f.write(f"NUEVA EJECUCIÓN: {datetime.datetime.now().isoformat()}\n")
            f.write(separator)
            
            # Escribir cada resultado de imagen
            for res in results:
                f.write(separator)
                f.write("RESULTADOS DE IMAGEN\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Modelo: {self.model_name}, Tokens máx: {self.max_tokens}\n")
                f.write(f"Imagen: {os.path.basename(res.get('image_path', 'N/A'))}\n")
                
                if "error" in res:
                    f.write(f"Error: {res['error']}\n")
                else:
                    f.write("\nMÉTRICAS:\n")
                    f.write(f"- Descripción: {res.get('processing_time', 0):.4f}s | ")
                    f.write(f"Total: {res.get('processing_time', 0):.4f}s | ")
                    f.write(f"Tokens: {res.get('token_count', 0)}\n")
                    
                    f.write("\nRESULTADOS:\n")
                    f.write("Pregunta: ¿Qué se observa en esta imagen?\n\n")
                    f.write(f"Descripción: {res.get('response', '')}\n")
                
                f.write(separator + "\n")
            
            # Añadir resumen final
            self._guardar_resumen_final(results)
                
        logger.info(f"Resultados añadidos en: {self.output_file}")
    

    def _guardar_resumen_final(self, results):
        """
        Guarda un resumen final de todos los resultados al final del archivo.
        """
        separator = "\n" + "=" * 80 + "\n"
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(separator)
            f.write("RESUMEN FINAL DE PROCESAMIENTO\n")
            f.write(f"Timestamp finalización: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Modelo: {self.model_name}, Tokens máx: {self.max_tokens}\n")
            
            # Estadísticas básicas
            total_images = len(results)
            successful_images = sum(1 for res in results if "error" not in res)
            failed_images = total_images - successful_images
            
            f.write(f"Total imágenes procesadas: {total_images}\n")
            f.write(f"Exitosas: {successful_images}, Errores: {failed_images}\n")
            
            # Tiempo y tokens totales
            total_time = sum(res.get("processing_time", 0) for res in results if "processing_time" in res)
            total_tokens = sum(res.get("token_count", 0) for res in results if "token_count" in res)
            
            f.write(f"Tiempo total de procesamiento: {total_time:.2f} segundos\n")
            
            if successful_images > 0:
                # Calcular promedios
                avg_time = total_time / successful_images
                avg_tokens = total_tokens / successful_images
                
                f.write("\nTiempos promedio por imagen:\n")
                f.write(f"- Total: {avg_time:.4f}s | ")
                f.write(f"Tokens: {avg_tokens:.1f}\n")
            
            f.write(separator)


    def process_interactive(self, image_path):
        """
        Procesa una imagen en modo interactivo (similar a moondream).
        """
        print(f"\n{'='*60}")
        print(f"Procesando: {os.path.basename(image_path)}")
        print(f"Modelo: {self.model_name}")
        
        try:
            # Procesar imagen
            print("Procesando imagen...")
            result = self.process_image(image_path)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                # Mostrar resultados de forma compacta
                print("\nTiempos: ", end="")
                print(f"Procesamiento={result['processing_time']:.2f}s | ", end="")
                print(f"Tokens estimados={result['token_count']}")
                
                print("\nDescripción:")
                print(f"{result['response']}")
            
            # Guardar resultado en formato común
            self.save_single_result(result)
            print(f"\nResultados guardados: {os.path.basename(self.output_file)}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    def save_single_result(self, img_res):
        """
        Guarda el resultado de una única imagen procesada en modo interactivo.
        Usa append para acumular resultados en múltiples ejecuciones.
        """
        # Verificar si el archivo existe para decidir si añadir encabezado
        file_exists = os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0
        
        with open(self.output_file, "a", encoding="utf-8") as f:
            # Solo añadir encabezado si el archivo es nuevo
            if not file_exists:
                f.write(f"Resultados de descripción de imágenes con Ollama - Modelo: {self.model_name} - max_tokens: {self.max_tokens}\n\n")
            
            separator = "=" * 80 + "\n"
            f.write(separator)
            f.write("RESULTADOS DE IMAGEN\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Modelo: {self.model_name}, Tokens máx: {self.max_tokens}\n")
            f.write(f"Imagen: {os.path.basename(img_res.get('image_path', 'N/A'))}\n")
            
            if "error" in img_res:
                f.write(f"Error: {img_res['error']}\n")
            else:
                f.write("\nMÉTRICAS:\n")
                f.write(f"- Descripción: {img_res.get('processing_time', 0):.4f}s | ")
                f.write(f"Total: {img_res.get('processing_time', 0):.4f}s | ")
                f.write(f"Tokens: {img_res.get('token_count', 0)}\n")
                
                f.write("\nRESULTADOS:\n")
                f.write("Pregunta: ¿Qué se observa en esta imagen?\n\n")
                f.write(f"Descripción: {img_res.get('response', '')}\n")
            
            f.write("\n" + separator)


def parse_args():
    """
    Parsea los argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description="Procesamiento de imágenes con LLM local (Ollama) usando hilos"
    )
    parser.add_argument("--image_dir", "-d", type=str, default="./imagenes_prueba",
                        help="Directorio con imágenes a procesar (default: ./imagenes_prueba)")
    parser.add_argument("--output", "-o", default=None,
                        help="Nombre del archivo de salida para resultados")
    parser.add_argument("--model", "-m", default="gemma3:4b",
                        help="Modelo a utilizar (default: gemma3:4b)")
    parser.add_argument("--temperature", "-t", type=float, default=0.5,
                        help="Temperatura base para el modelo (default: 0.5)")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Número máximo de tokens a generar (default: 4096)")
    parser.add_argument("--threads", type=int, default=None,
                        help="Número máximo de hilos a utilizar (default: automático)")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Limitar selección de tokens a los k más probables (default: 40)")
    parser.add_argument("--mode", choices=["single", "batch"], default="single",
                        help="Modo: 'single' (una imagen) o 'batch' (todas) (default: single)")
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = os.path.abspath(args.image_dir)
    logger.info(f"Directorio de imágenes: {image_dir}")

    # Buscar imágenes en el directorio con extensiones válidas
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    image_files = glob.glob(os.path.join(image_dir, "*"))
    image_files = [img for img in image_files if img.lower().endswith(valid_extensions)]

    if not image_files:
        logger.error(f"No se encontraron imágenes en el directorio: {image_dir}")
        return
    
    # Ordenar las imágenes para consistencia
    image_files = sorted(image_files)

    # Simplificar la generación del nombre de archivo de salida
    model_clean = args.model.replace(":", "_")
    output_file = args.output
    if output_file is None:
        mode_suffix = "single" if args.mode == "single" else "batch"
        output_file = f"./tests/results/ollama__{model_clean}__{mode_suffix}__tokens_{args.max_tokens}.txt"
    
    # Asegurarse de que el directorio existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Inicializar el procesador
    processor = OllamaImageProcessor(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=0.5,
        num_ctx=4096,
        top_k=20,
        output_file=output_file,
        mode=args.mode
    )

    # Calcular número de hilos si no se especifica (por defecto 70% de núcleos)
    if args.threads is None:
        cpu_count = os.cpu_count() or 1
        threads = max(1, int(cpu_count * 0.7))
    else:
        threads = args.threads

    # Lógica según el modo seleccionado
    if args.mode == "single":
        # Procesar solo la primera imagen
        image_path = image_files[0]
        print(f"Modo: Single | Imagen: {os.path.basename(image_path)}")
        print(f"Modelo: {args.model} | Tokens máx: {args.max_tokens}")
        
        processor.process_interactive(image_path)
    else:
        # Procesar todas las imágenes en batch
        print(f"Modo: Batch | Imágenes: {len(image_files)}")
        print(f"Modelo: {args.model} | Workers: {threads}")
        
        start_time = time.time()
        results = processor.process_batch(image_files, max_workers=threads)
        total_time = time.time() - start_time

        # Mostrar resumen de resultados
        success_count = sum(1 for res in results if "error" not in res)
        total_tokens = sum(res.get("token_count", 0) for res in results if "token_count" in res)

        print(f"Procesamiento completado en {total_time:.2f}s")
        print(f"Imágenes procesadas exitosamente: {success_count}/{len(results)}")
        print(f"Total de tokens generados: {total_tokens}")
        print(f"Resultados guardados en: {os.path.basename(output_file)}")

        # Guardar resultados
        processor.save_results(results)


if __name__ == "__main__":
    main()
