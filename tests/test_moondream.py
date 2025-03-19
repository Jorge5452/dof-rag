# /// script
# dependencies = [
#   "moondream>=0.0.6",
#   "tokenizers>=0.20.1,<0.21.0",
#   "sentencepiece>=0.2.0",
#   "sacremoses>=0.1.1",
#   "transformers>=4.46.3"
# ]
# ///

import moondream as md
from PIL import Image
import time
import datetime
import os
import glob
import gc
import threading
import concurrent.futures
import argparse
import logging
import colorama
from tqdm import tqdm
from colorama import Fore

# Inicializar colorama para colores en la terminal
colorama.init(autoreset=True)

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Importamos las librerías para traducción con MarianMT
from transformers import MarianMTModel, MarianTokenizer

# Cargamos el modelo de traducción (del inglés al español) de forma global para mayor eficiencia
translation_model_name = "Helsinki-NLP/opus-mt-en-es"
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)

# Definimos la pregunta por defecto para las imágenes
DEFAULT_QUESTION = "¿Qué se muestra en esta imagen? Describe todos los elementos visibles con detalle."

# Definimos los modelos disponibles
AVAILABLE_MODELS = {
    "small": "model/moondream-0_5b-int8.mf",  # Modelo pequeño (0.5b)
    "big": "model/moondream-2b-int8.mf"       # Modelo grande (2b)
}

def translate_text(text):
    """Traduce texto del inglés al español usando MarianMT."""
    if not text:
        return ""
    inputs = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = translation_model.generate(**inputs)
    translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

class MoondreamImageProcessor:
    """Clase para procesar imágenes con Moondream."""
    
    def __init__(self, model_path="model/moondream-0_5b-int8.mf", output_file=None, mode="single"):
        """Inicializa el procesador de imágenes de Moondream.
        
        Args:
            model_path: Ruta al modelo Moondream a utilizar
            output_file: Archivo donde guardar los resultados
            mode: Modo de procesamiento ("batch" o "single")
        """
        self.model_path = model_path
        self.mode = mode
        
        # Extraer el nombre del modelo de la ruta para usar en el nombre del archivo
        model_name = os.path.basename(model_path).split('.')[0]
        
        # Generar nombre de archivo de salida si no se especifica
        if output_file is None:
            mode_suffix = "single" if mode == "single" else "batch"
            self.output_file = f"tests/results/{model_name}_{mode_suffix}.txt"
        else:
            self.output_file = output_file
        
        # Asegurar que el directorio de salida existe
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Lock para sincronizar acceso al modelo
        self.model_lock = threading.Lock()
        self.model = None
    
    def initialize_model(self):
        """Inicializa el modelo Moondream."""
        try:
            init_start = time.time()
            self.model = md.vl(model=self.model_path)
            init_time = time.time() - init_start
            logger.info(f"{Fore.GREEN}✓ Modelo {os.path.basename(self.model_path)} inicializado en {init_time:.2f}s")
            return init_time
        except Exception as e:
            logger.error(f"{Fore.RED}✗ Error inicializando el modelo: {str(e)}")
            raise
    
    def close_model(self):
        """Cierra el modelo y libera recursos."""
        try:
            if self.model and hasattr(self.model, "close"):
                self.model.close()
            del self.model
            self.model = None
            gc.collect()
            logger.info(f"{Fore.CYAN}ℹ Modelo liberado")
        except Exception as e:
            logger.error(f"{Fore.RED}✗ Error cerrando el modelo: {str(e)}")
    
    def process_image(self, image_path, question):
        """Procesa una única imagen usando el modelo compartido."""
        result = {"image_path": image_path}
        start = time.time()
        
        try:
            image = Image.open(image_path)
            result["image_size"] = f"{image.width}x{image.height}"
            result["image_format"] = image.format
        except Exception as e:
            result["error"] = f"Error abriendo imagen: {str(e)}"
            return result
        
        try:
            with self.model_lock:
                # Codificar la imagen
                encode_start = time.time()
                encoded_image = self.model.encode_image(image)
                encode_time = time.time() - encode_start

                # Generar caption
                caption_start = time.time()
                caption_result = self.model.caption(encoded_image)
                caption_time = time.time() - caption_start
                caption = caption_result.get("caption", "")

                # Ejecutar la query
                query_start = time.time()
                answer_result = self.model.query(encoded_image, question)
                query_time = time.time() - query_start
                answer = answer_result.get("answer", "")
            
            total_time = time.time() - start
            result["encode_time"] = encode_time
            result["caption_time"] = caption_time
            result["query_time"] = query_time
            result["total_time"] = total_time
            result["caption"] = caption
            result["answer"] = answer
            result["question"] = question

            # Traducir las respuestas (caption y answer) del inglés al español
            result["caption_trad"] = translate_text(caption)
            result["answer_trad"] = translate_text(answer)
        except Exception as e:
            result["error"] = f"Error procesando imagen: {str(e)}"
        
        return result
    
    def guardar_resultado_imagen(self, img_res):
        """Guarda el resultado de una imagen en el archivo de resultados."""
        separator = "\n" + "-" * 60 + "\n"
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(separator)
            f.write("RESULTADOS DE IMAGEN\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Modelo: {os.path.basename(self.model_path)}\n")
            f.write(f"Imagen: {os.path.basename(img_res.get('image_path', 'N/A'))}\n")
            
            if "error" in img_res:
                f.write("Error: " + img_res["error"] + "\n")
            else:
                f.write("\nMÉTRICAS:\n")
                f.write(f"- Encode: {img_res.get('encode_time', 0):.4f}s | ")
                f.write(f"Caption: {img_res.get('caption_time', 0):.4f}s | ")
                f.write(f"Query: {img_res.get('query_time', 0):.4f}s | ")
                f.write(f"Total: {img_res.get('total_time', 0):.4f}s\n")
                
                f.write("\nRESULTADOS:\n")
                f.write(f"Pregunta: {img_res.get('question', '')}\n\n")
                f.write(f"Descripción: {img_res.get('caption_trad', '')}\n\n")
                f.write(f"Respuesta: {img_res.get('answer_trad', '')}\n")
            f.write(separator)

    def process_interactive(self, image_path, question):
        """Procesa una imagen en modo interactivo."""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}Procesando: {Fore.YELLOW}{os.path.basename(image_path)}")
        print(f"{Fore.CYAN}Modelo: {Fore.YELLOW}{os.path.basename(self.model_path)}")
        
        try:
            # Inicializar modelo si no está inicializado
            if self.model is None:
                print(f"{Fore.CYAN}Inicializando modelo...")
                self.initialize_model()
            
            # Procesar imagen
            print(f"{Fore.CYAN}Procesando imagen...")
            result = self.process_image(image_path, question)
            
            if "error" in result:
                print(f"{Fore.RED}Error: {result['error']}")
            else:
                # Mostrar resultados de forma compacta
                print(f"\n{Fore.GREEN}Tiempos: ", end="")
                print(f"Encode={result['encode_time']:.2f}s | ", end="")
                print(f"Caption={result['caption_time']:.2f}s | ", end="")
                print(f"Query={result['query_time']:.2f}s | ", end="")
                print(f"Total={result['total_time']:.2f}s")
                
                print(f"\n{Fore.GREEN}Descripción: {Fore.WHITE}{result['caption_trad']}")
                print(f"\n{Fore.GREEN}Pregunta: {Fore.WHITE}{question}")
                print(f"{Fore.GREEN}Respuesta: {Fore.WHITE}{result['answer_trad']}")
            
            # Guardar resultado
            self.guardar_resultado_imagen(result)
            print(f"\n{Fore.CYAN}Resultados guardados: {os.path.basename(self.output_file)}")
            
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}")

    def process_batch(self, image_files, question, max_workers=None):
        """Procesa múltiples imágenes en paralelo."""
        results = {
            "model_path": self.model_path,
            "images": [],
            "start_time": datetime.datetime.now().isoformat()
        }
        
        start_total = time.time()
        total_images = len(image_files)
        
        # Calcular el número de trabajadores
        if max_workers is None:
            num_cpu = os.cpu_count() or 1
            max_workers = max(1, int(num_cpu * 0.7))
        
        # Inicializar archivo de resultados
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("TEST MOONDREAM - BATCH\n")
            f.write(f"Modelo: {os.path.basename(self.model_path)}\n")
            f.write(f"Inicio: {results['start_time']}\n")
            f.write(f"Imágenes: {total_images}\n")
            f.write(f"Pregunta: {question}\n")
            f.write(f"Workers: {max_workers}\n")
        
        try:
            # Inicializar modelo
            init_time = self.initialize_model()
            results["init_time"] = init_time
            
            print(f"{Fore.CYAN}Procesando {total_images} imágenes con {max_workers} workers...")
            
            # Usar tqdm para mostrar una barra de progreso
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.process_image, img_path, question): img_path 
                           for img_path in image_files}
                
                # Barra de progreso con tqdm
                with tqdm(total=total_images, desc="Procesando") as progress_bar:
                    for future in concurrent.futures.as_completed(futures):
                        image_result = future.result()
                        results["images"].append(image_result)
                        
                        # Guardar resultado
                        self.guardar_resultado_imagen(image_result)
                        
                        # Actualizar barra de progreso
                        progress_bar.update(1)
                        
                        # Mostrar info mínima
                        img_name = os.path.basename(image_result['image_path'])
                        if "error" in image_result:
                            tqdm.write(f"{Fore.RED}Error: {img_name}")
                        else:
                            tqdm.write(f"{Fore.GREEN}OK: {img_name} ({image_result.get('total_time', 0):.2f}s)")
            
            # Agregar estadísticas finales
            total_time = time.time() - start_total
            results["total_time"] = total_time
            results["end_time"] = datetime.datetime.now().isoformat()
            
            # Guardar resumen final
            self._guardar_resumen(results)
            
            # Mostrar resumen compacto
            successful = sum(1 for img in results.get('images', []) if 'error' not in img)
            failed = len(results.get('images', [])) - successful
            print(f"\n{Fore.GREEN}Completado: {successful} OK, {failed} errores")
            print(f"{Fore.GREEN}Tiempo total: {results.get('total_time', 0):.2f} segundos")
            print(f"{Fore.GREEN}Resultados: {os.path.basename(self.output_file)}")
            
        except Exception as e:
            results["error"] = f"Error en el procesamiento: {str(e)}"
            logger.error(f"{Fore.RED}{results['error']}")
        finally:
            # Cerrar modelo
            self.close_model()
        
    def _guardar_resumen(self, results):
        """Guarda un resumen de los resultados."""
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write("\n" + "-" * 60 + "\n")
            f.write("RESUMEN FINAL DE PROCESAMIENTO\n")
            f.write("-" * 60 + "\n")
            f.write(f"Modelo: {os.path.basename(self.model_path)}\n")
            f.write(f"Timestamp finalización: {datetime.datetime.now().isoformat()}\n")
            
            # Estadísticas de imágenes
            total_images = len(results.get('images', []))
            successful_images = sum(1 for img in results.get('images', []) if 'error' not in img)
            failed_images = total_images - successful_images
            
            f.write(f"Total imágenes procesadas: {total_images}\n")
            f.write(f"Exitosas: {successful_images}, Errores: {failed_images}\n")
            
            # Tiempo total destacado
            f.write(f"Tiempo total de procesamiento: {results.get('total_time', 0):.2f} segundos\n")
            f.write(f"Tiempo de inicialización: {results.get('init_time', 0)::.2f} segundos\n")
            
            if successful_images > 0:
                # Calcular tiempos promedio de forma compacta
                avg_encode = sum(img.get('encode_time', 0) for img in results.get('images', []) 
                               if 'error' not in img) / successful_images
                avg_caption = sum(img.get('caption_time', 0) for img in results.get('images', []) 
                                if 'error' not in img) / successful_images
                avg_query = sum(img.get('query_time', 0) for img in results.get('images', []) 
                              if 'error' not in img) / successful_images
                avg_total = sum(img.get('total_time', 0) for img in results.get('images', []) 
                              if 'error' not in img) / successful_images
                
                f.write("\nTiempos promedio por imagen:\n")
                f.write(f"- Encode: {avg_encode:.4f}s | ")
                f.write(f"Caption: {avg_caption:.4f}s | ")
                f.write(f"Query: {avg_query:.4f}s | ")
                f.write(f"Total: {avg_total:.4f}s\n")
            
            f.write("-" * 60 + "\n")


def main():
    """Función principal que maneja los parámetros de línea de comandos."""
    # Banner simple
    print(f"\n{Fore.CYAN}TEST MOONDREAM")
    print(f"{Fore.CYAN}{'-' * 30}")
    
    parser = argparse.ArgumentParser(
        description="Procesador de imágenes con Moondream"
    )
    
    # Modo de procesamiento
    parser.add_argument("--mode", choices=["single", "batch"], default="single",
                      help="Modo: 'single' (una imagen) o 'batch' (todas)")
    
    # Tamaño del modelo
    parser.add_argument("--model", choices=["small", "big"], default="small",
                      help="Modelo: 'small' (0.5b) o 'big' (2b)")
    
    # Otros parámetros
    parser.add_argument("-q", "--question", type=str, default=DEFAULT_QUESTION,
                      help="Pregunta para las imágenes")
    parser.add_argument("-d", "--dir", type=str, default="imagenes_prueba",
                      help="Directorio de imágenes")
    parser.add_argument("-o", "--output", type=str, default=None,
                      help="Archivo de salida")
    parser.add_argument("-w", "--workers", type=int, default=None,
                      help="Número de workers (batch)")
    
    args = parser.parse_args()
    
    # Obtener la ruta del modelo
    model_path = AVAILABLE_MODELS[args.model]
    
    # Obtener las imágenes a procesar
    image_dir = os.path.abspath(args.dir)
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + 
                        glob.glob(os.path.join(image_dir, "*.jpeg")) + 
                        glob.glob(os.path.join(image_dir, "*.png")))
    
    if not image_files:
        print(f"{Fore.RED}Error: No se encontraron imágenes en {image_dir}")
        return
    
    # Crear el procesador de imágenes
    processor = MoondreamImageProcessor(model_path=model_path, output_file=args.output, mode=args.mode)
    
    if args.mode == "single":
        # Procesar solo la primera imagen
        image_path = image_files[0]
        print(f"{Fore.YELLOW}Modo: Single | Imagen: {os.path.basename(image_path)}")
        print(f"{Fore.YELLOW}Modelo: {args.model} | Pregunta: {args.question[:30]}...")
        
        processor.process_interactive(image_path, args.question)
    else:
        # Procesar todas las imágenes
        print(f"{Fore.YELLOW}Modo: Batch | Imágenes: {len(image_files)}")
        print(f"{Fore.YELLOW}Modelo: {args.model} | Workers: {args.workers or 'Auto'}")
        
        processor.process_batch(image_files, args.question, max_workers=args.workers)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Interrumpido por el usuario.")
    except Exception as e:
        print(f"\n{Fore.RED}Error: {str(e)}")
        raise
