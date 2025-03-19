import os
import time

class FileUtil:
    """
    Clase para iterar en directorios y procesar imágenes en lotes usando un cliente IA.

    Atributos:
        root_directory (str): Directorio raíz donde buscar las imágenes.
        client: Instancia del cliente (por ejemplo, GeminiClient) que procesa las imágenes.
        batch_size (int): Cantidad de imágenes a procesar en cada lote (por defecto: 10).
        cooling_period (int): Tiempo de espera en segundos entre cada lote (por defecto: 5).
    """

    # Extensiones de imagen válidas
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    def __init__(self, root_directory: str, client, batch_size: int = 10, cooling_period: int = 5):
        self.root_directory = root_directory
        self.client = client
        self.batch_size = batch_size
        self.cooling_period = cooling_period
        self.interrupt_signal = False

    def get_all_images(self) -> list:
        """
        Recorre recursivamente el directorio raíz y retorna una lista de rutas absolutas de archivos de imagen válidas.

        Returns:
            list: Lista de rutas absolutas de imágenes encontradas.
        """
        image_files = []
        for dirpath, _, filenames in os.walk(self.root_directory):
            for file in filenames:
                if file.lower().endswith(self.valid_extensions):
                    image_files.append(os.path.join(dirpath, file))
        return image_files
    
    def save_checkpoint(self, index: int, checkpoint_file="checkpoint.txt"):
        """Guarda el índice actual en un archivo de checkpoint."""
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                f.write(str(index))
            print(f"Checkpoint guardado en el índice {index}.")
        except Exception as e:
            print(f"Error al guardar checkpoint: {str(e)}")

    def process_images(self, force_overwrite: bool = False):
        """
        Procesa las imágenes encontradas en el directorio en lotes.
        Revisa si existe un checkpoint para reanudar el procesamiento.
        Se detiene el procesamiento si se activa la señal de interrupción, guardando un checkpoint.
        """
        images = self.get_all_images()
        total = len(images)
        print(f"Se encontraron {total} imágenes en '{self.root_directory}'.")
        
        # Leer checkpoint (si existe)
        checkpoint_file = "checkpoint.txt"
        start_index = 0
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    cp = int(f.read().strip())
                    start_index = cp
                    print(f"Reanudando desde el checkpoint en la imagen {start_index}.")
            except Exception as e:
                print("Error leyendo el checkpoint, iniciando desde el principio. Error:", str(e))
        
        for i in range(start_index, total, self.batch_size):
            batch = images[i:i+self.batch_size]
            print(f"Procesando lote {i // self.batch_size + 1} (imágenes: {len(batch)})...")
            for image_path in batch:
                # Revisar si se ha activado la señal de interrupción
                if self.interrupt_signal:
                    print("Procesamiento interrumpido. Guardando checkpoint...")
                    self.save_checkpoint(i, checkpoint_file)
                    self.interrupt_signal = False  # Reiniciar la señal
                    return
                result = self.client.process_imagen(image_path, force_overwrite=force_overwrite)
                if "error" in result:
                    print(f"Error en {image_path}: {result['error']}")
                else:
                    print(f"Procesada {image_path}: {result['status']}")
            if i + self.batch_size < total:
                print(f"Esperando {self.cooling_period} segundos antes del siguiente lote...")
                time.sleep(self.cooling_period)
        print("Procesamiento de imágenes completado.")
        # Si el procesamiento finaliza correctamente, se elimina el checkpoint.
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)


    def retry_failed_images(self, error_file="error_images.txt", force_overwrite: bool = True):
        """
        Reintenta el procesamiento de las imágenes cuyas rutas están registradas en 'error_images.txt'.
        """
        if not os.path.exists(error_file):
            print("No se encontró el archivo de imágenes fallidas.")
            return
        
        with open(error_file, 'r', encoding='utf-8') as f:
            failed_images = [line.strip() for line in f if line.strip()]
        
        print(f"Reintentando {len(failed_images)} imágenes fallidas...")
        for image_path in failed_images:
            result = self.client.process_imagen(image_path, force_overwrite=force_overwrite)
            if "error" in result:
                print(f"Reintento fallido en {image_path}: {result['error']}")
            else:
                print(f"Reintento exitoso en {image_path}: {result['status']}")
                
                
   
    def show_failed_images(self, error_file="error_images.txt"):
        """
        Muestra el contenido del archivo de imágenes fallidas.
        """
        if not os.path.exists(error_file):
            print("No se encontró el archivo de imágenes fallidas.")
            return
        
        with open(error_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print("=== Imágenes Fallidas ===")
        print(content)