import ipywidgets as widgets
from IPython.display import display, HTML

class ProcessingControls:
    """Controles de interfaz para procesar imágenes con Gemini."""
    
    def __init__(self, client_instance, file_util_instance):
        """
        Inicializa la interfaz.
        
        Parameters:
            client_instance: Instancia de GeminiClient.
            file_util_instance: Instancia de FileUtil.
        """
        self.client = client_instance
        self.file_util = file_util_instance
        
        # Área de salida para mensajes y logs
        self.output_area = widgets.Output()
        
        # Crear widgets
        self._create_widgets()
        self._setup_callbacks()
    
    def _create_widgets(self):
        """Crea los widgets de la interfaz."""
        self.dir_input = widgets.Text(
            value=self.file_util.root_directory,
            placeholder='Ingrese la ruta del directorio',
            description='Directorio:',
            style={'description_width': 'initial'},
            layout={'width': '80%'}
        )
        self.batch_slider = widgets.IntSlider(
            value=self.file_util.batch_size,
            min=1,
            max=20,
            description='Tamaño de lote:',
            continuous_update=False
        )
        self.cooling_slider = widgets.IntSlider(
            value=self.file_util.cooling_period,
            min=1,
            max=30,
            description='Enfriamiento (seg):',
            continuous_update=False
        )
        self.process_button = widgets.Button(
            description='Procesar Imágenes',
            button_style='success',
            tooltip='Procesar imágenes'
        )
        self.interrupt_button = widgets.Button(
            description='Detener Procesamiento',
            button_style='danger',
            tooltip='Detener el procesamiento y guardar checkpoint'
        )
        self.resume_button = widgets.Button(
            description='Reanudar Procesamiento',
            button_style='info',
            tooltip='Reanudar procesamiento desde el checkpoint'
        )
        self.retry_button = widgets.Button(
            description='Reintentar Fallidos',
            button_style='warning',
            tooltip='Reintentar imágenes fallidas'
        )
        self.show_failed_button = widgets.Button(
            description='Ver Fallidos',
            button_style='',
            tooltip='Mostrar rutas de imágenes fallidas'
        )
    
    def _setup_callbacks(self):
        """Configura los callbacks de los widgets."""
        self.process_button.on_click(self._on_process_clicked)
        self.interrupt_button.on_click(self._on_interrupt_clicked)
        self.resume_button.on_click(self._on_resume_clicked)
        self.retry_button.on_click(self._on_retry_clicked)
        self.show_failed_button.on_click(self._on_show_failed_clicked)
    
    def _on_process_clicked(self, b):
        """Callback para iniciar el procesamiento de imágenes."""
        self.file_util.root_directory = self.dir_input.value.strip()
        self.file_util.batch_size = self.batch_slider.value
        self.file_util.cooling_period = self.cooling_slider.value
        
        with self.output_area:
            print("Iniciando procesamiento...")
            self.file_util.process_images(force_overwrite=False)
    
    def _on_interrupt_clicked(self, b):
        """Callback para detener el procesamiento (activa la señal de interrupción)."""
        self.file_util.interrupt_signal = True
        with self.output_area:
            print("Se ha solicitado detener el procesamiento. Se guardará el checkpoint.")
    
    def _on_resume_clicked(self, b):
        """Callback para reanudar el procesamiento (se lee el checkpoint, si existe)."""
        with self.output_area:
            print("Reanudando procesamiento desde el checkpoint (si existe)...")
            self.file_util.process_images(force_overwrite=False)
    
    def _on_retry_clicked(self, b):
        """Callback para reintentar el procesamiento de imágenes fallidas."""
        with self.output_area:
            print("Reintentando imágenes fallidas...")
            self.file_util.retry_failed_images()
    
    def _on_show_failed_clicked(self, b):
        """Callback para mostrar las rutas de imágenes fallidas."""
        with self.output_area:
            self.file_util.show_failed_images()
    
    def display(self):
        """Muestra la interfaz completa."""
        display(HTML("<h3>Procesamiento de Imágenes con Gemini</h3>"))
        display(self.dir_input)
        display(widgets.HBox([self.batch_slider, self.cooling_slider]))
        display(widgets.HBox([self.process_button, self.interrupt_button, self.resume_button]))
        display(widgets.HBox([self.retry_button, self.show_failed_button]))
        display(self.output_area)

def create_processing_interface(client_instance, file_util_instance):
    """
    Crea y muestra la interfaz de procesamiento.
    
    Parameters:
        client_instance: Instancia del cliente Gemini.
        file_util_instance: Instancia de FileUtil.
    
    Returns:
        ProcessingControls: Instancia de los controles.
    """
    controls = ProcessingControls(client_instance, file_util_instance)
    controls.display()
    return controls
