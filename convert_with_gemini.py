import google.generativeai as genai
from pdf2image import convert_from_path
import pathlib
import os
import time

# Configurar la API de Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Ruta de Poppler (solo si no está en PATH)
poppler_path = r"C:\poppler-24.08.0\Library\bin"

# Definir rutas de entrada y salida
input_pdf = "dof/2024/03/18032024-MAT.pdf"
output_md = "dof_markdown_gemini/2024/03/18032024-MAT.md"

# Asegurar que la carpeta de salida exista
pathlib.Path(output_md).parent.mkdir(parents=True, exist_ok=True)

# Convertir el PDF a imágenes (con Poppler)
images = convert_from_path(input_pdf, poppler_path=poppler_path)

# Función para extraer Markdown usando Gemini
def extract_markdown_from_image(image):
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    response = model.generate_content(
        [image, "Extrae el contenido de esta página y conviértelo en Markdown."]
    )
    
    markdown = response.text if response and response.text else ""
    
    # Si Gemini se detiene, pedirle que continúe
    while response.candidates and not markdown:
        time.sleep(2)  # Espera 2 segundos antes de reintentar
        response = model.generate_content("Continúa por favor.")
        markdown = response.text if response and response.text else ""
    
    return markdown

# Extraer Markdown de cada página
markdown_pages = [extract_markdown_from_image(img) for img in images]

# Unir todas las páginas y guardar el archivo Markdown
final_markdown = "\n\n".join(markdown_pages)
pathlib.Path(output_md).write_text(final_markdown, encoding="utf-8")

print(f"✅ Conversión completada. Markdown guardado en: {output_md}")
