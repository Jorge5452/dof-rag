import fitz  # PyMuPDF se importa como 'fitz'
import os
import time

def pdf_to_markdown(pdf_path, output_dir):
    # Medir tiempo de inicio
    start_time = time.time()

    # Verificar que el archivo PDF existe
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"El archivo {pdf_path} no existe.")

    # Abrir el PDF con PyMuPDF
    doc = fitz.open(pdf_path)
    markdown_content = ""

    # Extraer texto de cada página y formatearlo como Markdown básico
    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text")  # Extrae texto plano
        # Añadir un encabezado por página
        markdown_content += f"## Página {page_num}\n\n"
        markdown_content += text + "\n\n"

    # Cerrar el documento
    doc.close()

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{base_filename}.md")

    # Guardar el contenido en un archivo Markdown
    with open(output_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_content)

    # Medir tiempo de finalización
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Conversión completada en {elapsed_time:.2f} segundos.")
    print(f"Archivo Markdown guardado en: {output_path}")
    return elapsed_time

if __name__ == "__main__":
    # Ruta del archivo PDF y directorio de salida
    pdf_path = "dof/2024/04/01042024-MAT.pdf"
    output_dir = "dof_markdown_pymupdf/2024/04/"

    try:
        pdf_to_markdown(pdf_path, output_dir)
    except Exception as e:
        print(f"Error: {e}")