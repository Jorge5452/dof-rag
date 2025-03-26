import pymupdf4llm
import pathlib
import os

def pdf_to_markdown_llm(pdf_path, output_dir):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"El archivo {pdf_path} no existe.")

    # Extraer el contenido en Markdown con PyMuPDF4LLM
    md_text = pymupdf4llm.to_markdown(pdf_path)

    # Crear la carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Guardar el Markdown en un archivo
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{base_filename}.md")
    pathlib.Path(output_path).write_bytes(md_text.encode("utf-8"))

    print(f"Conversión completada. Archivo Markdown guardado en: {output_path}")
    return output_path

if __name__ == "__main__":
    pdf_path = "dof/2024/04/12042024-MAT.pdf"
    output_dir = "dof_markdown_pymupdf4llm/2024/04/"

    try:
        pdf_to_markdown_llm(pdf_path, output_dir)
    except Exception as e:
        print(f"Error: {e}")
