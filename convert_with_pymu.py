import fitz
import os
import time
from PIL import Image

def pdf_to_markdown(pdf_path, output_dir):
    start_time = time.time()
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"El archivo {pdf_path} no existe.")
    
    doc = fitz.open(pdf_path)
    markdown_content = "# Documento Convertido\n\n"
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    for page_num, page in enumerate(doc, 1):
        markdown_content += f"## Página {page_num}\n\n"
        
        text = page.get_text("text")
        if text.strip():
            markdown_content += text + "\n\n"
        
        for img_index, img in enumerate(page.get_images(full=True), 1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"page{page_num}_img{img_index}.{image_ext}"
            image_path = os.path.join(image_dir, image_filename)
            
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
            
            markdown_content += f"![Imagen {img_index}](images/{image_filename})\n\n"
    
    doc.close()
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{base_filename}.md")
    
    with open(output_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_content)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Conversión completada en {elapsed_time:.2f} segundos.")
    print(f"Archivo Markdown guardado en: {output_path}")
    return elapsed_time

if __name__ == "__main__":
    pdf_path = "dof/2024/04/12042024-MAT.pdf"
    output_dir = "dof_markdown_pymupdf/2024/04/"
    
    try:
        pdf_to_markdown(pdf_path, output_dir)
    except Exception as e:
        print(f"Error: {e}")