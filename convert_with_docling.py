from docling.document_converter import DocumentConverter
import pathlib

# Definir rutas de entrada y salida
input_pdf = "dof/2024/03/18032024-MAT.pdf"
output_md = "dof_markdown_docling/2024/03/18032024-MAT.md"

# Asegurar que la ruta de salida exista
output_path = pathlib.Path(output_md)
output_path.parent.mkdir(parents=True, exist_ok=True)

# Crear instancia del convertidor
converter = DocumentConverter()

# Convertir el PDF a un objeto DoclingDocument
docling_doc = converter.convert(input_pdf)

# Extraer el contenido en formato Markdown
markdown_text = docling_doc.document.export_to_markdown()

# Guardar el contenido en un archivo Markdown
output_path.write_text(markdown_text, encoding="utf-8")

print(f"✅ Conversión completada. Markdown guardado en: {output_md}")
