"""
Exportador de chunks a archivos de texto para visualización y análisis.

Este módulo permite exportar los chunks almacenados en la base de datos
a archivos de texto plano ubicados en la misma ruta que los archivos
Markdown originales. Incluye metadatos del documento y detalles de cada chunk.
"""

import os
import logging
import gc
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Configurar logging
logger = logging.getLogger(__name__)

class ChunkExporter:
    """
    Clase para exportar chunks de documentos a archivos de texto.
    
    Extrae información de chunks desde la base de datos y genera
    archivos de texto con formato legible para visualización y análisis.
    """
    
    def __init__(self, db_instance):
        """
        Inicializa el exportador con una instancia de base de datos.
        
        Args:
            db_instance: Instancia de base de datos vectorial
        """
        self.db = db_instance
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # Directorio base para exportaciones si no se especifica uno
        self.output_base_dir = "exported_chunks"
        # Opciones de formato
        self.include_separators = True
        self.include_chunk_number = True
    
    def export_document_chunks(self, document_path: str, output_path: Optional[str] = None) -> bool:
        """
        Exporta los chunks de un documento específico a un archivo de texto.
        
        Args:
            document_path: Ruta del documento Markdown original
            output_path: Ruta de salida para el archivo de texto. Si es None,
                         se usará document_path + ".txt"
                         
        Returns:
            bool: True si la exportación fue exitosa, False en caso contrario
        """
        try:
            # Normalizar la ruta para búsqueda en la base de datos
            normalized_path = os.path.normpath(document_path)
            
            # Buscar el documento en la base de datos
            document = self.find_document_by_path(normalized_path)
            if not document:
                self.logger.warning(f"No se encontró documento para: {document_path}")
                return False
            
            # Determinar ruta de salida si no se proporcionó
            if output_path is None:
                output_path = document_path + ".txt"
            
            self.logger.info(f"Exportando chunks del documento: {document_path} -> {output_path}")
            
            # Asegurar que el directorio de salida existe
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Obtener y procesar chunks por lotes para manejar documentos grandes
            offset = 0
            limit = 100  # Obtener chunks en lotes para documentos grandes
            total_chunks = 0
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Escribir encabezado y metadatos
                f.write(self.format_document_metadata(document))
                
                # Escribir chunks por lotes
                while True:
                    self.logger.debug(f"Obteniendo lote de chunks: offset={offset}, limit={limit}")
                    batch = self.db.get_chunks_by_document(document['id'], offset, limit)
                    
                    if not batch:
                        break
                    
                    total_chunks += len(batch)
                    
                    # Escribir los chunks de este lote
                    for i, chunk in enumerate(batch, offset + 1):
                        chunk_text = self.format_chunk(chunk, i)
                        f.write(chunk_text)
                    
                    # Liberar memoria
                    del batch
                    if offset % 500 == 0:  # Forzar GC periódicamente para documentos grandes
                        gc.collect()
                    
                    offset += limit
                
                # Escribir resumen al final
                f.write(f"\n\n================ RESUMEN ================\n")
                f.write(f"Total de chunks: {total_chunks}\n")
                f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            self.logger.info(f"Exportación completada: {output_path} con {total_chunks} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al exportar chunks para {document_path}: {e}")
            return False
    
    def find_document_by_path(self, document_path: str) -> Optional[Dict[str, Any]]:
        """
        Busca un documento en la base de datos por su ruta de archivo.
        
        Args:
            document_path: Ruta del archivo a buscar
            
        Returns:
            Dict con información del documento o None si no se encuentra
        """
        try:
            self.logger.debug(f"Buscando documento con ruta: {document_path}")
            
            # Ejecutar consulta en la base de datos
            # La mayoría de nuestras implementaciones tienen un cursor
            cursor = self.db._cursor
            
            # Buscar el documento por ruta exacta
            cursor.execute(
                "SELECT id, title, url, file_path, created_at FROM documents WHERE file_path = ?", 
                (document_path,)
            )
            doc = cursor.fetchone()
            
            if doc:
                # Convertir a diccionario si es una fila de SQLite
                if hasattr(doc, 'keys'):
                    return dict(doc)
                else:
                    # Crear diccionario manualmente
                    return {
                        'id': doc[0],
                        'title': doc[1],
                        'url': doc[2],
                        'file_path': doc[3],
                        'created_at': doc[4]
                    }
            else:
                # Intentar búsqueda alternativa - con coincidencia parcial de ruta
                cursor.execute(
                    "SELECT id, title, url, file_path, created_at FROM documents WHERE file_path LIKE ?", 
                    (f"%{os.path.basename(document_path)}%",)
                )
                doc = cursor.fetchone()
                
                if doc:
                    if hasattr(doc, 'keys'):
                        return dict(doc)
                    else:
                        return {
                            'id': doc[0],
                            'title': doc[1],
                            'url': doc[2],
                            'file_path': doc[3],
                            'created_at': doc[4]
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error al buscar documento: {e}")
            return None
    
    def format_document_metadata(self, document: Dict[str, Any]) -> str:
        """
        Formatea los metadatos del documento para el archivo de salida.
        
        Args:
            document: Diccionario con información del documento
            
        Returns:
            String formateado con los metadatos
        """
        lines = []
        lines.append("================ METADATOS DEL DOCUMENTO ================\n")
        lines.append(f"ID: {document.get('id', 'N/A')}")
        lines.append(f"Título: {document.get('title', 'Sin título')}")
        lines.append(f"Ruta: {document.get('file_path', 'N/A')}")
        
        # Formatear fecha si está disponible
        created_at = document.get('created_at')
        if created_at:
            # Intentar convertir a formato legible si es timestamp
            try:
                if isinstance(created_at, (int, float)):
                    created_at = datetime.fromtimestamp(created_at).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                pass
            lines.append(f"Fecha de procesamiento: {created_at}")
        
        lines.append("\n=================== CHUNKS GENERADOS ===================\n")
        
        return "\n".join(lines)
    
    def format_chunk(self, chunk: Dict[str, Any], chunk_num: int) -> str:
        """
        Formatea un chunk para el archivo de salida.
        
        Args:
            chunk: Diccionario con información del chunk
            chunk_num: Número secuencial del chunk
            
        Returns:
            String formateado con la información del chunk
        """
        lines = []
        lines.append(f"\n----- CHUNK #{chunk_num} (ID: {chunk.get('id', 'N/A')}) -----\n")
        
        # Información del chunk
        if 'page' in chunk and chunk['page']:
            lines.append(f"Página: {chunk['page']}")
        
        if 'header' in chunk and chunk['header']:
            lines.append(f"Encabezado: {chunk['header']}")
        
        # Contenido del chunk
        lines.append(f"\nContenido:\n{'-' * 50}")
        lines.append(chunk.get('text', ''))
        lines.append(f"{'-' * 50}\n")
        
        return "\n".join(lines)
    
    def export_all_chunks_from_db(self) -> Dict[str, bool]:
        """
        Exporta los chunks de todos los documentos en la base de datos.
        
        Returns:
            Diccionario con las rutas de documentos y si su exportación fue exitosa
        """
        try:
            # Obtener todos los documentos de la base de datos
            cursor = self.db._cursor
            cursor.execute("SELECT id, title, url, file_path, created_at FROM documents")
            
            documents = []
            for row in cursor.fetchall():
                if hasattr(row, 'keys'):
                    documents.append(dict(row))
                else:
                    documents.append({
                        'id': row[0],
                        'title': row[1],
                        'url': row[2],
                        'file_path': row[3],
                        'created_at': row[4]
                    })
            
            results = {}
            
            # Exportar chunks para cada documento
            for doc in documents:
                file_path = doc.get('file_path')
                if file_path:
                    result = self.export_document_chunks(file_path)
                    results[file_path] = result
                
                # Liberar recursos cada pocos documentos
                if len(results) % 10 == 0:
                    gc.collect()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error al exportar todos los chunks: {e}")
            return {}

    def export_chunk(self, chunk, filename_prefix, metadata=None):
        """
        Exporta un único chunk a un archivo de texto.
        
        Args:
            chunk: El chunk a exportar
            filename_prefix: Prefijo para el nombre del archivo
            metadata: Metadatos adicionales opcionales
            
        Returns:
            Ruta al archivo generado
        """
        try:
            # Asegurar que el directorio existe
            os.makedirs(self.output_base_dir, exist_ok=True)
            
            # Crear nombre de archivo
            filename = os.path.join(self.output_base_dir, f"{filename_prefix}_{chunk.id if hasattr(chunk, 'id') else 'unnamed'}.txt")
            
            with open(filename, 'w', encoding='utf-8') as f:
                # Escribir metadatos si existen
                if metadata:
                    f.write("================ METADATOS ================\n")
                    for key, value in metadata.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                # Escribir separador superior si está habilitado
                if self.include_separators:
                    f.write("="*40 + "\n")
                
                # Escribir número de chunk si está habilitado
                if self.include_chunk_number:
                    f.write(f"Chunk #{getattr(chunk, 'id', 1)}\n\n")
                
                # Escribir encabezado si existe
                if hasattr(chunk, 'header') and chunk.header:
                    f.write(f"Encabezado: {chunk.header}\n\n")
                
                # Escribir texto del chunk
                f.write(chunk.text)
                
                # Escribir separador inferior si está habilitado
                if self.include_separators:
                    f.write("\n" + "="*40)
            
            self.logger.info(f"Chunk exportado a: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error al exportar chunk: {e}")
            return None
    
    def export_chunks(self, chunks, base_filename, metadata=None):
        """
        Exporta múltiples chunks a archivos de texto.
        
        Args:
            chunks: Lista de chunks a exportar
            base_filename: Nombre base para los archivos
            metadata: Metadatos adicionales opcionales
            
        Returns:
            Lista de rutas a los archivos generados
        """
        filenames = []
        
        for i, chunk in enumerate(chunks):
            filename = self.export_chunk(
                chunk, 
                f"{base_filename}_{i+1}",
                metadata
            )
            if filename:
                filenames.append(filename)
        
        return filenames


def export_chunks_for_files(file_paths: str, db_instance) -> Dict[str, bool]:
    """
    Exporta los chunks para todos los archivos Markdown especificados.
    
    Args:
        file_paths: Ruta a un directorio o archivo individual
        db_instance: Instancia de base de datos vectorial
        
    Returns:
        Diccionario con las rutas procesadas y si su exportación fue exitosa
    """
    exporter = ChunkExporter(db_instance)
    results = {}
    
    try:
        if os.path.isdir(file_paths):
            # Recorrer recursivamente el directorio
            logger.info(f"Exportando chunks para todos los Markdown en: {file_paths}")
            
            for root, _, files in os.walk(file_paths):
                for file in files:
                    if file.lower().endswith('.md'):
                        md_path = os.path.join(root, file)
                        result = exporter.export_document_chunks(md_path)
                        results[md_path] = result
                
                # Liberar recursos periódicamente
                if len(results) % 10 == 0:
                    gc.collect()
                    
        elif os.path.isfile(file_paths) and file_paths.lower().endswith('.md'):
            # Exportar un solo archivo
            logger.info(f"Exportando chunks para archivo Markdown: {file_paths}")
            result = exporter.export_document_chunks(file_paths)
            results[file_paths] = result
        else:
            logger.warning(f"La ruta proporcionada no es un archivo Markdown o directorio válido: {file_paths}")
    
    except Exception as e:
        logger.error(f"Error al exportar chunks: {e}")
    
    # Liberar recursos
    del exporter
    gc.collect()
    
    # Ahora generar visualizaciones t-SNE para los mismos archivos
    try:
        from modulos.view_chunks.tsne_visualizer import visualize_tsne_for_files
        logger.info("Generando visualizaciones t-SNE para los archivos procesados...")
        visualize_tsne_for_files(file_paths, db_instance)
    except Exception as e:
        logger.error(f"Error al generar visualizaciones t-SNE: {e}")
    
    return results 