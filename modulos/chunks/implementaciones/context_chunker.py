import re
import logging
from typing import List, Dict, Any, Optional

from modulos.chunks.ChunkAbstract import ChunkAbstract
from config import config

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextChunker(ChunkAbstract):
    """
    Implementación de chunker basado en contexto.
    Divide el texto en chunks basados en la estructura semántica del contenido,
    utilizando encabezados y otras señales contextuales.
    """
    
    def __init__(self, embedding_model=None) -> None:
        """
        Constructor que inicializa el chunker con la configuración específica para chunking por contexto.
        
        Parámetros:
            embedding_model: Modelo de embeddings inicializado. Si es None, se debe asignar posteriormente.
        """
        super().__init__(embedding_model)
        
        # Obtener configuración específica para chunking por contexto
        self.context_config = self.chunks_config.get("context", {})
        
        # Parámetros de configuración con valores por defecto
        self.use_headers = self.context_config.get("use_headers", True)
        self.max_header_level = self.context_config.get("max_header_level", 3)
        self.max_chunk_size = self.context_config.get("max_chunk_size", 1500)
        
        logger.info(f"ContextChunker inicializado con max_header_level={self.max_header_level}, max_chunk_size={self.max_chunk_size}")
    
    def extract_headers(self, content: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Extrae los encabezados del contenido Markdown utilizando expresiones regulares.
        
        Parámetros:
            content: Contenido del archivo Markdown.
            **kwargs: Parámetros adicionales (opcional).
            
        Retorna:
            Lista de diccionarios con información de cada encabezado.
        """
        # Obtener parámetros de kwargs o usar valores por defecto
        max_header_level = kwargs.get("max_header_level", self.max_header_level)
        
        headers = []
        
        # Expresión regular para buscar encabezados en Markdown (# Título, ## Subtítulo, etc.)
        header_pattern = r'^(#{1,6})\s+(.+?)(?:\s+#+)?$'
        
        # Buscar encabezados en cada línea
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            level = len(match.group(1))  # Número de # determina el nivel
            
            # Verificar si el nivel está dentro del rango deseado
            if level <= max_header_level:
                header_text = match.group(2).strip()
                
                # Posición de inicio y fin en el contenido
                start_index = match.start()
                end_index = match.end()
                
                headers.append({
                    "header_text": header_text,
                    "level": level,
                    "start_index": start_index,
                    "end_index": end_index
                })
        
        logger.debug(f"Extraídos {len(headers)} encabezados")
        return headers
    
    def chunk(self, content: str, headers: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Divide el contenido en chunks basados en la estructura semántica y los encabezados.
        
        Parámetros:
            content: Contenido del archivo Markdown.
            headers: Lista de encabezados extraídos previamente.
            **kwargs: Parámetros adicionales (opcional).
            
        Retorna:
            Lista de diccionarios con los chunks generados.
        """
        # Obtener parámetros de kwargs o usar valores por defecto
        use_headers = kwargs.get("use_headers", self.use_headers)
        max_chunk_size = kwargs.get("max_chunk_size", self.max_chunk_size)
        
        chunks = []
        
        # Si no hay encabezados o no se utilizan, dividir por tamaño
        if not headers or not use_headers:
            # Dividir el contenido en párrafos
            paragraphs = re.split(r'\n\s*\n', content)
            
            current_chunk = ""
            current_page = 1
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Si añadir el párrafo excede el tamaño máximo, crear un nuevo chunk
                if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                    # Encontrar encabezado para la posición actual
                    start_pos = content.find(current_chunk)
                    header = self.find_header_for_position(start_pos, headers)
                    
                    chunks.append({
                        "text": current_chunk,
                        "header": header,
                        "page": str(current_page)
                    })
                    
                    current_chunk = paragraph
                    current_page += 1
                else:
                    # Añadir espacio si no es el primer párrafo del chunk
                    if current_chunk:
                        current_chunk += "\n\n"
                    current_chunk += paragraph
            
            # Añadir el último chunk si queda contenido
            if current_chunk:
                start_pos = content.find(current_chunk)
                header = self.find_header_for_position(start_pos, headers)
                
                chunks.append({
                    "text": current_chunk,
                    "header": header,
                    "page": str(current_page)
                })
            
            return chunks
        
        # Dividir por encabezados si se utilizan y existen
        # Ordenar los encabezados por posición
        ordered_headers = sorted(headers, key=lambda h: h["start_index"])
        
        # Añadir un último marcador de posición para facilitar la definición de los chunks
        if content:
            ordered_headers.append({
                "header_text": "",
                "level": 0,
                "start_index": len(content),
                "end_index": len(content)
            })
        
        # Si no hay encabezados ordenados, retornar un único chunk
        if not ordered_headers:
            chunks.append({
                "text": content,
                "header": "",
                "page": "1"
            })
            return chunks
        
        # Generar chunks basados en encabezados
        for i in range(len(ordered_headers) - 1):
            header = ordered_headers[i]
            next_header = ordered_headers[i + 1]
            
            # Determinar el texto del chunk desde el fin del encabezado actual hasta el inicio del siguiente
            chunk_start = header["end_index"]
            chunk_end = next_header["start_index"]
            chunk_text = content[chunk_start:chunk_end].strip()
            
            # Si el chunk no está vacío
            if chunk_text:
                # Construir el encabezado jerárquico
                hierarchical_header = self.build_hierarchical_header(header, ordered_headers[:i])
                
                # Asignar número de página (basado en número de encabezado)
                page_num = i + 1
                
                # Si el chunk es demasiado grande, subdividirlo
                if len(chunk_text) > max_chunk_size:
                    sub_chunks = self._subdivide_large_chunk(chunk_text, max_chunk_size)
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunks.append({
                            "text": sub_chunk,
                            "header": hierarchical_header,
                            "page": f"{page_num}.{j+1}"
                        })
                else:
                    chunks.append({
                        "text": chunk_text,
                        "header": hierarchical_header,
                        "page": str(page_num)
                    })
        
        logger.info(f"Generados {len(chunks)} chunks por contexto")
        return chunks
    
    def build_hierarchical_header(self, current_header: Dict[str, Any], previous_headers: List[Dict[str, Any]]) -> str:
        """
        Construye un encabezado jerárquico basado en el encabezado actual y los anteriores.
        
        Parámetros:
            current_header: Encabezado actual.
            previous_headers: Lista de encabezados anteriores.
            
        Retorna:
            String con el encabezado jerárquico.
        """
        # Nivel del encabezado actual
        current_level = current_header["level"]
        current_text = current_header["header_text"]
        
        # Buscar encabezados de nivel superior para construir la jerarquía
        relevant_headers = {}
        
        # Buscar entre los encabezados anteriores
        for header in previous_headers:
            level = header["level"]
            # Sólo considerar encabezados de nivel superior al actual
            if level < current_level:
                # Guardar el encabezado más reciente para cada nivel
                if level not in relevant_headers or header["start_index"] > relevant_headers[level]["start_index"]:
                    relevant_headers[level] = header
        
        # Construir la jerarquía
        hierarchy = []
        
        # Añadir encabezados de nivel superior en orden
        for level in sorted(relevant_headers.keys()):
            hierarchy.append(relevant_headers[level]["header_text"])
        
        # Añadir el encabezado actual
        hierarchy.append(current_text)
        
        # Unir la jerarquía
        return " > ".join(hierarchy)
    
    def _subdivide_large_chunk(self, text: str, max_size: int) -> List[str]:
        """
        Subdivide un chunk grande en sub-chunks más pequeños.
        
        Parámetros:
            text: Texto del chunk a subdividir.
            max_size: Tamaño máximo de cada sub-chunk.
            
        Retorna:
            Lista de sub-chunks.
        """
        sub_chunks = []
        
        # Dividir por párrafos
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Si el párrafo es demasiado grande por sí mismo, dividirlo por oraciones
            if len(paragraph) > max_size:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    if len(sentence) > max_size:
                        # Si una oración es muy larga, dividirla por tamaño
                        for i in range(0, len(sentence), max_size):
                            sub_chunks.append(sentence[i:i+max_size])
                    else:
                        # Si añadir la oración excede el tamaño máximo, crear un nuevo sub-chunk
                        if len(current_chunk) + len(sentence) > max_size and current_chunk:
                            sub_chunks.append(current_chunk)
                            current_chunk = sentence
                        else:
                            # Añadir espacio si no es la primera oración del chunk
                            if current_chunk:
                                current_chunk += " "
                            current_chunk += sentence
            else:
                # Si añadir el párrafo excede el tamaño máximo, crear un nuevo sub-chunk
                if len(current_chunk) + len(paragraph) > max_size and current_chunk:
                    sub_chunks.append(current_chunk)
                    current_chunk = paragraph
                else:
                    # Añadir espacio si no es el primer párrafo del chunk
                    if current_chunk:
                        current_chunk += "\n\n"
                    current_chunk += paragraph
        
        # Añadir el último sub-chunk si queda contenido
        if current_chunk:
            sub_chunks.append(current_chunk)
        
        return sub_chunks
