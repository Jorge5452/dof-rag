# -*- coding: utf-8 -*-
"""Context rendering utilities for RAG chat system UI."""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Constants for formatting
FRAGMENT_CONTAINER_HEIGHT = 200  # Fixed height for scroll container
DOCUMENT_EMOJI = "📄"
FRAGMENT_BULLET = "▪"
INDENT = "  "

# UX Design constants for better visual hierarchy
HEADER_SIZE = "###"  # Smaller than default h2
SUBHEADER_SIZE = "####"  # Even smaller for sections
SOURCE_SEPARATOR = "---"

def render_sources(sources: List[Dict[str, Any]]) -> str:
    """Render sources into formatted HTML for UI display with improved UX.
    
    Args:
        sources: List of source documents with their fragments
        
    Returns:
        Formatted HTML string ready for display with collapsible sections
    """
    try:
        if not sources:
            return """
<div style="text-align: center; padding: 2em; color: rgba(107, 114, 126, 0.8); font-style: italic; background: rgba(55, 65, 81, 0.1); border-radius: 8px; border: 2px solid rgba(107, 114, 126, 0.3);">
    No se encontraron fuentes relevantes para esta consulta.
</div>
"""
        
        # Create main container with better styling
        html_content = """
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif; line-height: 1.5;">
"""
        
        # Add header with count
        html_content += f"""
<div style="margin-bottom: 6px; margin-left: 0; padding: 6px; background: rgba(59, 130, 246, 0.8); border-radius: 6px; border-left: 3px solid rgba(37, 99, 235, 0.9);">
    <h4 style="margin: 0; margin-left: 0; color: #ffffff; font-size: 0.9em; font-weight: 600;">
        📚 Fuentes Consultadas ({len(sources)} documentos)
    </h4>
</div>
"""
        
        # Add each document section
        for i, source in enumerate(sources, 1):
            document_section = _format_document_section(source, i)
            html_content += document_section
        
        html_content += """
</div>
"""
        
        return html_content
        
    except Exception as e:
        logger.error(f"Failed to render sources: {e}")
        return f"""
<div style="color: #ef4444; padding: 1em; border: 1px solid #ef4444; border-radius: 8px; background: rgba(239, 68, 68, 0.1);">
    <h4 style="margin: 0 0 8px 0; color: #ef4444;">Error al renderizar las fuentes</h4>
    <p style="margin: 0; font-size: 0.9em;">{str(e)}</p>
</div>
"""

def _format_document_section(
    source: Dict[str, Any], 
    index: int
) -> str:
    """Format a single document section with collapsible UX design.
    
    Args:
        source: Source document dictionary
        index: Document index for display
        
    Returns:
        Formatted document section with better UX
    """
    try:
        title = source.get("title", "Documento sin título")
        chunks = source.get("chunks", [])
        url = source.get("url", "")
        file_path = source.get("file_path", "")
        
        # Calculate document statistics
        num_chunks = len(chunks)
        avg_similarity = sum(chunk.get("similarity", 0) for chunk in chunks) / num_chunks if chunks else 0
        best_similarity = max(chunk.get("similarity", 0) for chunk in chunks) if chunks else 0
        
        # Create collapsible document section using HTML details
        document_html = f"""
<details open style="margin-bottom: 6px; margin-left: 0; border: 1px solid rgba(139, 92, 246, 0.6); border-radius: 6px; padding: 0; background: rgba(30, 41, 59, 0.05);">
<summary style="cursor: pointer; font-weight: 600; font-size: 0.9em; color: rgba(248, 250, 252, 0.9); padding: 6px; margin: 0; background: rgba(139, 92, 246, 0.2); border-radius: 6px 6px 0 0; border: none;">
    {DOCUMENT_EMOJI} {title}
    <span style="font-size: 0.75em; color: rgba(226, 232, 240, 0.8); font-weight: 400;">
        ({num_chunks} fragmentos • Mejor: {best_similarity:.1%} • Prom: {avg_similarity:.1%})
    </span>
</summary>

<div style="margin-top: 0; padding: 6px; margin-left: 0; border-left: 2px solid rgba(59, 130, 246, 0.7); background: rgba(30, 41, 59, 0.02); border-radius: 0 0 6px 6px;">
"""
        
        # Add metadata if available
        if url:
            document_html += f"""
<div style="margin-bottom: 4px; margin-left: 0; font-size: 0.8em; color: rgba(248, 250, 252, 0.8); background: rgba(59, 130, 246, 0.1); padding: 3px; border-radius: 3px;">
    🔗 <a href="{url}" target="_blank" style="color: rgba(96, 165, 250, 0.9); text-decoration: none; font-weight: 500;">{url}</a>
</div>"""
        elif file_path:
            document_html += f"""
<div style="margin-bottom: 4px; margin-left: 0; font-size: 0.8em; color: rgba(248, 250, 252, 0.8); font-weight: 500; background: rgba(59, 130, 246, 0.1); padding: 3px; border-radius: 3px;">
    📁 {file_path}
</div>"""
        
        # Add chunks as nested collapsible sections
        for i, chunk in enumerate(chunks, 1):
            chunk_html = _format_chunk_collapsible(chunk, i)
            document_html += chunk_html
        
        document_html += """
</div>
</details>
"""
        
        return document_html
        
    except Exception as e:
        logger.error(f"Failed to format document section: {e}")
        return f"""
<div style="color: #ffffff; padding: 8px; border: 2px solid rgba(239, 68, 68, 0.8); border-radius: 4px; background: rgba(239, 68, 68, 0.2);">
    {DOCUMENT_EMOJI} Error al formatear documento: {str(e)}
</div>
"""

def _format_chunk_collapsible(chunk: Dict[str, Any], index: int) -> str:
    """Format a single chunk as a collapsible section with better UX.
    
    Args:
        chunk: Chunk dictionary
        index: Chunk index for display
        
    Returns:
        Formatted chunk HTML with collapsible design
    """
    try:
        text = chunk.get("text", "")
        header = chunk.get("header", "")
        similarity = chunk.get("similarity", 0)
        
        # Process markdown formatting in text
        formatted_text = _process_markdown_to_html(text)
        
        # Format similarity score with visual indicators
        similarity_percentage = f"{similarity:.1%}" if similarity > 0 else "N/A"
        
        # Create similarity indicator with better colors for dark themes
        if similarity >= 0.8:
            similarity_color = "#10b981"  # Emerald green - good contrast
            similarity_icon = "🟢"
        elif similarity >= 0.6:
            similarity_color = "#f59e0b"  # Amber - good contrast
            similarity_icon = "🟡"
        else:
            similarity_color = "#ef4444"  # Red - good contrast
            similarity_icon = "🔴"
        
        # Create chunk label without character count
        chunk_title = f"Fragmento {index}"
        if header:
            chunk_title += f" - {header}"
        
        # Create collapsible chunk with better styling
        chunk_html = f"""
<details style="margin: 2px 0; margin-left: 0; border: 1px solid rgba(168, 85, 247, 0.5); border-radius: 4px; background: rgba(30, 41, 59, 0.02);">
<summary style="cursor: pointer; padding: 4px; margin-left: 0; font-weight: 500; font-size: 0.8em; color: rgba(248, 250, 252, 0.9); background: rgba(168, 85, 247, 0.12); border-radius: 3px 3px 0 0;">
    {FRAGMENT_BULLET} {chunk_title}
    <span style="float: right; font-size: 0.75em; color: {similarity_color}; font-weight: 600;">
        {similarity_icon} {similarity_percentage}
    </span>
</summary>

<div class="source-content" style="padding: 4px; margin-left: 0; background: rgba(248, 250, 252, 0.01); border-top: 1px solid rgba(168, 85, 247, 0.3); max-height: {FRAGMENT_CONTAINER_HEIGHT}px; overflow-y: auto; line-height: 1.4; font-size: 0.75em; border-radius: 0 0 3px 3px;">
    <div style="background: rgba(248, 250, 252, 0.03); padding: 4px; margin-left: 0; border-radius: 2px; color: rgba(248, 250, 252, 0.95); font-weight: 400; border: 1px solid rgba(148, 163, 184, 0.2); font-family: 'Segoe UI', system-ui, sans-serif;">
        {formatted_text}
    </div>
</div>
</details>
"""
        
        return chunk_html
        
    except Exception as e:
        logger.error(f"Failed to format chunk: {e}")
        return f"""
<div style="color: #ffffff; padding: 8px; margin: 4px 0; border: 2px solid rgba(239, 68, 68, 0.8); border-radius: 4px; font-size: 0.9em; background: rgba(239, 68, 68, 0.2);">
    {FRAGMENT_BULLET} Error al formatear fragmento: {str(e)}
</div>
"""

def render_summary(sources: List[Dict[str, Any]]) -> str:
    """Render a comprehensive summary of the sources found.
    
    Args:
        sources: List of source documents
        
    Returns:
        Enhanced summary text with detailed metrics
    """
    try:
        if not sources:
            return "❌ **No se encontraron fuentes relevantes.**"
        
        num_documents = len(sources)
        total_chunks = sum(len(source.get("chunks", [])) for source in sources)
        
        # Calculate similarity statistics
        all_similarities = []
        for source in sources:
            for chunk in source.get("chunks", []):
                similarity = chunk.get("similarity", 0)
                if similarity > 0:
                    all_similarities.append(similarity)
        
        if all_similarities:
            avg_similarity = sum(all_similarities) / len(all_similarities)
            max_similarity = max(all_similarities)
        else:
            avg_similarity = max_similarity = 0
        
        # Quality assessment
        if avg_similarity >= 0.8:
            quality_icon = "🟢"
            quality_text = "Excelente"
        elif avg_similarity >= 0.6:
            quality_icon = "🟡"
            quality_text = "Buena"
        elif avg_similarity >= 0.4:
            quality_icon = "🟠"
            quality_text = "Regular"
        else:
            quality_icon = "🔴"
            quality_text = "Baja"
        
        # Count high-quality chunks
        high_quality_chunks = sum(1 for sim in all_similarities if sim >= 0.7)
        
        # Create improved summary with better visual design
        summary = f"""
<div style="margin-top: 8px; padding: 8px; background: rgba(239, 68, 68, 0.8); border-radius: 6px; border: 1px solid rgba(220, 38, 127, 0.6);">
    <h5 style="margin: 0 0 6px 0; color: #ffffff; font-size: 0.85em; font-weight: 600;">
        📊 Resumen de Consulta
    </h5>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 0.75em; color: rgba(248, 250, 252, 0.95);">
        <div><strong>{num_documents}</strong> docs</div>
        <div><strong>{total_chunks}</strong> frags</div>
        <div>Calidad: <span style="font-weight: 600;">{quality_icon} {quality_text}</span></div>
        <div>Alta: <strong>{high_quality_chunks}</strong></div>
        <div>Máx: <span style="color: #10b981; font-weight: 600;">{max_similarity:.1%}</span></div>
        <div>Prom: <span style="color: #60a5fa; font-weight: 600;">{avg_similarity:.1%}</span></div>
    </div>
</div>
"""
        return summary
        
    except Exception as e:
        logger.error(f"Failed to render summary: {e}")
        return "❌ **Error al generar resumen de fuentes.**"

def _process_markdown_to_html(text: str) -> str:
    """Convert basic Markdown formatting to HTML for better display.
    
    Args:
        text: Text with Markdown formatting
        
    Returns:
        Text with basic HTML formatting
    """
    if not text:
        return text
    
    # Make a copy to work with
    processed_text = text
    
    # First escape HTML characters to prevent XSS
    processed_text = processed_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Process headers first (order matters: longer patterns first)
    # #### Header
    processed_text = re.sub(r'^#### (.*?)$', r'<h4>\1</h4>', processed_text, flags=re.MULTILINE)
    # ### Header  
    processed_text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', processed_text, flags=re.MULTILINE)
    # ## Header
    processed_text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', processed_text, flags=re.MULTILINE)
    # # Header
    processed_text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', processed_text, flags=re.MULTILINE)
    
    # Process bold text (**text** and __text__)
    processed_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', processed_text)
    processed_text = re.sub(r'__(.*?)__', r'<strong>\1</strong>', processed_text)
    
    # Process italic text (*text* and _text_) - but avoid conflicts with bold
    processed_text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<em>\1</em>', processed_text)
    processed_text = re.sub(r'(?<!_)_([^_]+?)_(?!_)', r'<em>\1</em>', processed_text)
    
    # Convert line breaks to HTML breaks
    processed_text = processed_text.replace('\n', '<br>')
    
    return processed_text