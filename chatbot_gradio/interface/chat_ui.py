# -*- coding: utf-8 -*-
"""Modern Gradio chat interface for RAG chat system.

This module provides a clean, modern ChatInterface implementation using
Gradio Blocks with accordion panels and responsive design.
"""

import logging
from typing import Dict, List, Tuple

import gradio as gr

from core.rag_pipeline import RAGPipeline
from interface.render_context import render_sources, render_summary

logger = logging.getLogger(__name__)

# UI Configuration
APP_TITLE = "Chat RAG - Sistema de Consulta Avanzado"
APP_DESCRIPTION = """
🤖 **Sistema de Chat Inteligente con Tecnología RAG**

Haz preguntas y obtén respuestas respaldadas por fuentes documentales. 
El sistema utiliza técnicas avanzadas de recuperación para proporcionar respuestas precisas 
y contextuales basadas en tu base de conocimientos de documentos.
"""
CHAT_PLACEHOLDER = "Escribe tu pregunta aquí..."
MAX_INPUT_LENGTH = 2000
CHAT_HEIGHT = 600
SOURCES_HEIGHT = 320  # Reduced for more compact layout


def create_chat_interface(rag_pipeline: RAGPipeline) -> gr.Blocks:
    """Create the main chat interface using Gradio Blocks.
    
    Args:
        rag_pipeline: Configured RAG pipeline instance
        
    Returns:
        Configured Gradio Blocks interface
    """
    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate"
        ),
        fill_height=True,
        css="""
        .chat-container { min-height: 600px; }
        .sources-panel { 
            background: var(--background-fill-secondary); 
            border-radius: 8px; 
            padding: 8px; 
            border: 1px solid var(--border-color-primary);
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        
        /* Compact configuration section */
        .config-header {
            margin: 0 !important;
            padding: 2px 2px !important;
            font-size: 1em !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            min-height: auto;
        }
        .compact-slider {
            margin-top: 4px !important;
            margin-bottom: 8px !important;
        }
        .compact-slider label {
            font-size: 0.9em !important;
            margin-bottom: 2px !important;
        }
        .compact-slider .wrap {
            gap: 4px !important;
        }
        
        /* Optimized accordion styling */
        .sources-accordion {
            margin-top: 8px !important;
            margin-bottom: 0 !important;
        }
        .sources-accordion > .label-wrap {
            padding: 8px 12px !important;
            background: var(--background-fill-primary);
            border-radius: 6px;
            font-weight: 500;
        }
        .sources-accordion > .label-wrap:hover {
            background: var(--background-fill-secondary-dark);
        }
        
        /* Content area optimization */
        .accordion-content { 
            max-height: 450px; 
            overflow-y: auto; 
            line-height: 1.5;
            color: var(--body-text-color);
            padding: 4px !important;
            margin: 0 !important;
        }
        
        /* Summary styling */
        .sources-summary {
            margin-top: 8px !important;
            margin-bottom: 0 !important;
            padding: 4px !important;
        }
        
        /* Enhanced HTML content styling for dark theme compatibility */
        .sources-panel details {
            margin-bottom: 0.3em;
            margin-left: 0 !important;
            padding: 0 !important;
            border: 1px solid var(--border-color-accent);
            border-radius: 6px;
            background: var(--background-fill-primary);
        }
        .sources-panel summary {
            transition: background-color 0.2s ease;
            padding: 6px !important;
            margin: 0 !important;
            background: var(--background-fill-secondary);
            border-radius: 6px 6px 0 0;
            color: var(--body-text-color);
            font-weight: 500;
            font-size: 0.85em;
            border: none !important;
        }
        .sources-panel summary:hover {
            background: var(--background-fill-secondary-dark);
        }
        .sources-panel details[open] summary {
            border-bottom: none !important;
            border-radius: 6px 6px 0 0;
        }
        
        /* Improve text contrast in source content */
        .sources-panel .source-content {
            background: var(--background-fill-primary);
            color: var(--body-text-color);
            padding: 4px;
            margin-left: 0 !important;
            padding-left: 4px !important;
            border-radius: 0 0 4px 4px;
            font-size: 0.8em;
            line-height: 1.3;
        }
        
        /* Compact group styling */
        .sources-panel .group {
            padding: 6px !important;
            margin-bottom: 6px !important;
            background: var(--background-fill-primary);
            border-radius: 6px;
            border: 1px solid var(--border-color-primary);
        }
        
        /* Remove excessive margins from markdown elements */
        .sources-panel .markdown {
            margin: 0 !important;
        }
        .sources-panel .markdown h3,
        .sources-panel .markdown h4,
        .sources-panel .markdown h5 {
            margin-top: 0 !important;
            margin-bottom: 4px !important;
        }
        """
    ) as interface:
        
        # Header
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)
        
        # Main layout with responsive columns
        with gr.Row(equal_height=True):
            # Left column: Chat interface (2/3 width)
            with gr.Column(scale=2, elem_classes="chat-container"):
                chatbot = gr.Chatbot(
                    value=[],
                    height=CHAT_HEIGHT,
                    type="messages",
                    show_copy_button=True,
                    avatar_images=("👤", "🤖"),
                    bubble_full_width=False,
                    elem_id="main_chatbot"
                )
                
                # Input and controls
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder=CHAT_PLACEHOLDER,
                        lines=2,
                        max_lines=4,
                        show_label=False,
                        scale=4
                    )
                    submit_btn = gr.Button(
                        "Enviar", 
                        variant="primary", 
                        scale=1
                    )
                
                # Action buttons
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Limpiar", size="sm")
            
            # Right column: Sources and controls (1/3 width)
            with gr.Column(scale=1, elem_classes="sources-panel"):
                # Configuration panel (compact design)
                with gr.Group():
                    with gr.Row():
                        gr.Markdown("### ⚙️ Configuración", elem_classes="config-header")
                    max_sources_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Máximo de Fuentes",
                        info="Controla cuántas fuentes se consultan para cada pregunta",
                        interactive=True,
                        elem_classes="compact-slider"
                    )
                
                # Sources accordion with improved spacing
                with gr.Accordion("📄 Fuentes de Documentos", open=True, elem_classes="sources-accordion"):
                    sources_display = gr.HTML(
                        value="Aún no hay fuentes disponibles.",
                        elem_classes="accordion-content"
                    )
                    
                    sources_summary = gr.Markdown(
                        value="",
                        elem_id="sources_summary",
                        elem_classes="sources-summary"
                    )
        
        gr.Examples(
            examples=[
                ["¿Cuáles son los temas principales cubiertos en los documentos?"],
                ["¿Puedes explicar los conceptos clave mencionados?"],
                ["¿Qué metodología se discute en las fuentes?"],
                ["Resume los hallazgos más importantes."]
            ],
            inputs=chat_input
        )
        
        # Event handlers
        def process_message(message: str, history: List[Dict[str, str]], max_sources: int) -> Tuple:
            """Process user message and return updated components."""
            message = message.strip()
            if not message:
                return history, "", "Por favor, ingresa una pregunta.", ""
            
            try:
                # Validate and truncate input length
                if len(message) > MAX_INPUT_LENGTH:
                    message = message[:MAX_INPUT_LENGTH]
                    logger.warning(f"Input truncated to {MAX_INPUT_LENGTH} characters")
                
                # Add user message to history
                history.append({"role": "user", "content": message})
                
                # Get response from RAG pipeline with dynamic top_k
                response, sources = rag_pipeline.query(message, top_k=max_sources)
                
                # Add assistant response
                history.append({"role": "assistant", "content": response})
                
                # Format sources
                sources_text = render_sources(sources) if sources else "No se encontraron fuentes."
                summary_text = render_summary(sources) if sources else ""
                
                return history, "", sources_text, summary_text
                
            except Exception as e:
                error_msg = f"Error al procesar tu pregunta: {str(e)}"
                history.append({"role": "assistant", "content": error_msg})
                return (
                    history, 
                    "", 
                    "Error al recuperar fuentes.", 
                    ""
                )
        
        def clear_chat() -> Tuple:
            """Clear chat history and reset interface."""
            return [], "", "No hay fuentes disponibles.", ""
        
        # Wire up events
        submit_btn.click(
            fn=process_message,
            inputs=[chat_input, chatbot, max_sources_slider],
            outputs=[chatbot, chat_input, sources_display, sources_summary]
        )
        
        chat_input.submit(
            fn=process_message,
            inputs=[chat_input, chatbot, max_sources_slider],
            outputs=[chatbot, chat_input, sources_display, sources_summary]
        )
        
        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot, chat_input, sources_display, sources_summary]
        )
    
    return interface


def launch_ui(
    rag_pipeline: RAGPipeline,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False
) -> None:
    """Launch the Gradio chat interface.
    
    Args:
        rag_pipeline: Configured RAG pipeline instance
        server_name: Server host name
        server_port: Server port number
        share: Whether to create a public link
    """
    try:
        logger.info("Launching modern chat interface...")
        
        # Create the interface
        interface = create_chat_interface(rag_pipeline)
        
        # Launch
        logger.info(f"Starting server on {server_name}:{server_port}")
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            show_error=True,
            inbrowser=True
        )
        
    except Exception as e:
        logger.error(f"Failed to launch UI: {e}")
        raise


if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("Use the main application entry point instead.")
