"""
Utilidades para formateo de salidas en consola.

Este módulo contiene funciones para formatear y colorear las salidas
del sistema RAG en la terminal.
"""

import logging
from typing import List, Dict, Any
from colorama import init, Fore, Style, Back

# Inicializar colorama para que funcione en todas las plataformas
init(autoreset=True)

# Colores y estilos para mejorar la experiencia visual
C_TITLE = Back.BLUE + Fore.WHITE + Style.BRIGHT
C_SUBTITLE = Fore.BLUE + Style.BRIGHT
C_SUCCESS = Fore.GREEN + Style.BRIGHT
C_ERROR = Fore.RED + Style.BRIGHT
C_WARNING = Fore.YELLOW
C_HIGHLIGHT = Fore.MAGENTA + Style.BRIGHT
C_COMMAND = Fore.YELLOW
C_PARAM = Fore.GREEN
C_INFO = Style.RESET_ALL
C_VALUE = Fore.CYAN + Style.BRIGHT
C_PROMPT = Style.BRIGHT + Fore.GREEN
C_RESET = Style.RESET_ALL
C_SEPARATOR = Style.DIM + Fore.BLUE

# Exportar las constantes para uso en otros módulos
__all__ = [
    'C_TITLE', 'C_SUBTITLE', 'C_SUCCESS', 'C_ERROR', 'C_WARNING',
    'C_HIGHLIGHT', 'C_COMMAND', 'C_PARAM', 'C_INFO', 'C_VALUE',
    'C_PROMPT', 'C_RESET', 'C_SEPARATOR',
    'print_header', 'print_separator', 'print_status', 'print_formatted_response',
    'print_command_help', 'print_useful_commands'
]

def print_header(title: str) -> None:
    """
    Imprime un encabezado formateado.
    
    Args:
        title: Título del encabezado
    """
    print("\n" + Style.BRIGHT + "="*80)
    print(C_TITLE + title)
    print(Style.BRIGHT + "="*80)

def print_separator(char="─", width=80):
    """
    Imprime un separador visual en la consola
    
    Args:
        char: Carácter a usar para el separador
        width: Ancho del separador
    """
    print(C_SEPARATOR + char * width)

def print_status(status: str, message: str):
    """
    Imprime un mensaje de estado con formato adecuado
    
    Args:
        status: Tipo de estado ("success", "error", "warning", "info")
        message: Mensaje a mostrar
    """
    if status == "success":
        icon = f"{C_SUCCESS}✓{C_RESET}"
        status_color = C_SUCCESS
    elif status == "error":
        icon = f"{C_ERROR}✗{C_RESET}"
        status_color = C_ERROR
    elif status == "warning":
        icon = f"{C_WARNING}!{C_RESET}"
        status_color = C_WARNING
    elif status == "info":
        icon = f"{C_INFO}ℹ{C_RESET}"
        status_color = C_INFO
    else:
        icon = f"{C_INFO}•{C_RESET}"
        status_color = C_INFO
        
    print(f"{icon} {status_color}{message}{C_RESET}")

def print_formatted_response(title: str, response: str) -> None:
    """
    Imprime una respuesta formateada con un título.
    
    Args:
        title: Título de la respuesta
        response: Texto de la respuesta
    """
    # Formatear la respuesta para mostrar respuesta y contexto separados
    if "=======================  RESPUESTA  =======================" in response:
        parts = response.split("=======================  RESPUESTA  =======================")
        if len(parts) > 1:
            # Extraer la parte de respuesta (sin encabezado)
            response_text = parts[1].split("=======================  CONTEXTO  =======================")[0].strip()
            context_text = response.split("=======================  CONTEXTO  =======================")
            
            # Imprimir solo la respuesta primero
            print("\n" + C_TITLE + " RESPUESTA " + C_RESET)
            print_separator()
            print(response_text)
            print_separator()
            
            # Imprimir contexto si existe y no está vacío
            if len(context_text) > 1 and context_text[1].strip():
                print("\n" + C_TITLE + " CONTEXTO UTILIZADO " + C_RESET)
                print_separator()
                context_content = context_text[1].strip()
                
                # Limitar la longitud del contexto si es muy largo
                if len(context_content) > 1500:
                    context_lines = context_content.split('\n')
                    # Mostrar solo las primeras líneas significativas
                    shortened = '\n'.join(context_lines[:20])
                    print(f"{shortened}\n\n{C_VALUE}[...contexto adicional omitido...]{C_RESET}")
                else:
                    print(context_content)
                print_separator()
        else:
            # Fallback al formato original
            print("\n" + C_TITLE + f" {title} " + C_RESET)
            print_separator()
            print(response)
            print_separator()
    else:
        # Fallback al formato original
        print("\n" + C_TITLE + f" {title} " + C_RESET)
        print_separator()
        print(response)
        print_separator()

def print_command_help(commands: List[str]) -> None:
    """
    Imprime la ayuda de comandos.
    
    Args:
        commands: Lista de comandos con formato
    """
    for command in commands:
        print(command)
    print(Style.BRIGHT + "=" * 80 + "\n")

def print_useful_commands() -> None:
    """
    Imprime una lista de comandos útiles.
    """
    print(Style.BRIGHT + "="*80)
    print(f"{C_SUBTITLE}COMANDOS ÚTILES:")
    print("  • Para consultar usando la base de datos más reciente:")
    print(f"    {C_COMMAND}python run.py --query {C_PARAM}\"tu pregunta\"")
    print("  • Para consultar usando una base de datos específica por índice:")
    print(f"    {C_COMMAND}python run.py --query {C_PARAM}\"tu pregunta\" --db-index {C_PARAM}<número>")
    print("  • Para ver esta lista de nuevo:")
    print(f"    {C_COMMAND}python run.py --list-dbs")
    print("  • Para mostrar las bases de datos antes de preguntar:")
    print(f"    {C_COMMAND}python run.py --query {C_PARAM}\"tu pregunta\" --show-dbs")
    print("  • Para optimizar una base de datos específica:")
    print(f"    {C_COMMAND}python run.py --optimize-db {C_PARAM}<número>")
    print("  • Para optimizar todas las bases de datos:")
    print(f"    {C_COMMAND}python run.py --optimize-all")
    print("  • Para ver estadísticas de las bases de datos:")
    print(f"    {C_COMMAND}python run.py --db-stats")
    print("  • Para modo interactivo:")
    print(f"    {C_COMMAND}python run.py --query")
    print(Style.BRIGHT + "="*80 + "\n") 