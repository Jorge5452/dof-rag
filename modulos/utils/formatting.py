"""
Console output formatting utilities.

This module contains functions to format and colorize the
RAG system outputs in the terminal.
"""

from typing import List
from colorama import init, Fore, Style, Back

# Initialize colorama to work across all platforms
init(autoreset=True)

# Colors and styles to improve visual experience
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

# Export constants for use in other modules
__all__ = [
    'C_TITLE', 'C_SUBTITLE', 'C_SUCCESS', 'C_ERROR', 'C_WARNING',
    'C_HIGHLIGHT', 'C_COMMAND', 'C_PARAM', 'C_INFO', 'C_VALUE',
    'C_PROMPT', 'C_RESET', 'C_SEPARATOR',
    'print_header', 'print_separator', 'print_status', 'print_formatted_response',
    'print_command_help', 'print_useful_commands'
]

def print_header(title: str) -> None:
    """
    Prints a formatted header.
    
    Args:
        title: Header title
    """
    print("\n" + Style.BRIGHT + "="*80)
    print(C_TITLE + title)
    print(Style.BRIGHT + "="*80)

def print_separator(char="─", width=80):
    """
    Prints a visual separator in the console
    
    Args:
        char: Character to use for the separator
        width: Width of the separator
    """
    print(C_SEPARATOR + char * width)

def print_status(status: str, message: str):
    """
    Prints a status message with appropriate formatting
    
    Args:
        status: Status type ("success", "error", "warning", "info")
        message: Message to display
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
    Prints a formatted response with a title.
    
    Args:
        title: Response title
        response: Response text
    """
    # Format the response to display response and context separately
    if "=======================  RESPONSE  =======================" in response:
        parts = response.split("=======================  RESPONSE  =======================")
        if len(parts) > 1:
            # Extract the response part (without header)
            response_text = parts[1].split("=======================  CONTEXT  =======================")[0].strip()
            context_text = response.split("=======================  CONTEXT  =======================")
            
            # Print only the response first
            print("\n" + C_TITLE + " RESPONSE " + C_RESET)
            print_separator()
            print(response_text)
            print_separator()
            
            # Print context if it exists and isn't empty
            if len(context_text) > 1 and context_text[1].strip():
                print("\n" + C_TITLE + " CONTEXT USED " + C_RESET)
                print_separator()
                context_content = context_text[1].strip()
                
                # Limit context length if very long
                if len(context_content) > 1500:
                    context_lines = context_content.split('\n')
                    # Show only the first significant lines
                    shortened = '\n'.join(context_lines[:20])
                    print(f"{shortened}\n\n{C_VALUE}[...additional context omitted...]{C_RESET}")
                else:
                    print(context_content)
                print_separator()
        else:
            # Fallback to original format
            print("\n" + C_TITLE + f" {title} " + C_RESET)
            print_separator()
            print(response)
            print_separator()
    else:
        # Fallback to original format
        print("\n" + C_TITLE + f" {title} " + C_RESET)
        print_separator()
        print(response)
        print_separator()

def print_command_help(commands: List[str]) -> None:
    """
    Prints command help.
    
    Args:
        commands: List of formatted commands
    """
    for command in commands:
        print(command)
    print(Style.BRIGHT + "=" * 80 + "\n")

def print_useful_commands() -> None:
    """
    Prints a list of useful commands.
    """
    print(Style.BRIGHT + "="*80)
    print(f"{C_SUBTITLE}USEFUL COMMANDS:")
    print("  • To query using the most recent database:")
    print(f"    {C_COMMAND}python run.py --query {C_PARAM}\"your question\"")
    print("  • To query using a specific database by index:")
    print(f"    {C_COMMAND}python run.py --query {C_PARAM}\"your question\" --db-index {C_PARAM}<number>")
    print("  • To see this list again:")
    print(f"    {C_COMMAND}python run.py --list-dbs")
    print("  • To show databases before asking:")
    print(f"    {C_COMMAND}python run.py --query {C_PARAM}\"your question\" --show-dbs")
    print("  • To optimize a specific database:")
    print(f"    {C_COMMAND}python run.py --optimize-db {C_PARAM}<number>")
    print("  • To optimize all databases:")
    print(f"    {C_COMMAND}python run.py --optimize-all")
    print("  • To view database statistics:")
    print(f"    {C_COMMAND}python run.py --db-stats")
    print("  • For interactive mode:")
    print(f"    {C_COMMAND}python run.py --query")
    print(Style.BRIGHT + "="*80 + "\n")