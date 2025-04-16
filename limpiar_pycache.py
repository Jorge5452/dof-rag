#!/usr/bin/env python3
# limpiar_pycache.py

import os
import shutil
from pathlib import Path

def limpiar_pycache(directorio_raiz="."):
    """
    Elimina todos los directorios __pycache__ recursivamente a partir del directorio raíz.
    
    Args:
        directorio_raiz (str): Ruta del directorio desde donde empezar la búsqueda.
    
    Returns:
        int: Número de directorios __pycache__ eliminados.
    """
    contador = 0
    directorio_raiz = Path(directorio_raiz)
    
    # Recorrer recursivamente todos los directorios
    for directorio_actual, subdirectorios, archivos in os.walk(directorio_raiz, topdown=False):
        for subdirectorio in subdirectorios:
            if subdirectorio == "__pycache__":
                ruta_pycache = Path(directorio_actual) / subdirectorio
                # Eliminar el directorio __pycache__ y su contenido
                shutil.rmtree(ruta_pycache)
                print(f"Eliminado: {ruta_pycache}")
                contador += 1
                
    return contador

if __name__ == "__main__":
    # Usar el directorio actual como raíz
    directorio_actual = Path.cwd()
    
    print(f"Buscando directorios __pycache__ en {directorio_actual}...")
    total_eliminados = limpiar_pycache(directorio_actual)
    
    if total_eliminados > 0:
        print(f"\nSe eliminaron {total_eliminados} directorios __pycache__.")
    else:
        print("\nNo se encontraron directorios __pycache__ para eliminar.")