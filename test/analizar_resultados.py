#!/usr/bin/env python3
"""
Script para analizar y visualizar los resultados de las pruebas de chunkers.

Uso:
    python -m test.analizar_resultados --dir <directorio_resultados> --out <directorio_analisis>

Este script procesa los archivos de resultados generados por las pruebas de chunkers,
genera gráficos comparativos y estadísticas, y los guarda en el directorio especificado.
"""

import os
import argparse
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Añadir el directorio raíz al path para permitir importaciones relativas
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

def extraer_metricas_archivo(archivo):
    """
    Extrae las métricas de un archivo de resultados.
    
    Args:
        archivo: Ruta al archivo de resultados
        
    Returns:
        Lista de diccionarios con las métricas de cada documento procesado
    """
    metricas = []
    config_por_metodo = {}
    
    with open(archivo, 'r', encoding='utf-8') as f:
        contenido = f.read()
    
    # Dividir por secciones de documentos
    secciones = re.split(r'={80}', contenido)
    
    # Primera sección contiene el nombre del método
    if len(secciones) > 1:
        metodo_match = re.search(r'MÉTODO DE CHUNKING: (\w+)', secciones[1])
        metodo = metodo_match.group(1).lower() if metodo_match else "desconocido"
        
        # Procesar cada sección de documento
        for i in range(1, len(secciones), 2):
            if i+1 < len(secciones):
                seccion = secciones[i] + secciones[i+1]
                
                # Extraer datos del documento
                doc_match = re.search(r'DOCUMENTO: (.+?)\n', seccion)
                documento = doc_match.group(1) if doc_match else "desconocido"
                
                # Extraer configuración detallada del chunker
                config = {}
                config_section = re.search(r'CONFIGURACIÓN DETALLADA DEL CHUNKER:\n-+\n(.*?)\n\nMÉTRICAS:', seccion, re.DOTALL)
                if config_section:
                    config_text = config_section.group(1)
                    for line in config_text.split('\n'):
                        if ': ' in line:
                            key, value = line.split(': ', 1)
                            if key.startswith('- '):
                                key = key[2:]
                            config[key] = value
                
                # Si no hay configuración detallada, buscar en la sección original
                if not config:
                    config_section = re.search(r'CONFIGURACIÓN DEL CHUNKER:\n-+\n(.*?)\n\n', seccion, re.DOTALL)
                    if config_section:
                        config_text = config_section.group(1)
                        for line in config_text.split('\n'):
                            if ': ' in line:
                                key, value = line.split(': ', 1)
                                if key.startswith('- '):
                                    key = key[2:]
                                config[key] = value
                
                # Guardar configuración por método
                if metodo not in config_por_metodo:
                    config_por_metodo[metodo] = config
                
                # Extraer métricas
                total_match = re.search(r'- Total de chunks: (\d+)', seccion)
                prom_match = re.search(r'- Longitud promedio: ([\d\.]+)', seccion)
                min_match = re.search(r'- Longitud mínima: (\d+)', seccion)
                max_match = re.search(r'- Longitud máxima: (\d+)', seccion)
                dim_match = re.search(r'- Dimensión de embedding: (\d+)', seccion)
                
                # Extraer info de cada chunk
                chunks_info = []
                chunk_sections = re.findall(r'CHUNK \d+:.*?-{40}.*?={80}', seccion, re.DOTALL)
                for chunk_text in chunk_sections:
                    header_match = re.search(r'- Header: (.*?)\n', chunk_text)
                    page_match = re.search(r'- Page: (.*?)\n', chunk_text)
                    length_match = re.search(r'- Longitud: (\d+) caracteres', chunk_text)
                    text_match = re.search(r'TEXTO COMPLETO:\n-{40}\n(.*?)\n={80}', chunk_text, re.DOTALL)
                    
                    if header_match and length_match:
                        chunks_info.append({
                            'header': header_match.group(1),
                            'page': page_match.group(1) if page_match else "",
                            'length': int(length_match.group(1)) if length_match else 0,
                            'text': text_match.group(1).strip() if text_match else ""
                        })
                
                metrica = {
                    'metodo': metodo,
                    'documento': Path(documento).name,
                    'total_chunks': int(total_match.group(1)) if total_match else 0,
                    'longitud_promedio': float(prom_match.group(1)) if prom_match else 0,
                    'longitud_minima': int(min_match.group(1)) if min_match else 0,
                    'longitud_maxima': int(max_match.group(1)) if max_match else 0,
                    'dimension_embedding': int(dim_match.group(1)) if dim_match else 0,
                    'configuracion': config,
                    'chunks': chunks_info
                }
                metricas.append(metrica)
    
    return metricas, config_por_metodo

def generar_informe_comparativo(directorio):
    """
    Genera un informe comparativo de los diferentes métodos de chunking.
    
    Args:
        directorio: Directorio con los archivos de resultados
        
    Returns:
        DataFrame con las métricas comparativas y diccionario con configuraciones
    """
    directorio_path = Path(directorio)
    # Buscar archivos de resultados con el nuevo patrón de nombres
    archivos_resultados = list(directorio_path.glob('*_*_results.txt'))
    
    if not archivos_resultados:
        # Si no encuentra archivos con el nuevo patrón, intentar con el antiguo
        archivos_resultados = list(directorio_path.glob('*_results.txt'))
        
    if not archivos_resultados:
        print(f"No se encontraron archivos de resultados en {directorio}")
        return None, {}, []
    
    # Extraer métricas de todos los archivos
    todas_metricas = []
    todas_configs = {}
    
    for archivo in archivos_resultados:
        # Extraer métricas sin pasar el segundo argumento incorrecto
        metricas, configs = extraer_metricas_archivo(archivo)
        todas_metricas.extend(metricas)
        todas_configs.update(configs)
    
    # Crear DataFrame
    if todas_metricas:
        # Extraer solo las métricas principales para el DataFrame
        df_metricas = []
        for metrica in todas_metricas:
            df_metricas.append({
                'metodo': metrica['metodo'],
                'documento': metrica['documento'],
                'total_chunks': metrica['total_chunks'],
                'longitud_promedio': metrica['longitud_promedio'],
                'longitud_minima': metrica['longitud_minima'],
                'longitud_maxima': metrica['longitud_maxima'],
                'dimension_embedding': metrica['dimension_embedding']
            })
        
        df = pd.DataFrame(df_metricas)
        return df, todas_configs, todas_metricas
    
    return None, {}, []

def generar_visualizaciones(df, directorio_salida):
    """
    Genera visualizaciones comparativas de los resultados.
    
    Args:
        df: DataFrame con las métricas
        directorio_salida: Directorio donde guardar las visualizaciones
    """
    if df is None or df.empty:
        print("No hay datos suficientes para generar visualizaciones")
        return
    
    os.makedirs(directorio_salida, exist_ok=True)
    
    # Agrupar por método y documento
    df_agrupado = df.groupby(['metodo', 'documento']).mean().reset_index()
    
    # 1. Gráfico de barras: Total de chunks por método y documento
    plt.figure(figsize=(12, 6))
    df_pivot = df_agrupado.pivot(index='documento', columns='metodo', values='total_chunks')
    df_pivot.plot(kind='bar')
    plt.title('Número de chunks por método y documento')
    plt.ylabel('Número de chunks')
    plt.xlabel('Documento')
    plt.tight_layout()
    plt.savefig(Path(directorio_salida) / 'total_chunks_por_metodo.png')
    
    # 2. Gráfico de barras: Longitud promedio por método
    plt.figure(figsize=(10, 6))
    df.groupby('metodo')['longitud_promedio'].mean().plot(kind='bar')
    plt.title('Longitud promedio de chunks por método')
    plt.ylabel('Longitud promedio (caracteres)')
    plt.tight_layout()
    plt.savefig(Path(directorio_salida) / 'longitud_promedio_por_metodo.png')
    
    # 3. Gráfico de dispersión: Relación entre número de chunks y longitud promedio
    plt.figure(figsize=(10, 6))
    for metodo in df['metodo'].unique():
        df_metodo = df[df['metodo'] == metodo]
        plt.scatter(df_metodo['total_chunks'], df_metodo['longitud_promedio'], label=metodo, alpha=0.7)
    
    plt.title('Relación entre número de chunks y longitud promedio')
    plt.xlabel('Número de chunks')
    plt.ylabel('Longitud promedio (caracteres)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(directorio_salida) / 'relacion_chunks_longitud.png')
    
    print(f"Visualizaciones guardadas en {directorio_salida}")

def generar_reporte_texto(df, configs, metricas_completas, ruta_salida):
    """
    Genera un reporte de texto con estadísticas comparativas.
    
    Args:
        df: DataFrame con las métricas
        configs: Diccionario con configuraciones por método
        metricas_completas: Lista con todas las métricas y detalles
        ruta_salida: Ruta donde guardar el reporte
    """
    if df is None or df.empty:
        print("No hay datos suficientes para generar el reporte")
        return
    
    with open(ruta_salida, 'w', encoding='utf-8') as f:
        f.write("REPORTE COMPARATIVO DE MÉTODOS DE CHUNKING\n")
        f.write("=" * 80 + "\n\n")
        
        # Escribir configuraciones por método
        f.write("CONFIGURACIONES POR MÉTODO\n")
        f.write("-" * 50 + "\n\n")
        
        for metodo, config in configs.items():
            f.write(f"Método: {metodo.upper()}\n")
            for key, value in config.items():
                f.write(f"  - {key}: {value}\n")
            f.write("\n")
        
        # Estadísticas generales por método
        f.write("\nESTADÍSTICAS POR MÉTODO\n")
        f.write("-" * 50 + "\n\n")
        
        for metodo in df['metodo'].unique():
            df_metodo = df[df['metodo'] == metodo]
            f.write(f"Método: {metodo.upper()}\n")
            f.write(f"- Documentos procesados: {df_metodo['documento'].nunique()}\n")
            f.write(f"- Total de chunks generados: {df_metodo['total_chunks'].sum()}\n")
            f.write(f"- Promedio de chunks por documento: {df_metodo['total_chunks'].mean():.2f}\n")
            f.write(f"- Longitud promedio de chunks: {df_metodo['longitud_promedio'].mean():.2f} caracteres\n")
            f.write(f"- Longitud mínima encontrada: {df_metodo['longitud_minima'].min()} caracteres\n")
            f.write(f"- Longitud máxima encontrada: {df_metodo['longitud_maxima'].max()} caracteres\n\n")
        
        # Comparativa por documento
        f.write("\nCOMPARATIVA POR DOCUMENTO\n")
        f.write("-" * 50 + "\n\n")
        
        for documento in df['documento'].unique():
            f.write(f"Documento: {documento}\n")
            f.write("-" * 40 + "\n")
            df_doc = df[df['documento'] == documento]
            
            for metodo in df_doc['metodo'].unique():
                df_metodo_doc = df_doc[df_doc['metodo'] == metodo]
                f.write(f"  • Método {metodo.upper()}:\n")
                f.write(f"    - Chunks generados: {df_metodo_doc['total_chunks'].values[0]}\n")
                f.write(f"    - Longitud promedio: {df_metodo_doc['longitud_promedio'].values[0]:.2f} caracteres\n")
                f.write(f"    - Rango de longitudes: {df_metodo_doc['longitud_minima'].values[0]} - {df_metodo_doc['longitud_maxima'].values[0]} caracteres\n")
                
                # Ejemplos de chunks (primeros 2 si hay)
                for metrica in metricas_completas:
                    if metrica['metodo'] == metodo and metrica['documento'] == documento:
                        if 'chunks' in metrica and metrica['chunks']:
                            f.write(f"    - Ejemplos de chunks:\n")
                            for i, chunk in enumerate(metrica['chunks'][:2]):
                                f.write(f"      * Chunk {i+1}:\n")
                                f.write(f"        Header: {chunk['header']}\n")
                                f.write(f"        Page: {chunk['page']}\n")
                                f.write(f"        Longitud: {chunk['length']} caracteres\n")
                        break
            
            f.write("\n")
        
        # Conclusiones
        f.write("\nCONCLUSIONES\n")
        f.write("-" * 50 + "\n\n")
        
        # Método con más chunks en promedio
        metodo_mas_chunks = df.groupby('metodo')['total_chunks'].mean().idxmax()
        promedio_mas_chunks = df.groupby('metodo')['total_chunks'].mean().max()
        
        # Método con chunks más largos en promedio
        metodo_chunks_largos = df.groupby('metodo')['longitud_promedio'].mean().idxmax()
        promedio_chunks_largos = df.groupby('metodo')['longitud_promedio'].mean().max()
        
        f.write(f"- El método {metodo_mas_chunks.upper()} genera más chunks en promedio ({promedio_mas_chunks:.2f} chunks por documento).\n")
        f.write(f"- El método {metodo_chunks_largos.upper()} genera chunks más largos en promedio ({promedio_chunks_largos:.2f} caracteres).\n")
        
        # Añadir observaciones sobre las configuraciones
        f.write("\nOBSERVACIONES SOBRE CONFIGURACIONES:\n")
        for metodo, config in configs.items():
            f.write(f"- Método {metodo.upper()}:\n")
            if metodo == 'character':
                chunk_size = config.get('chunk_size', 'N/A')
                chunk_overlap = config.get('chunk_overlap', 'N/A')
                f.write(f"  * Utiliza tamaño de chunk de {chunk_size} caracteres con solapamiento de {chunk_overlap} caracteres.\n")
            elif metodo == 'token':
                max_tokens = config.get('max_tokens', 'N/A')
                token_overlap = config.get('token_overlap', 'N/A')
                tokenizer = config.get('tokenizer', 'N/A')
                f.write(f"  * Utiliza {max_tokens} tokens máximos por chunk con solapamiento de {token_overlap} tokens.\n")
                f.write(f"  * Tokenizador: {tokenizer}\n")
            elif metodo == 'context':
                max_chunk_size = config.get('max_chunk_size', 'N/A')
                max_header_level = config.get('max_header_level', 'N/A')
                f.write(f"  * Tamaño máximo de chunk: {max_chunk_size} caracteres.\n")
                f.write(f"  * Nivel máximo de encabezados considerado: {max_header_level}\n")
    
    print(f"Reporte guardado en {ruta_salida}")

def main():
    parser = argparse.ArgumentParser(description="Analiza y compara resultados de diferentes chunkers")
    parser.add_argument("--dir", type=str, default="test/resultados_pruebas", 
                        help="Directorio con los archivos de resultados (default: test/resultados_pruebas)")
    parser.add_argument("--out", type=str, default="test/analisis_resultados",
                        help="Directorio para guardar los resultados del análisis (default: test/analisis_resultados)")
    
    args = parser.parse_args()
    
    # Convertir rutas relativas a absolutas si es necesario
    base_dir = Path(__file__).parent.parent
    results_dir = Path(args.dir)
    output_dir = Path(args.out)
    
    if not results_dir.is_absolute():
        results_dir = base_dir / results_dir
    
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar informe comparativo
    df, configs, metricas_completas = generar_informe_comparativo(results_dir)
    
    if df is not None:
        # Guardar datos procesados en CSV
        df.to_csv(output_dir / 'metricas_chunkers.csv', index=False)
        
        # Generar reporte de texto
        generar_reporte_texto(df, configs, metricas_completas, output_dir / 'reporte_comparativo.txt')
        
        # Generar visualizaciones
        generar_visualizaciones(df, output_dir)
        
        print(f"Análisis completado. Resultados guardados en {output_dir}")
    else:
        print("No se pudieron generar análisis debido a falta de datos.")

if __name__ == "__main__":
    main()
