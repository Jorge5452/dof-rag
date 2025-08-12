#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "duckdb",
#     "numpy",
#     "matplotlib",
#     "pandas",
#     "requests",
#     "python-dotenv",
#     "google-genai",
#     "sentence-transformers",
#     "tqdm",
#     "scikit-learn"
# ]
# ///

"""
compare_embeddings.py - OPTIMIZED VERSION

Compares embedding model performance across DuckDB databases.
Optimized for maintainability with reduced code complexity.
"""

import argparse
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import duckdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load environment
load_dotenv()

# High contrast color palette for t-SNE visualizations
HIGH_CONTRAST_COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',  # Primary colors
    '#FF8000', '#8000FF', '#FF0080', '#80FF00', '#0080FF', '#FFB000',  # Secondary colors  
    '#800000', '#008000', '#000080', '#808000', '#800080', '#008080',  # Dark colors
    '#FF4000', '#4000FF', '#FF0040', '#40FF00', '#0040FF', '#FF8040'   # Additional colors
]

# Global model colors - consistent across all visualizations
MODEL_COLORS = {
    'nomic': '#2ca02c',    # Green
    'gemini': '#1f77b4',   # Blue  
    'jina': '#ff7f0e'      # Orange
}

# Configuration - merged settings
CONFIG = {
    "similarity_metric": "coseno",
    "relevance_thresholds": {"high": 75.0, "medium": 50.0},
    "models": {
        "nomic": {"name": "nomic-ai/modernbert-embed-base", "dim": 768, "prefix": "search_query: "},
        "gemini": {"name": "gemini-embedding-001", "dim": 1536, "prefix": "RETRIEVAL_QUERY"},
        "jina": {"name": "jina-embeddings-v4", "dim": 1024, "url": "https://api.jina.ai/v1/embeddings"}
    },
    "files": {
        "db_folder": "dof_db",
        "results_folder": "results", 
        "report_name": "compare_embeddings_optimized.txt"
    }
}

class EmbeddingEncoder:
    """Unified embedding encoder for all models."""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.config = CONFIG["models"][model_type]
        self._model = None
        self._lock = threading.Lock()
        
    def encode(self, text: str) -> List[float]:
        """Encode text to embedding vector."""
        if not self._model:
            self._lazy_init()
            
        if self.model_type == "nomic":
            with self._lock:
                return self._model.encode(f"{self.config['prefix']}{text}").tolist()
        elif self.model_type == "gemini":
            return self._encode_gemini(text)
        elif self.model_type == "jina":
            return self._encode_jina(text)
            
    def _lazy_init(self):
        """Initialize model on first use."""
        if self.model_type == "nomic":
            self._model = SentenceTransformer(self.config["name"], trust_remote_code=True)
        elif self.model_type == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY required")
            self._model = genai.Client(api_key=api_key)
            
    def _encode_gemini(self, text: str) -> List[float]:
        """Encode using Gemini API with error handling."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                config = types.EmbedContentConfig(
                    output_dimensionality=self.config["dim"],
                    task_type=self.config["prefix"]
                )
                result = self._model.models.embed_content(
                    model=self.config["name"], contents=[text], config=config
                )
                return result.embeddings[0].values
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️  Gemini API error (intento {attempt + 1}/{max_retries}). Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Gemini API falló después de {max_retries} intentos: {e}")
                    raise
        
    def _encode_jina(self, text: str) -> List[float]:
        """Encode using Jina API with robust error handling and retries."""
        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            raise ValueError("JINA_API_KEY required")
        
        max_retries = 3
        base_timeout = 60  # Increased base timeout
        
        for attempt in range(max_retries):
            try:
                # Exponential backoff for timeout
                timeout = base_timeout * (2 ** attempt)
                
                response = requests.post(
                    self.config["url"],
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": self.config["name"],
                        "task": "retrieval.query", 
                        "dimensions": self.config["dim"],
                        "input": [text]
                    },
                    timeout=timeout
                )
                response.raise_for_status()
                
                embedding = response.json()["data"][0]["embedding"]
                arr = np.array(embedding, dtype=np.float32)
                norm = np.linalg.norm(arr)
                return (arr / norm if norm > 0 else arr).tolist()
                
            except requests.exceptions.ReadTimeout as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️  Jina API timeout (intento {attempt + 1}/{max_retries}). Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Jina API falló después de {max_retries} intentos: {e}")
                    raise
                    
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️  Error de conexión Jina API (intento {attempt + 1}/{max_retries}). Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Jina API conexión falló después de {max_retries} intentos: {e}")
                    raise
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [429, 502, 503, 504] and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️  Jina API error HTTP {e.response.status_code} (intento {attempt + 1}/{max_retries}). Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Jina API HTTP error: {e}")
                    raise
                    
            except Exception as e:
                print(f"❌ Jina API error inesperado: {e}")
                raise

class DatabaseClient:
    """Simplified database client for vector search."""
    
    def __init__(self, db_path: str, use_sql: bool = False):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
        self.conn = duckdb.connect(db_path)
        self.use_sql = use_sql
        
    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        if self.use_sql:
            return self._search_sql(query_embedding, top_k)
        return self._search_manual(query_embedding, top_k)
        
    def _search_sql(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search using DuckDB native functions."""
        dim = len(query_embedding)
        metric = CONFIG["similarity_metric"]
        
        if metric == "cosine":
            sql = f"""
            SELECT c.id, c.document_id, d.title, c.text, c.page_number, c.embedding,
                   array_cosine_similarity(c.embedding::FLOAT[{dim}], ?::FLOAT[{dim}]) AS similarity
            FROM chunks c LEFT JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
            ORDER BY similarity DESC LIMIT {top_k}
            """
        else:
            sql = f"""
            SELECT c.id, c.document_id, d.title, c.text, c.page_number, c.embedding,
                   array_distance(c.embedding::FLOAT[{dim}], ?::FLOAT[{dim}]) AS similarity
            FROM chunks c LEFT JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
            ORDER BY similarity ASC LIMIT {top_k}
            """
            
        results = []
        for row in self.conn.execute(sql, [query_embedding]).fetchall():
            chunk_id, doc_id, title, text, page, embedding, sim = row
            relevance = self._calc_relevance(sim)
            results.append({
                "chunk_id": chunk_id, "document_id": doc_id, "title": title,
                "text": text, "page_number": page, "embedding": embedding,
                "similarity": float(sim), "relevance": relevance,
                "relevance_category": self._categorize_relevance(relevance)
            })
        return results
        
    def _search_manual(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search using manual numpy calculations."""
        rows = self.conn.execute("""
            SELECT c.id, c.document_id, d.title, c.text, c.page_number, c.embedding
            FROM chunks c LEFT JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
        """).fetchall()
        
        if not rows:
            return []
            
        q_vec = np.asarray(query_embedding, dtype=np.float32)
        embeddings = np.stack([row[5] for row in rows]).astype(np.float32)
        
        if CONFIG["similarity_metric"] == "cosine":
            q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
            emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            similarities = emb_norms.dot(q_norm)
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        else:
            distances = np.linalg.norm(embeddings - q_vec, axis=1)
            top_indices = np.argpartition(distances, top_k)[:top_k]
            top_indices = top_indices[np.argsort(distances[top_indices])]
            
        results = []
        for idx in top_indices:
            chunk_id, doc_id, title, text, page, embedding = rows[idx]
            
            if CONFIG["similarity_metric"] == "cosine":
                sim_value = float(similarities[idx])
            else:
                sim_value = float(distances[idx])
            
            relevance = self._calc_relevance(sim_value)
            results.append({
                "chunk_id": chunk_id, "document_id": doc_id, "title": title,
                "text": text, "page_number": page, "embedding": embedding,
                "similarity": sim_value, "relevance": relevance,
                "relevance_category": self._categorize_relevance(relevance)
            })
        return results
        
    def _calc_relevance(self, similarity: float) -> float:
        """Convert similarity to relevance percentage."""
        if CONFIG["similarity_metric"] == "cosine":
            return np.clip(similarity * 100.0, 0.0, 100.0)
        else:
            relevance = 100.0 / (1.0 + similarity / 10.0)
            return np.clip(relevance, 0.0, 100.0)
            
    def _categorize_relevance(self, relevance: float) -> str:
        """Categorize relevance level."""
        thresholds = CONFIG["relevance_thresholds"]
        if relevance >= thresholds["high"]:
            return "Relevante"
        elif relevance >= thresholds["medium"]:
            return "Aceptable"
        return "Poco relevante"

class PerformanceTracker:
    """Simplified performance tracking with question embeddings storage."""
    
    def __init__(self):
        self.data = {}
        self.detailed_timings = []
        self.question_embeddings = {}  # Store question embeddings for t-SNE overlay
        
    def record(self, model: str, metric: str, value: float):
        """Record a metric value."""
        if model not in self.data:
            self.data[model] = {}
        if metric not in self.data[model]:
            self.data[model][metric] = []
        self.data[model][metric].append(value)
        
    def record_question_details(self, question_num: int, model: str, 
                              embedding_time: float, search_time: float, chunks_processed: int,
                              question_text: str = "", question_embedding: List[float] = None):
        """Record detailed timing information and question embeddings."""
        self.detailed_timings.append({
            "question_num": question_num, "model": model,
            "embedding_time": embedding_time, "search_time": search_time,
            "total_time": embedding_time + search_time,
            "chunks_processed": chunks_processed
        })
        
        # Store question embedding for t-SNE overlay
        if question_embedding is not None:
            key = (question_num, model)
            self.question_embeddings[key] = {
                "question_text": question_text,
                "embedding": question_embedding,
                "question_num": question_num
            }
        
    def get_summary(self) -> Dict:
        """Get performance summary with standard deviation and additional metrics."""
        summary = {}
        for model, metrics in self.data.items():
            if "embedding_time" in metrics and "search_time" in metrics:
                embed_times = metrics["embedding_time"]
                search_times = metrics["search_time"]
                total_times = [e + s for e, s in zip(embed_times, search_times)]
                
                # Get chunks processed info
                model_timings = [t for t in self.detailed_timings if t["model"] == model]
                total_chunks = sum(t["chunks_processed"] for t in model_timings) if model_timings else 0
                
                summary[model] = {
                    "embedding_avg": np.mean(embed_times),
                    "embedding_std": np.std(embed_times) if len(embed_times) > 1 else 0,
                    "search_avg": np.mean(search_times),
                    "search_std": np.std(search_times) if len(search_times) > 1 else 0,
                    "total_avg": np.mean(total_times),
                    "total_std": np.std(total_times) if len(total_times) > 1 else 0,
                    "queries_per_second": 1.0 / np.mean(total_times) if total_times else 0,
                    "chunks_per_second": total_chunks / sum(search_times) if search_times else 0,
                    "total_queries": len(embed_times),
                    "total_chunks_processed": total_chunks
                }
        return summary
    
    def get_question_embeddings_for_model(self, model: str) -> List[Dict]:
        """Get question embeddings for specific model for t-SNE overlay."""
        model_questions = []
        for (question_num, question_model), data in self.question_embeddings.items():
            if question_model == model:
                model_questions.append({
                    "question_num": question_num,
                    "question_text": data["question_text"], 
                    "embedding": data["embedding"]
                })
        # Sort by question number for consistent ordering
        return sorted(model_questions, key=lambda x: x["question_num"])

def generate_comprehensive_visualizations(performance_data: Dict, tracker: PerformanceTracker, output_dir: str, databases: Dict = None, use_sql: bool = False):
    """Generate comprehensive individual visualizations including t-SNE."""
    if not performance_data:
        print("⚠️  No hay datos de rendimiento disponibles")
        return
        
    print("📊 Generando visualizaciones...")
    
    # Add suffix based on SQL usage
    method_suffix = "_native" if use_sql else "_manual"
    
    # Create organized directory structure - FIXED: Don't duplicate 'visualizations'
    metrics_dir = os.path.join(output_dir, f"metrics{method_suffix}")
    tsne_dir = os.path.join(output_dir, f"tsne{method_suffix}")
    
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(tsne_dir, exist_ok=True)
    
    # Generate performance metrics visualizations
    generate_performance_comparison(performance_data, metrics_dir, method_suffix)
    generate_performance_boxplots(tracker, metrics_dir, method_suffix)
    generate_timing_scatter(tracker, metrics_dir, method_suffix)
    generate_timing_histograms(tracker, metrics_dir, method_suffix)
    generate_velocity_comparison(performance_data, metrics_dir, method_suffix)
    generate_throughput_comparison(performance_data, metrics_dir, method_suffix)
    
    # Generate t-SNE visualizations if databases available
    if databases:
        try:
            # Check if sklearn is available
            import importlib.util
            sklearn_spec = importlib.util.find_spec("sklearn")
            if sklearn_spec is not None:
                print("🔍 Generando visualizaciones t-SNE...")
                generate_enhanced_tsne_visualizations(tracker, tsne_dir, databases, method_suffix)
            else:
                print("⚠️  Visualizaciones t-SNE deshabilitadas - sklearn no disponible")
        except ImportError:
            print("⚠️  Visualizaciones t-SNE deshabilitadas - sklearn no disponible")
    else:
        print("⚠️  t-SNE saltado - bases de datos no disponibles")
    
    print(f"✅ Visualizaciones guardadas en: {output_dir} (método: {'nativo' if use_sql else 'manual'})")

def generate_performance_comparison(performance_data: Dict, output_dir: str, method_suffix=""):
    """Generate grouped bar chart comparing 3 models across metrics."""
    print("Generando gráfica de comparación de rendimiento...")
    
    if not performance_data:
        print("No hay datos de rendimiento disponibles")
        return
    
    # Extract data
    models = list(performance_data.keys())
    metrics = ['embedding_avg', 'search_avg', 'total_avg']
    metric_labels = ['Embedding', 'Search', 'Total']
    
    # Convert to milliseconds for better readability
    data = {}
    for model in models:
        data[model] = []
        for metric in metrics:
            value = performance_data[model].get(metric, 0) * 1000  # Convert to ms
            data[model].append(value)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Bar positions
    x = np.arange(len(metric_labels))
    width = 0.25
    
    # Create bars for each model
    for i, model in enumerate(models):
        color = MODEL_COLORS.get(model, '#gray')
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data[model], width, 
                     label=model.upper(), color=color, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    # Customize plot
    ax.set_xlabel('Métricas de Rendimiento', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tiempo (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Rendimiento por Modelo', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'performance_comparison{method_suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_performance_boxplots(tracker: PerformanceTracker, output_dir: str, method_suffix=""):
    """Generate box plots showing distribution of times across models."""
    print("Generando box plots de distribución de tiempos...")
    
    detailed_timings = tracker.detailed_timings
    if not detailed_timings:
        print("No hay datos detallados de timing disponibles")
        return
    
    # Organize data by model and metric
    models = ['nomic', 'gemini', 'jina']
    metrics = ['embedding_time', 'search_time', 'total_time']
    metric_labels = ['Embedding Time', 'Search Time', 'Total Time']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        # Prepare data for each model
        box_data = []
        box_labels = []
        box_colors = []
        
        for model in models:
            model_data = [t[metric] * 1000 for t in detailed_timings if t['model'] == model]  # Convert to ms
            if model_data:
                box_data.append(model_data)
                box_labels.append(model.upper())
                box_colors.append(MODEL_COLORS.get(model, '#gray'))
        
        # Create box plot
        if box_data:
            box_plot = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(box_plot['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        # Customize subplot
        ax.set_title(f'{label} Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tiempo (ms)', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics annotations
        if box_data:
            for i, data in enumerate(box_data):
                median_val = np.median(data)
                ax.text(i+1, median_val, f'{median_val:.1f}', 
                       ha='center', va='bottom', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.suptitle('Distribución de Tiempos por Modelo y Métrica', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'performance_boxplots{method_suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Box plots guardados: {output_path}")

def generate_timing_scatter(tracker: PerformanceTracker, output_dir: str, method_suffix=""):
    """Generate scatter plot: Embedding Time vs Search Time with trend lines."""
    print("Generando scatter plot de tiempos...")
    
    detailed_timings = tracker.detailed_timings
    if not detailed_timings:
        print("No hay datos detallados de timing disponibles")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    models = ['nomic', 'gemini', 'jina']
    
    for model in models:
        # Extract data for this model
        model_data = [t for t in detailed_timings if t['model'] == model]
        if not model_data:
            continue
            
        embedding_times = [t['embedding_time'] * 1000 for t in model_data]  # Convert to ms
        search_times = [t['search_time'] * 1000 for t in model_data]
        
        color = MODEL_COLORS.get(model, '#gray')
        
        # Create scatter plot
        ax.scatter(embedding_times, search_times, 
                  color=color, alpha=0.7, s=60, 
                  label=f'{model.upper()} (n={len(model_data)})')
        
        # Add trend line if we have enough points
        if len(embedding_times) >= 2:
            try:
                # Calculate linear regression
                z = np.polyfit(embedding_times, search_times, 1)
                p = np.poly1d(z)
                
                # Generate points for trend line
                x_trend = np.linspace(min(embedding_times), max(embedding_times), 100)
                y_trend = p(x_trend)
                
                ax.plot(x_trend, y_trend, color=color, linestyle='--', alpha=0.8, linewidth=2)
                
                # Add correlation coefficient
                corr = np.corrcoef(embedding_times, search_times)[0, 1]
                ax.text(0.02, 0.98 - models.index(model) * 0.05, 
                       f'{model.upper()}: r={corr:.3f}',
                       transform=ax.transAxes, color=color, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
            except Exception as e:
                print(f"Error calculando tendencia para {model}: {e}")
    
    # Customize plot
    ax.set_xlabel('Embedding Time (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Search Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Embedding Time vs Search Time\n(con líneas de tendencia)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'timing_scatter{method_suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scatter plot guardado: {output_path}")

def generate_timing_histograms(tracker: PerformanceTracker, output_dir: str, method_suffix=""):
    """Generate histograms showing timing distribution per question by model."""
    print("Generando histogramas de distribución por pregunta...")
    
    detailed_timings = tracker.detailed_timings
    if not detailed_timings:
        print("No hay datos detallados de timing disponibles")
        return
    
    models = ['nomic', 'gemini', 'jina']
    
    fig, axes = plt.subplots(1, len(models), figsize=(18, 6))
    if len(models) == 1:
        axes = [axes]
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        
        # Extract total times for this model (convert to ms)
        model_data = [t for t in detailed_timings if t['model'] == model]
        if not model_data:
            ax.text(0.5, 0.5, f'No data for {model.upper()}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{model.upper()} - Sin datos', fontsize=12, fontweight='bold')
            continue
            
        total_times = [t['total_time'] * 1000 for t in model_data]  # Convert to ms
        
        # Create histogram
        color = MODEL_COLORS.get(model, '#gray')
        n, bins, patches = ax.hist(total_times, bins=min(10, len(total_times)), 
                                  color=color, alpha=0.7, edgecolor='black')
        
        # Add statistics
        mean_time = np.mean(total_times)
        std_time = np.std(total_times)
        
        # Add vertical line for mean
        ax.axvline(mean_time, color='red', linestyle='--', linewidth=2, 
                  label=f'Media: {mean_time:.1f}ms')
        
        # Customize subplot
        ax.set_title(f'{model.upper()} - Distribución de Tiempos\n'
                    f'({len(model_data)} consultas)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Tiempo Total (ms)', fontsize=10)
        ax.set_ylabel('Frecuencia', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f'Media: {mean_time:.1f}ms\nDesv: {std_time:.1f}ms\nMin: {min(total_times):.1f}ms\nMax: {max(total_times):.1f}ms'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
               fontsize=9)
    
    plt.suptitle('Histogramas de Distribución de Tiempos por Modelo\n(Variabilidad entre preguntas)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'timing_histograms{method_suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histogramas guardados: {output_path}")

def generate_velocity_comparison(performance_data: Dict, output_dir: str, method_suffix=""):
    """Generate velocity comparison chart (queries/second) with values on bars."""
    print("Generando gráfica de velocidad (queries/second)...")
    
    if not performance_data:
        print("No hay datos de rendimiento disponibles")
        return
    
    # Extract velocity data
    models = list(performance_data.keys())
    velocities = []
    colors = []
    
    for model in models:
        velocity = performance_data[model].get('queries_per_second', 0)
        velocities.append(velocity)
        colors.append(MODEL_COLORS.get(model, '#gray'))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create bars
    bars = ax.bar(range(len(models)), velocities, color=colors, alpha=0.8, 
                 edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, velocity) in enumerate(zip(bars, velocities)):
        height = bar.get_height()
        ax.annotate(f'{velocity:.2f}', 
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Modelos de Embedding', fontsize=12, fontweight='bold')
    ax.set_ylabel('Velocidad (consultas/segundo)', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Velocidad de Procesamiento\n(Queries por Segundo)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([model.upper() for model in models])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add performance ranking
    velocity_ranking = sorted(zip(models, velocities), key=lambda x: x[1], reverse=True)
    ranking_text = "Ranking de Velocidad:\n"
    for i, (model, vel) in enumerate(velocity_ranking):
        ranking_text += f"{i+1}. {model.upper()}: {vel:.2f} q/s\n"
    
    ax.text(0.02, 0.98, ranking_text, transform=ax.transAxes, 
           verticalalignment='top', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
           fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'velocity_comparison{method_suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_throughput_comparison(performance_data: Dict, output_dir: str, method_suffix=""):
    """Generate throughput comparison chart (chunks processed/second) with efficiency metrics."""
    print("Generando gráfica de throughput (chunks/second)...")
    
    if not performance_data:
        print("No hay datos de rendimiento disponibles")
        return
    
    # Extract throughput data
    models = list(performance_data.keys())
    throughputs = []
    chunks_per_ms = []
    colors = []
    
    for model in models:
        throughput = performance_data[model].get('chunks_per_second', 0)
        throughputs.append(throughput)
        
        # Calculate chunks per millisecond for efficiency
        search_avg = performance_data[model].get('search_avg', 1)  # Avoid division by zero
        total_chunks = performance_data[model].get('total_chunks_processed', 0)
        total_queries = performance_data[model].get('total_queries', 1)
        chunks_ms = (total_chunks / total_queries) / (search_avg * 1000) if search_avg > 0 and total_queries > 0 else 0
        chunks_per_ms.append(chunks_ms)
        
        colors.append(MODEL_COLORS.get(model, '#gray'))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Subplot 1: Throughput (chunks/second)
    bars1 = ax1.bar(range(len(models)), throughputs, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, throughput in zip(bars1, throughputs):
        height = bar.get_height()
        ax1.annotate(f'{throughput:.1f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('Modelos de Embedding', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Throughput (chunks/segundo)', fontsize=12, fontweight='bold')
    ax1.set_title('Throughput de Búsqueda\n(Chunks procesados por segundo)', 
                 fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([model.upper() for model in models])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: Efficiency (chunks/ms)
    bars2 = ax2.bar(range(len(models)), chunks_per_ms, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, efficiency in zip(bars2, chunks_per_ms):
        height = bar.get_height()
        ax2.annotate(f'{efficiency:.3f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Modelos de Embedding', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Eficiencia (chunks/ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Eficiencia de Búsqueda\n(Chunks por milisegundo)', 
                 fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([model.upper() for model in models])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add combined ranking
    throughput_ranking = sorted(zip(models, throughputs), key=lambda x: x[1], reverse=True)
    ranking_text = "Ranking Throughput:\n"
    for i, (model, thr) in enumerate(throughput_ranking):
        ranking_text += f"{i+1}. {model.upper()}: {thr:.1f} ch/s\n"
    
    fig.text(0.02, 0.98, ranking_text, transform=fig.transFigure, 
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
            fontsize=10)
    
    plt.suptitle('Comparación de Throughput y Eficiencia de Búsqueda', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'throughput_comparison{method_suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_enhanced_tsne_visualizations(tracker: PerformanceTracker, output_dir: str, databases: Dict, method_suffix=""):
    """Generate enhanced t-SNE visualizations with high contrast colors and question overlay."""
    import os
    # Silence joblib warnings by setting environment variable
    os.environ['LOKY_MAX_CPU_COUNT'] = '8'
    
    models_processed = 0
    
    # Generate for each model
    for model_name, db_client in databases.items():
        try:
            # Get sample of chunks from database (limit for performance)
            chunks_query = """
            SELECT c.id, c.document_id, d.title, c.text, c.page_number, c.embedding
            FROM chunks c
            LEFT JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
            LIMIT 1000
            """
            
            rows = db_client.conn.execute(chunks_query).fetchall()
            if not rows:
                continue
                
            # Extract embeddings and metadata
            embeddings = []
            doc_ids = []
            for row in rows:
                chunk_id, doc_id, title, text, page, embedding = row
                if embedding is not None:
                    embeddings.append(np.array(embedding, dtype=np.float32))
                    doc_ids.append(str(doc_id) if doc_id else "unknown")
            
            if len(embeddings) < 2:
                continue
                
            # Stack embeddings for t-SNE
            X = np.vstack(embeddings)
            
            # Apply PCA for dimensionality reduction if needed
            if X.shape[1] > 50:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=50, random_state=42)
                X_pca = pca.fit_transform(X)
            else:
                X_pca = X
            
            # Apply t-SNE (suppress verbose output)
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings)-1), 
                       random_state=42, verbose=0)
            X_2d = tsne.fit_transform(X_pca)
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            
            # Generate chunks-only visualization
            _generate_chunks_tsne(model_name, X_2d, doc_ids, MODEL_COLORS, output_dir, method_suffix)
            
            # Generate overlay visualization with questions
            question_data = tracker.get_question_embeddings_for_model(model_name)
            if question_data:
                _generate_overlay_tsne(model_name, X, X_pca, question_data, output_dir, method_suffix)
            
            models_processed += 1
            
        except Exception as e:
            print(f"⚠️ Error procesando {model_name}: {e}")
            continue
    
    print(f"🎯 t-SNE completado: {models_processed} modelos procesados")

def _generate_chunks_tsne(model_name: str, X_2d: np.ndarray, doc_ids: List[str], 
                         model_colors: Dict, output_dir: str, method_suffix=""):
    """Generate chunks-only t-SNE visualization."""
    plt.figure(figsize=(12, 10))
    
    # Color by document if available
    unique_docs = np.unique(doc_ids)
    if len(unique_docs) > 1:
        # Use high contrast colors for multiple documents
        colors = [HIGH_CONTRAST_COLORS[i % len(HIGH_CONTRAST_COLORS)] for i in range(len(unique_docs))]
        for idx, doc in enumerate(unique_docs):
            mask = np.array(doc_ids) == doc
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       alpha=0.6, s=15, color=colors[idx], 
                       label=f"Doc {doc}", edgecolors='gray', linewidth=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        color = MODEL_COLORS.get(model_name, '#808080')
        plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7, s=15, color=color,
                   edgecolors='gray', linewidth=0.3)
    
    plt.title(f't-SNE: Distribución de Chunks - {model_name.upper()}', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Dimensión t-SNE 1', fontsize=12)
    plt.ylabel('Dimensión t-SNE 2', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'tsne_{model_name}_chunks{method_suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def _generate_overlay_tsne(model_name: str, chunk_embeddings: np.ndarray, chunk_embeddings_pca: np.ndarray,
                          question_data: List[Dict], output_dir: str, method_suffix=""):
    """Generate t-SNE visualization with question overlay."""
    try:
        # Extract question embeddings
        question_embeddings = []
        question_texts = []
        
        for q_data in question_data:
            embedding = np.array(q_data["embedding"], dtype=np.float32)
            question_embeddings.append(embedding)
            question_texts.append(f"Q{q_data['question_num']+1}")
        
        if not question_embeddings:
            return
            
        question_embeddings = np.vstack(question_embeddings)
        
        # Apply same PCA transformation to questions if needed
        if chunk_embeddings.shape[1] > 50:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50, random_state=42)
            pca.fit(chunk_embeddings)  # Fit on chunks
            question_embeddings_pca = pca.transform(question_embeddings)
        else:
            question_embeddings_pca = question_embeddings
        
        # Combine chunks and questions for joint t-SNE
        all_embeddings = np.vstack([chunk_embeddings_pca, question_embeddings_pca])
        
        # Apply t-SNE to combined data
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=min(30, len(all_embeddings)-1), 
                   random_state=42, verbose=0)
        all_2d = tsne.fit_transform(all_embeddings)
        
        # Split back into chunks and questions
        n_chunks = len(chunk_embeddings_pca)
        chunks_2d = all_2d[:n_chunks]
        questions_2d = all_2d[n_chunks:]
        
        # Create overlay visualization
        plt.figure(figsize=(14, 10))
        
        # Plot chunks as background (gray, low alpha)
        plt.scatter(chunks_2d[:, 0], chunks_2d[:, 1], 
                   alpha=0.3, s=15, color='#CCCCCC', 
                   edgecolors='gray', linewidth=0.2, zorder=1)
        
        # Plot questions with high contrast colors
        for i, (x, y) in enumerate(questions_2d):
            color = HIGH_CONTRAST_COLORS[i % len(HIGH_CONTRAST_COLORS)]
            
            # Plot question point
            plt.scatter(x, y, alpha=0.9, s=120, color=color, 
                       edgecolors='black', linewidth=2, zorder=10)
            
            # Add question label with background
            plt.annotate(question_texts[i], (x, y), 
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=11, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor=color, 
                                edgecolor='black', linewidth=2, alpha=0.9),
                        zorder=11)
        
        plt.title(f't-SNE: Chunks + Preguntas Overlay - {model_name.upper()}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Dimensión t-SNE 1', fontsize=12)
        plt.ylabel('Dimensión t-SNE 2', fontsize=12)
        
        # Add legend for questions
        legend_text = f"Preguntas ({len(question_texts)}): " + ", ".join(question_texts)
        plt.figtext(0.02, 0.02, legend_text, fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, f'tsne_{model_name}_overlay{method_suffix}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Error en overlay t-SNE para {model_name}: {e}")
        if 'plt' in locals():
            plt.close()

class EmbeddingComparator:
    """Main comparison class - optimized."""
    
    def __init__(self, db_paths: Dict[str, str], use_sql: bool = False):
        self.encoders = {}
        self.databases = {}
        self.tracker = PerformanceTracker()
        
        for model_name, db_path in db_paths.items():
            if os.path.exists(db_path):
                self.encoders[model_name] = EmbeddingEncoder(model_name)
                self.databases[model_name] = DatabaseClient(db_path, use_sql)
        
        print(f"Initialized {len(self.databases)} models: {list(self.databases.keys())}")
        
    def process_question(self, question: str, top_k: int, question_num: int = 0) -> Dict[str, Any]:
        """Process a single question across all models with error handling."""
        results = {"question": question, "question_num": question_num}
        
        for model_name in self.databases.keys():
            try:
                # Generate embedding
                embedding_start = time.time()
                question_embedding = self.encoders[model_name].encode(question)
                embedding_time = time.time() - embedding_start
                
                # Search database
                search_start = time.time()
                chunks = self.databases[model_name].search(question_embedding, top_k)
                search_time = time.time() - search_start
                
                # Record metrics with question embedding for t-SNE overlay
                self.tracker.record(model_name, "embedding_time", embedding_time)
                self.tracker.record(model_name, "search_time", search_time)
                self.tracker.record(model_name, "chunks_processed", len(chunks))
                
                # Convert to list if numpy array, otherwise use as-is
                embedding_list = question_embedding.tolist() if hasattr(question_embedding, 'tolist') else question_embedding
                self.tracker.record_question_details(question_num, model_name, embedding_time, search_time, 
                                                    len(chunks), question, embedding_list)
                
                results[model_name] = {
                    "chunks": chunks,
                    "embedding_time": embedding_time,
                    "search_time": search_time,
                    "total_time": embedding_time + search_time
                }
                
            except Exception as e:
                print(f"❌ Error procesando pregunta {question_num} con modelo {model_name}: {e}")
                print("   Continuando con otros modelos...")
                
                # Record failed attempt
                results[model_name] = {
                    "chunks": [],
                    "embedding_time": 0.0,
                    "search_time": 0.0,
                    "total_time": 0.0,
                    "error": str(e)
                }
                
        return results
        
    def compare_questions(self, questions: List[str], top_k: int = 5, 
                         parallel: bool = False, workers: int = 3) -> List[Dict[str, Any]]:
        """Compare all questions."""
        if parallel and len(questions) > 1:
            return self._compare_parallel(questions, top_k, workers)
        return self._compare_sequential(questions, top_k)
        
    def _compare_sequential(self, questions: List[str], top_k: int) -> List[Dict[str, Any]]:
        """Sequential processing."""
        results = []
        for i, question in enumerate(tqdm(questions, desc="Processing")):
            result = self.process_question(question, top_k, i)
            results.append(result)
        return results
        
    def _compare_parallel(self, questions: List[str], top_k: int, workers: int) -> List[Dict[str, Any]]:
        """Parallel processing."""
        results = [None] * len(questions)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {executor.submit(self.process_question, q, top_k, i): i 
                             for i, q in enumerate(questions)}
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
                    
        return [r for r in results if r]

def save_comprehensive_report(results: List[Dict], output_path: str, performance: Dict):
    """Generate comprehensive report with detailed tables and metrics like the original."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write("REPORTE COMPARATIVO DE BÚSQUEDA VECTORIAL - VERSIÓN COMPLETA\n")
        f.write("=" * 100 + "\n")
        
        # PERFORMANCE METRICS SECTION (if available)
        if performance:
            f.write("MÉTRICAS DE RENDIMIENTO POR MODELO\n")
            f.write("=" * 100 + "\n")
            
            # Performance comparison table with enhanced metrics
            f.write("| Modelo | Avg Embedding (ms) | Avg Search (ms) | Total Avg (ms) | Queries/sec | Chunks/sec | Calls |\n")
            f.write("|--------|-------------------|-----------------|----------------|-------------|------------|-------|\n")
            
            for model, stats in performance.items():
                emb_avg = stats.get("embedding_avg", 0) * 1000  # convert to ms
                search_avg = stats.get("search_avg", 0) * 1000
                total_avg = stats.get("total_avg", 0) * 1000
                calls = stats.get("total_queries", 0)
                queries_per_sec = stats.get("queries_per_second", 0)
                chunks_per_sec = stats.get("chunks_per_second", 0)
                
                f.write(f"| {model:<6} | {emb_avg:>16.2f} | {search_avg:>14.2f} | "
                       f"{total_avg:>13.2f} | {queries_per_sec:>10.2f} | {chunks_per_sec:>9.1f} | {calls:>5} |\n")
            
            f.write("\nDETALLE POR MODELO:\n")
            f.write("-" * 50 + "\n")
            
            for model, stats in performance.items():
                f.write(f"\n{model.upper()}:\n")
                f.write("  Embedding Generation:\n")
                f.write(f"    • Promedio: {stats.get('embedding_avg', 0)*1000:.2f}ms\n")
                f.write(f"    • Desviación estándar: {stats.get('embedding_std', 0)*1000:.2f}ms\n")
                f.write(f"    • Total llamadas: {stats.get('total_queries', 0)}\n")
                
                f.write("  Vector Search:\n")
                f.write(f"    • Promedio: {stats.get('search_avg', 0)*1000:.2f}ms\n")
                f.write(f"    • Desviación estándar: {stats.get('search_std', 0)*1000:.2f}ms\n")
                
                f.write("  Total (Embedding + Search):\n")
                f.write(f"    • Promedio: {stats.get('total_avg', 0)*1000:.2f}ms\n")
                f.write(f"    • Desviación estándar: {stats.get('total_std', 0)*1000:.2f}ms\n")
                
                f.write("  Métricas de Eficiencia:\n")
                f.write(f"    • Consultas por segundo: {stats.get('queries_per_second', 0):.2f}\n")
                f.write(f"    • Chunks por segundo: {stats.get('chunks_per_second', 0):.1f}\n")
                f.write(f"    • Total chunks procesados: {stats.get('total_chunks_processed', 0)}\n")
            
            f.write("\n" + "=" * 100 + "\n")
        
        # INTERPRETATION SECTION
        f.write("INTERPRETACIÓN DE RELEVANCIA:\n")
        f.write("• Relevante (≥75%): Fragmentos muy cercanos/alineados a la consulta\n")
        f.write("• Aceptable (50%-75%): Fragmentos con relación; útiles pero menos precisos\n")
        f.write("• Poco relevante (<50%): Fragmentos alejados o poco alineados\n")
        f.write(f"\nMétrica utilizada: {CONFIG['similarity_metric'].upper()}\n")
        f.write("=" * 100 + "\n\n")
        
        # Process each question
        for result in results:
            qnum = result["question_num"]
            question = result["question"]
            
            f.write("=" * 100 + "\n")
            f.write(f"PREGUNTA {qnum}: {question}\n")
            f.write("-" * 100 + "\n")
            
            # Write results for each model
            for model_name in ["nomic", "gemini", "jina"]:
                if model_name in result:
                    chunks = result[model_name]["chunks"]
                    if not chunks:
                        continue
                    
                    f.write(f"\n{model_name.upper()}\n")
                    f.write("| Doc ID | Chunk ID | Title                 | Página | Relevancia (%) | Categoría      | Similarity |\n")
                    f.write("|--------|----------|-----------------------|--------|----------------|----------------|------------|\n")
                    
                    for chunk in chunks[:5]:  # Show top 5
                        title_short = str(chunk.get('title', ''))[:21]
                        relevance_pct = chunk.get('relevance', 0.0)
                        category = chunk.get('relevance_category', 'N/A')
                        similarity = chunk.get('similarity', 0.0)
                        doc_id = chunk.get('document_id', 0)
                        chunk_id = chunk.get('chunk_id', 0)
                        page_num = chunk.get('page_number', 0)
                        
                        f.write(f"| {doc_id:<6} | {chunk_id:<8} | {title_short:<21} | "
                               f"{page_num:<6} | {relevance_pct:>13.1f} | {category:<14} | {similarity:>10.4f} |\n")
                    
                    # Statistics
                    if chunks:
                        stats = generate_relevance_stats(chunks[:5])
                        f.write(f"\nEstadísticas {model_name}:\n")
                        f.write(f"• Relevancia promedio: {stats['avg_relevance']:.1f}%\n")
                        f.write(f"• Rango: {stats['min_relevance']:.1f}% - {stats['max_relevance']:.1f}%\n")
                        f.write("• Distribución: ")
                        for cat, pct in stats['categories'].items():
                            f.write(f"{cat} {pct:.0f}%, ")
                        f.write("\n")
                        
                        f.write(f"\nTexto del mejor fragmento {model_name}:\n")
                        best_text = chunks[0]["text"].replace('\n', ' ').replace('\r', ' ').strip()
                        f.write(f'"{best_text}"\n')
            
            # PERFORMANCE METRICS PER QUESTION
            f.write(f"\nMÉTRICAS DE RENDIMIENTO (Pregunta {result['question_num']}):\n")
            f.write("| Modelo | Embedding (ms) | Search (ms) | Total (ms) |\n")
            f.write("|--------|----------------|-------------|------------|\n")
            
            for model in ["nomic", "gemini", "jina"]:
                if model in result:
                    timing = result[model]
                    emb_ms = timing["embedding_time"] * 1000
                    search_ms = timing["search_time"] * 1000
                    total_ms = timing["total_time"] * 1000
                    f.write(f"| {model:<6} | {emb_ms:>13.2f} | {search_ms:>10.2f} | {total_ms:>9.2f} |\n")
            
            f.write("\n" + "-" * 100 + "\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("VISUALIZACIONES GENERADAS\n")
        f.write("=" * 100 + "\n")
        f.write("Se han generado visualizaciones comprehensivas en el directorio 'visualizations/'\n")
        f.write("para análisis detallado de rendimiento y distribución de embeddings.\n")
        f.write("=" * 100 + "\n")

def generate_relevance_stats(results_list: List[Dict]) -> Dict:
    """Generate relevance statistics for results."""
    if not results_list:
        return {"total": 0, "categories": {}, "avg_relevance": 0.0}
    
    relevances = [r.get('relevance', 0.0) for r in results_list]
    
    # Generate categories using the existing _categorize_relevance logic
    categories = []
    thresholds = CONFIG["relevance_thresholds"]
    for rel in relevances:
        if rel >= thresholds["high"]:
            categories.append("Relevante")
        elif rel >= thresholds["medium"]:
            categories.append("Aceptable")
        else:
            categories.append("Poco relevante")
    
    category_counts = {}
    for cat in categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    total = len(results_list)
    category_percentages = {
        cat: (count / total) * 100.0 for cat, count in category_counts.items()
    }
    
    return {
        "total": total,
        "categories": category_percentages,
        "avg_relevance": np.mean(relevances),
        "min_relevance": np.min(relevances),
        "max_relevance": np.max(relevances)
    }

def load_questions(input_path: str) -> List[str]:
    """Load questions from file."""
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
        col = 'question' if 'question' in df.columns else 'Question'
        return df[col].dropna().tolist()
    elif input_path.endswith('.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return [q.get('question', q.get('Question', '')) for q in data if q.get('question') or q.get('Question')]
        return [q.get('question', q.get('Question', '')) for q in data.get('questions', [])]
    else:
        raise ValueError("Input must be CSV or JSON file")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Compare embedding models - Optimized")
    parser.add_argument('--input_path', required=True, help='Questions file (CSV/JSON)')
    parser.add_argument('--db_nomic', default=f"{CONFIG['files']['db_folder']}/db_nomic_clean.duckdb")
    parser.add_argument('--db_gemini', default=f"{CONFIG['files']['db_folder']}/db_gemini_clean.duckdb") 
    parser.add_argument('--db_jina', default=f"{CONFIG['files']['db_folder']}/db_jinav4.duckdb")
    parser.add_argument('--output_path', default=f"{CONFIG['files']['results_folder']}/{CONFIG['files']['report_name']}")
    parser.add_argument('--top_k', type=int, default=5, help='Top K results per model')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel workers')
    parser.add_argument('--use_sql', action='store_true', help='Use SQL native similarity functions')
    
    args = parser.parse_args()
    
    try:
        # Setup
        db_paths = {
            "nomic": args.db_nomic,
            "gemini": args.db_gemini, 
            "jina": args.db_jina
        }
        
        print(f"Configuration: {CONFIG['similarity_metric']} similarity, SQL={args.use_sql}")
        
        # Load questions
        questions = load_questions(args.input_path)
        print(f"Loaded {len(questions)} questions")
        
        # Run comparison
        start_time = time.time()
        comparator = EmbeddingComparator(db_paths, use_sql=args.use_sql)
        
        results = comparator.compare_questions(
            questions, top_k=args.top_k, 
            parallel=args.parallel, workers=args.workers
        )
        
        performance = comparator.tracker.get_summary()
        comparison_time = time.time() - start_time
        
        # Generate outputs with method suffix
        method_suffix = "_native" if args.use_sql else "_manual"
        base_path, ext = os.path.splitext(args.output_path)
        report_path = f"{base_path}{method_suffix}{ext}"
        save_comprehensive_report(results, report_path, performance)
        
        # Generate comprehensive visualizations
        viz_dir = os.path.join(os.path.dirname(args.output_path), "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        generate_comprehensive_visualizations(performance, comparator.tracker, viz_dir, comparator.databases, args.use_sql)
        
        print(f"\n✅ Completado en {comparison_time:.1f}s")
        print(f"📄 Reporte: {report_path}")
        print(f"📊 Visualizaciones: {viz_dir}")
        
        # Show performance summary with emojis
        if performance:
            print("\n🚀 Resumen de rendimiento:")
            for model, metrics in performance.items():
                speed_emoji = "🔥" if metrics['total_avg'] < 2.0 else "⚡" if metrics['total_avg'] < 5.0 else "🐌"
                print(f"  {speed_emoji} {model.upper()}: {metrics['total_avg']*1000:.0f}ms promedio, {metrics['queries_per_second']:.2f} q/s")
                
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
