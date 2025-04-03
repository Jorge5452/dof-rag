# %%
"""
DOF Document Embedding Generator with Contextual Retrieval

This script processes markdown files from the Mexican Official Gazette (DOF),
extracts their content, splits them into semantic chunks, generates contextualized
vector embeddings using Gemini, and stores them in a SQLite database with vector
search capabilities. It also integrates BM25 for lexical matching.
"""

import os
from datetime import datetime
import typer
from fastlite import database
from sqlite_vec import load, serialize_float32
from semantic_text_splitter import MarkdownSplitter
from tokenizers import Tokenizer
from tqdm import tqdm
import google.generativeai as genai
from rank_bm25 import BM25Okapi
import numpy as np
import time
from google.api_core import exceptions  # For handling Gemini-specific errors

# Load Gemini API key from environment variables
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except TypeError:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it with 'export GEMINI_API_KEY=your-key'.")

# Tokenizer for splitting (using a lightweight tokenizer)
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
splitter = MarkdownSplitter.from_huggingface_tokenizer(tokenizer, 2048)

# Database initialization
db_path = "dof_db/db.sqlite"
os.makedirs(os.path.dirname(db_path), exist_ok=True)  # Ensure directory exists
db = database(db_path)
db.conn.enable_load_extension(True)
load(db.conn)
db.conn.enable_load_extension(False)

# Schema setup
db.t.documents.create(
    id=int, title=str, url=str, file_path=str, created_at=datetime, pk="id", ignore=True
)
db.t.documents.create_index(["url"], unique=True, if_not_exists=True)

db.t.chunks.create(
    id=int,
    document_id=int,
    text=str,
    embedding=bytes,
    bm25_context=str,
    created_at=datetime,
    pk="id",
    foreign_keys=[("document_id", "documents")],
    ignore=True,
)

# Gemini model for context generation
context_model = genai.GenerativeModel("gemini-2.0-flash")  # Adjust model as needed

def get_url_from_filename(filename: str) -> str:
    """Generate URL from filename."""
    base_filename = os.path.basename(filename).replace(".md", "")
    if len(base_filename) >= 8:
        year = base_filename[4:8]
        pdf_filename = f"{base_filename}.pdf"
        return f"https://diariooficial.gob.mx/abrirPDF.php?archivo={pdf_filename}&anio={year}&repo=repositorio/"
    raise ValueError(f"Expected filename like 23012025-MAT.md but got {filename}")

def get_gemini_embedding(text: str) -> np.ndarray:
    """Generate embeddings using Gemini API."""
    try:
        result = genai.embed_content(model="models/embedding-001", content=text)
        return np.array(result["embedding"], dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding with Gemini API: {str(e)}")

def generate_context(file_path: str, document_content: str, chunk: str, max_retries: int = 5) -> str:
    """Generate succinct context for a chunk using Gemini API with retry on rate limit."""
    prompt = f"""Here is the chunk we want to situate within the whole document
            <chunk>
            {chunk}
            </chunk>
            Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
            """
    for attempt in range(max_retries):
        try:
            response = context_model.generate_content(prompt)
            return response.text.strip()
        except exceptions.ResourceExhausted as e:  # Specifically catch 429 errors
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                print(f"Rate limit hit for {file_path}. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                print(f"Max retries reached for {file_path}: {str(e)}. Falling back to basic context.")
        except Exception as e:
            print(f"Error generating context for chunk in {file_path}: {str(e)}. Falling back to basic context.")
            break
    # Fallback to basic context if all retries fail
    title = os.path.splitext(os.path.basename(file_path))[0]
    return f"This chunk is from {title} (DOF document)."

def process_file(file_path: str):
    """Process a markdown file with Contextual Retrieval."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Skipping.")
        return
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}. Skipping.")
        return

    # Metadata extraction
    title = os.path.splitext(os.path.basename(file_path))[0]
    try:
        url = get_url_from_filename(file_path)
    except ValueError as e:
        print(f"Error generating URL for {file_path}: {str(e)}. Skipping.")
        return

    # Delete existing document to avoid duplicates
    db.t.documents.delete_where("url = ?", [url])
    doc = db.t.documents.insert(
        title=title, url=url, file_path=file_path, created_at=datetime.now()
    )

    # Split content into chunks
    chunks = splitter.chunks(content)
    if not chunks:
        print(f"Warning: No chunks generated for {file_path}. Skipping.")
        return

    # Prepare BM25 corpus (tokenized chunks)
    tokenized_corpus = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # Process chunks with Contextual Retrieval
    for i, chunk in enumerate(tqdm(chunks, desc=f"Processing {file_path}")):
        context = generate_context(file_path, content, chunk)
        contextualized_chunk = f"{context} {chunk}"

        # Generate embedding with Gemini
        try:
            embedding = get_gemini_embedding(contextualized_chunk)
        except RuntimeError as e:
            print(f"Error embedding chunk {i} in {file_path}: {str(e)}. Skipping chunk.")
            continue

        # Store in database
        db.t.chunks.insert(
            document_id=doc["id"],
            text=chunk,
            embedding=serialize_float32(embedding),
            bm25_context=contextualized_chunk,
            created_at=datetime.now(),
        )

def process_directory(directory_path: str):
    """Recursively process all markdown files in a directory."""
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a directory.")
        return
    for entry in tqdm(os.listdir(directory_path), desc=f"Processing {directory_path}"):
        entry_path = os.path.join(directory_path, entry)
        if os.path.isfile(entry_path) and entry_path.lower().endswith(".md"):
            process_file(entry_path)
        elif os.path.isdir(entry_path):
            process_directory(entry_path)

def main(root_dir: str):
    process_directory(root_dir)

if __name__ == "__main__":
    typer.run(main)