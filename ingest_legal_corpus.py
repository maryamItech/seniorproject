# import os
import argparse
from pathlib import Path
from typing import Any
import chromadb
import pickle

try:
    from config import (
        VECTOR_DB_PATH,
        CHROMA_COLLECTION_NAME,
        EMBEDDING_MODEL,
        CLEANED_TEXT_FILE,
        LEGAL_DATABASES
    )
    from embeddings import ArabicEmbedder
    from chunker import chunk_legal_articles
    from bm25_search import BM25Index
except ModuleNotFoundError:
    from .config import (
        VECTOR_DB_PATH, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL, CLEANED_TEXT_FILE, LEGAL_DATABASES
    )
    from .embeddings import ArabicEmbedder
    from .chunker import chunk_legal_articles
    from .bm25_search import BM25Index

def ingest_all():
    print("--- Starting Syrian Laws Processing Pipeline ---")

    # 1. Initialize Models
    embedder = ArabicEmbedder(model_name=EMBEDDING_MODEL)

    # 2. Read the unified cleaned text file
    text_file_path = Path(__file__).resolve().parent / "data" / CLEANED_TEXT_FILE
    if not text_file_path.exists():
        print(f"Error: File {text_file_path} not found.")
        return

    with open(text_file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # 3. Split text into specific laws based on headers in the file
    # Note: The file contains "Civil Law", "Penal Code", etc.
    all_chunks = []

    # Partition the file into major segments based on law names
    # Assuming each law starts with its name on a standalone line
    laws_to_process = [
        ("القانون المدني", "civil"),
        ("قانون العقوبات", "penal"),
        ("قانون الأحوال الشخصية", "personal_status")
    ]

    for law_display_name, law_key in laws_to_process:
        print(f"Processing: {law_display_name}...")

    # Extract the specific section for this law from the text file (optional if file is pre-segmented)
    # Or pass the full text to the smart Chunker for processing
        law_chunks = chunk_legal_articles(
            text=full_text,
            law_name=law_display_name,
            source=CLEANED_TEXT_FILE
        )

        # Filter chunks to ensure they follow the correct law (if the file is mixed)
        # This step relies on the Chunker's ability to identify delimiters
        all_chunks.extend(law_chunks)

    print(f"Successfully extracted {len(all_chunks)} legal articles.")

    # 4. Storage in ChromaDB
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    # Delete the old collection to create a clean new one
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Prepare data for ingestion
    documents = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    ids = [f"doc_{i}" for i in range(len(all_chunks))]

    print("Generating Embeddings (GPU will be used if available)...")
    embeddings = embedder.encode_passages(documents)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )
    print("Storage in ChromaDB completed successfully.")

    # 5. Build and store BM25 index for fast lexical search
    print("Building BM25 index...")
    bm25_index = BM25Index()
    bm25_index.build(all_chunks)

    bm25_path = Path(VECTOR_DB_PATH) / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_index, f)

    print(f"BM25 index saved at: {bm25_path}")
    print("--- Process completed successfully! ---")

if __name__ == "__main__":
    ingest_all()