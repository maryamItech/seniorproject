"""
Text-based ingestion pipeline for Syrian legal RAG.
"""

import os
import pickle
import re
import shutil
from pathlib import Path
from typing import Any

import chromadb

try:
    from config import CHROMA_COLLECTION_NAME, DATA_DIR, VECTOR_DB_PATH
    from embeddings import ArabicEmbedder
    from bm25_search import BM25Index
    from text_normalization import clean_text
except ModuleNotFoundError:
    from config import CHROMA_COLLECTION_NAME, DATA_DIR, VECTOR_DB_PATH
    from embeddings import ArabicEmbedder
    from bm25_search import BM25Index
    from text_normalization import clean_text

BM25_INDEX_PATH = Path(__file__).resolve().parent / "bm25_index.pkl"
MAX_ARTICLE_CHARS = 1500
CHUNK_OVERLAP = 180


def load_bm25_index(bm25_instance: BM25Index) -> bool:
    if not BM25_INDEX_PATH.exists():
        print(f"--- [Warning] BM25 file not found at {BM25_INDEX_PATH} ---")
        return False
    try:
        with open(BM25_INDEX_PATH, "rb") as f:
            loaded = pickle.load(f)
        bm25_instance.index     = getattr(loaded, "index",     None)
        bm25_instance.chunks    = list(getattr(loaded, "chunks",    []) or [])
        bm25_instance.tokenized = list(getattr(loaded, "tokenized", []) or [])
        if bm25_instance.index is None or not bm25_instance.chunks:
            print("--- [Warning] BM25 loaded with missing payload ---")
            return False
        print(f"--- [Success] BM25 Index loaded correctly ({len(bm25_instance.chunks)} chunks) ---")
        return True
    except Exception as e:
        print(f"--- [Error] Failed to load BM25 index: {e} ---")
        bm25_instance.index     = None
        bm25_instance.chunks    = []
        bm25_instance.tokenized = []
        return False


def _clean_legal_lines(text: str) -> str:
    if not text:
        return ""
    normalized = clean_text(text, apply_reversal_fix=False).replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return normalized


def _to_western_digits(value: str) -> str:
    return value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))


def _extract_article_number(article_text: str, fallback: int) -> int:
    normalized = _to_western_digits(article_text[:50])
    match = re.search(r"الماد[هة]\s+(\d+)", normalized)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return fallback


def _split_long_article(article_text: str, max_chars: int = MAX_ARTICLE_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(article_text) <= max_chars:
        return [article_text]
    parts = []
    start = 0
    while start < len(article_text):
        end = min(start + max_chars, len(article_text))
        segment = article_text[start:end].strip()
        if segment:
            parts.append(segment)
        if end >= len(article_text):
            break
        start = end - overlap
    return parts


def load_chunks_from_cleaned_text(data_dir: Path, filename: str) -> list[dict[str, Any]]:
    file_path = data_dir / filename
    if not file_path.exists():
        print(f"  [Error] File not found: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        full_content = f.read()

    law_sections = re.split(r"={5,}", full_content)
    law_sections = [s.strip() for s in law_sections if s.strip()]
    print(f"  [Info] Found {len(law_sections)} sections in file")

    law_mapping = [
        {"name": "القانون المدني",  "file": "civil_law.pdf"},
        {"name": "الأحوال الشخصية", "file": "personal_statlaw.pdf"},
        {"name": "قانون العقوبات",  "file": "sy_penalcode.pdf"},
    ]

    chunks = []
    law_idx = 0

    for i, section in enumerate(law_sections):
        if law_idx >= len(law_mapping):
            break
        arabic_chars = len(re.findall(r"[\u0600-\u06FF]", section))
        total_chars  = max(1, len(section.replace(" ", "").replace("\n", "")))
        if arabic_chars / total_chars < 0.3:
            print(f"  [Skip] Section {i} — corrupted")
            continue

        law_name    = law_mapping[law_idx]["name"]
        source_file = law_mapping[law_idx]["file"]
        print(f"\n--- Processing Law: {law_name} (section {i}) ---")
        law_idx += 1

        articles = re.split(r"(الماد[هة]\s+\d+)", section)
        processed_articles = []
        for j in range(1, len(articles), 2):
            header  = articles[j]
            content = articles[j + 1] if j + 1 < len(articles) else ""
            processed_articles.append(header + content)

        print(f"  [Info] Found {len(processed_articles)} articles")

        for idx, art_text in enumerate(processed_articles):
            clean_art = _clean_legal_lines(art_text)
            if not clean_art:
                continue
            art_num   = _extract_article_number(clean_art, idx + 1)
            sub_parts = _split_long_article(clean_art)
            for p_idx, p_text in enumerate(sub_parts):
                full_entry = f"[المصدر: {law_name}]\n{p_text}"
                chunks.append({
                    "text": full_entry,
                    "metadata": {
                        "source":         source_file,
                        "law_name":       law_name,
                        "article_number": art_num,
                        "part_index":     p_idx,
                        "total_parts":    len(sub_parts),
                    }
                })

    print(f"\n  [Total] {len(chunks)} chunks created")
    return chunks


def run_ingest():
    vector_db_abs = os.path.abspath(str(VECTOR_DB_PATH))
    vector_db_dir = Path(vector_db_abs)
    if vector_db_dir.exists():
        shutil.rmtree(vector_db_dir, ignore_errors=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    vector_db_dir.mkdir(parents=True, exist_ok=True)

    print("Loading chunks from cleaned text file...")
    chunks = load_chunks_from_cleaned_text(DATA_DIR, "all_laws_cleaned_v2.txt")
    if not chunks:
        return

    print("Creating embeddings...")
    embedder = ArabicEmbedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode_passages(texts, show_progress=True)

    print("Building BM25 index...")
    bm25 = BM25Index()
    bm25.build(chunks)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print("Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=vector_db_abs)
    collection = client.create_collection(
        CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    ids = [
        f"{c['metadata']['source']}_art_{c['metadata']['article_number']}_{i}"
        for i, c in enumerate(chunks)
    ]
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[c["metadata"] for c in chunks]
    )
    print("--- Ingestion Done Successfully ---")


if __name__ == "__main__":
    run_ingest()