
"""
Embeddings for Arabic legal passages and queries.
Uses SentenceTransformers with multilingual-e5-large.
Optimized for GPU acceleration (RTX 3050).
"""

import os
import time
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

try:
    from config import EMBEDDING_MODEL
    from text_normalization import clean_text
except ModuleNotFoundError:
    from config import EMBEDDING_MODEL
    from text_normalization import clean_text

try:
    import streamlit as st
except Exception:
    st = None

def _build_embedding_model(model_name: str) -> SentenceTransformer:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "100")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "100")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Loading Embedding Model on: {device} ---")

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            return SentenceTransformer(model_name, device=device)
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(2 + attempt)
    raise RuntimeError(f"Failed to load model '{model_name}'.") from last_exc

if st:
    _get_cached_embedding_model = st.cache_resource(show_spinner=False)(_build_embedding_model)
else:
    def _get_cached_embedding_model(model_name: str) -> SentenceTransformer:
        return _build_embedding_model(model_name)

class ArabicEmbedder:
    """Embedder for Arabic legal text with E5 prefix convention."""

    PASSAGE_PREFIX = "passage: "
    QUERY_PREFIX = "query: "

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = _get_cached_embedding_model(model_name)

    def encode_passages(self, texts: list[str], batch_size: int = 32, show_progress: bool = True):
        prefixed = [self.PASSAGE_PREFIX + clean_text(t, apply_reversal_fix=False) for t in texts]
        return self.model.encode(prefixed, batch_size=batch_size, show_progress_bar=show_progress)

    def encode_query(self, query: str) -> np.ndarray:
        prefixed = self.QUERY_PREFIX + clean_text(query, apply_reversal_fix=False)
        return self.model.encode([prefixed])[0]