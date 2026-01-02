"""Embedding utilities for BGE model (using BAAI/bge-small-en).

This is a simple, stable wrapper around SentenceTransformer.
No Visualized_BGE, no bge-base-en-v1.5, and no GPU management logic here.
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class BGEEmbeddings:
    """Simple wrapper around SentenceTransformer for BGE embeddings."""

    def __init__(self, model_name: str = "BAAI/bge-small-en"):
        """
        Initialize BGE embedding model.

        Args:
            model_name: HuggingFace model name (default: "BAAI/bge-small-en")
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query with BGE prefix for search."""
        return (
            self.model.encode(
                "Represent this sentence for searching: " + query,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
        )

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Embed documents with BGE."""
        return (
            self.model.encode(
                documents,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
        )


































