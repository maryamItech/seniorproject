"""ScienceQA helper tool for retrieving similar QA samples."""
import sys
from pathlib import Path

# Ensure project root is in path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
from typing import List, Dict, Optional, Union

import numpy as np
from langchain_core.documents import Document

# Pandas will be imported lazily when needed to avoid import errors at module level

from utils.embeddings import BGEEmbeddings
from tools.image_processing import ImagePreprocessor
from embeddings.image_embeddings import ClipImageEmbeddings
from config.settings import (
    PROJECT_ROOT,
    EMBEDDING_MODEL_NAME,
    SCIENCEQA_IMAGE_INDEX_PATH,
    SCIENCEQA_IMAGE_META_PATH,
    SCIENCEQA_METADATA_PATH,
    SCIENCEQA_IMAGES_DIR,
)
import faiss

logger = logging.getLogger(__name__)


def _import_pandas():
    """Lazy import of pandas with helpful error message."""
    try:
        import pandas as pd
        return pd
    except ImportError as e:
        raise ImportError(
            "pandas is required but not installed. "
            "Please install all dependencies with: pip install -r requirements.txt\n"
            "Or install pandas directly: pip install pandas>=2.0.0"
        ) from e


def ensure_scienceqa_dataset(csv_path: Path) -> None:
    """Download ScienceQA dataset from Hugging Face if CSV doesn't exist."""
    if csv_path.exists():
        return
    
    pd = _import_pandas()  # Import pandas when needed
    logger.info(f"CSV file not found at {csv_path}. Downloading from Hugging Face...")
    
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets library is required to download ScienceQA dataset. "
            "Please install all dependencies with: pip install -r requirements.txt\n"
            "Or install datasets directly: pip install datasets>=2.16.0"
        ) from e
    
    # Load the dataset from Hugging Face
    logger.info("Loading ScienceQA dataset from HuggingFace (derek-thomas/ScienceQA)...")
    ds = load_dataset("derek-thomas/ScienceQA")
    
    # Combine all splits into a single DataFrame
    all_data = []
    for split_name in ["train", "validation", "test"]:
        if split_name in ds:
            split_data = ds[split_name]
            for item in split_data:
                row = dict(item)
                row["split"] = split_name
                all_data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Ensure data directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    logger.info(f"Saving dataset to {csv_path}...")
    df.to_csv(csv_path, index=False)
    logger.info(f"Dataset saved successfully. Total rows: {len(df)}")


class ScienceQARetriever:
    """Retriever for ScienceQA dataset (text + image)."""

    def __init__(
        self,
        csv_path: Optional[Path] = None,
        embedding_model_name: str = None,
        image_index_path: Path = SCIENCEQA_IMAGE_INDEX_PATH,
        image_meta_path: Path = SCIENCEQA_IMAGE_META_PATH,
    ):
        # Import pandas lazily when actually needed
        pd = _import_pandas()
        
        self.csv_path = csv_path or SCIENCEQA_METADATA_PATH
        # Ensure dataset exists, download if needed
        ensure_scienceqa_dataset(self.csv_path)
        
        self.embedding_model_name = embedding_model_name or EMBEDDING_MODEL_NAME
        self.embedding_model = BGEEmbeddings(self.embedding_model_name)
        self.preprocessor = ImagePreprocessor()
        self.df = pd.read_csv(self.csv_path)
        self._pd = pd  # Store pandas reference for use in other methods

        self.image_index_path = image_index_path
        self.image_meta_path = image_meta_path
        self.image_embedder = None
        self.image_index = None
        self.image_meta = []

        # Text index for questions
        self.text_index_path = PROJECT_ROOT / "embeddings" / "scienceqa_text.faiss"
        self._build_text_index()
        self._load_image_index()

    def _build_text_index(self):
        """Build FAISS index for ScienceQA questions."""
        if self.text_index_path.exists():
            self.text_index = faiss.read_index(str(self.text_index_path))
            return
        questions = self.df["question"].fillna("").tolist()
        embeddings = self.embedding_model.embed_documents(questions)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        self.text_index = faiss.IndexFlatIP(dimension)
        self.text_index.add(embeddings)
        faiss.write_index(self.text_index, str(self.text_index_path))

    def _load_image_index(self):
        """Load image index + metadata if available."""
        if self.image_index_path.exists() and self.image_meta_path.exists():
            self.image_index = faiss.read_index(str(self.image_index_path))
            with open(self.image_meta_path, "r", encoding="utf-8") as f:
                self.image_meta = json.load(f)
        else:
            logger.warning("ScienceQA image index not found. Run scripts/build_image_index.py")

    def _embed_image(self, image) -> np.ndarray:
        if self.image_embedder is None:
            self.image_embedder = ClipImageEmbeddings()
        return self.image_embedder.embed_image(image)

    def retrieve_similar(
        self,
        query: Union[str, np.ndarray, Path],
        k: int = 5,
    ) -> List[Dict]:
        """Retrieve similar ScienceQA samples based on text or image."""
        # Image path or array
        if isinstance(query, (str, Path)) and Path(query).exists():
            if self.image_index is None:
                return [{"message": "Image index not built. Run scripts/build_image_index.py"}]
            image = Path(query)
            emb = self._embed_image(image)
            emb = emb.reshape(1, -1).astype("float32")
            faiss.normalize_L2(emb)
            scores, indices = self.image_index.search(emb, k)
            return self._collect_image_results(scores, indices)

        if isinstance(query, np.ndarray):
            if self.image_index is None:
                return [{"message": "Image index not built. Run scripts/build_image_index.py"}]
            emb = query.reshape(1, -1).astype("float32")
            faiss.normalize_L2(emb)
            scores, indices = self.image_index.search(emb, k)
            return self._collect_image_results(scores, indices)

        # Text path -> fallback to text index
        query_embedding = self.embedding_model.embed_query(str(query))
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_embedding)
        scores, indices = self.text_index.search(query_embedding, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.df):
                row = self.df.iloc[idx]
                image_path = self._image_path_for_row(row, idx)
                results.append(
                    {
                        "question": row.get("question", ""),
                        "choices": row.get("choices", ""),
                        "answer": row.get("answer", ""),
                        "explanation": row.get("explanation", ""),
                        "similarity_score": float(scores[0][i]),
                        "index": int(idx),
                        "image_path": image_path,
                        "split": row.get("split", "train"),
                    }
                )
        return results

    def _image_path_for_row(self, row, idx: int) -> Optional[str]:
        split = row.get("split", "train")
        candidate = SCIENCEQA_IMAGES_DIR / split / f"{idx}.png"
        return str(candidate) if candidate.exists() else None

    def _collect_image_results(self, scores, indices):
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.image_meta):
                meta = self.image_meta[idx]
                results.append(
                    {
                        "question": meta.get("question", ""),
                        "choices": meta.get("choices", ""),
                        "answer": meta.get("answer", ""),
                        "explanation": meta.get("explanation", ""),
                        "similarity_score": float(scores[0][i]),
                        "index": int(idx),
                        "image_path": meta.get("image_path"),
                        "split": meta.get("split"),
                    }
                )
        return results

    def format_results(self, results: List[Dict]) -> str:
        """Format retrieval results as string."""
        formatted = "Similar ScienceQA Samples:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"Sample {i} (Similarity: {result.get('similarity_score', 0):.3f}):\n"
            formatted += f"Question: {result.get('question', 'N/A')}\n"
            if result.get("choices"):
                formatted += f"Choices: {result.get('choices')}\n"
            formatted += f"Answer: {result.get('answer', 'N/A')}\n"
            if result.get("explanation"):
                formatted += f"Explanation: {result.get('explanation')}\n"
            if result.get("image_path"):
                formatted += f"Image: {result.get('image_path')}\n"
            formatted += "\n"
        return formatted.strip()


def scienceqa_helper(query: Union[str, np.ndarray, Path], k: int = 5) -> str:
    """Wrapper function for LangChain tool."""
    retriever = ScienceQARetriever()
    results = retriever.retrieve_similar(query, k)
    return retriever.format_results(results)

