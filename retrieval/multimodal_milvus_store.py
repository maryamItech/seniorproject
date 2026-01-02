"""Multimodal Milvus vector store for Visualized_BGE embeddings.

This store supports both text and image search using the multimodal_visualized_bge collection.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np
from langchain_core.documents import Document
from pymilvus import MilvusClient

# Add FlagEmbedding to path
PROJECT_ROOT = Path(__file__).parent.parent
FLAG_EMBEDDING_DIR = PROJECT_ROOT / "FlagEmbedding-master"
if str(FLAG_EMBEDDING_DIR / "research") not in sys.path:
    sys.path.insert(0, str(FLAG_EMBEDDING_DIR / "research"))

from visual_bge.modeling import Visualized_BGE
from config.settings import (
    MILVUS_URI,
    MILVUS_TOKEN,
    MILVUS_MULTIMODAL_COLLECTION_NAME,
    PROJECT_ROOT,
)
from utils.performance_profiler import get_profiler


class MultimodalMilvusStore:
    """Multimodal Milvus vector store for Visualized_BGE embeddings."""

    def __init__(
        self,
        client: MilvusClient,
        collection_name: str,
        model_weight_path: Path,
        model_name: str = "BAAI/bge-base-en-v1.5",
    ):
        """
        Initialize multimodal Milvus vector store.
        
        Args:
            client: Milvus client connection
            collection_name: Name of the Milvus collection
            model_weight_path: Path to Visualized_BGE model weight file
            model_name: BGE model name
        """
        self.client = client
        self.collection_name = collection_name
        
        # Initialize Visualized_BGE model
        print(f"[MultimodalMilvusStore] Loading Visualized_BGE model...")
        self.embedding_model = Visualized_BGE(
            model_name_bge=model_name,
            model_weight=str(model_weight_path)
        )
        self.embedding_model.eval()
        print(f"[MultimodalMilvusStore] Model loaded successfully")
        
        # Load collection into memory
        try:
            self.client.load_collection(self.collection_name)
        except Exception:
            pass

    def similarity_search_with_score(
        self, 
        query: Union[str, Path], 
        k: int = 4,
        search_type: str = "text"  # "text" or "image"
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with scores.
        
        Args:
            query: Text query string or image path
            k: Number of results to return
            search_type: "text" to search text_embedding, "image" to search image_embedding
        """
        profiler = get_profiler()
        
        # Generate embedding based on search type
        with profiler.stage("5_embedding_generation", {
            "search_type": search_type,
            "query_type": "text" if search_type == "text" else "image"
        }):
            if search_type == "text":
                if not isinstance(query, str):
                    raise ValueError("Text search requires string query")
                query_embedding = self.embedding_model.encode(text=query)
                anns_field = "text_embedding"
            elif search_type == "image":
                if isinstance(query, str):
                    query = Path(query)
                query_embedding = self.embedding_model.encode(image=str(query))
                anns_field = "image_embedding"
            else:
                raise ValueError(f"Invalid search_type: {search_type}. Use 'text' or 'image'")
            
            query_embedding = query_embedding[0].cpu().detach().numpy().astype("float32")
        
        # Prepare search parameters and execute search
        search_params = {
            "metric_type": "COSINE",
            "params": {},
        }
        
        # Execute search (this is part of retrieval, already timed in rag_chain.retrieve)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding.tolist()],
            anns_field=anns_field,
            limit=k,
            search_params=search_params,
            output_fields=["id", "text", "image_path", "question", "answer", "choices", "explanation", "split"],
        )
        
        # Convert to LangChain Documents
        docs_with_scores: List[Tuple[Document, float]] = []
        
        if not results:
            return docs_with_scores
        
        hits = results[0]
        for hit in hits:
            # Extract distance from Milvus (COSINE distance: 0=best, 1=worst)
            # Return distance (not similarity) so rag_chain can handle conversion consistently
            distance = float(hit.get("distance", hit.get("score", 0.0)))
            
            # Extract metadata
            record_id = hit.get("id", hit.get("primary_key", None))
            text = hit.get("text", "")
            image_path = hit.get("image_path", "")
            question = hit.get("question", "")
            answer = hit.get("answer", "")
            choices = hit.get("choices", "")
            explanation = hit.get("explanation", "")
            split = hit.get("split", "")
            
            # Create document with full context
            page_content = text if text else f"Question: {question}\nAnswer: {answer}"
            
            # Calculate similarity for metadata (for display purposes)
            similarity = 1.0 - distance if distance <= 1.0 else distance
            
            doc = Document(
                page_content=page_content,
                metadata={
                    "id": record_id,
                    "score": similarity,  # Store similarity in metadata for display
                    "distance": distance,  # Store original distance
                    "image_path": image_path,
                    "question": question,
                    "answer": answer,
                    "choices": choices,
                    "explanation": explanation,
                    "split": split,
                }
            )
            # Return (doc, distance) so rag_chain can convert it properly
            docs_with_scores.append((doc, distance))
        
        return docs_with_scores

    def similarity_search(
        self, 
        query: Union[str, Path], 
        k: int = 4,
        search_type: str = "text"
    ) -> List[Document]:
        """Search for similar documents and return only Documents."""
        return [doc for doc, _ in self.similarity_search_with_score(query, k, search_type)]

    @classmethod
    def load(
        cls,
        model_weight_path: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ) -> "MultimodalMilvusStore":
        """
        Load multimodal Milvus vector store using defaults from config.
        
        Args:
            model_weight_path: Path to Visualized_BGE model weight (default: PROJECT_ROOT/Visualized_base_en_v1.5.pth)
            collection_name: Name of Milvus collection (default: from settings)
        """
        if model_weight_path is None:
            model_weight_path = PROJECT_ROOT / "Visualized_base_en_v1.5.pth"
        
        if collection_name is None:
            collection_name = MILVUS_MULTIMODAL_COLLECTION_NAME
        
        client = MilvusClient(
            uri=MILVUS_URI,
            token=MILVUS_TOKEN,
            timeout=60,
        )
        
        if not client.has_collection(collection_name):
            raise RuntimeError(
                f"Milvus collection '{collection_name}' does not exist. "
                "Run 'scripts/upload_multimodal_to_milvus.py' first to create and populate it."
            )
        
        return cls(
            client=client,
            collection_name=collection_name,
            model_weight_path=model_weight_path,
        )


