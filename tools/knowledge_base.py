"""Knowledge base tool with fallback to web search."""
import sys
from pathlib import Path

# Ensure project root is in path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Tuple, Optional
import logging
from langchain_core.documents import Document
from retrieval.multimodal_milvus_store import MultimodalMilvusStore
from tools.web_search import WebSearchTool, look_on_web_with_params
from config.settings import RETRIEVAL_K, KB_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

class KnowledgeBaseTool:
    """Knowledge base tool with automatic web fallback."""
    
    def __init__(self, vector_store: MultimodalMilvusStore, similarity_threshold: float = KB_SIMILARITY_THRESHOLD):
        """Initialize knowledge base tool.
        
        Args:
            vector_store: Multimodal Milvus vector store instance
            similarity_threshold: Minimum similarity score to use KB results
        """
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.web_search = WebSearchTool()
    
    def search(
        self,
        query: str,
        k: int = RETRIEVAL_K,
        auto_fallback: bool = True
    ) -> str:
        """Search knowledge base, with optional web fallback.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            auto_fallback: Whether to automatically fallback to web if similarity is low
        
        Returns:
            Answer from KB or web search results
        """
        logger.info("tool=look_in_your_knowledge query=%s k=%s", query, k)
        # Search knowledge base (text search by default)
        results = self.vector_store.similarity_search_with_score(query, k, search_type="text")
        
        if not results:
            if auto_fallback:
                return f"No results in knowledge base. Searching web...\n\n{self.web_search.search(query)}"
            return "No results found in knowledge base."
        
        # Check if similarity scores are above threshold
        # Milvus COSINE metric returns distance in [0, 2], where:
        # distance = 1 - cosine_similarity for normalized vectors.
        best_distance = results[0][1]
        similarity = 1.0 - min(max(best_distance, 0.0), 1.0)
        
        if similarity < self.similarity_threshold and auto_fallback:
            # Low similarity, fallback to web
            kb_context = self._format_kb_results(results)
            web_results = self.web_search.search(query)
            return (
                f"Knowledge base similarity ({similarity:.2f}) below threshold ({self.similarity_threshold}).\n\n"
                f"KB Results:\n{kb_context}\n\n"
                f"Web Results:\n{web_results}"
            )
        
        # High similarity, return KB results
        return self._format_kb_results(results)
    
    def _format_kb_results(self, results: List[Tuple[Document, float]]) -> str:
        """Format knowledge base results as string."""
        formatted = "Knowledge Base Results:\n\n"
        for i, (doc, score) in enumerate(results, 1):
            similarity = 1.0 - min(score, 1.0)  # Approximate similarity
            src = doc.metadata.get("index", "N/A")
            formatted += f"{i}. (Similarity: {similarity:.3f}, Source: {src})\n{doc.page_content}\n\n"
        return formatted.strip()


def look_in_your_knowledge(query: str, vector_store: Optional[MultimodalMilvusStore] = None) -> str:
    """Wrapper function for LangChain tool."""
    if vector_store is None:
        vector_store = MultimodalMilvusStore.load()
    tool = KnowledgeBaseTool(vector_store)
    return tool.search(query)


def look_in_your_knowledge_with_threshold(
    query: str,
    threshold: float = KB_SIMILARITY_THRESHOLD,
    vector_store: Optional[MultimodalMilvusStore] = None,
    max_results: int = RETRIEVAL_K,
) -> str:
    """Wrapper with threshold override for LangChain tool."""
    if vector_store is None:
        vector_store = MultimodalMilvusStore.load()
    tool = KnowledgeBaseTool(vector_store, similarity_threshold=threshold)
    return tool.search(query, k=max_results, auto_fallback=True)


