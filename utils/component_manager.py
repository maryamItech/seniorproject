"""Component Manager for RAG System - Singleton Pattern for Expensive Resources.

This module provides a centralized way to initialize and cache expensive components
that should only be loaded once across the entire application lifecycle:
- Embedding models (BGE)
- LLM instances (OpenRouter)
- Milvus client connections
- Vector store instances

All components are cached using Streamlit's @st.cache_resource decorator when used
in Streamlit, or using a simple singleton pattern for non-Streamlit contexts.
"""

import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from langchain_ollama import ChatOllama
except ImportError:
    # Fallback for older versions
    from langchain_community.chat_models import ChatOllama
from pymilvus import MilvusClient
from retrieval.multimodal_milvus_store import MultimodalMilvusStore
from config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TEMPERATURE_CORRECTION,
    TEMPERATURE_GENERATION,
    MILVUS_URI,
    MILVUS_TOKEN,
    MILVUS_MULTIMODAL_COLLECTION_NAME,
    PROJECT_ROOT,
)


class ComponentManager:
    """
    Singleton manager for expensive RAG system components.
    
    This ensures that:
    1. Embedding models are loaded only once
    2. LLM instances are created only once
    3. Milvus client connections are reused
    4. Vector store instances are cached
    
    Usage in Streamlit:
        Use the cached functions (get_embedding_model, get_llm, etc.)
        which use @st.cache_resource for proper caching.
    
    Usage outside Streamlit:
        Use ComponentManager.get_instance() to get the singleton instance.
    """
    
    _instance: Optional['ComponentManager'] = None
    
    def __init__(self):
        """Initialize component manager (private - use get_instance())."""
        self._correction_llm: Optional[ChatOllama] = None
        self._generation_llm: Optional[ChatOllama] = None
        self._milvus_client: Optional[MilvusClient] = None
        self._vector_store: Optional[MultimodalMilvusStore] = None
    
    @classmethod
    def get_instance(cls) -> 'ComponentManager':
        """Get singleton instance of ComponentManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    
    def get_correction_llm(self) -> ChatOllama:
        """Get or create LLM for question correction/rephrasing (cached)."""
        if self._correction_llm is None:
            print("[ComponentManager] Initializing correction LLM (Ollama)...")
            # ChatOllama doesn't support num_gpu directly - GPU is handled by Ollama server
            # Make sure Ollama server is configured to use GPU (check GPU_SETUP.md)
            self._correction_llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=TEMPERATURE_CORRECTION,
            )
            print(f"[ComponentManager] Correction LLM initialized: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
        return self._correction_llm
    
    def get_generation_llm(self) -> ChatOllama:
        """Get or create LLM for answer generation (cached)."""
        if self._generation_llm is None:
            print("[ComponentManager] Initializing generation LLM (Ollama)...")
            # ChatOllama doesn't support num_gpu directly - GPU is handled by Ollama server
            # Make sure Ollama server is configured to use GPU (check GPU_SETUP.md)
            self._generation_llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=TEMPERATURE_GENERATION,
            )
            print(f"[ComponentManager] Generation LLM initialized: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
        return self._generation_llm
    
    def get_milvus_client(self) -> MilvusClient:
        """Get or create Milvus client connection (cached)."""
        if self._milvus_client is None:
            print("[ComponentManager] Connecting to Milvus...")
            self._milvus_client = MilvusClient(
                uri=MILVUS_URI,
                token=MILVUS_TOKEN,
                timeout=60,
            )
            print(f"[ComponentManager] Milvus client connected: {MILVUS_URI}")
        return self._milvus_client
    
    def get_vector_store(self) -> MultimodalMilvusStore:
        """Get or create multimodal vector store instance (cached)."""
        if self._vector_store is None:
            print("[ComponentManager] Loading multimodal vector store...")
            client = self.get_milvus_client()
            model_weight_path = PROJECT_ROOT / "Visualized_base_en_v1.5.pth"
            
            if not client.has_collection(MILVUS_MULTIMODAL_COLLECTION_NAME):
                raise RuntimeError(
                    f"Milvus collection '{MILVUS_MULTIMODAL_COLLECTION_NAME}' does not exist. "
                    "Run 'scripts/upload_multimodal_to_milvus.py' first to create and populate it."
                )
            
            self._vector_store = MultimodalMilvusStore(
                client=client,
                collection_name=MILVUS_MULTIMODAL_COLLECTION_NAME,
                model_weight_path=model_weight_path,
            )
            print(f"[ComponentManager] Multimodal vector store loaded: {MILVUS_MULTIMODAL_COLLECTION_NAME}")
        return self._vector_store


# Streamlit-specific cached functions
# These use @st.cache_resource to ensure components are cached across reruns
def _get_streamlit_cache():
    """Get Streamlit cache decorator if available."""
    try:
        import streamlit as st
        return st.cache_resource
    except ImportError:
        # Return a no-op decorator if Streamlit is not available
        def noop_decorator(func):
            return func
        return noop_decorator


# Try to use Streamlit caching if available
try:
    import streamlit as st
    
    @st.cache_resource
    def get_correction_llm() -> ChatOllama:
        """Get correction LLM (cached with Streamlit)."""
        return ComponentManager.get_instance().get_correction_llm()
    
    @st.cache_resource
    def get_generation_llm() -> ChatOllama:
        """Get generation LLM (cached with Streamlit)."""
        return ComponentManager.get_instance().get_generation_llm()
    
    @st.cache_resource
    def get_milvus_client() -> MilvusClient:
        """Get Milvus client (cached with Streamlit)."""
        return ComponentManager.get_instance().get_milvus_client()
    
    @st.cache_resource
    def get_vector_store() -> MultimodalMilvusStore:
        """Get multimodal vector store (cached with Streamlit)."""
        return ComponentManager.get_instance().get_vector_store()
    
except ImportError:
    # Fallback for non-Streamlit contexts
    def get_correction_llm() -> ChatOllama:
        """Get correction LLM (singleton pattern)."""
        return ComponentManager.get_instance().get_correction_llm()
    
    def get_generation_llm() -> ChatOllama:
        """Get generation LLM (singleton pattern)."""
        return ComponentManager.get_instance().get_generation_llm()
    
    def get_milvus_client() -> MilvusClient:
        """Get Milvus client (singleton pattern)."""
        return ComponentManager.get_instance().get_milvus_client()
    
    def get_vector_store() -> MultimodalMilvusStore:
        """Get multimodal vector store (singleton pattern)."""
        return ComponentManager.get_instance().get_vector_store()


