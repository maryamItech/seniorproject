"""Configuration settings for the RAG project."""
import os
from pathlib import Path

# Optionally load .env in development
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Safe to ignore if dotenv is not installed; advise in README
    pass

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# LLM Configuration
# Primary LLM provider: Ollama (Qwen3-VL for multimodal reasoning)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "Qwen3-vl:4b")  # Using Qwen3-vl:4b exclusively
# Force GPU usage (Ollama should auto-detect, but this ensures it)
OLLAMA_NUM_GPU = os.getenv("OLLAMA_NUM_GPU", "1")  # Use 1 GPU
OLLAMA_NUM_THREAD = os.getenv("OLLAMA_NUM_THREAD", "4")  # CPU threads for fallback

# Embedding Model Configuration
# BGE model for text embeddings (used by BGEEmbeddings)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en")

# OpenRouter settings (optional fallback - not used by default)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# OpenRouter model for RAG Evaluation
OPENROUTER_EVAL_MODEL = os.getenv("OPENROUTER_EVAL_MODEL", "qwen/qwen-2.5-vl-7b-instruct:free")

# Web Search API Keys (optional - for web search functionality)
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")  # Serper API for Google search
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # Tavily API for search
DUCKDUCKGO_API_KEY = os.getenv("DUCKDUCKGO_API_KEY", "")  # DuckDuckGo (usually not needed)

# Image Generation API Keys (optional - for image generation functionality)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # OpenAI API for DALL-E image generation
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "")  # Stability AI API for image generation

# Knowledge base similarity threshold
KB_SIMILARITY_THRESHOLD = float(os.getenv("KB_SIMILARITY_THRESHOLD", "0.7"))

# Paths
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.json"
VECTORS_PATH = EMBEDDINGS_DIR / "vectors.npy"

# ScienceQA assets
SCIENCEQA_IMAGES_DIR = PROJECT_ROOT / "scienceqa_images"
SCIENCEQA_METADATA_PATH = PROJECT_ROOT / "data" / "science_qa.csv"
SCIENCEQA_IMAGE_INDEX_PATH = EMBEDDINGS_DIR / "scienceqa_image_index.faiss"
SCIENCEQA_IMAGE_META_PATH = EMBEDDINGS_DIR / "scienceqa_image_meta.json"

# RAG Configuration
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
TEMPERATURE_CORRECTION = float(os.getenv("TEMPERATURE_CORRECTION", "0.0"))
TEMPERATURE_GENERATION = float(os.getenv("TEMPERATURE_GENERATION", "0.2"))

# Milvus / Zilliz Cloud configuration (used instead of FAISS)
MILVUS_URI = os.getenv(
    "MILVUS_URI",
    "https://in03-f434be2ef99e1da.serverless.aws-eu-central-1.cloud.zilliz.com",
)
MILVUS_TOKEN = os.getenv(
    "MILVUS_TOKEN",
    "eebf94aa84c711d8569514d5cf98686225f3f53bbc2112cc45e5870"
    "44b5849f74a08f625660e500ae01b888e23fd6c5e1cf9bb22",
)

# Mistral AI Configuration (for RAG Evaluation)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

# Multimodal Collection (Visualized_BGE - text + image embeddings)
MILVUS_MULTIMODAL_COLLECTION_NAME = os.getenv(
    "MILVUS_MULTIMODAL_COLLECTION_NAME",
    "multimodal_visualized_bge",
)

# Dev mode flag (used to enable plotting/visualizations)
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"
