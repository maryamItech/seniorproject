"""
Configuration for the Arabic Legal RAG system.
Optimized for Unified Cleaned Text.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

VECTOR_DB_PATH = str(BASE_DIR / "legal_db_v2_clean")

CLEANED_TEXT_FILE = "all_laws_cleaned_v2.txt"
AMBIGUITY_ROUTE = "ambiguity_check"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_DIM = 1024
#RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL = "BAAI/bge-reranker-base"
# ChromaDB
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "legal_collection_v2")

# LLM provider settings
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "Qwen3-vl:4b")

# GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

# Provider selection: default to local Ollama, with optional overrides/fallbacks.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
if LLM_PROVIDER == "groq":
    LLM_API_BASE = os.getenv("LLM_API_BASE", GROQ_API_BASE)
    LLM_API_KEY = os.getenv("LLM_API_KEY", GROQ_API_KEY)
    LLM_MODEL = os.getenv("LLM_MODEL", GROQ_MODEL)
elif LLM_PROVIDER == "openrouter":
    LLM_API_BASE = os.getenv("LLM_API_BASE", OPENROUTER_API_BASE)
    LLM_API_KEY = os.getenv("LLM_API_KEY", OPENROUTER_API_KEY)
    LLM_MODEL = os.getenv("LLM_MODEL", OPENROUTER_MODEL)
else:
    LLM_API_BASE = os.getenv("LLM_API_BASE", OLLAMA_API_BASE)
    LLM_API_KEY = os.getenv("LLM_API_KEY", OLLAMA_API_KEY)
    LLM_MODEL = os.getenv("LLM_MODEL", OLLAMA_MODEL)

LEGAL_DATABASES = {
    "civil": "القانون المدني",
    "penal": "قانون العقوبات",
    "personal_status": "الأحوال الشخصية",
    "all": "جميع القوانين",
}

TOP_K_VECTOR = 7
TOP_K_BM25 = 7
TOP_K_HYBRID = 5
TOP_K_RERANK = 3


ARTICLE_PATTERN = r"(?=الماد[هة]\s+[\d\u0660-\u0669]+)"
