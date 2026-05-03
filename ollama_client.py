# """
# Provider-agnostic chat completion client.
# Supports Ollama, Groq, and OpenRouter through the OpenAI-compatible SDK.
# """
# import sys
# sys.stdout.reconfigure(encoding='utf-8')
# sys.stderr.reconfigure(encoding='utf-8')
# import os
# from typing import Optional

# from dotenv import load_dotenv
# load_dotenv(encoding="utf-8")
# from openai import OpenAI

# load_dotenv()

# try:
#     from config import (
#         GROQ_API_KEY,
#         LLM_PROVIDER,
#         LLM_API_BASE,
#         LLM_API_KEY,
#         LLM_MODEL,
#         OLLAMA_API_BASE,
#         OPENROUTER_API_BASE,
#     )
# except ModuleNotFoundError:  # pragma: no cover
#     from .config import (
#         GROQ_API_KEY,
#         LLM_PROVIDER,
#         LLM_API_BASE,
#         LLM_API_KEY,
#         LLM_MODEL,
#         OLLAMA_API_BASE,
#         OPENROUTER_API_BASE,
#     )

# try:
#     import streamlit as st
# except Exception:  # pragma: no cover - streamlit may be unavailable in scripts
#     st = None


# def _build_client(key: str, base_url: str) -> OpenAI:
#     return OpenAI(
#         base_url=base_url,
#         api_key=key,
#     )


# if st:
#     _get_cached_client = st.cache_resource(show_spinner=False)(_build_client)
# else:
#     def _get_cached_client(key: str, base_url: str) -> OpenAI:
#         return _build_client(key, base_url)


# def chat_completion(
#     messages: list[dict[str, str]],
#     api_key: Optional[str] = None,
#     model: Optional[str] = None,
#     temperature: float = 0.3,
#     max_tokens: int = 100,
#     stream: bool = False,
# ) -> str:
#     """
#     Call the configured chat completion provider via the OpenAI-compatible client.

#     Args:
#         messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
#         api_key: Override configured API key
#         model: Override default model
#         temperature: Sampling temperature (lower = more deterministic)
#         max_tokens: Maximum generated tokens for cost control
#         stream: Whether to stream response (returns full text when stream=False)

#     Returns:
#         Assistant reply text
#     """
#     inferred_provider = LLM_PROVIDER or "ollama"
#     default_local_key = "ollama" if ("localhost:11434" in LLM_API_BASE or LLM_API_BASE.startswith(OLLAMA_API_BASE)) else ""
#     key = (
#         api_key
#         or LLM_API_KEY
#         or os.getenv("LLM_API_KEY")
#         or os.getenv("OLLAMA_API_KEY")
#         or os.getenv("GROQ_API_KEY")
#         or os.getenv("OPENROUTER_API_KEY")
#         or default_local_key
#     )
#     if not key:
#         raise ValueError("No LLM API key found. Set OLLAMA_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY.")

#     m = (
#         model
#         or LLM_MODEL
#         or os.getenv("LLM_MODEL")
#         or os.getenv("OLLAMA_MODEL")
#         or os.getenv("GROQ_MODEL")
#         or os.getenv("OPENROUTER_MODEL")
#         or "Qwen3-vl:4b"
#     )
#     client = _get_cached_client(key, LLM_API_BASE)
#     request_kwargs = {
#         "model": m,
#         "messages": messages,
#         "temperature": temperature,
#         "max_tokens": max_tokens,
#         "stream": stream,
#     }
#     if inferred_provider == "openrouter" or LLM_API_BASE.startswith(OPENROUTER_API_BASE):
#         request_kwargs["extra_headers"] = {
#             "HTTP-Referer": "http://localhost",
#             "X-OpenRouter-Title": "Legal-RAG",
#         }
#         request_kwargs["extra_body"] = {
#             "transforms": ["middle-out"],
#         }

#     try:
#         resp = client.chat.completions.create(**request_kwargs)
#     except Exception as exc:
#         error_text = str(exc)
#         if "Error code: 402" in error_text:
#             provider_name = "Groq" if (GROQ_API_KEY or "groq" in LLM_API_BASE.lower()) else "OpenRouter"
#             raise RuntimeError(
#                 f"{provider_name} rejected the request بسبب الرصيد أو حدود الحساب. "
#                 f"خفّض max_tokens أو تحقق من المفتاح/الرصيد. التفاصيل: {error_text}"
#             ) from exc
#         if "Error code: 401" in error_text:
#             raise RuntimeError(
#                 f"فشل التوثيق مع المزود الحالي ({inferred_provider}). "
#                 f"تحقق من LLM_PROVIDER والمفتاح وإعدادات base_url. التفاصيل: {error_text}"
#             ) from exc
#         if "Connection error" in error_text or "Failed to establish a new connection" in error_text:
#             raise RuntimeError(
#                 "تعذر الاتصال بـ Ollama المحلي. تأكد أن خدمة Ollama تعمل وأن الموديل تم سحبه محليًا."
#             ) from exc
#         raise

#     if stream:
#         return "".join(chunk.choices[0].delta.content or "" for chunk in resp)
#     return resp.choices[0].message.content or ""
"""
Ollama local client for LLM chat completions.
Uses Qwen3-vl:4b running locally via Ollama.
"""
import os
import re
from typing import Optional
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "Qwen3-vl:4b")


def _strip_thinking(text: str) -> str:
    """Removes model thinking blocks <think>...</think> and returns only the final answer."""
    # Remove <think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove any "thinking" key inside JSON-like strings
    text = re.sub(r'"thinking"\s*:\s*".*?"(?=\s*[,}])', "", text, flags=re.DOTALL)
    return text.strip()


def chat_completion(
    messages: list[dict[str, str]],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1000,
    stream: bool = False,
) -> str:
    m = model or OLLAMA_MODEL

    # ✅ Add /no_think in the system prompt to prevent long thinking chains
    has_system = any(msg.get("role") == "system" for msg in messages)
    if not has_system:
        messages = [
            {"role": "system", "content": "/no_think Answer directly without extended thinking."},
            *messages,
        ]

    payload = {
        "model":     m,
        "messages": messages,
        "stream":   False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=180,  # Increased timeout to accommodate model reasoning time
        )
        resp.raise_for_status()
        data    = resp.json()
        content = data.get("message", {}).get("content", "") or ""

        # ✅ Clean thinking tags before returning the response
        return _strip_thinking(content)

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to Ollama.\n"
            "Please run in your terminal:\n"
            "  ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Connection timed out — the model is responding slowly.\n"
            "Try reducing max_tokens or restart the Ollama service."
        )
    except Exception as exc:
        raise RuntimeError(f"Ollama error: {exc}")
################################################################################
# """
# PDF loader for Arabic legal documents.
# Extracts and normalizes Arabic text from PDF files.
# """

# import re
# import unicodedata
# from pathlib import Path
# from typing import Optional
# import arabic_reshaper
# from bidi.algorithm import get_display

# try:
#     import pdfplumber
#     PDFPLUMBER_AVAILABLE = True
# except ImportError:
#     PDFPLUMBER_AVAILABLE = False

# try:
#     from pypdf import PdfReader
#     PYPDF_AVAILABLE = True
# except ImportError:
#     PYPDF_AVAILABLE = False


# def normalize_arabic_text(text: str, apply_reversal_fix: bool = False) -> str:
#     """
#     Normalize Arabic text for consistency.
#     - Normalize Alef variants
#     - Remove tatweel (kashida)
#     - Normalize whitespace
#     - Remove zero-width characters
#     """
#     if not text or not text.strip():
#         return ""

#     # Normalize compatibility forms (including Arabic presentation forms)
#     text = unicodedata.normalize("NFKC", text)

#     # Attempt recovery for visually broken Arabic fragments only when requested.
#     if apply_reversal_fix and re.search(r"[\uFB50-\uFDFF\uFE70-\uFEFC]", text):
#         text = get_display(arabic_reshaper.reshape(text))

#     # Normalize Alef variants to standard Alef (ا)
#     text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")

#     # Remove tatweel (kashida) - U+0640
#     text = text.replace("\u0640", "")

#     # Remove zero-width characters
#     text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
#     text = text.replace("\ufeff", "")  # BOM

#     # Normalize multiple whitespace to single space
#     text = re.sub(r"\s+", " ", text)

#     # Strip leading/trailing whitespace
#     return text.strip()


# def load_pdf_with_pdfplumber(pdf_path: Path) -> str:
#     """Extract text using pdfplumber (better for complex layouts)."""
#     full_text = []
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 full_text.append(page_text)
#     return "\n".join(full_text)


# def load_pdf_with_pypdf(pdf_path: Path) -> str:
#     """Extract text using pypdf (fallback)."""
#     reader = PdfReader(pdf_path)
#     full_text = []
#     for page in reader.pages:
#         page_text = page.extract_text()
#         if page_text:
#             full_text.append(page_text)
#     return "\n".join(full_text)


# def load_pdf(
#     pdf_path: str | Path,
#     normalize: bool = True,
#     apply_reversal_fix: bool = False,
# ) -> str:
#     """
#     Load PDF and extract all text.
#     Uses pdfplumber if available, otherwise pypdf.

#     Args:
#         pdf_path: Path to PDF file
#         normalize: Whether to normalize Arabic text

#     Returns:
#         Extracted and optionally normalized text
#     """
#     path = Path(pdf_path)
#     if not path.exists():
#         raise FileNotFoundError(f"PDF not found: {path}")

#     if PDFPLUMBER_AVAILABLE:
#         raw_text = load_pdf_with_pdfplumber(path)
#     elif PYPDF_AVAILABLE:
#         raw_text = load_pdf_with_pypdf(path)
#     else:
#         raise ImportError("Install pdfplumber or pypdf: pip install pdfplumber pypdf")

#     if normalize:
#         raw_text = normalize_arabic_text(raw_text, apply_reversal_fix=apply_reversal_fix)

#     return raw_text
