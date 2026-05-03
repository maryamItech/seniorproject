# #

# """
# Reranker using cross-encoder.
# Optimized for NVIDIA RTX GPU and Arabic Legal Chunks.
# """

# from typing import Any
# import torch
# from sentence_transformers import CrossEncoder

# try:
#     from config import RERANKER_MODEL, TOP_K_RERANK
# except ModuleNotFoundError:
#     from .config import RERANKER_MODEL, TOP_K_RERANK

# try:
#     import streamlit as st
# except Exception:
#     st = None

# def _build_reranker_model(model_name: str) -> CrossEncoder:
#     # التحقق من توفر الـ GPU لاستخدامه في تسريع إعادة الترتيب
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"--- Reranker loading on: {device} ---")

#     return CrossEncoder(
#         model_name,
#         device=device,
#         # إضافة max_length لضمان عدم تجاوز حدود الموديل في المواد القانونية الطويلة
#         max_length=512
#     )

# if st:
#     _get_cached_reranker_model = st.cache_resource(show_spinner=False)(_build_reranker_model)
# else:
#     def _get_cached_reranker_model(model_name: str) -> CrossEncoder:
#         return _build_reranker_model(model_name)

# class LegalReranker:
#     """Cross-encoder reranker for legal relevance."""

#     def __init__(self, model_name: str = RERANKER_MODEL, top_k: int = TOP_K_RERANK):
#         self.model = _get_cached_reranker_model(model_name)
#         self.top_k = top_k

#     def rerank(
#         self,
#         query: str,
#         chunks: list[dict[str, Any]],
#     ) -> list[dict[str, Any]]:
#         """
#         Rerank chunks by relevance to query.
#         Returns top_k most relevant articles.
#         """
#         if not chunks:
#             return []

#         # استخراج النصوص فقط للمقارنة
#         pairs = [(query, c.get("text", "")) for c in chunks]

#         # التنبؤ بدرجات الصلة (Relevance Scores)
#         # استخدام batch_size صغير لتوفير ذاكرة الـ VRAM
#         scores = self.model.predict(pairs, batch_size=8, show_progress_bar=False)

#         # دمج الدرجات مع النصوص الأصلية
#         indexed = []
#         for i in range(len(chunks)):
#             indexed.append((float(scores[i]), chunks[i]))

#         # الترتيب من الأعلى درجة (الأكثر صلة) إلى الأقل
#         indexed.sort(key=lambda x: x[0], reverse=True)

#         # إرجاع أفضل النتائج (عادة أفضل 3 مواد)
#         return [chunk for _, chunk in indexed[: self.top_k]]
"""
Reranker using cross-encoder.
Optimized for NVIDIA RTX GPU and Arabic Legal Chunks.
Fixed: added minimum relevance score threshold.
"""

from typing import Any

import torch
from sentence_transformers import CrossEncoder

try:
    from config import RERANKER_MODEL, TOP_K_RERANK
except ModuleNotFoundError:
    from .config import RERANKER_MODEL, TOP_K_RERANK

try:
    import streamlit as st
except Exception:
    st = None

# ✅ Minimum relevance threshold — prevents low-relevance articles from appearing
MIN_RERANK_SCORE = -2.0


def _build_reranker_model(model_name: str) -> CrossEncoder:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Reranker loading on: {device} ---")
    return CrossEncoder(
        model_name,
        device=device,
        max_length=512,
    )


if st:
    _get_cached_reranker_model = st.cache_resource(show_spinner=False)(_build_reranker_model)
else:
    def _get_cached_reranker_model(model_name: str) -> CrossEncoder:
        return _build_reranker_model(model_name)


class LegalReranker:
    """Cross-encoder reranker for legal relevance."""

    def __init__(self, model_name: str = RERANKER_MODEL, top_k: int = TOP_K_RERANK):
        self.model = _get_cached_reranker_model(model_name)
        self.top_k = top_k

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not chunks:
            return []

        pairs  = [(query, c.get("text", "")) for c in chunks]
        scores = self.model.predict(pairs, batch_size=8, show_progress_bar=False)

        # Sort results from highest to lowest relevance
        indexed = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        # ✅ Filter out articles with highly negative relevance scores
        filtered = [
            chunk
            for score, chunk in indexed
            if float(score) >= MIN_RERANK_SCORE
        ]

        # If filtering removes all results -> fallback to original sorted list
        final = filtered or [chunk for _, chunk in indexed]
        return final[: self.top_k]