# import logging
# import os
# import re
# from typing import Any

# import chromadb

# try:
#     from config import (
#         CHROMA_COLLECTION_NAME,
#         TOP_K_BM25,
#         TOP_K_HYBRID,
#         TOP_K_VECTOR,
#         VECTOR_DB_PATH,
#     )
#     from bm25_search import BM25Index
#     from embeddings import ArabicEmbedder
#     from text_normalization import clean_text
# except ModuleNotFoundError:  # pragma: no cover
#     from .config import (
#         CHROMA_COLLECTION_NAME,
#         TOP_K_BM25,
#         TOP_K_HYBRID,
#         TOP_K_VECTOR,
#         VECTOR_DB_PATH,
#     )
#     from .bm25_search import BM25Index
#     from .embeddings import ArabicEmbedder
#     from .text_normalization import clean_text

# logger = logging.getLogger(__name__)


# class HybridRetriever:
#     MIN_SIMILARITY_SCORE = 0.0

#     def __init__(
#         self,
#         embedder: ArabicEmbedder,
#         bm25_index: BM25Index,
#         top_k_hybrid: int = TOP_K_HYBRID,
#     ):
#         self.embedder = embedder
#         self.bm25_index = bm25_index
#         self.top_k_hybrid = top_k_hybrid
#         self.collection_error_message = ""
#         self.db_path = os.path.abspath(str(VECTOR_DB_PATH))
#         self.chroma_client = chromadb.PersistentClient(
#             path=self.db_path,
#             settings=chromadb.Settings(allow_reset=True, anonymized_telemetry=False),
#         )
#         self.collection = self._init_collection_with_recovery()
#         self._ensure_collection_accessible()

#     def _init_collection_with_recovery(self):
#         try:
#             return self.chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
#         except Exception as exc:
#             logger.exception("Initial collection setup failed: %s", exc)
#             return self.chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)

#     def _ensure_collection_accessible(self):
#         if self.collection is None:
#             self.collection_error_message = "Collection was not initialized."
#             return
#         try:
#             _ = self.collection.count()
#             self.collection_error_message = ""
#         except Exception as exc:
#             self.collection_error_message = f"Collection access error: {exc}"
#             self.collection = None

#     def _normalize_scores(self, results: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
#         if not results:
#             return []
#         scores = [r[1] for r in results]
#         lo, hi = min(scores), max(scores)
#         if hi - lo == 0:
#             return [(r[0], 1.0) for r in results]
#         return [(r[0], (r[1] - lo) / (hi - lo)) for r in results]

#     @staticmethod
#     def _source_filter(database_route: str) -> dict[str, Any] | None:
#         route_to_sources = {
#             "civil": ["القانون المدني"],
#             "penal": ["قانون العقوبات"],
#             "personal_status": ["الأحوال الشخصية"],
#             "all": [],
#         }
#         sources = route_to_sources.get(database_route, [])
#         if not sources:
#             return None
#         return {"law_name": {"$in": sources}}

#     @staticmethod
#     def _metadata_matches_route(metadata: dict[str, Any], database_route: str) -> bool:
#         if database_route == "all":
#             return True
#         route_map = {
#             "civil": "القانون المدني",
#             "penal": "قانون العقوبات",
#             "personal_status": "الأحوال الشخصية",
#         }
#         target_law = route_map.get(database_route)
#         return metadata.get("law_name") == target_law

#     def _extract_cross_references(self, chunks: list[dict[str, Any]]) -> list[int]:
#         article_ids: set[int] = set()
#         for chunk in chunks:
#             text = str(chunk.get("text", "") or "")
#             for match in re.finditer(r"(?:الماد[هة]|Article)\s*(\d+)", text, flags=re.IGNORECASE):
#                 article_ids.add(int(match.group(1)))
#         return sorted(article_ids)

#     def fetch_by_article_numbers(self, article_numbers: list[int], database_route: str = "all") -> list[dict[str, Any]]:
#         if self.collection is None or not article_numbers:
#             return []

#         results: list[dict[str, Any]] = []
#         where_route = self._source_filter(database_route)

#         for n in article_numbers[:10]:
#             where_article = {"article_number": {"$eq": n}}
#             where = {"$and": [where_route, where_article]} if where_route else where_article

#             try:
#                 query = self.collection.get(where=where, limit=1)
#                 docs = query.get("documents") or []
#                 metas = query.get("metadatas") or []

#                 for i, doc in enumerate(docs):
#                     meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
#                     results.append({"text": doc or "", "metadata": meta})
#             except Exception as exc:
#                 logger.error("Error fetching article %s: %s", n, exc)
#         return results

#     def retrieve(self, query: str | list[str], database_route: str = "all") -> list[dict[str, Any]]:
#         self._ensure_collection_accessible()
#         if self.collection is None:
#             return []

#         try:
#             queries = query if isinstance(query, list) else [query]
#             queries = [clean_text(q, apply_reversal_fix=False) for q in queries if q]
#             if not queries:
#                 return []

#             where_filter = self._source_filter(database_route)

#             def _vector_search(active_where: dict[str, Any] | None) -> list[tuple[dict[str, Any], float]]:
#                 collected: list[tuple[dict[str, Any], float]] = []
#                 for q in queries:
#                     q_emb = self.embedder.encode_query(q)
#                     emb_list = q_emb.tolist() if hasattr(q_emb, "tolist") else list(q_emb)

#                     vector_results = self.collection.query(
#                         query_embeddings=[emb_list],
#                         n_results=TOP_K_VECTOR,
#                         where=active_where,
#                         include=["documents", "metadatas", "distances"],
#                     )

#                     docs_outer = vector_results.get("documents") or []
#                     metas_outer = vector_results.get("metadatas") or []
#                     dists_outer = vector_results.get("distances") or []

#                     docs = docs_outer[0] if docs_outer and docs_outer[0] else []
#                     metas = metas_outer[0] if metas_outer and metas_outer[0] else []
#                     dists = dists_outer[0] if dists_outer and dists_outer[0] else []

#                     safe_len = min(len(docs), len(metas), len(dists))
#                     for i in range(safe_len):
#                         doc = docs[i] or ""
#                         meta = metas[i] if isinstance(metas[i], dict) else {}
#                         dist = dists[i] if dists[i] is not None else 999.0
#                         score = 1.0 / (1.0 + float(dist))
#                         if score >= self.MIN_SIMILARITY_SCORE:
#                             collected.append(({"text": doc, "metadata": meta}, score))

#                 return collected

#             vector_chunks = _vector_search(where_filter)
#             if where_filter is not None and len(vector_chunks) < 2:
#                 vector_chunks = _vector_search(None)

#             bm25_raw: list[tuple[dict[str, Any], float]] = []
#             for q in queries:
#                 try:
#                     bm25_hits = self.bm25_index.search(q, top_k=TOP_K_BM25) or []
#                 except Exception as exc:
#                     logger.warning("BM25 search failed for query segment: %s", exc)
#                     bm25_hits = []
#                 bm25_raw.extend(bm25_hits)

#             if where_filter:
#                 bm25_raw = [
#                     res
#                     for res in bm25_raw
#                     if self._metadata_matches_route(res[0].get("metadata", {}), database_route)
#                 ]
#                 if len(bm25_raw) < 2:
#                     bm25_raw = []
#                     for q in queries:
#                         try:
#                             bm25_hits = self.bm25_index.search(q, top_k=max(TOP_K_BM25, 5)) or []
#                         except Exception as exc:
#                             logger.warning("BM25 fallback search failed for query segment: %s", exc)
#                             bm25_hits = []
#                         bm25_raw.extend(bm25_hits)

#             vector_norm = self._normalize_scores(vector_chunks)
#             bm25_norm = self._normalize_scores(bm25_raw)

#             fused: dict[str, dict[str, Any]] = {}
#             for rank, (chunk, _) in enumerate(vector_norm, start=1):
#                 key = str(chunk.get("text", ""))[:200]
#                 fused[key] = fused.get(key, {"chunk": chunk, "score": 0.0})
#                 fused[key]["score"] += 1.0 / (rank + 60)

#             for rank, (chunk, _) in enumerate(bm25_norm, start=1):
#                 key = str(chunk.get("text", ""))[:200]
#                 fused[key] = fused.get(key, {"chunk": chunk, "score": 0.0})
#                 fused[key]["score"] += 1.0 / (rank + 60)

#             sorted_chunks = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
#             top_chunks = [item["chunk"] for item in sorted_chunks[:TOP_K_HYBRID]]
#             if not top_chunks and vector_chunks:
#                 top_chunks = [
#                     chunk
#                     for chunk, _ in sorted(vector_chunks, key=lambda x: x[1], reverse=True)[: max(TOP_K_HYBRID, 5)]
#                 ]

#             refs = self._extract_cross_references(top_chunks)
#             hop_docs = self.fetch_by_article_numbers(refs, database_route=database_route)
#             final_pool = top_chunks + hop_docs

#             seen: set[str] = set()
#             final_results: list[dict[str, Any]] = []
#             for c in final_pool:
#                 text = str(c.get("text", "") or "")
#                 fingerprint = text[:200]
#                 if not fingerprint or fingerprint in seen:
#                     continue
#                 final_results.append({"text": text, "metadata": c.get("metadata", {}) or {}})
#                 seen.add(fingerprint)

#             return final_results[:TOP_K_HYBRID]
#         except Exception as exc:
#             logger.exception("Hybrid retrieve failed: %s", exc)
#             return []

#     def get_collection_count(self) -> int:
#         self._ensure_collection_accessible()
#         if self.collection is None:
#             return 0
#         try:
#             return int(self.collection.count())
#         except Exception:
#             return 0
"""
Hybrid Retriever: Vector + BM25 with RRF fusion.
Fixed: RRF score threshold + disabled noisy cross-references.
"""

import logging
import os
import re
from typing import Any

import chromadb

try:
    from config import (
        CHROMA_COLLECTION_NAME,
        TOP_K_BM25,
        TOP_K_HYBRID,
        TOP_K_VECTOR,
        VECTOR_DB_PATH,
    )
    from bm25_search import BM25Index
    from embeddings import ArabicEmbedder
    from text_normalization import clean_text
except ModuleNotFoundError:
    from .config import (
        CHROMA_COLLECTION_NAME,
        TOP_K_BM25,
        TOP_K_HYBRID,
        TOP_K_VECTOR,
        VECTOR_DB_PATH,
    )
    from .bm25_search import BM25Index
    from .embeddings import ArabicEmbedder
    from .text_normalization import clean_text

logger = logging.getLogger(__name__)

MIN_RRF_SCORE = 0.008


class HybridRetriever:
    MIN_SIMILARITY_SCORE = 0.0

    def __init__(
        self,
        embedder: ArabicEmbedder,
        bm25_index: BM25Index,
        top_k_hybrid: int = TOP_K_HYBRID,
    ):
        self.embedder = embedder
        self.bm25_index = bm25_index
        self.top_k_hybrid = top_k_hybrid
        self.collection_error_message = ""
        self.db_path = os.path.abspath(str(VECTOR_DB_PATH))
        self.chroma_client = chromadb.PersistentClient(
            path=self.db_path,
            settings=chromadb.Settings(allow_reset=True, anonymized_telemetry=False),
        )
        self.collection = self._init_collection_with_recovery()
        self._ensure_collection_accessible()

    def _init_collection_with_recovery(self):
        try:
            return self.chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        except Exception as exc:
            logger.exception("Initial collection setup failed: %s", exc)
            return self.chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)

    def _ensure_collection_accessible(self):
        if self.collection is None:
            self.collection_error_message = "Collection was not initialized."
            return
        try:
            _ = self.collection.count()
            self.collection_error_message = ""
        except Exception as exc:
            self.collection_error_message = f"Collection access error: {exc}"
            self.collection = None

    def _normalize_scores(self, results: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
        if not results:
            return []
        scores = [r[1] for r in results]
        lo, hi = min(scores), max(scores)
        if hi - lo == 0:
            return [(r[0], 1.0) for r in results]
        return [(r[0], (r[1] - lo) / (hi - lo)) for r in results]

    @staticmethod
    def _source_filter(database_route: str) -> dict[str, Any] | None:
        route_to_sources = {
            "civil": ["القانون المدني"],
            "penal": ["قانون العقوبات"],
            "personal_status": ["الأحوال الشخصية"],
            "all": [],
        }
        sources = route_to_sources.get(database_route, [])
        if not sources:
            return None
        return {"law_name": {"$in": sources}}

    @staticmethod
    def _metadata_matches_route(metadata: dict[str, Any], database_route: str) -> bool:
        if database_route == "all":
            return True
        route_map = {
            "civil": "القانون المدني",
            "penal": "قانون العقوبات",
            "personal_status": "الأحوال الشخصية",
        }
        target_law = route_map.get(database_route)
        return metadata.get("law_name") == target_law

    def fetch_by_article_numbers(
        self, article_numbers: list[int], database_route: str = "all"
    ) -> list[dict[str, Any]]:

        return []

    def retrieve(self, query: str | list[str], database_route: str = "all") -> list[dict[str, Any]]:
        self._ensure_collection_accessible()
        if self.collection is None:
            return []

        try:
            queries = query if isinstance(query, list) else [query]
            queries = [clean_text(q, apply_reversal_fix=False) for q in queries if q]
            if not queries:
                return []

            where_filter = self._source_filter(database_route)

            def _vector_search(active_where: dict[str, Any] | None) -> list[tuple[dict[str, Any], float]]:
                collected: list[tuple[dict[str, Any], float]] = []
                for q in queries:
                    q_emb = self.embedder.encode_query(q)
                    emb_list = q_emb.tolist() if hasattr(q_emb, "tolist") else list(q_emb)

                    vector_results = self.collection.query(
                        query_embeddings=[emb_list],
                        n_results=TOP_K_VECTOR,
                        where=active_where,
                        include=["documents", "metadatas", "distances"],
                    )

                    docs_outer = vector_results.get("documents") or []
                    metas_outer = vector_results.get("metadatas") or []
                    dists_outer = vector_results.get("distances") or []

                    docs  = docs_outer[0]  if docs_outer  and docs_outer[0]  else []
                    metas = metas_outer[0] if metas_outer and metas_outer[0] else []
                    dists = dists_outer[0] if dists_outer and dists_outer[0] else []

                    safe_len = min(len(docs), len(metas), len(dists))
                    for i in range(safe_len):
                        doc   = docs[i] or ""
                        meta  = metas[i] if isinstance(metas[i], dict) else {}
                        dist  = dists[i] if dists[i] is not None else 999.0
                        score = 1.0 / (1.0 + float(dist))
                        if score >= self.MIN_SIMILARITY_SCORE:
                            collected.append(({"text": doc, "metadata": meta}, score))

                return collected

            vector_chunks = _vector_search(where_filter)
            if where_filter is not None and len(vector_chunks) < 2:
                vector_chunks = _vector_search(None)

            bm25_raw: list[tuple[dict[str, Any], float]] = []
            for q in queries:
                try:
                    bm25_hits = self.bm25_index.search(q, top_k=TOP_K_BM25) or []
                except Exception as exc:
                    logger.warning("BM25 search failed: %s", exc)
                    bm25_hits = []
                bm25_raw.extend(bm25_hits)

            if where_filter:
                bm25_raw = [
                    res for res in bm25_raw
                    if self._metadata_matches_route(res[0].get("metadata", {}), database_route)
                ]
                if len(bm25_raw) < 2:
                    bm25_raw = []
                    for q in queries:
                        try:
                            bm25_hits = self.bm25_index.search(q, top_k=max(TOP_K_BM25, 5)) or []
                        except Exception as exc:
                            logger.warning("BM25 fallback failed: %s", exc)
                            bm25_hits = []
                        bm25_raw.extend(bm25_hits)

            vector_norm = self._normalize_scores(vector_chunks)
            bm25_norm   = self._normalize_scores(bm25_raw)

            fused: dict[str, dict[str, Any]] = {}
            for rank, (chunk, _) in enumerate(vector_norm, start=1):
                key = str(chunk.get("text", ""))[:200]
                fused[key] = fused.get(key, {"chunk": chunk, "score": 0.0})
                fused[key]["score"] += 1.0 / (rank + 60)

            for rank, (chunk, _) in enumerate(bm25_norm, start=1):
                key = str(chunk.get("text", ""))[:200]
                fused[key] = fused.get(key, {"chunk": chunk, "score": 0.0})
                fused[key]["score"] += 1.0 / (rank + 60)

            sorted_chunks = sorted(fused.values(), key=lambda x: x["score"], reverse=True)

            filtered_chunks = [item for item in sorted_chunks if item["score"] >= MIN_RRF_SCORE]
            top_chunks = [
                item["chunk"]
                for item in (filtered_chunks or sorted_chunks)[:TOP_K_HYBRID]
            ]

            if not top_chunks and vector_chunks:
                top_chunks = [
                    chunk
                    for chunk, _ in sorted(vector_chunks, key=lambda x: x[1], reverse=True)[: max(TOP_K_HYBRID, 5)]
                ]

            final_pool = top_chunks

            seen: set[str] = set()
            final_results: list[dict[str, Any]] = []
            for c in final_pool:
                text        = str(c.get("text", "") or "")
                fingerprint = text[:200]
                if not fingerprint or fingerprint in seen:
                    continue
                final_results.append({"text": text, "metadata": c.get("metadata", {}) or {}})
                seen.add(fingerprint)

            return final_results[:TOP_K_HYBRID]

        except Exception as exc:
            logger.exception("Hybrid retrieve failed: %s", exc)
            return []

    def get_collection_count(self) -> int:
        self._ensure_collection_accessible()
        if self.collection is None:
            return 0
        try:
            return int(self.collection.count())
        except Exception:
            return 0