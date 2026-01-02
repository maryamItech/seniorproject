"""RAG chain for retrieval and answer generation."""
import sys
from pathlib import Path

# Ensure project root is in path for local imports
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
try:
    from langchain_ollama import ChatOllama
except ImportError:
    # Fallback for older versions
    from langchain_community.chat_models import ChatOllama
from typing import List, Tuple, Optional, Dict, Any
import json
import re
import os
from config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TEMPERATURE_GENERATION,
    RETRIEVAL_K,
    EMBEDDING_MODEL_NAME,
)
from retrieval.multimodal_milvus_store import MultimodalMilvusStore
from utils.performance_profiler import get_profiler


class RAGChain:
    """RAG chain for question answering with retrieval."""
    
    def __init__(self, vector_store: MultimodalMilvusStore, llm: Optional[ChatOllama] = None):
        """
        Initialize RAG chain with vector store.
        
        Args:
            vector_store: Milvus vector store instance
            llm: Optional pre-initialized LLM instance (for reuse)
        """
        self.vector_store = vector_store
        # Lower threshold to 0.3 to allow more results (can be adjusted)
        # Cosine similarity: 0.3 is still reasonable for educational content
        self.similarity_threshold = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.3"))
        self.last_debug = {}
        # Use provided LLM or create new one
        if llm is not None:
            self.llm = llm
        else:
            # Use Ollama LLM (Qwen3-VL by default)
            self.llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=TEMPERATURE_GENERATION,
            )
        
        # Create prompt template (Multimodal RAG with Qwen3-VL)
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are a multimodal reasoning assistant running locally using the Qwen3-VL-4B model via Ollama.\n\n"
                    "You are part of a Retrieval-Augmented Generation (RAG) system.\n"
                    "You do NOT retrieve documents.\n"
                    "You do NOT generate embeddings.\n"
                    "You do NOT store data.\n\n"
                    "Your ONLY responsibility is understanding, reasoning, and explanation.\n\n"
                    "────────────────────────\n"
                    "INPUT MODES\n"
                    "────────────────────────\n\n"
                    "The user input may be ONE of the following:\n"
                    "1. Text only\n"
                    "2. Image only\n"
                    "3. Text + Image together\n\n"
                    "The input format is fixed and must be respected exactly.\n\n"
                    "────────────────────────\n"
                    "TEXT-ONLY MODE\n"
                    "────────────────────────\n\n"
                    "If the input contains ONLY text:\n\n"
                    "- Assume that retrieved documents (context) are already included in the prompt.\n"
                    "- Use ONLY the provided context.\n"
                    "- Do NOT use external knowledge.\n"
                    "- Do NOT hallucinate missing facts.\n"
                    "- Generate a clear, concise final answer.\n\n"
                    "If the context is insufficient:\n"
                    "- Explicitly state that the information is insufficient.\n\n"
                    "────────────────────────\n"
                    "IMAGE-ONLY MODE\n"
                    "────────────────────────\n\n"
                    "If the input contains ONLY an image:\n\n"
                    "- Carefully analyze the visual content.\n"
                    "- Identify any visible text, symbols, labels, numbers, or diagrams.\n"
                    "- If readable text exists:\n"
                    "  - Extract and return the text clearly.\n"
                    "- If no readable text exists:\n"
                    "  - Clearly state that the image contains no extractable text.\n\n"
                    "Do NOT generate embeddings.\n"
                    "Do NOT describe irrelevant details.\n\n"
                    "────────────────────────\n"
                    "TEXT + IMAGE MODE\n"
                    "────────────────────────\n\n"
                    "If both text and image are provided:\n\n"
                    "- First, analyze the image.\n"
                    "- Then, logically connect it to the text.\n"
                    "- Use strict visual reasoning.\n"
                    "- Do NOT assume unseen information.\n\n"
                    "────────────────────────\n"
                    "OUTPUT RULES\n"
                    "────────────────────────\n\n"
                    "- Be concise, factual, and structured.\n"
                    "- No markdown unless necessary.\n"
                    "- No extra commentary.\n"
                    "- No system explanations.\n\n"
                    "You are NOT:\n"
                    "- an embedding model\n"
                    "- a retrieval system\n"
                    "- a memory system\n\n"
                    "You ONLY reason over what is given."
                ),
            ),
            (
                "human",
                (
                    "Question:\n"
                    "{question}\n\n"
                    "Retrieved Chunks:\n"
                    "{context}\n\n"
                    "Task:\n"
                    "Produce an educational, concise, and coherent answer using ONLY the retrieved chunks above. "
                    "If important details are missing, clearly highlight these limitations in your answer."
                ),
            ),
        ])
        
        # Create document chain using LCEL (LangChain Expression Language)
        # This approach works with all LangChain versions
        self.output_parser = StrOutputParser()
        
        # Try to use create_stuff_documents_chain if available
        try:
            from langchain.chains.combine_documents import create_stuff_documents_chain
            self.document_chain = create_stuff_documents_chain(
                llm=self.llm,
                prompt=self.prompt
            )
            self.use_chain_function = True
        except ImportError:
            # Fallback: Use LCEL approach with proper document formatting
            from langchain_core.runnables import RunnableLambda
            
            def format_docs(input_dict):
                """Format documents for the prompt."""
                docs = input_dict["context"]
                question = input_dict["question"]
                context_text = "\n\n".join(doc.page_content for doc in docs)
                return {"context": context_text, "question": question}
            
            self.document_chain = (
                RunnableLambda(format_docs)
                | self.prompt
                | self.llm
                | self.output_parser
            )
            self.use_chain_function = False

    def diagnostic_report(
        self,
        original_question: str,
        corrected_question: str,
        rephrased_question: str,
        top_k: int = RETRIEVAL_K,
    ) -> str:
        """
        Build a structured diagnostic report for zero/poor retrieval cases.
        This does NOT attempt to answer the question.
        """
        client = self.vector_store.client
        collection = self.vector_store.collection_name

        # 1) Data existence
        collections = []
        total_vectors = "unknown"
        try:
            collections = client.list_collections()
        except Exception as e:
            collections = [f"error: {e}"]
        try:
            stats = client.get_collection_stats(collection)
            total_vectors = stats.get("row_count", "unknown")
        except Exception as e:
            total_vectors = f"error: {e}"

        # 2) Embedding consistency
        query_dim = "unknown"
        query_vec = None
        try:
            # Use Visualized_BGE encode method for text
            query_text = rephrased_question or original_question
            query_vec_tensor = self.vector_store.embedding_model.encode(text=query_text)
            query_vec = query_vec_tensor[0].cpu().detach().numpy().astype("float32")
            query_dim = len(query_vec)
        except Exception as e:
            query_dim = f"error: {e}"
        doc_model = EMBEDDING_MODEL_NAME
        query_model = EMBEDDING_MODEL_NAME

        # 3) Collection match
        target_collection = collection
        available_collections = collections

        # 5) Similarity search settings
        similarity_metric = "COSINE"
        similarity_threshold = "none"
        search_top_k = top_k

        # 6) Raw search results (best-effort, without filters)
        raw_scores = []
        raw_ids = []
        try:
            if isinstance(query_dim, int) and query_vec is not None:
                raw_results = client.search(
                    collection_name=collection,
                    data=[query_vec.tolist()],
                    anns_field="text_embedding",  # Use text_embedding field for text queries
                    limit=min(top_k, 5),
                    search_params={"metric_type": "COSINE", "params": {}},
                    output_fields=["id"],
                )
                if raw_results and raw_results[0]:
                    for hit in raw_results[0]:
                        raw_scores.append(hit.get("distance", hit.get("score", None)))
                        raw_ids.append(hit.get("id", None))
        except Exception as e:
            raw_scores = [f"error: {e}"]

        # 7) Metadata filters (none applied)
        metadata_filters = "none"

        # Determine reason
        reason = "No relevant documents were retrieved; strict RAG mode blocks answering."
        failing_step = 1
        suggested_fix = "Check data existence and embedding/collection consistency."

        if self.last_debug:
            passed = self.last_debug.get("passed_threshold_count", 0)
            raw_cnt = self.last_debug.get("raw_results_count", 0)
            if raw_cnt > 0 and passed == 0:
                reason = "No vectors passed similarity threshold"
                failing_step = 5
                suggested_fix = "Lower threshold or verify embedding consistency."
            elif raw_cnt == 0:
                reason = "Vector search returned zero results"
                failing_step = 1
                suggested_fix = "Verify collection data and search parameters."

        report = {
            "reason_for_zero_retrieval": reason,
            "failing_step_number": failing_step,
            "suggested_fix": suggested_fix,
            "checks": {
                "data_existence": {
                    "target_collection": target_collection,
                    "total_vectors": total_vectors,
                    "available_collections": available_collections,
                },
                "embedding_consistency": {
                    "document_embedding_model": doc_model,
                    "query_embedding_model": query_model,
                    "embedding_dimension": query_dim,
                },
                "collection_match": {
                    "target_collection": target_collection,
                    "available_collections": available_collections,
                },
                "query_quality": {
                    "original_question": original_question,
                    "corrected_question": corrected_question,
                    "rephrased_question": rephrased_question,
                },
                "similarity_search": {
                    "metric": similarity_metric,
                    "threshold": similarity_threshold,
                    "top_k": search_top_k,
                },
                "raw_search_results": {
                    "top_k_scores": raw_scores,
                    "document_ids": raw_ids,
                },
                "metadata_filters": metadata_filters,
                "last_debug": self.last_debug,
            },
        }

        import json

        return json.dumps(report, indent=2)
    
    def retrieve(self, query: str, k: int = RETRIEVAL_K, search_type: str = "text", image_path: Optional[str] = None) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents."""
        profiler = get_profiler()
        
        # If image_path is provided and search_type is "image", use image_path as query
        if search_type == "image" and image_path:
            query_for_search = image_path
        else:
            query_for_search = query
        
        with profiler.stage("6_vector_database_retrieval", {
            "search_type": search_type,
            "k": k,
            "has_image": bool(image_path)
        }):
            raw_results = self.vector_store.similarity_search_with_score(query_for_search, k, search_type=search_type)

            debug_entries = []
            filtered: List[Tuple[Document, float]] = []

            for doc, score in raw_results:
                # Milvus may return cosine distance (0=best) or cosine similarity (1=best).
                # Use a robust heuristic: accept if either (1 - distance) >= threshold OR score >= threshold.
                distance = float(score)
                similarity_from_distance = 1.0 - distance
                similarity_direct = distance  # if the backend already returns similarity
                similarity = max(similarity_from_distance, similarity_direct)
                doc_id = None
                has_metadata = False
                missing_id = False
                try:
                    has_metadata = bool(doc.metadata)
                    doc_id = doc.metadata.get("index") if has_metadata else None
                    missing_id = doc.metadata.get("missing_id", False) if has_metadata else False
                except Exception:
                    has_metadata = False

                debug_entries.append(
                    {
                        "score": distance,
                        "similarity_used": similarity,
                        "similarity_from_distance": similarity_from_distance,
                        "similarity_direct": similarity_direct,
                        "doc_id": doc_id,
                        "has_metadata": has_metadata,
                        "missing_id": missing_id,
                        "has_content": bool(getattr(doc, "page_content", "")),
                    }
                )

                # Keep vectors if similarity passes threshold (either interpretation).
                # Allow empty content if id missing to avoid false zero retrieval; warn later.
                if similarity >= self.similarity_threshold:
                    # Return (doc, similarity) for consistency with UI display
                    filtered.append((doc, similarity))

            self.last_debug = {
            "collection": self.vector_store.collection_name,
            "threshold": self.similarity_threshold,
            "top_k_requested": k,
            "raw_results_count": len(raw_results),
            "passed_threshold_count": len(filtered),
                "entries": debug_entries,
            }

            # Print debug info for transparency
            print("[RAGChain][DEBUG] Retrieval summary:")
            print(json.dumps(self.last_debug, indent=2))
            
            # If no results passed threshold but we have raw results, log warning
            if len(filtered) == 0 and len(raw_results) > 0:
                print(f"[RAGChain][WARNING] No results passed threshold {self.similarity_threshold}")
                print(f"[RAGChain][WARNING] Raw results: {len(raw_results)}")
                print(f"[RAGChain][WARNING] Top similarity scores: {[entry.get('similarity_used', 'N/A') for entry in debug_entries[:3]]}")
            elif len(filtered) < len(raw_results):
                print(f"[RAGChain][INFO] Filtered {len(raw_results)} results to {len(filtered)} (threshold: {self.similarity_threshold})")

        return filtered
    
    def generate_answer(self, query: str, retrieved_docs: List[Tuple[Document, float]] = None) -> str:
        """
        Generate answer using retrieved context.
        OPTIMIZED: Uses ollama.chat() directly for faster inference (no LangChain overhead).
        """
        profiler = get_profiler()
        
        if retrieved_docs is None:
            retrieved_docs = self.retrieve(query)

        # Strict RAG: no answer if nothing retrieved
        if not retrieved_docs:
            return "No answer was generated because no relevant documents were retrieved from the knowledge base."

        # Warn (but still allow) if any metadata is missing
        missing_meta = any(
            not getattr(doc, "metadata", None) or doc.metadata.get("index") is None
            for doc, _ in retrieved_docs
        )
        if missing_meta:
            print("[RAGChain][WARN] Some retrieved vectors have missing metadata; proceeding with available content.")
        
        # Extract documents from tuples and truncate to 400 chars per document (optimization)
        docs = [doc for doc, _ in retrieved_docs]
        context_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content[:400]  # Truncate to 400 chars per document
            context_parts.append(f"[Document {i}]:\n{content}")
        context_text = "\n\n".join(context_parts)
        
        # Context selection and filtering (already part of retrieval, but we can time the filtering)
        with profiler.stage("7_context_selection_filtering", {
            "retrieved_count": len(retrieved_docs),
            "filtered_count": len(docs),
            "total_context_length": len(context_text)
        }):
            # Filtering logic is already done in retrieve(), this is just extraction
            pass
        
        # OPTIMIZED: Use ollama.chat() directly (much faster than LangChain wrapper)
        try:
            import ollama
            from config.settings import OLLAMA_MODEL
            
            print(f"[RAGChain][INFO] Generating answer using ollama.chat() directly (optimized for speed)")
            
            # Educational assistant prompt - allows basic knowledge completion if context is partial
            system_prompt = """You are an educational assistant answering a student question using a
Retrieval-Augmented Generation (RAG) system.

You will be given:
- A refined question
- Retrieved educational context

INSTRUCTIONS:
1. Base your answer primarily on the retrieved context.
2. If the context is partial or high-level:
   - Complete the explanation using basic, well-known educational knowledge.
3. Do NOT refuse to answer.
4. Do NOT mention missing documents or context limitations.
5. Do NOT invent advanced or highly technical details.
6. Be concise, clear, and student-friendly.

RESPONSE STYLE:
- Short definition
- Clear explanation
- Simple language
- No meta-commentary"""
            
            user_prompt = f"""QUESTION:
{query}

RETRIEVED CONTEXT:
{context_text}

Answer:"""

            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            answer = response["message"]["content"].strip()
            return answer
            
        except Exception as e:
            error_msg = str(e)
            print(f"[RAGChain][ERROR] Error generating answer with ollama.chat(): {error_msg}")
            # Fallback to LangChain chain if ollama.chat() fails
            try:
                print(f"[RAGChain][INFO] Falling back to LangChain chain")
                if self.use_chain_function:
                    result = self.document_chain.invoke({
                        "context": docs,
                        "question": query
                    })
                else:
                    result = self.document_chain.invoke({
                        "context": docs,
                        "question": query
                    })
                return result if isinstance(result, str) else str(result)
            except Exception as fallback_error:
                return f"Error generating answer: {error_msg} (fallback also failed: {str(fallback_error)})"
    
    def __call__(self, query: str) -> dict:
        """Run full RAG pipeline."""
        # Retrieve
        retrieved_docs = self.retrieve(query)
        
        # Generate answer
        answer = self.generate_answer(query, retrieved_docs)
        
        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs
        }

