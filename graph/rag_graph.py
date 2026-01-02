"""LangGraph state machine for orchestrating the RAG workflow."""
import sys
from pathlib import Path

# Ensure project root is in path for local imports
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import TypedDict, List, Optional, Literal, Dict, Any
from langgraph.graph import StateGraph, END
import networkx as nx

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

from chains.question_processing import CombinedQuestionProcessingChain
from chains.rag_chain import RAGChain
from retrieval.multimodal_milvus_store import MultimodalMilvusStore
from tools.langchain_tools import create_all_tools
from tools.web_search import look_on_web
# from tools.knowledge_base import look_in_your_knowledge  # Removed - not used in TEXT-ONLY pipeline (single retrieval call only)
# from tools.ocr_tool import OCRTool  # No longer needed - using Qwen3-VL for OCR
from tools.scienceqa import scienceqa_helper
from tools.image_generation import generate_image
from tools.image_processing import ImagePreprocessor
from config.settings import DEV_MODE, RETRIEVAL_K, OLLAMA_MODEL, OLLAMA_NUM_GPU, OLLAMA_NUM_THREAD
from utils.performance_profiler import PerformanceProfiler, get_profiler
try:
    from langchain_ollama import ChatOllama
except ImportError:
    # Fallback for older versions
    from langchain_community.chat_models import ChatOllama


class RAGState(TypedDict):
    original_question: str
    corrected_question: str
    rephrased_question: str
    question_parts: List[str]
    retrieved_docs: List
    answer: str
    error: str
    input_type: Literal["text", "image", "scientific", "image_generation"]
    tool_results: dict
    processed_image: Optional[str]
    ocr_text: Optional[str]
    used_tools: List[str]
    user_text_before_ocr: Optional[str]  # Store original user text before OCR
    image_content_type: Optional[Literal["textual", "visual"]]  # Image content type: textual or visual (deprecated, use processing_mode)
    processing_mode: Optional[Literal["text_only", "visual_only", "text_and_visual"]]  # Processing mode determined by perception
    contains_text: Optional[bool]  # Boolean decision: does image contain readable text? (IMAGE PIPELINE)
    selected_images: Optional[List[Dict]]  # Selected images for display from retrieved documents
    visual_description: Optional[str]  # Textual description generated from visual image
    detected_questions_count: Optional[int]  # Number of detected questions (Step 5 - UI Feedback)


class RAGGraph:
    def __init__(
        self,
        vector_store: MultimodalMilvusStore,
        correction_llm: Optional[ChatOllama] = None,
        generation_llm: Optional[ChatOllama] = None,
        profiler: Optional[PerformanceProfiler] = None,
    ):
        """
        Initialize RAG graph with reusable components.
        
        Args:
            vector_store: Multimodal Milvus vector store instance
            correction_llm: Optional pre-initialized LLM for question processing
            generation_llm: Optional pre-initialized LLM for answer generation
            profiler: Optional performance profiler instance
        """
        self.vector_store = vector_store
        # Use provided components or create new ones (for backward compatibility)
        self.question_processor = CombinedQuestionProcessingChain(
            llm=correction_llm
        )
        self.rag_chain = RAGChain(vector_store, llm=generation_llm)
        self.tools = create_all_tools(vector_store)
        self.preprocessor = ImagePreprocessor()
        self.graph_edges = []
        self.profiler = profiler or get_profiler()
        self.graph = self._build_graph()
    

    def _record_edge(self, src: str, dst: str, label: str = ""):
        self.graph_edges.append((src, dst, label))

    def _build_graph(self):
        workflow = StateGraph(RAGState)

        workflow.add_node("routing_node", self._route_input)
        workflow.add_node("correct_question", self._correct_question)
        workflow.add_node("rephrase_question", self._rephrase_question)
        workflow.add_node("split_question", self._split_question)
        workflow.add_node("kb_query_node", self._check_knowledge_base)
        workflow.add_node("presentation_control_node", self._select_images_for_presentation)
        workflow.add_node("web_search_node", self._search_web)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("combine_results", self._combine_results)
        workflow.add_node("image_processing_node", self._process_image)
        workflow.add_node("perception_routing_node", self._perception_and_routing)
        workflow.add_node("visual_understanding_node", self._visual_understanding)
        workflow.add_node("ocr_node", self._extract_text_from_image)
        workflow.add_node("scienceqa_retriever_node", self._scienceqa_retrieval)
        workflow.add_node("image_generation_node", self._generate_image_tool)

        workflow.set_entry_point("routing_node")

        workflow.add_conditional_edges(
            "routing_node",
            self._route_input_type,
            {
                "text": "correct_question",
                "image": "image_processing_node",
                "scientific": "scienceqa_retriever_node",
                "image_generation": "image_generation_node",
            },
        )

        self._record_edge("routing_node", "correct_question", "text")
        self._record_edge("routing_node", "image_processing_node", "image")
        self._record_edge("routing_node", "scienceqa_retriever_node", "scientific")
        self._record_edge("routing_node", "image_generation_node", "image_generation")

        workflow.add_edge("correct_question", "rephrase_question")
        workflow.add_edge("rephrase_question", "split_question")
        workflow.add_edge("split_question", "kb_query_node")

        # TEXT PIPELINE: Skip presentation control and web search (STRICT TEXT-ONLY)
        # IMAGE PIPELINE: Skip presentation control (goes directly to answer generation)
        workflow.add_conditional_edges(
            "kb_query_node",
            self._should_skip_presentation_control,
            {
                "skip_presentation": "generate_answer",  # TEXT-ONLY & IMAGE PIPELINE: skip presentation control
                "use_presentation": "presentation_control_node",  # Legacy path (not used for TEXT-ONLY)
            },
        )
        
        workflow.add_conditional_edges(
            "presentation_control_node",
            self._should_use_web,
            {
                "use_kb": "generate_answer",
                "use_web": "web_search_node",
            },
        )

        workflow.add_edge("web_search_node", "generate_answer")
        workflow.add_edge("generate_answer", "combine_results")
        workflow.add_edge("combine_results", END)

        workflow.add_edge("image_processing_node", "perception_routing_node")
        
        # After Perception & Routing, determine path based on contains_text boolean
        workflow.add_conditional_edges(
            "perception_routing_node",
            self._route_after_perception,
            {
                "textual_path": "ocr_node",  # contains_text=true → extract text
                "visual_path": "kb_query_node",  # contains_text=false → skip visual understanding, go directly to retrieval
            }
        )
        
        # Note: visual_understanding_node is kept for backward compatibility but is no longer used in IMAGE PIPELINE
        # When contains_text=false, we use image directly for embedding (no text conversion needed)
        workflow.add_edge("visual_understanding_node", "kb_query_node")
        
        # After OCR, skip text processing for image-only queries (no user text)
        workflow.add_conditional_edges(
            "ocr_node",
            self._should_process_text_after_ocr,
            {
                "skip_processing": "kb_query_node",  # Skip text processing for image-only
                "process_text": "correct_question"  # Process text for text + image
            }
        )

        workflow.add_edge("scienceqa_retriever_node", "kb_query_node")
        workflow.add_edge("image_generation_node", END)

        return workflow.compile()

    # ==================== Nodes ====================

    def _needs_text_processing(self, text: str) -> dict:
        """
        فحص النص لتحديد ما إذا كان يحتاج تصحيح/إعادة صياغة/تقسيم.
        
        Returns:
            dict with keys: needs_correction, needs_rephrasing, needs_splitting
        """
        # التعامل مع None والقيم الفارغة
        if text is None:
            return {
                "needs_correction": False,
                "needs_rephrasing": False,
                "needs_splitting": False,
                "reason": "empty_text"
            }
        
        text = str(text).strip() if text else ""
        
        if not text:
            return {
                "needs_correction": False,
                "needs_rephrasing": False,
                "needs_splitting": False,
                "reason": "empty_text"
            }
        
        # فحص بسيط: إذا كان النص قصير جدًا أو طويل جدًا، قد يحتاج معالجة
        # لكن سنعتمد على فحص بسيط أولاً
        
        # فحص الأخطاء الإملائية الشائعة (مثال بسيط)
        # يمكن تحسين هذا لاحقًا
        common_typos = {
            "فوتوسينثيسيس": "التمثيل الضوئي",
            "فوتوسينثيس": "التمثيل الضوئي",
        }
        
        has_typo = any(typo in text for typo in common_typos.keys())
        
        # فحص إذا كان النص يحتوي على أكثر من سؤال (علامات استفهام متعددة)
        question_marks = text.count("؟") + text.count("?")
        needs_splitting = question_marks > 1
        
        # فحص إذا كان النص غير واضح (قصير جدًا أو طويل جدًا بدون علامات ترقيم)
        is_unclear = len(text) < 5 or (len(text) > 200 and "؟" not in text and "?" not in text)
        
        # إذا كان النص واضحًا وبدون أخطاء واضحة، لا حاجة للمعالجة
        if not has_typo and not needs_splitting and not is_unclear:
            return {
                "needs_correction": False,
                "needs_rephrasing": False,
                "needs_splitting": False,
                "reason": "text_is_clear"
            }
        
        return {
            "needs_correction": has_typo,
            "needs_rephrasing": is_unclear,
            "needs_splitting": needs_splitting,
            "reason": f"typo={has_typo}, unclear={is_unclear}, splitting={needs_splitting}"
        }

    def _route_input(self, state: RAGState) -> RAGState:
        """
        Determine input type. If the UI/handler already set input_type, respect it.
        Otherwise fall back to simple heuristic on the question text.
        """
        with self.profiler.stage("1_user_input_reception", {
            "has_text": bool(state.get("original_question")),
            "has_image": bool(state.get("processed_image")),
            "input_type": state.get("input_type", "unknown")
        }):
            if state.get("input_type"):
                return state

            q = state.get("original_question", "").lower()
            if "image" in q and ("generate" in q or "create" in q):
                state["input_type"] = "image_generation"
            elif any(k in q for k in ["physics", "chemistry", "biology", "science"]):
                state["input_type"] = "scientific"
            elif q.endswith((".png", ".jpg", ".jpeg")):
                state["input_type"] = "image"
            else:
                state["input_type"] = "text"
        return state

    def _route_input_type(self, state: RAGState) -> str:
        return state.get("input_type", "text")
    
    def _should_skip_presentation_control(self, state: RAGState) -> str:
        """
        TEXT-ONLY PIPELINE: Skip presentation control (STRICT - no images).
        IMAGE PIPELINE: Skip presentation control for image queries.
        """
        input_type = state.get("input_type", "text")
        processed_image = state.get("processed_image")
        
        # TEXT-ONLY PIPELINE: STRICT - Skip presentation control (no images, no LLM calls)
        if input_type == "text" and not processed_image:
            print("[RAGGraph][INFO] TEXT-ONLY PIPELINE: Skipping presentation control (STRICT - no images)")
            return "skip_presentation"
        
        # IMAGE PIPELINE: Skip presentation control for image-only queries
        if input_type == "image" and processed_image and not state.get("user_text_before_ocr"):
            print("[RAGGraph][INFO] IMAGE PIPELINE: Skipping presentation control")
            return "skip_presentation"
        
        # Legacy path (should not be reached for TEXT-ONLY)
        print("[RAGGraph][WARNING] Using presentation control (legacy path)")
        return "use_presentation"
    
    def _should_process_text_after_ocr(self, state: RAGState) -> str:
        """
        Determine whether text should be processed after OCR.
        
        Rules:
        - Image only (no user text): skip processing
        - Text + image: process user text only
        """
        user_text_before_ocr_raw = state.get("user_text_before_ocr") or ""
        user_text_before_ocr = str(user_text_before_ocr_raw).strip() if user_text_before_ocr_raw else ""
        ocr_text_raw = state.get("ocr_text") or ""
        ocr_text = str(ocr_text_raw).strip() if ocr_text_raw else ""
        
        # If there is user text before OCR, it must be processed
        if user_text_before_ocr:
            # Text + image - process user text
            print(f"[RAGGraph][INFO] Text + Image: processing user text '{user_text_before_ocr[:50]}...'")
            return "process_text"
        elif ocr_text:
            # Image only - skip processing (use extracted text as-is)
            print(f"[RAGGraph][INFO] Image-only: skipping text processing after OCR (using extracted text as-is)")
            return "skip_processing"
        else:
            # No text and no OCR - skip processing
            print(f"[RAGGraph][INFO] No text found - skipping text processing")
            return "skip_processing"

    def _correct_question(self, state: RAGState) -> RAGState:
        """
        Text processing: correction/rephrasing/splitting only when needed.
        
        Rules:
        - Text only: check first, process only if necessary
        - Image only: no processing (this function is skipped)
        - Text + image: check user text only, process only if necessary
        """
        # Use original user text (before merging OCR) for processing
        user_text_before_ocr = state.get("user_text_before_ocr") or ""
        user_text = str(user_text_before_ocr).strip() if user_text_before_ocr else ""
        if not user_text:
            # If there is no user text, use original_question
            original_question = state.get("original_question") or ""
            user_text = str(original_question).strip() if original_question else ""
        
        original_q = user_text  # User text for processing
        input_type = state.get("input_type", "text")
        processed_image = state.get("processed_image")
        ocr_text_raw = state.get("ocr_text") or ""
        ocr_text = str(ocr_text_raw).strip() if ocr_text_raw else ""
        
        with self.profiler.stage("4_text_correction_rephrasing", {
            "has_text": bool(original_q),
            "has_ocr_text": bool(ocr_text),
            "input_type": input_type
        }):
            # Case 1: Image only without text (should not reach here, but for safety)
            if not original_q and input_type == "image":
                print("[RAGGraph][INFO] Image-only query with no text - skipping text processing")
                state["corrected_question"] = ""
                state["rephrased_question"] = ""
                state["question_parts"] = []
                return state
            
            # Case 2: TEXT-ONLY PIPELINE - Step 2: Question Correction (ALWAYS)
            if input_type == "text" and not processed_image:
                print("[RAGGraph][INFO] TEXT-ONLY PIPELINE: Step 2 - Question Correction (ALWAYS)")
                try:
                    # Step 2: ALWAYS correct spelling and grammar (mandatory step)
                    corrected = self.question_processor.correction_chain.correct(original_q)
                    print(f"[RAGGraph][INFO] Step 2 - Corrected: '{original_q}' -> '{corrected}'")
                    state["corrected_question"] = corrected
                    
                    # Note: Rephrasing (Step 3) and Splitting (Step 4) will be done in separate nodes
                    # Don't set rephrased_question or question_parts here
                    
                except Exception as e:
                    print(f"[RAGGraph][ERROR] Step 2 - Correction failed: {e}")
                    state["corrected_question"] = original_q
                return state
            
            # Case 3: Text + image queries - NO normalization (keep text as-is)
            # TEXT-ONLY PIPELINE ONLY: correction/rephrase/split are NOT applied to text+image queries
            print("[RAGGraph][INFO] Text + Image pipeline: Keeping user text as-is (NO normalization - only for TEXT-ONLY)")
            state["corrected_question"] = original_q
            state["rephrased_question"] = f"{original_q}\n\n{ocr_text}" if ocr_text else original_q
            state["question_parts"] = [original_q]
        return state

    def _rephrase_question(self, state: RAGState) -> RAGState:
        """
        TEXT-ONLY PIPELINE: Step 3 - Question Rephrasing.
        
        Rephrase the corrected question to be:
        - Clear
        - Grammatically correct
        - Explicit and unambiguous
        - Preserve the original intent
        
        Output: rephrased_question
        """
        input_type = state.get("input_type", "text")
        processed_image = state.get("processed_image")
        
        # TEXT-ONLY PIPELINE: Step 3 - ALWAYS rephrase for clarity (mandatory step)
        if input_type == "text" and not processed_image:
            corrected = state.get("corrected_question") or ""
            original = state.get("original_question") or ""
            text_to_rephrase = corrected if corrected else original
            
            if text_to_rephrase:
                try:
                    print(f"[RAGGraph][INFO] TEXT-ONLY PIPELINE: Step 3 - Question Rephrasing (ALWAYS)")
                    rephrased = self.question_processor.rephrasing_chain.rephrase(text_to_rephrase)
                    state["rephrased_question"] = rephrased
                    print(f"[RAGGraph][INFO] Step 3 - Rephrased: '{text_to_rephrase}' -> '{rephrased}'")
                except Exception as e:
                    print(f"[RAGGraph][ERROR] Step 3 - Rephrasing failed: {e}")
                    state["rephrased_question"] = text_to_rephrase
            else:
                state["rephrased_question"] = original
        else:
            # For non-TEXT-ONLY queries, pass through unchanged
            state["rephrased_question"] = state.get("rephrased_question") or state.get("corrected_question") or state.get("original_question") or ""
        
        return state

    def _split_question(self, state: RAGState) -> RAGState:
        """
        TEXT-ONLY PIPELINE: Step 4 - Question Splitting.
        
        Analyze the rephrased question:
        - If it contains multiple independent questions: Split it
        - If it contains only one question: Keep it as single-item list
        
        Output: question_parts (list of questions)
        """
        input_type = state.get("input_type", "text")
        processed_image = state.get("processed_image")
        
        # TEXT-ONLY PIPELINE: Step 4 - ALWAYS analyze and split (mandatory step)
        if input_type == "text" and not processed_image:
            rephrased = state.get("rephrased_question") or ""
            corrected = state.get("corrected_question") or ""
            original = state.get("original_question") or ""
            text_to_split = rephrased if rephrased else (corrected if corrected else original)
            
            if text_to_split:
                try:
                    print(f"[RAGGraph][INFO] TEXT-ONLY PIPELINE: Step 4 - Question Splitting (ALWAYS)")
                    parts = self.question_processor.splitting_chain.split(text_to_split)
                    state["question_parts"] = parts
                    
                    # Step 5: UI Feedback - Display number of detected questions
                    num_questions = len(parts)
                    print(f"[RAGGraph][INFO] Step 4 - Split: Detected Questions: {num_questions}")
                    print(f"[RAGGraph][INFO] Step 5 - UI Feedback: Detected Questions: {num_questions}")
                    
                    # Store for UI display
                    state["detected_questions_count"] = num_questions
                except Exception as e:
                    print(f"[RAGGraph][ERROR] Step 4 - Splitting failed: {e}")
                    state["question_parts"] = [text_to_split]
                    state["detected_questions_count"] = 1
            else:
                state["question_parts"] = [original]
                state["detected_questions_count"] = 1
        else:
            # For non-TEXT-ONLY queries, pass through unchanged
            rephrased = state.get("rephrased_question") or ""
            corrected = state.get("corrected_question") or ""
            original = state.get("original_question") or ""
            state["question_parts"] = [rephrased if rephrased else (corrected if corrected else original)]
            state["detected_questions_count"] = 1
        
        return state

    def _check_knowledge_base(self, state: RAGState) -> RAGState:
        """
        IMAGE PIPELINE: Strict retrieval rules - no cross-modal comparisons.
        
        RULES:
        1. IF contains_text == true:
           - Use OCR-extracted text to create TEXT embedding
           - Search ONLY against text_embedding field in Milvus
           - Retrieve top-k TEXT documents
           - Do NOT search image_embedding field
        
        2. IF contains_text == false:
           - Convert image directly to IMAGE embedding
           - Search ONLY against image_embedding field in Milvus
           - Retrieve top-k IMAGE documents
           - Do NOT search text_embedding field
           - Do NOT convert image to text
        
        3. TEXT PIPELINE (unchanged):
           - Text only: retrieve texts using text embedding
           - Text + image: retrieve texts only (image is context, not search query)
        """
        rephrased = state.get("rephrased_question") or ""
        corrected = state.get("corrected_question") or ""
        original = state.get("original_question") or ""
        
        # Get first available text
        q_raw = rephrased or corrected or original
        q = str(q_raw).strip() if q_raw else ""
        
        input_type = state.get("input_type", "text")
        processed_image = state.get("processed_image")
        contains_text = state.get("contains_text")  # Boolean decision from perception
        ocr_text = state.get("ocr_text") or ""
        user_text_before_ocr = state.get("user_text_before_ocr") or ""
        
        with self.profiler.stage("5_embedding_generation_and_retrieval", {
            "has_text": bool(q),
            "has_image": bool(processed_image),
            "contains_text": contains_text,
            "k": RETRIEVAL_K
        }):
            # ========== IMAGE PIPELINE ==========
            if processed_image and not user_text_before_ocr:
                # Image-only query (no user text)
                if contains_text is True:
                    # Rule: contains_text == true → OCR text → TEXT embedding → text_embedding search
                    if not ocr_text:
                        print(f"[RAGGraph][WARNING] contains_text=true but no OCR text found")
                        state["retrieved_docs"] = []
                        return state
                    
                    print(f"[RAGGraph][INFO] IMAGE PIPELINE (contains_text=true): Using OCR text for TEXT embedding → text_embedding search")
                    print(f"[RAGGraph][INFO] OCR text (first 100 chars): '{ocr_text[:100]}...'")
                    
                    # TEXT embedding → text_embedding search ONLY
                    text_docs = self.rag_chain.retrieve(
                        query=ocr_text, 
                        k=RETRIEVAL_K, 
                        search_type="text",  # Search text_embedding field
                        image_path=None
                    )
                    state["retrieved_docs"] = text_docs
                    print(f"[RAGGraph][INFO] Retrieved {len(text_docs)} TEXT documents (from text_embedding search)")
                    
                elif contains_text is False:
                    # Rule: contains_text == false → IMAGE embedding → image_embedding search
                    print(f"[RAGGraph][INFO] IMAGE PIPELINE (contains_text=false): Using image for IMAGE embedding → image_embedding search")
                    
                    # IMAGE embedding → image_embedding search ONLY
                    image_docs = self.rag_chain.retrieve(
                        query=processed_image, 
                        k=RETRIEVAL_K, 
                        search_type="image",  # Search image_embedding field
                        image_path=processed_image
                    )
                    state["retrieved_docs"] = image_docs
                    print(f"[RAGGraph][INFO] Retrieved {len(image_docs)} IMAGE documents (from image_embedding search)")
                    
                else:
                    # contains_text is None (should not happen, but handle gracefully)
                    print(f"[RAGGraph][WARNING] contains_text is None, defaulting to image embedding search")
                    image_docs = self.rag_chain.retrieve(
                        query=processed_image, 
                        k=RETRIEVAL_K, 
                        search_type="image",
                        image_path=processed_image
                    )
                    state["retrieved_docs"] = image_docs
                    print(f"[RAGGraph][INFO] Retrieved {len(image_docs)} IMAGE documents (fallback)")
            
            # ========== TEXT PIPELINE (STRICT TEXT-ONLY) ==========
            else:
                # TEXT-ONLY PIPELINE: Steps 6-8
                # Step 6: Text Embedding (done in retrieve)
                # Step 7: Similarity Measurement (done in retrieve)
                # Step 8: Document Retrieval
                if q:
                    print(f"[RAGGraph][INFO] TEXT PIPELINE: Step 6-8 - Text Embedding, Similarity, Retrieval")
                    print(f"[RAGGraph][INFO] Query: '{q[:50]}...'")
                    
                    # Step 6-8: Single retrieval call (TEXT embeddings only, cosine similarity, top-K)
                    text_docs = self.rag_chain.retrieve(
                        query=q, 
                        k=RETRIEVAL_K,  # Top-K documents
                        search_type="text",  # TEXT embeddings ONLY (never image embeddings)
                        image_path=None  # No image processing
                    )
                    state["retrieved_docs"] = text_docs
                    
                    # Set tool_results for web search fallback decision (minimal overhead)
                    if text_docs:
                        state["tool_results"] = {"knowledge_base": "retrieved"}  # Simple indicator
                    else:
                        state["tool_results"] = {"knowledge_base": "fallback"}  # Signal for web search
                    
                    print(f"[RAGGraph][INFO] Step 8 - Retrieved {len(text_docs)} text documents (TEXT embeddings only, cosine similarity)")
                else:
                    print(f"[RAGGraph][INFO] TEXT PIPELINE: No text query - no documents retrieved")
                    state["retrieved_docs"] = []
                    state["tool_results"] = {"knowledge_base": "fallback"}
        
        return state

    def _select_images_for_presentation(self, state: RAGState) -> RAGState:
        """
        Presentation Control Module - selects appropriate images for display from retrieved documents.
        
        Rules:
        - Reviews retrieved documents only
        - Selects images directly relevant to the query or input image
        - Does not select images from irrelevant documents
        - Prefers images that visually support the answer
        - Selects at most 3 images
        - Returns JSON only without explanations
        """
        retrieved_docs = state.get("retrieved_docs", [])
        processed_image = state.get("processed_image")
        input_type = state.get("input_type", "text")
        user_text_before_ocr = state.get("user_text_before_ocr") or ""
        original_q = state.get("original_question") or ""
        
        # STRICT RULE: TEXT-ONLY queries - Image selection is STRICTLY FORBIDDEN
        # Skip all image processing for TEXT input_type to minimize latency
        if input_type == "text" and not processed_image:
            print(f"[RAGGraph][INFO] Presentation Control (TEXT-ONLY): Image selection STRICTLY FORBIDDEN - returning empty list (no LLM calls, no processing)")
            state["selected_images"] = []
            return state
        
        # Collect all images from retrieved documents (for IMAGE queries only)
        available_images = []
        for doc, score in retrieved_docs:
            if hasattr(doc, 'metadata') and doc.metadata:
                image_path = doc.metadata.get("image_path")
                doc_id = doc.metadata.get("id")
                if image_path and doc_id:
                    # Verify image exists
                    from pathlib import Path
                    img_path = Path(image_path)
                    if img_path.exists():
                        available_images.append({
                            "image_path": str(image_path),
                            "doc_id": str(doc_id),
                            "similarity": score,
                            "doc_content": doc.page_content[:200]  # First 200 chars for context
                        })
        
        if not available_images:
            # No images available
            state["selected_images"] = []
            print("[RAGGraph][INFO] Presentation Control: No images found in retrieved documents")
            return state
        
        with self.profiler.stage("6_presentation_control", {
            "available_images_count": len(available_images),
            "has_user_image": bool(processed_image),
            "has_user_text": bool(user_text_before_ocr or original_q)
        }):
            try:
                import ollama
                import json
                from config.settings import OLLAMA_MODEL
                
                # Build list of available images for the model
                images_list = []
                for i, img_info in enumerate(available_images[:10], 1):  # Max 10 images to save memory
                    images_list.append(f"Image {i}:\n  - Path: {img_info['image_path']}\n  - Document ID: {img_info['doc_id']}\n  - Similarity: {img_info['similarity']:.2f}\n  - Content preview: {img_info['doc_content']}")
                
                images_context = "\n\n".join(images_list)
                
                # Build prompt using f-string to avoid KeyError with curly braces
                user_query_text = user_text_before_ocr or original_q or "Analyze the input image"
                presentation_prompt = f"""You are a presentation control module in a multimodal Retrieval-Augmented Generation (RAG) system.

━━━━━━━━━━━━━━━━━━━━━━
IMAGE SELECTION RULES
━━━━━━━━━━━━━━━━━━━━━━
- Review the retrieved documents only.
- Select images that are directly relevant to the user's query or the input image.
- Do NOT select images from irrelevant documents, even if they are highly ranked.
- Prefer images that visually support or clarify the answer.

━━━━━━━━━━━━━━━━━━━━━━
SELECTION CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━
- Select at most 3 images.
- If fewer relevant images exist, select only those.
- If no relevant images exist, return an empty result.

━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT)
━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a JSON object in the following format:

{{
  "show_images": [
    {{
      "image_id": "<image_path>",
      "source_doc_id": "<document_id>"
    }}
  ]
}}

━━━━━━━━━━━━━━━━━━━━━━
STRICT PROHIBITIONS
━━━━━━━━━━━━━━━━━━━━━━
- Do NOT include explanations.
- Do NOT include comments.
- Do NOT include text outside the JSON.
- Do NOT infer relevance using external knowledge.
- Use only the retrieved documents' metadata.

━━━━━━━━━━━━━━━━━━━━━━
FAIL-SAFE RULE
━━━━━━━━━━━━━━━━━━━━━━
If no images are suitable for display, return:

{{
  "show_images": []
}}

━━━━━━━━━━━━━━━━━━━━━━
USER QUERY:
━━━━━━━━━━━━━━━━━━━━━━
{user_query_text}

━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE IMAGES FROM RETRIEVED DOCUMENTS:
━━━━━━━━━━━━━━━━━━━━━━
{images_context}

━━━━━━━━━━━━━━━━━━━━━━
TASK:
━━━━━━━━━━━━━━━━━━━━━━
Select the most relevant images (max 3) based on the user query and return ONLY the JSON object."""
                
                print(f"[RAGGraph][INFO] Presentation Control: Selecting images from {len(available_images)} available images")
                
                # Send request to Qwen3-VL:4b
                messages = [{"role": "user", "content": presentation_prompt}]
                
                # If there is an input image, add it to context
                if processed_image:
                    messages[0]["images"] = [processed_image]
                
                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=messages
                )
                
                # Extract JSON from response
                response_text = response["message"]["content"].strip()
                
                # Try to extract JSON from response (may contain additional text)
                try:
                    # Search for JSON in response
                    import re
                    json_match = re.search(r'\{[^{}]*"show_images"[^{}]*\}', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                    else:
                        # Try to parse full response
                        json_str = response_text
                    
                    result = json.loads(json_str)
                    
                    # Validate format
                    if "show_images" in result and isinstance(result["show_images"], list):
                        # Verify that selected images actually exist
                        valid_images = []
                        for img_entry in result["show_images"]:
                            if "image_id" in img_entry and "source_doc_id" in img_entry:
                                # Search for image in available list
                                found = False
                                for avail_img in available_images:
                                    if (avail_img["image_path"] == img_entry["image_id"] or 
                                        avail_img["doc_id"] == img_entry["source_doc_id"]):
                                        valid_images.append({
                                            "image_id": avail_img["image_path"],
                                            "source_doc_id": avail_img["doc_id"]
                                        })
                                        found = True
                                        break
                                if not found:
                                    print(f"[RAGGraph][WARNING] Selected image not found: {img_entry}")
                        
                        state["selected_images"] = valid_images[:3]  # Max 3 images
                        print(f"[RAGGraph][INFO] Presentation Control: Selected {len(state['selected_images'])} images for display")
                    else:
                        state["selected_images"] = []
                        print(f"[RAGGraph][WARNING] Presentation Control: Invalid JSON format, no images selected")
                        
                except json.JSONDecodeError as e:
                    print(f"[RAGGraph][ERROR] Presentation Control: Failed to parse JSON response: {e}")
                    print(f"[RAGGraph][DEBUG] Response text: {response_text[:200]}")
                    state["selected_images"] = []
                    
            except Exception as e:
                print(f"[RAGGraph][ERROR] Presentation Control failed: {e}")
                import traceback
                traceback.print_exc()
                state["selected_images"] = []
        
        return state

    def _should_use_web(self, state: RAGState) -> str:
        """
        TEXT-ONLY PIPELINE: Never use web search (go directly to answer generation).
        Web search is only for legacy/fallback paths.
        """
        input_type = state.get("input_type", "text")
        processed_image = state.get("processed_image")
        
        # TEXT-ONLY PIPELINE: STRICT - Never use web search (skip web_search_node)
        if input_type == "text" and not processed_image:
            print("[RAGGraph][INFO] TEXT-ONLY PIPELINE: Skipping web search (STRICT - go directly to answer generation)")
            return "use_kb"
        
        # For other paths, check if fallback is needed
        kb = state.get("tool_results", {}).get("knowledge_base", "").lower()
        return "use_web" if "fallback" in kb else "use_kb"

    def _search_web(self, state: RAGState) -> RAGState:
        rephrased_raw = state.get("rephrased_question") or ""
        q = str(rephrased_raw).strip() if rephrased_raw else ""
        if not q:
            # Fallback to other question fields
            corrected_raw = state.get("corrected_question") or ""
            original_raw = state.get("original_question") or ""
            q = str(corrected_raw).strip() if corrected_raw else (str(original_raw).strip() if original_raw else "")
        state["tool_results"]["web"] = look_on_web(q)
        return state

    def _generate_answer(self, state: RAGState) -> RAGState:
        input_type = state.get("input_type", "text")
        rephrased_raw = state.get("rephrased_question") or ""
        corrected_raw = state.get("corrected_question") or ""
        original_raw = state.get("original_question") or ""
        
        # Get first available text
        q_raw = rephrased_raw or corrected_raw or original_raw
        q = str(q_raw).strip() if q_raw else ""
        
        processed_image = state.get("processed_image")
        
        print(f"[RAGGraph][DEBUG] _generate_answer called:")
        print(f"  - input_type: {input_type}")
        print(f"  - processed_image: {processed_image}")
        print(f"  - retrieved_docs count: {len(state.get('retrieved_docs', []))}")
        print(f"  - question: {q[:100] if q else 'None'}")
        
        # ALWAYS use ollama.chat() directly for image queries (more memory efficient)
        with self.profiler.stage("8_llm_inference_answer_generation", {
            "input_type": input_type,
            "has_image": bool(processed_image),
            "retrieved_docs_count": len(state.get('retrieved_docs', [])),
            "model": OLLAMA_MODEL
        }):
            if input_type == "image" and processed_image:
                # IMAGE PIPELINE: Answer generation based on contains_text decision
                try:
                    import ollama
                    
                    retrieved_docs = state.get("retrieved_docs", [])
                    contains_text = state.get("contains_text")  # Boolean decision
                    ocr_text = state.get("ocr_text") or ""
                    user_text_before_ocr = state.get("user_text_before_ocr") or ""
                    original_q_raw = state.get("original_question") or ""
                    original_q = str(original_q_raw).strip() if original_q_raw else ""
                    
                    if retrieved_docs:
                        # Format retrieved documents as snippets
                        snippet_parts = []
                        for i, (doc, score) in enumerate(retrieved_docs[:3], 1):  # Max 3 documents (optimized)
                            doc_content = doc.page_content[:200]  # Max 200 chars (optimized)
                            snippet_parts.append(f"Snippet {i} (Similarity: {score:.2f}):\n{doc_content}")
                        snippets_text = "\n".join(snippet_parts)
                        
                        # Extract question and choices from retrieved documents
                        question_text = ""
                        choices_text = ""
                        if retrieved_docs:
                            first_doc, _ = retrieved_docs[0]
                            if hasattr(first_doc, 'metadata') and first_doc.metadata:
                                question_text = first_doc.metadata.get("question", "")
                                choices_text = first_doc.metadata.get("choices", "")
                        
                        # Fallback to OCR text or original question
                        if not question_text:
                            question_text = ocr_text if ocr_text else original_q
                        
                        # RULE: contains_text == true → send ONLY text + TEXT docs (NO images)
                        if contains_text is True:
                            print("[RAGGraph][INFO] IMAGE PIPELINE (contains_text=true): Generating answer with OCR text + TEXT documents only (NO images)")
                            
                            generation_prompt = f"""You are a precise question-answering assistant specialized in image-based retrieval. Follow these rules strictly:

1. Process the uploaded image only. Do NOT access or modify any original text data.
2. Detect if the image contains text:
   - If text is present, extract it accurately (OCR).
   - If no text is present, process the image visually (embedding).
3. Convert the OCR text (if exists) or the image into embedding and retrieve only relevant textual documents from the database.
4. Use only the retrieved textual snippets to answer. Ignore any unrelated documents or images.
5. Answer in **one short sentence**. Do not explain, summarize, quote, or analyze.
6. Focus strictly on the main question and the provided choices.
7. Do NOT include images in the output. Only text is allowed.
8. If unsure, choose the snippet with the highest similarity score.

Question: {question_text}

Choices: {choices_text if choices_text else 'N/A'}

Retrieved Text Snippets: 
{snippets_text}"""
                            
                            # Send ONLY text prompt (NO images)
                            response = ollama.chat(
                                model=OLLAMA_MODEL,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": generation_prompt
                                        # NO images parameter
                                    }
                                ]
                            )
                            
                            state["answer"] = response["message"]["content"]
                            state["used_tools"].append("qwen3vl_text_only_mode")
                            print("[RAGGraph][INFO] Successfully generated answer using text + TEXT documents (no images)")
                            return state
                        
                        # RULE: contains_text == false → send ONLY text + TEXT documents (NO images)
                        elif contains_text is False:
                            print("[RAGGraph][INFO] IMAGE PIPELINE (contains_text=false): Generating answer with TEXT documents from image embedding search (NO images)")
                            
                            generation_prompt = f"""You are a precise question-answering assistant specialized in image-based retrieval. Follow these rules strictly:

1. Process the uploaded image only. Do NOT access or modify any original text data.
2. Detect if the image contains text:
   - If text is present, extract it accurately (OCR).
   - If no text is present, process the image visually (embedding).
3. Convert the OCR text (if exists) or the image into embedding and retrieve only relevant textual documents from the database.
4. Use only the retrieved textual snippets to answer. Ignore any unrelated documents or images.
5. Answer in **one short sentence**. Do not explain, summarize, quote, or analyze.
6. Focus strictly on the main question and the provided choices.
7. Do NOT include images in the output. Only text is allowed.
8. If unsure, choose the snippet with the highest similarity score.

Question: {question_text}

Choices: {choices_text if choices_text else 'N/A'}

Retrieved Text Snippets: 
{snippets_text}"""
                            
                            # Send ONLY text prompt (NO images) - TEXT documents from image embedding search
                            response = ollama.chat(
                                model=OLLAMA_MODEL,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": generation_prompt
                                        # NO images parameter - text-only generation
                                    }
                                ]
                            )
                            
                            state["answer"] = response["message"]["content"]
                            state["used_tools"].append("qwen3vl_image_only_mode")
                            print("[RAGGraph][INFO] Successfully generated answer using TEXT documents from image embedding search (no images)")
                            return state
                        
                        else:
                            # Fallback: contains_text is None (should not happen)
                            print("[RAGGraph][WARNING] contains_text is None, defaulting to image + documents")
                            generation_prompt = f"""Answer based on the image and retrieved documents.

Question: {question_text}
{f'Choices: {choices_text}' if choices_text else ''}

Relevant snippet(s): 
{snippets_text}"""
                            
                            response = ollama.chat(
                                model=OLLAMA_MODEL,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": generation_prompt,
                                        "images": [processed_image]
                                    }
                                ]
                            )
                            state["answer"] = response["message"]["content"]
                            state["used_tools"].append("qwen3vl_fallback")
                            return state
                    else:
                        # No retrieved documents - handle error case
                        print("[RAGGraph][WARNING] No retrieved documents found")
                        state["answer"] = "No relevant documents found in the knowledge base to answer this query."
                        state["used_tools"].append("qwen3vl_no_docs")
                        return state
                        
                except Exception as e:
                    print(f"[RAGGraph][ERROR] Failed to generate answer for image query: {e}")
                    import traceback
                    traceback.print_exc()
                    state["answer"] = f"Error analyzing image: {str(e)}"
                    return state
            
            # For text-only queries, handle based on retrieved docs
            if not state.get("retrieved_docs"):
                # No retrieved docs for text query
                print("[RAGGraph][INFO] Text-only query with no retrieved docs - using diagnostic report")
                state["answer"] = self.rag_chain.diagnostic_report(
                    original_question=state.get("original_question", ""),
                    corrected_question=state.get("corrected_question", ""),
                    rephrased_question=q,
                    top_k=RETRIEVAL_K,
                )
                return state
            
            # TEXT-ONLY PIPELINE: Step 9 - LLM Answer Generation (ONLY LLM CALL)
            print("[RAGGraph][INFO] TEXT PIPELINE: Step 9 - LLM Answer Generation (ONLY LLM CALL)")
            print(f"[RAGGraph][INFO] Sending to LLM: rephrased_question + {len(state['retrieved_docs'])} retrieved documents")
            state["answer"] = self.rag_chain.generate_answer(q, state["retrieved_docs"])
        return state

    def _combine_results(self, state: RAGState) -> RAGState:
        """
        TEXT-ONLY PIPELINE: Final output formatting.
        
        Return ONLY:
        - The final text answer
        - The number of detected questions
        - Nothing else (no evaluation unless explicitly requested)
        """
        input_type = state.get("input_type", "text")
        processed_image = state.get("processed_image")
        
        with self.profiler.stage("9_final_response_formatting", {
            "has_answer": bool(state.get("answer")),
            "answer_length": len(state.get("answer", "")),
            "detected_questions": state.get("detected_questions_count", 1)
        }):
                # Final output: answer + detected_questions_count
            if input_type == "text" and not processed_image:
                print(f"[RAGGraph][INFO] TEXT-ONLY PIPELINE: Final output - Answer + {state.get('detected_questions_count', 1)} detected question(s)")
        return state

    def _process_image(self, state: RAGState) -> RAGState:
        # processed_image should already be set from run() method (image_path parameter)
        processed_image = state.get("processed_image")
        with self.profiler.stage("2_image_preprocessing", {
            "image_path": processed_image if processed_image else None
        }):
            if not processed_image:
                # This should not happen if image_path was passed correctly
                print(f"[RAGGraph][WARNING] processed_image not set in state. Available keys: {list(state.keys())}")
            else:
                # Verify image file exists
                from pathlib import Path
                img_path = Path(processed_image)
                if img_path.exists():
                    print(f"[RAGGraph][INFO] Processing image: {processed_image}")
                    # Actual preprocessing would happen here if needed
                    # For now, we just verify the file exists
                else:
                    print(f"[RAGGraph][ERROR] Image file not found: {processed_image}")
                    state["error"] = f"Image file not found: {processed_image}"
        return state

    def _perception_and_routing(self, state: RAGState) -> RAGState:
        """
        IMAGE PIPELINE: Text Presence Detection (Optimized).
        
        Uses Qwen3-VL-4B ONLY to detect whether the image contains readable text.
        Output: boolean contains_text decision.
        
        Rules:
        - Do NOT perform document comparison at this stage.
        - Do NOT analyze or compare image embeddings.
        - Output: contains_text = true / false
        """
        processed_image = state.get("processed_image")
        original_q_raw = state.get("original_question") or ""
        original_q = str(original_q_raw).strip() if original_q_raw else ""
        
        # Store original user text before OCR (if exists)
        if original_q and not state.get("user_text_before_ocr"):
            state["user_text_before_ocr"] = original_q
        
        # If image exists, analyze it for text presence ONLY
        if processed_image:
            with self.profiler.stage("3_perception_routing", {
                "image_path": processed_image,
                "method": "Qwen3-VL:4b",
                "purpose": "text_presence_detection"
            }):
                try:
                    import ollama
                    import json
                    import re
                    from config.settings import OLLAMA_MODEL
                    
                    print("[RAGGraph][INFO] Perception: Detecting text presence in image...")
                    
                    perception_prompt = """You are a deterministic multimodal module inside a RAG system.

Input: a resized image (max resolution: 384×384).

Your tasks (in one call):

1. Detect if the image contains readable text.
   - Output a boolean field: "contains_text" (true or false).
   - Do NOT describe the image or infer context.

2. If contains_text == true:
   - Extract ONLY the visible text exactly as it appears.
   - Do NOT paraphrase, correct, or interpret the text.
   - Output as plain text in the "extracted_text" field.

3. If contains_text == false:
   - Do NOT attempt OCR.
   - Leave "extracted_text" empty.

Requirements:
- Perform both Perception and OCR in a single inference call.
- Use GPU acceleration if available for faster image encoding.
- Output strictly in JSON format, no extra text.

JSON format:
{
  "contains_text": true | false,
  "extracted_text": "..."
}"""
                    
                    response = ollama.chat(
                        model=OLLAMA_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": perception_prompt,
                                "images": [processed_image]
                            }
                        ]
                    )
                    
                    response_text = response["message"]["content"].strip()
                    
                    # Extract JSON from response
                    try:
                        # Search for JSON in response (may contain both contains_text and extracted_text)
                        json_match = re.search(r'\{[^{}]*"contains_text"[^{}]*\}', response_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                        else:
                            json_str = response_text
                        
                        result = json.loads(json_str)
                        contains_text = result.get("contains_text", False)
                        extracted_text = result.get("extracted_text", "").strip()
                        
                        # Validation: Log the raw response for debugging
                        print(f"[RAGGraph][INFO] Perception raw response: {response_text[:200]}")
                        print(f"[RAGGraph][INFO] Perception parsed result: contains_text={contains_text}, extracted_text_length={len(extracted_text)}")
                        
                        # Store boolean decision
                        state["contains_text"] = bool(contains_text)
                        state["processing_mode"] = "text_only" if contains_text else "visual_only"
                        state["image_content_type"] = "textual" if contains_text else "visual"
                        
                        # If OCR text was extracted in perception step, store it
                        if contains_text and extracted_text:
                            state["ocr_text"] = extracted_text
                            print(f"[RAGGraph][INFO] OCR text extracted in perception step (length: {len(extracted_text)} chars): '{extracted_text[:100]}...'")
                        
                        print(f"[RAGGraph][INFO] Perception result: contains_text={contains_text}")
                        
                    except json.JSONDecodeError as e:
                        print(f"[RAGGraph][ERROR] Failed to parse JSON from perception response: {e}")
                        print(f"[RAGGraph][DEBUG] Response text: {response_text[:200]}")
                        # Default to false (assume visual-only) if unclear
                        state["contains_text"] = False
                        state["processing_mode"] = "visual_only"
                        state["image_content_type"] = "visual"
                        print("[RAGGraph][WARNING] Defaulting to contains_text=false (visual-only)")
                    
                except Exception as e:
                    print(f"[RAGGraph][ERROR] Perception & Routing failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Default to false on error
                    state["contains_text"] = False
                    state["processing_mode"] = "visual_only"
                    state["image_content_type"] = "visual"
        else:
            # No image: no need for perception
            state["contains_text"] = None
            state["processing_mode"] = None
            state["image_content_type"] = None
        
        return state
    
    def _route_after_perception(self, state: RAGState) -> str:
        """
        IMAGE PIPELINE: Route based on contains_text boolean decision.
        
        Returns:
            "textual_path": if contains_text == true (needs OCR)
            "visual_path": if contains_text == false (direct image embedding)
        """
        contains_text = state.get("contains_text")
        original_q = state.get("original_question") or ""
        has_user_text = bool(str(original_q).strip())
        
        # Route based on contains_text boolean
        if contains_text is True:
            print(f"[RAGGraph][INFO] Routing: TEXTUAL path (contains_text=true) - proceeding to OCR")
            return "textual_path"
        elif contains_text is False:
            print(f"[RAGGraph][INFO] Routing: VISUAL path (contains_text=false) - skipping OCR, using image embedding directly")
            return "visual_path"
        
        # Fallback if contains_text not set (should not happen)
        processing_mode = state.get("processing_mode")
        if processing_mode == "visual_only":
            print(f"[RAGGraph][INFO] Routing: VISUAL path (fallback mode={processing_mode})")
            return "visual_path"
        else:
            print(f"[RAGGraph][INFO] Routing: TEXTUAL path (fallback mode={processing_mode})")
            return "textual_path"
    
    def _visual_understanding(self, state: RAGState) -> RAGState:
        """
        Visual Understanding Assistant - converts visual images to textual descriptions.
        
        Analyzes the image and generates a clear textual description (1-3 sentences)
        that can be used as a query for information retrieval.
        """
        processed_image = state.get("processed_image")
        original_q = state.get("original_question") or ""
        
        if not processed_image:
            print("[RAGGraph][ERROR] No image path provided for visual understanding")
            state["visual_description"] = ""
            return state
        
        # IMAGE PIPELINE: Skip visual understanding if contains_text == true
        contains_text = state.get("contains_text")
        if contains_text is True:
            print(f"[RAGGraph][INFO] Skipping visual understanding (contains_text=true)")
            state["visual_description"] = ""
            return state
        
        with self.profiler.stage("4_visual_understanding", {
            "method": "Qwen3-VL:4b",
            "image_path": processed_image,
            "image_content_type": state.get("image_content_type")
        }):
            try:
                import ollama
                from config.settings import OLLAMA_MODEL
                
                print("[RAGGraph][INFO] Visual Understanding: Analyzing image to generate descriptive text...")
                
                visual_prompt = """You are a visual understanding assistant. Your task is to analyze the input image and generate a clear textual description that can be used as a query for information retrieval. Follow these instructions strictly:

1️⃣ Analyze the image carefully. Extract all important visual information, including:
   - Objects or entities present.
   - Actions or events occurring.
   - Any visible text (use OCR if present).
   - Scene context, environment, or background details.

2️⃣ Generate a descriptive text (1-3 sentences) summarizing the image content **in plain English**.
   - Be precise and concise.
   - Avoid speculation beyond what is visible in the image.
   - Use complete sentences.

3️⃣ **Output descriptive text only**.

4️⃣ Additional Notes:
   - This text will later be passed to a RAG system for retrieval and answer generation.
   - Do **not** attempt to answer any question yet.
   - You **do not need to rephrase, correct, or split the text** unless it is unclear; keep it faithful to the visual content."""
                
                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": visual_prompt,
                            "images": [processed_image]
                        }
                    ]
                )
                
                description = response["message"]["content"].strip()
                
                # Store visual description
                state["visual_description"] = description
                
                # For image-only queries, use description as the query
                if not original_q:
                    state["original_question"] = description
                    print(f"[RAGGraph][INFO] Visual Understanding: Generated description (will be used as query): '{description[:100]}...'")
                else:
                    # For text + image, description can be added as context
                    print(f"[RAGGraph][INFO] Visual Understanding: Generated description: '{description[:100]}...'")
                
            except Exception as e:
                print(f"[RAGGraph][ERROR] Visual Understanding failed: {e}")
                import traceback
                traceback.print_exc()
                state["visual_description"] = ""
                # On failure, if no user text, set empty question
                if not original_q:
                    state["original_question"] = ""
        
        return state

    def _extract_text_from_image(self, state: RAGState) -> RAGState:
        """
        Extract text from image using Qwen3-VL:4b.
        
        Your only role is:
        - Extract all visible text in the image accurately
        - Output the text exactly as is without any modification
        
        Strictly prohibited:
        - Rephrasing or summarizing
        - Spelling or language correction
        - Splitting text into sentences or paragraphs
        - Adding titles or explanations
        - Interpreting text or answering questions
        """
        processed_image = state.get("processed_image")
        original_q_raw = state.get("original_question") or ""
        original_q = str(original_q_raw).strip() if original_q_raw else ""
        input_type = state.get("input_type", "text")
        
        # Store original user text before OCR (if available)
        if original_q and not state.get("user_text_before_ocr"):
            state["user_text_before_ocr"] = original_q
        
        if not processed_image:
            print("[RAGGraph][ERROR] No image path provided for OCR")
            state["ocr_text"] = ""
            return state
        
        # Check if OCR text already provided by perception module
        if state.get("ocr_text") and state.get("ocr_text").strip():
            print("[RAGGraph][INFO] OCR text already provided by perception module, skipping OCR extraction")
            return state
        
        # IMAGE PIPELINE: Only perform OCR if contains_text == true
        contains_text = state.get("contains_text")
        if contains_text is False:
            print(f"[RAGGraph][INFO] Skipping OCR (contains_text=false)")
            state["ocr_text"] = ""
            return state
        
        with self.profiler.stage("4_ocr_extraction", {
            "method": "Qwen3-VL:4b",
            "image_path": processed_image,
            "has_user_text": bool(original_q),
            "input_type": input_type,
            "contains_text": state.get("contains_text")
        }):
            # Case: text + image - check if image contains additional information
            if original_q and input_type == "image":
                print(f"[RAGGraph][INFO] Text + Image input: user text='{original_q}', extracting text from image for additional context")
            
            try:
                import ollama
                from config.settings import OLLAMA_MODEL
                
                # Use deterministic OCR prompt (only called if contains_text=true from perception)
                # This is a fallback in case perception didn't extract text in the first call
                ocr_prompt = """You are a deterministic multimodal module inside a RAG system.

Input: a resized image (max resolution: 384×384).

Your tasks (in one call):

1. Detect if the image contains readable text.
   - Output a boolean field: "contains_text" (true or false).
   - Do NOT describe the image or infer context.

2. If contains_text == true:
   - Extract ONLY the visible text exactly as it appears.
   - Do NOT paraphrase, correct, or interpret the text.
   - Output as plain text in the "extracted_text" field.

3. If contains_text == false:
   - Do NOT attempt OCR.
   - Leave "extracted_text" empty.

Requirements:
- Perform both Perception and OCR in a single inference call.
- Use GPU acceleration if available for faster image encoding.
- Output strictly in JSON format, no extra text.

JSON format:
{
  "contains_text": true | false,
  "extracted_text": "..."
}"""
                
                print("[RAGGraph][INFO] Extracting raw text from image using Qwen3-VL:4b (OCR mode)...")
                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": ocr_prompt,
                            "images": [processed_image]
                        }
                    ]
                )
                
                response_text = response["message"]["content"].strip()
                
                # Extract JSON from response
                try:
                    import re
                    import json
                    # Search for JSON in response (may contain both contains_text and extracted_text)
                    json_match = re.search(r'\{[^{}]*"contains_text"[^{}]*\}', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                    else:
                        json_str = response_text
                    
                    result = json.loads(json_str)
                    ocr_contains_text = result.get("contains_text", False)
                    extracted_text = result.get("extracted_text", "").strip()
                    
                    # VALIDATION: If OCR says contains_text=false or extracted_text is empty,
                    # correct the decision to contains_text=false
                    if not ocr_contains_text or not extracted_text or len(extracted_text.strip()) == 0:
                        print(f"[RAGGraph][WARNING] OCR confirmed no text (contains_text={ocr_contains_text}, extracted_length={len(extracted_text)}). Correcting decision to contains_text=false")
                        state["contains_text"] = False
                        state["processing_mode"] = "visual_only"
                        state["image_content_type"] = "visual"
                        state["ocr_text"] = ""
                        print(f"[RAGGraph][INFO] Corrected routing: Will use visual embedding path instead")
                        # Return early to skip text processing
                        return state
                    
                    # Store extracted text
                    state["ocr_text"] = extracted_text
                    print(f"[RAGGraph][INFO] OCR extracted text (length: {len(extracted_text)} chars): '{extracted_text[:100]}...'")
                    
                except json.JSONDecodeError as e:
                    print(f"[RAGGraph][ERROR] Failed to parse JSON from OCR response: {e}")
                    print(f"[RAGGraph][DEBUG] Response text: {response_text[:200]}")
                    # If JSON parsing fails, try to extract text directly
                    extracted_text = response_text.strip()
                    # Clean result (remove "NO_TEXT" if present)
                    if extracted_text.upper() == "NO_TEXT" or extracted_text.upper() == '"NO_TEXT"':
                        extracted_text = ""
                    
                    # VALIDATION: If OCR extracted no text, correct the decision
                    if not extracted_text or len(extracted_text.strip()) == 0:
                        print(f"[RAGGraph][WARNING] OCR extracted no text. Correcting decision to contains_text=false")
                        state["contains_text"] = False
                        state["processing_mode"] = "visual_only"
                        state["image_content_type"] = "visual"
                        state["ocr_text"] = ""
                        print(f"[RAGGraph][INFO] Corrected routing: Will use visual embedding path instead")
                        return state
                    
                    # Store extracted text
                    state["ocr_text"] = extracted_text
                    print(f"[RAGGraph][INFO] OCR extracted text (length: {len(extracted_text)} chars): '{extracted_text[:100]}...'")
                
                # Case 1: Image only (no user text)
                if not original_q:
                    if extracted_text:
                        # Use extracted text as-is (no processing)
                        state["original_question"] = extracted_text
                        print(f"[RAGGraph][INFO] Image-only: Raw text extracted (will be used as-is, no processing): '{extracted_text[:100]}...'")
                    else:
                        # No text in image
                        print(f"[RAGGraph][INFO] Image-only: No text extracted from image")
                        state["original_question"] = ""
                
                # Case 2: Text + image
                elif original_q:
                    if extracted_text:
                        # Merge texts directly (Concatenation) without modification
                        # User text is the base, extracted text is added as additional context
                        combined_text = f"{original_q}\n\n{extracted_text}"
                        state["original_question"] = combined_text
                        print(f"[RAGGraph][INFO] Text + Image: User text='{original_q}', OCR text='{extracted_text[:50]}...', combined (no processing)")
                    else:
                        # No text in image - use user text only
                        print(f"[RAGGraph][INFO] Text + Image: No text in image, using user text only: '{original_q}'")
                        # User text is already in original_question
                
            except Exception as e:
                print(f"[RAGGraph][ERROR] Qwen3-VL OCR failed: {e}")
                import traceback
                traceback.print_exc()
                state["ocr_text"] = ""
                # في حالة الفشل، إذا كان هناك نص من المستخدم، احتفظ به
                if not original_q:
                    state["original_question"] = ""
        
        return state

    def _scienceqa_retrieval(self, state: RAGState) -> RAGState:
        state["tool_results"] = {"scienceqa": scienceqa_helper(state["original_question"])}
        return state

    def _generate_image_tool(self, state: RAGState) -> RAGState:
        state["answer"] = generate_image(state["original_question"], None, None)
        return state

    # ==================== Main Run Method ====================

    def run(self, question: str, input_type: str = "text", image_path: str = None) -> dict:
        """Execute the RAG workflow with a question and optional image path."""
        # Reset and start profiling for this query
        from utils.performance_profiler import PerformanceProfiler
        if isinstance(self.profiler, PerformanceProfiler):
            # Reset the profiler for a fresh start
            self.profiler.stages.clear()
            self.profiler.start_time = None
            self.profiler.end_time = None
            self.profiler.active_stages.clear()
        self.profiler.start()
        
        initial_state: RAGState = {
            "original_question": question,
            "corrected_question": "",
            "rephrased_question": "",
            "question_parts": [],
            "retrieved_docs": [],
            "answer": "",
            "error": "",
            "input_type": input_type or "text",
            "tool_results": {},
            "processed_image": image_path,
            "ocr_text": None,
            "used_tools": [],
            "user_text_before_ocr": question if question else None,  # Store original text
            "image_content_type": None,  # Image content type: textual or visual (deprecated, use processing_mode)
            "processing_mode": None,  # Processing mode: text_only, visual_only, text_and_visual
            "selected_images": None,  # Selected images for display
            "visual_description": None,  # Textual description from visual understanding
            "detected_questions_count": None  # Number of detected questions (Step 5 - UI Feedback)
        }
        
        try:
            result = self.graph.invoke(initial_state)
            
            # End profiling
            self.profiler.end()
            
            # Print performance report
            self.profiler.print_report(detailed=False)
            
            # Convert result to dict format expected by Streamlit app
            result_dict = {
                "original_question": result.get("original_question", question),
                "corrected_question": result.get("corrected_question", ""),
                "rephrased_question": result.get("rephrased_question", ""),
                "question_parts": result.get("question_parts", []),
                "retrieved_docs": result.get("retrieved_docs", []),
                "answer": result.get("answer", ""),
                "error": result.get("error", ""),
                "input_type": result.get("input_type", "text"),
                "used_tools": result.get("used_tools", []),
                "tool_results": result.get("tool_results", {}),
                "selected_images": result.get("selected_images", []),  # Selected images for display
                "visual_description": result.get("visual_description", ""),  # Textual description from visual understanding
                "ocr_text": result.get("ocr_text", ""),  # Extracted text from OCR
                "processing_mode": result.get("processing_mode", None),  # Processing mode determined by perception
                "detected_questions_count": result.get("detected_questions_count", 1),  # Number of detected questions (Step 5 - UI Feedback)
                "performance_report": self.profiler.get_report_dict()  # Add performance data
            }
            
            return result_dict
        except Exception as e:
            # End profiling even on error
            self.profiler.end()
            self.profiler.print_report(detailed=False)
            
            return {
                "original_question": question,
                "corrected_question": "",
                "rephrased_question": "",
                "question_parts": [],
                "retrieved_docs": [],
                "answer": "",
                "error": str(e),
                "input_type": "text",
                "used_tools": [],
                "tool_results": {},
                "selected_images": [],
                "performance_report": self.profiler.get_report_dict()
            }

    # ==================== Visualization ====================

    def print_graph_structure(self):
        print("\n" + "=" * 80)
        print("LANGGRAPH STRUCTURE")
        print("=" * 80)

        for src, dst, label in self.graph_edges:
            print(f"{src:25s} → {dst:25s} {label}")

        if not MATPLOTLIB_AVAILABLE:
            print("(Graph visualization skipped: matplotlib not installed)")
            return

        try:
            G = nx.DiGraph()
            for src, dst, label in self.graph_edges:
                G.add_edge(src, dst, label=label)

            pos = nx.spring_layout(G, seed=42)
            plt.figure(figsize=(12, 9))
            nx.draw(G, pos, with_labels=True, node_size=1600, font_size=8)
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels={(u, v): d["label"] for u, v, d in G.edges(data=True)},
                font_size=7,
            )

            if DEV_MODE:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            print(f"(Graph visualization unavailable: {e})")
