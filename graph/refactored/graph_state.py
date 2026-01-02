"""Comprehensive GraphState for multimodal RAG system."""
from typing import TypedDict, List, Optional, Literal, Dict, Tuple, Any
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Comprehensive state object for the multimodal RAG graph.
    
    Tracks all state throughout the pipeline execution.
    """
    # ==================== INPUT ====================
    input_type: Literal["text", "image", "text_and_image"]
    question: Optional[str]  # Original question text
    image_path: Optional[str]  # Path to input image
    
    # ==================== IMAGE PROCESSING ====================
    processing_mode: Optional[Literal["text_only", "visual_only", "text_and_visual"]]
    """Processing mode determined by image perception:
    - text_only: Image contains readable text only
    - visual_only: Image contains visual elements only
    - text_and_visual: Image contains both text and visual elements
    """
    extracted_text: Optional[str]  # OCR-extracted text from image
    visual_description: Optional[str]  # Textual description of visual content
    
    # ==================== EMBEDDING ====================
    embedding_type: Optional[Literal["text", "image", "multimodal"]]
    """Type of embedding to generate:
    - text: Text embedding only
    - image: Image embedding only
    - multimodal: Both text and image embeddings
    """
    text_query: Optional[str]  # Final text query for embedding generation
    
    # ==================== RETRIEVAL ====================
    retrieved_documents: List[Tuple[Document, float]]
    """Retrieved documents with similarity scores.
    Each tuple: (Document, similarity_score)
    """
    
    # ==================== GENERATION ====================
    generated_answer: Optional[str]  # Final generated answer
    
    # ==================== EVALUATION ====================
    ground_truth: Optional[str]  # Ground truth answer for evaluation
    choices: Optional[str]  # Choices for multiple-choice questions
    
    ragas_scores: Optional[Dict[str, Any]]
    """RAGAS evaluation scores:
    {
        "relevance": "Relevant | Partially Relevant | Irrelevant",
        "recall_at_k": bool,
        "correctness": "Correct | Partially Correct | Incorrect",
        "reason": str
    }
    """
    
    llm_judge_result: Optional[Dict[str, Any]]
    """LLM-as-Judge evaluation result:
    {
        "evaluation": "Correct | Partially Correct | Incorrect",
        "reason": str
    }
    """
    
    final_evaluation: Optional[Dict[str, Any]]
    """Final merged evaluation result combining RAGAS and LLM-as-Judge"""
    
    # ==================== METADATA ====================
    errors: List[str]  # List of errors encountered
    execution_trace: List[str]  # Ordered list of executed node names
    node_status: Dict[str, Literal["pending", "running", "completed", "skipped", "error"]]
    """Status of each node:
    - pending: Not yet executed
    - running: Currently executing
    - completed: Successfully completed
    - skipped: Skipped due to conditions
    - error: Failed with error
    """




