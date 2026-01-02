"""LangChain Tool wrappers for all tools."""
import sys
from pathlib import Path

# Ensure project root is in path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.tools import Tool
from typing import Optional
from retrieval.multimodal_milvus_store import MultimodalMilvusStore
from tools.web_search import look_on_web_with_params
from tools.knowledge_base import look_in_your_knowledge_with_threshold
from tools.ocr_tool import ocr_function
from tools.image_processing import ImagePreprocessor
from tools.scienceqa import scienceqa_helper
from tools.image_generation import generate_image
from config.settings import KB_SIMILARITY_THRESHOLD


def create_all_tools(vector_store: Optional[MultimodalMilvusStore] = None) -> list:
    """Create all LangChain tools.
    
    Args:
        vector_store: Optional Multimodal Milvus vector store (will load if not provided)
    
    Returns:
        List of LangChain Tool objects
    """
    if vector_store is None:
        vector_store = MultimodalMilvusStore.load()
    
    # Image preprocessing tool
    def image_preprocessing_tool(image_path: str, resize: Optional[str] = None, grayscale: bool = False) -> str:
        """Preprocess image: resize, grayscale, normalize, denoise."""
        try:
            preprocessor = ImagePreprocessor()
            resize_tuple = None
            if resize:
                w, h = map(int, resize.split("x"))
                resize_tuple = (h, w)  # Height, width for OpenCV

            payload = preprocessor.preprocess(
                image_path,
                resize=resize_tuple,
                grayscale=grayscale,
                normalize=True,
                denoise=True,
                return_array=True,
            )

            # Save processed image
            from PIL import Image
            output_path = Path(image_path).parent / f"processed_{Path(image_path).name}"
            Image.fromarray(payload["image"]).save(output_path)

            meta = payload["metadata"]
            return f"Image processed and saved to: {output_path} (orig={meta['orig_size']}, new={meta['new_size']})"
        except Exception as e:
            return f"Error preprocessing image: {str(e)}"

    # Knowledge base tool with vector store
    def kb_tool_wrapper(query: str, threshold: float = KB_SIMILARITY_THRESHOLD, max_results: int = 3) -> str:
        return look_in_your_knowledge_with_threshold(query, threshold=threshold, vector_store=vector_store, max_results=max_results)
    
    tools = [
        Tool(
            name="look_on_web",
            func=look_on_web_with_params,
            description="Search the web for information. Inputs: query (str), max_results (int, optional). Uses configured provider via env (SERPER_API_KEY/TAVILY_API_KEY)."
        ),
        Tool(
            name="look_in_your_knowledge",
            func=kb_tool_wrapper,
            description="Search the knowledge base (Milvus). If similarity below threshold, auto-fallback to web search. Inputs: query (str), threshold (float, optional), max_results (int, optional)."
        ),
        Tool(
            name="ocr_tool",
            func=ocr_function,
            description="Extract text from images using OCR. Input should be a path to an image file."
        ),
        Tool(
            name="image_preprocessing_tool",
            func=image_preprocessing_tool,
            description="Preprocess images: resize, optional grayscale, normalize, denoise. Input: image path, optional resize (e.g., '1024x1024'), grayscale flag."
        ),
        Tool(
            name="scienceqa_helper",
            func=scienceqa_helper,
            description="Retrieve similar ScienceQA samples for a question or image. Inputs: question text or image path."
        ),
        Tool(
            name="generate_image",
            func=generate_image,
            description="Generate images from text prompts. Input should be a prompt string, optional style, and optional resolution (e.g., '1024x1024')."
        )
    ]
    
    return tools


