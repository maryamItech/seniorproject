"""LangChain Tool for OCR text extraction using Qwen3-VL."""
import sys
from pathlib import Path
from typing import Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import OLLAMA_MODEL


class OCRInput(BaseModel):
    """Input for OCR tool."""
    image_path: str = Field(description="Path to the image file to extract text from")


class OCRTool(BaseTool):
    """Tool for extracting text from images using Qwen3-VL:4b."""
    
    name: str = "ocr_extraction"
    description: str = (
        "Extract raw text from images using OCR. "
        "Returns the text exactly as it appears in the image without any modifications, "
        "rephrasing, or corrections. Use this when an image contains readable text."
    )
    args_schema: Type[BaseModel] = OCRInput
    
    def _run(
        self,
        image_path: str,
        run_manager: Optional[Any] = None,
    ) -> str:
        """Extract text from image using Qwen3-VL:4b."""
        try:
            import ollama
            
            ocr_prompt = """You are a text extraction tool. Your task is to extract ONLY the raw text from the image.

Rules:
1. Extract all visible text in the image accurately.
2. Output the text exactly as is without any modification.
3. Do NOT rephrase, summarize, correct spelling, or add explanations.
4. Do NOT interpret or answer questions from the text.
5. Output ONLY the raw text.

If there is no text in the image, return "NO_TEXT" only."""

            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": ocr_prompt,
                        "images": [image_path]
                    }
                ]
            )
            
            extracted_text = response["message"]["content"].strip()
            
            # Clean "NO_TEXT" responses
            if extracted_text.upper() == "NO_TEXT" or not extracted_text:
                return ""
            
            return extracted_text
            
        except Exception as e:
            error_msg = f"OCR extraction failed: {str(e)}"
            if run_manager:
                run_manager.on_tool_error(error_msg)
            raise Exception(error_msg)
    
    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[Any] = None,
    ) -> str:
        """Async version of OCR extraction."""
        return self._run(image_path, run_manager)




