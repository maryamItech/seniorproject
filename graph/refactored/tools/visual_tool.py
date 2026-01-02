"""LangChain Tool for visual understanding using Qwen3-VL."""
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


class VisualUnderstandingInput(BaseModel):
    """Input for visual understanding tool."""
    image_path: str = Field(description="Path to the image file to analyze")


class VisualUnderstandingTool(BaseTool):
    """Tool for generating textual descriptions from visual images using Qwen3-VL:4b."""
    
    name: str = "visual_understanding"
    description: str = (
        "Analyze visual images and generate a clear textual description (1-3 sentences) "
        "that can be used as a query for information retrieval. "
        "Use this when an image contains visual elements (diagrams, illustrations, etc.) "
        "but no readable text."
    )
    args_schema: Type[BaseModel] = VisualUnderstandingInput
    
    def _run(
        self,
        image_path: str,
        run_manager: Optional[Any] = None,
    ) -> str:
        """Generate textual description from visual image."""
        try:
            import ollama
            
            visual_prompt = """You are a visual understanding assistant. Your task is to analyze the input image and generate a clear textual description that can be used as a query for information retrieval.

Instructions:
1. Analyze the image carefully. Extract all important visual information:
   - Objects or entities present
   - Actions or events occurring
   - Any visible text (use OCR if present)
   - Scene context, environment, or background details

2. Generate a descriptive text (1-3 sentences) summarizing the image content in plain English.
   - Be precise and concise
   - Avoid speculation beyond what is visible in the image
   - Use complete sentences

3. Output descriptive text only.

Additional Notes:
- This text will later be passed to a RAG system for retrieval and answer generation
- Do NOT attempt to answer any question yet
- Do NOT rephrase, correct, or split the text unless it is unclear; keep it faithful to the visual content"""

            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": visual_prompt,
                        "images": [image_path]
                    }
                ]
            )
            
            description = response["message"]["content"].strip()
            return description
            
        except Exception as e:
            error_msg = f"Visual understanding failed: {str(e)}"
            if run_manager:
                run_manager.on_tool_error(error_msg)
            raise Exception(error_msg)
    
    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[Any] = None,
    ) -> str:
        """Async version of visual understanding."""
        return self._run(image_path, run_manager)




