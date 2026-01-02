"""OCR tool for extracting text from images."""
import sys
from pathlib import Path

# Ensure project root is in path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import numpy as np
from typing import Union, Optional, Dict, Any
from PIL import Image
from tools.image_processing import ImagePreprocessor

# Try to import OCR libraries
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False


class OCRTool:
    """OCR tool for text extraction from images."""
    
    def __init__(self, ocr_engine: str = None):
        """Initialize OCR tool.
        
        Args:
            ocr_engine: "tesseract" or "paddleocr"
        """
        self.ocr_engine = ocr_engine or os.getenv("OCR_BACKEND", "tesseract")
        self.preprocessor = ImagePreprocessor()
        
        if ocr_engine == "tesseract" and not TESSERACT_AVAILABLE:
            raise ImportError(
                "Tesseract not available. Install with: pip install pytesseract. "
                "Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract"
            )
        
        if ocr_engine == "paddleocr" and not PADDLEOCR_AVAILABLE:
            raise ImportError(
                "PaddleOCR not available. Install with: pip install paddleocr"
            )
        
        if ocr_engine == "paddleocr" and PADDLEOCR_AVAILABLE:
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    def extract_text(
        self,
        image_path: Union[str, Path, np.ndarray],
        preprocess: bool = True,
        **preprocess_kwargs
    ) -> Dict[str, Any]:
        """Extract text from image and return text + confidence."""
        # Preprocess if requested
        if preprocess:
            payload = self.preprocessor.preprocess(
                image_path,
                resize=(1024, 1024),
                grayscale=True,
                normalize=True,
                denoise=True,
                **preprocess_kwargs
            )
            image = payload["image"]
        else:
            if isinstance(image_path, (str, Path)):
                image = np.array(Image.open(image_path))
            else:
                image = image_path
            payload = {"image": image, "metadata": {}}
        
        # Perform OCR
        if self.ocr_engine == "tesseract":
            pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
            text = pytesseract.image_to_string(pil_image)
            return {"text": text.strip(), "confidence": None, "engine": "tesseract"}
        
        elif self.ocr_engine == "paddleocr":
            if isinstance(image, np.ndarray):
                result = self.paddle_ocr.ocr(image, cls=True)
            else:
                result = self.paddle_ocr.ocr(str(image_path), cls=True)
            
            if result and result[0]:
                # Extract text from PaddleOCR result and confidence
                texts = [line[1][0] for line in result[0]]
                confidences = [line[1][1] for line in result[0] if len(line) > 1]
                avg_conf = float(np.mean(confidences)) if confidences else None
                return {"text": "\n".join(texts), "confidence": avg_conf, "engine": "paddleocr"}
            return {"text": "", "confidence": None, "engine": "paddleocr"}
        
        raise ValueError(f"Unknown OCR engine: {self.ocr_engine}")


def ocr_function(image_path: str) -> str:
    """Wrapper function for LangChain tool."""
    ocr = OCRTool()
    result = ocr.extract_text(image_path)
    return f"Text: {result.get('text','')}\nConfidence: {result.get('confidence')}"


