"""Image preprocessing utilities."""
import sys
from pathlib import Path

# Ensure project root is in path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import Union, Optional, Tuple, Dict, Any
from PIL import Image
import cv2


class ImagePreprocessor:
    """Image preprocessing utilities."""
    
    @staticmethod
    def resize(image: np.ndarray, max_size: Tuple[int, int] = (1024, 1024), keep_aspect: bool = True) -> np.ndarray:
        """Resize image while keeping aspect ratio."""
        if keep_aspect:
            h, w = image.shape[:2]
            max_h, max_w = max_size
            
            # Calculate scaling factor
            scale = min(max_h / h, max_w / w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize
            if len(image.shape) == 3:
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized
        else:
            return cv2.resize(image, max_size, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    @staticmethod
    def normalize(image: np.ndarray, method: str = "min_max") -> np.ndarray:
        """Normalize image values."""
        if method == "min_max":
            min_val, max_val = image.min(), image.max()
            if max_val > min_val:
                return (image - min_val) / (max_val - min_val)
            return image
        elif method == "z_score":
            mean, std = image.mean(), image.std()
            if std > 0:
                return (image - mean) / std
            return image
        return image
    
    @staticmethod
    def remove_noise(image: np.ndarray, method: str = "gaussian") -> np.ndarray:
        """Remove noise from image."""
        if method == "gaussian":
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == "median":
            return cv2.medianBlur(image, 5)
        elif method == "bilateral":
            if len(image.shape) == 3:
                return cv2.bilateralFilter(image, 9, 75, 75)
            else:
                # Bilateral filter works on color images, convert to color first
                color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                filtered = cv2.bilateralFilter(color_img, 9, 75, 75)
                return cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
        return image
    
    @staticmethod
    def preprocess(
        image_path: Union[str, Path, np.ndarray, bytes, Image.Image],
        resize: Optional[Tuple[int, int]] = None,
        grayscale: bool = False,
        normalize: bool = True,
        denoise: bool = True,
        return_array: bool = True
    ) -> Dict[str, Any]:
        """Complete image preprocessing pipeline.

        Returns a dict with processed image and metadata.
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_path, bytes):
            image = np.array(Image.open(Path(image_path)))
        elif isinstance(image_path, Image.Image):
            image = np.array(image_path.convert("RGB"))
        else:
            image = image_path.copy()

        orig_size = (image.shape[0], image.shape[1])

        # Resize
        if resize:
            image = ImagePreprocessor.resize(image, resize)
        
        # Grayscale
        if grayscale:
            image = ImagePreprocessor.to_grayscale(image)
        
        # Denoise
        if denoise:
            image = ImagePreprocessor.remove_noise(image)
        
        # Normalize
        if normalize:
            image = ImagePreprocessor.normalize(image)
            # Convert to uint8 for display
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
        
        new_size = (image.shape[0], image.shape[1])

        payload = {
            "image": image if return_array else Image.fromarray(image),
            "metadata": {
                "orig_size": orig_size,
                "new_size": new_size,
                "grayscale": grayscale,
                "normalize": normalize,
                "denoise": denoise,
                "resize": resize,
            },
        }
        return payload


