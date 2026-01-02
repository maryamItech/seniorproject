"""Image generation tool."""
import sys
from pathlib import Path

# Ensure project root is in path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import base64
from typing import Optional, Tuple
from pathlib import Path
from config.settings import OPENAI_API_KEY, STABILITY_API_KEY, PROJECT_ROOT


class ImageGenerator:
    """Image generation tool with multiple provider support."""
    
    def __init__(self, provider: str = "openai"):
        """Initialize image generator.
        
        Args:
            provider: "openai" (DALL-E) or "stability" (Stable Diffusion)
        """
        self.provider = provider
        self._setup_provider()
    
    def _setup_provider(self):
        """Setup the image generation provider."""
        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set. Cannot use DALL-E.")
        elif self.provider == "stability":
            if not STABILITY_API_KEY:
                raise ValueError("STABILITY_API_KEY not set. Cannot use Stability AI.")
    
    def generate(
        self,
        prompt: str,
        style: Optional[str] = None,
        resolution: Tuple[int, int] = (1024, 1024),
        output_path: Optional[Path] = None
    ) -> str:
        """Generate image from prompt.
        
        Args:
            prompt: Image generation prompt
            style: Optional style modifier
            resolution: Image resolution (width, height)
            output_path: Optional path to save image
        
        Returns:
            Path to generated image or base64 encoded image
        """
        # Add style to prompt if provided
        full_prompt = prompt
        if style:
            full_prompt = f"{prompt}, {style} style"
        
        if self.provider == "openai":
            return self._generate_openai(full_prompt, resolution, output_path)
        elif self.provider == "stability":
            return self._generate_stability(full_prompt, resolution, output_path)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _generate_openai(
        self,
        prompt: str,
        resolution: Tuple[int, int],
        output_path: Optional[Path]
    ) -> str:
        """Generate image using OpenAI DALL-E."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Map resolution to DALL-E supported sizes
            width, height = resolution
            size_map = {
                (1024, 1024): "1024x1024",
                (512, 512): "512x512",
                (256, 256): "256x256"
            }
            size = size_map.get(resolution, "1024x1024")
            
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                n=1,
                response_format="url"
            )
            
            image_url = response.data[0].url
            
            # Download and save image
            import requests
            img_response = requests.get(image_url)
            img_data = img_response.content
            
            if output_path is None:
                output_path = PROJECT_ROOT / "generated_images" / f"generated_{hash(prompt) % 10000}.png"
                output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, "wb") as f:
                f.write(img_data)
            
            # Also return base64 for API responses
            base64_img = base64.b64encode(img_data).decode("utf-8")
            return f"Image generated and saved to: {output_path}\nBase64: {base64_img[:100]}..."
        except ImportError:
            return "OpenAI library not available. Install with: pip install openai"
        except Exception as e:
            return f"Error generating image: {str(e)}"
    
    def _generate_stability(
        self,
        prompt: str,
        resolution: Tuple[int, int],
        output_path: Optional[Path]
    ) -> str:
        """Generate image using Stability AI."""
        try:
            import requests
            
            url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
            headers = {
                "Authorization": f"Bearer {STABILITY_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7,
                "width": resolution[0],
                "height": resolution[1],
                "samples": 1
            }
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Extract base64 image
            base64_img = data["artifacts"][0]["base64"]
            img_data = base64.b64decode(base64_img)
            
            if output_path is None:
                output_path = PROJECT_ROOT / "generated_images" / f"generated_{hash(prompt) % 10000}.png"
                output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, "wb") as f:
                f.write(img_data)
            
            return f"Image generated and saved to: {output_path}\nBase64: {base64_img[:100]}..."
        except Exception as e:
            return f"Error generating image with Stability AI: {str(e)}"


def generate_image(
    prompt: str,
    style: Optional[str] = None,
    resolution: Optional[Tuple[int, int]] = None
) -> str:
    """Wrapper function for LangChain tool."""
    if resolution is None:
        resolution = (1024, 1024)
    
    # Try OpenAI first, fallback to Stability
    try:
        generator = ImageGenerator(provider="openai")
        return generator.generate(prompt, style, resolution)
    except ValueError:
        try:
            generator = ImageGenerator(provider="stability")
            return generator.generate(prompt, style, resolution)
        except ValueError:
            return "No image generation API keys configured. Please set OPENAI_API_KEY or STABILITY_API_KEY."






