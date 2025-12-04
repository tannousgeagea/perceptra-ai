#!/usr/bin/env python3
"""
Example Applications using the Unified AI API

Demonstrates various use cases and integration patterns.
"""

import json
import time
from pathlib import Path
from typing import List, Dict
import sys
from pathlib import Path
root = Path(__file__).parent.parent

sys.path.append(f"{root}/app")

from sdk.client import UnifiedAIClient, quick_generate, quick_chat #type:ignore


class ImageAnalyzer:
    """
    Analyze images using multimodal models.
    """
    
    def __init__(self, provider: str = "ollama", model: str = "llava"):
        self.client = UnifiedAIClient()
        self.provider = provider
        self.model = model
    
    def analyze_image(
        self,
        image_path: Path,
        question: str = "What's in this image?"
    ) -> str:
        """Analyze an image"""
        response = self.client.generate_with_images(
            prompt=question,
            images=[image_path],
            provider=self.provider,
            model=self.model
        )
        
        return response['content']
    
    def describe_image(self, image_path: Path) -> str:
        """Get detailed description of image"""
        return self.analyze_image(
            image_path,
            "Provide a detailed description of this image."
        )
    
    def detect_objects(self, image_path: Path) -> str:
        """Detect objects in image"""
        return self.analyze_image(
            image_path,
            "List all objects you can see in this image."
        )


def example_image_analyzer():
    """Example: Image analysis"""
    print("=== Image Analyzer Example ===\n")
    print("Note: Requires a multimodal model like llava")
    print("Skipping actual execution (no test images available)\n")
    
    # analyzer = ImageAnalyzer()
    # 
    # image_path = Path("test_image.jpg")
    # if image_path.exists():
    #     description = analyzer.describe_image(image_path)
    #     print(f"Description: {description}")