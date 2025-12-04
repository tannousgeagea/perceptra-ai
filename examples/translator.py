
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

class MultiLanguageTranslator:
    """
    Translate text between languages using AI.
    """
    
    def __init__(self, provider: str = None):
        self.client = UnifiedAIClient()
        self.provider = provider
    
    def translate(
        self,
        text: str,
        target_language: str,
        source_language: str = "auto"
    ) -> str:
        """Translate text"""
        if source_language == "auto":
            prompt = f"Translate the following text to {target_language}:\n\n{text}"
        else:
            prompt = f"Translate from {source_language} to {target_language}:\n\n{text}"
        
        response = self.client.generate(
            prompt=prompt,
            provider=self.provider,
            temperature=0.3
        )
        
        return response['content']


def example_translator():
    """Example: Translation"""
    print("=== Multi-Language Translator Example ===\n")
    
    translator = MultiLanguageTranslator()
    
    text = "Hello, how are you today?"
    languages = ["Spanish", "French", "German"]
    
    print(f"Original (English): {text}\n")
    
    for lang in languages:
        translation = translator.translate(text, lang)
        print(f"{lang}: {translation}")