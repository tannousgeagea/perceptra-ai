
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

class DocumentSummarizer:
    """
    Summarize documents using AI.
    Can handle long documents by chunking.
    """
    
    def __init__(self, provider: str = None):
        self.client = UnifiedAIClient()
        self.provider = provider
    
    def summarize(
        self,
        text: str,
        style: str = "concise"
    ) -> str:
        """Summarize text"""
        prompt = f"""Summarize the following text in a {style} style:

{text}

Summary:"""
        
        response = self.client.generate(
            prompt=prompt,
            provider=self.provider,
            temperature=0.3,
            max_tokens=500
        )
        
        return response['content']
    
    def summarize_file(self, file_path: Path) -> str:
        """Summarize a text file"""
        text = file_path.read_text()
        return self.summarize(text)


def example_document_summarizer():
    """Example: Document summarization"""
    print("=== Document Summarizer Example ===\n")
    
    summarizer = DocumentSummarizer()
    
    sample_text = """
    Artificial Intelligence (AI) has transformed numerous industries over 
    the past decade. From healthcare to finance, AI systems are being deployed 
    to automate tasks, analyze data, and make predictions. Machine learning, 
    a subset of AI, enables systems to learn from data without explicit programming.
    Deep learning, using neural networks, has achieved remarkable success in 
    image recognition, natural language processing, and game playing.
    """
    
    summary = summarizer.summarize(sample_text, style="concise")
    print(f"Original length: {len(sample_text)} characters")
    print(f"Summary length: {len(summary)} characters")
    print(f"\nSummary:\n{summary}")