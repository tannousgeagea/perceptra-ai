
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



class BatchProcessor:
    """
    Process multiple prompts in batch.
    """
    
    def __init__(self, provider: str = None):
        self.client = UnifiedAIClient()
        self.provider = provider
    
    def process_batch(
        self,
        prompts: List[str],
        show_progress: bool = True
    ) -> List[Dict]:
        """Process multiple prompts"""
        results = []
        
        for i, prompt in enumerate(prompts, 1):
            if show_progress:
                print(f"Processing {i}/{len(prompts)}...", end="\r")
            
            try:
                response = self.client.generate(
                    prompt=prompt,
                    provider=self.provider
                )
                results.append({
                    "prompt": prompt,
                    "response": response['content'],
                    "model": response['model']
                })
            except Exception as e:
                results.append({
                    "prompt": prompt,
                    "error": str(e)
                })
        
        if show_progress:
            print()  # Clear progress line
        
        return results


def example_batch_processor():
    """Example: Batch processing"""
    print("=== Batch Processor Example ===\n")
    
    processor = BatchProcessor()
    
    prompts = [
        "What is 2+2?",
        "Name a color",
        "Say 'hello' in Spanish"
    ]
    
    results = processor.process_batch(prompts)
    
    print("Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Prompt: {result['prompt']}")
        if 'response' in result:
            print(f"   Response: {result['response']}")
        else:
            print(f"   Error: {result['error']}")