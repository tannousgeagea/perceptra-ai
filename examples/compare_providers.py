
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

class ProviderComparison:
    """
    Compare responses from different providers for the same prompt.
    """
    
    def __init__(self):
        self.client = UnifiedAIClient()
    
    def compare_providers(
        self,
        prompt: str,
        providers: List[str] = None
    ) -> Dict[str, str]:
        """Get responses from multiple providers"""
        if providers is None:
            # Get available providers
            available = self.client.list_providers()
            providers = [p['name'] for p in available if p['available']]
        
        results = {}
        
        for provider in providers:
            try:
                print(f"Querying {provider}...")
                response = self.client.generate(
                    prompt=prompt,
                    provider=provider,
                    temperature=0.7
                )
                results[provider] = {
                    "content": response['content'],
                    "model": response['model']
                }
            except Exception as e:
                results[provider] = {
                    "error": str(e)
                }
        
        return results


def example_provider_comparison():
    """Example: Compare providers"""
    print("=== Provider Comparison Example ===\n")
    
    comparator = ProviderComparison()
    prompt = "What is the meaning of life?"
    
    results = comparator.compare_providers(prompt)
    
    print(f"Prompt: {prompt}\n")
    for provider, result in results.items():
        print(f"\n{provider.upper()}:")
        if 'content' in result:
            print(f"Model: {result['model']}")
            print(f"Response: {result['content'][:200]}...")
        else:
            print(f"Error: {result['error']}")