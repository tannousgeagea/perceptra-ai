
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



class StreamingHandler:
    """
    Handle streaming responses with various display options.
    """
    
    def __init__(self, provider: str = None):
        self.client = UnifiedAIClient()
        self.provider = provider
    
    def stream_with_typing_effect(self, prompt: str, delay: float = 0.05):
        """Stream response with typing effect"""
        print("Response: ", end="", flush=True)
        
        for chunk in self.client.generate(
            prompt=prompt,
            provider=self.provider,
            stream=True
        ):
            print(chunk, end="", flush=True)
            time.sleep(delay)
        
        print()  # New line at end
    
    def stream_with_callback(self, prompt: str, callback):
        """Stream response and call callback with each chunk"""
        full_response = ""
        
        for chunk in self.client.generate(
            prompt=prompt,
            provider=self.provider,
            stream=True
        ):
            full_response += chunk
            callback(chunk, full_response)
        
        return full_response


def example_streaming():
    """Example: Streaming responses"""
    print("=== Streaming Handler Example ===\n")
    
    handler = StreamingHandler()
    
    # Typing effect
    print("With typing effect:")
    handler.stream_with_typing_effect(
        "Count from 1 to 5 slowly",
        delay=0.1
    )
    
    # With callback
    print("\nWith callback (showing progress):")
    
    def progress_callback(chunk, full_text):
        print(f"[{len(full_text)} chars] {chunk}", end="", flush=True)
    
    handler.stream_with_callback(
        "Say hello",
        callback=progress_callback
    )
    print()
