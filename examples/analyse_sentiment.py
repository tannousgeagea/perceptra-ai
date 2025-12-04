
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


class SentimentAnalyzer:
    """
    Analyze sentiment of text.
    """
    
    def __init__(self, provider: str = None):
        self.client = UnifiedAIClient()
        self.provider = provider
    
    def analyze(self, text: str) -> Dict[str, any]:
        """Analyze sentiment"""
        prompt = f"""Analyze the sentiment of the following text.
Respond with ONLY a JSON object containing:
- sentiment: positive, negative, or neutral
- confidence: 0-1
- explanation: brief explanation

Text: {text}

JSON:"""
        
        response = self.client.generate(
            prompt=prompt,
            provider=self.provider,
            temperature=0.1
        )
        
        try:
            # Try to parse JSON from response
            content = response['content']
            # Remove markdown code blocks if present
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            return json.loads(content.strip())
        except:
            return {"error": "Could not parse response"}


def example_sentiment_analyzer():
    """Example: Sentiment analysis"""
    print("=== Sentiment Analyzer Example ===\n")
    
    analyzer = SentimentAnalyzer()
    
    texts = [
        "I absolutely love this product! It's amazing!",
        "This is the worst experience I've ever had.",
        "The weather is nice today."
    ]
    
    for text in texts:
        result = analyzer.analyze(text)
        print(f"Text: {text}")
        if 'error' not in result:
            print(f"Sentiment: {result.get('sentiment', 'unknown')}")
            print(f"Confidence: {result.get('confidence', 0)}")
            print(f"Explanation: {result.get('explanation', '')}")
        else:
            print(f"Error: {result['error']}")
        print()