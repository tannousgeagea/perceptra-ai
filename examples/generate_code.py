

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


class CodeGenerator:
    """
    Generate code snippets from natural language descriptions.
    """
    
    def __init__(self, provider: str = None):
        self.client = UnifiedAIClient()
        self.provider = provider
    
    def generate_code(
        self,
        description: str,
        language: str = "Python"
    ) -> str:
        """Generate code from description"""
        prompt = f"""Write a {language} function that {description}.
Include docstring and comments.

Code:"""
        
        response = self.client.generate(
            prompt=prompt,
            provider=self.provider,
            temperature=0.2,
            max_tokens=500
        )
        
        return response['content']
    
    def explain_code(self, code: str) -> str:
        """Explain what code does"""
        prompt = f"""Explain what this code does in simple terms:

{code}

Explanation:"""
        
        response = self.client.generate(
            prompt=prompt,
            provider=self.provider,
            temperature=0.3
        )
        
        return response['content']


def example_code_generator():
    """Example: Code generation"""
    print("=== Code Generator Example ===\n")
    
    generator = CodeGenerator()
    
    # Generate code
    description = "calculates the factorial of a number recursively"
    code = generator.generate_code(description)
    
    print(f"Description: {description}")
    print(f"\nGenerated Code:\n{code}")
    
    # Explain code
    explanation = generator.explain_code(code)
    print(f"\nExplanation:\n{explanation}")