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

from sdk.client import UnifiedAIClient, quick_generate, quick_chat


# ====================
# EXAMPLE 1: SIMPLE CHATBOT
# ====================

class SimpleChatbot:
    """
    A simple chatbot that maintains conversation history
    and can use any AI provider.
    """
    
    def __init__(self, provider: str = None, model: str = None):
        self.client = UnifiedAIClient()
        self.provider = provider
        self.model = model
        self.conversation_history = []
    
    def chat(self, user_message: str) -> str:
        """Send a message and get a response"""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Get response
        response = self.client.chat(
            messages=self.conversation_history,
            provider=self.provider,
            model=self.model,
            temperature=0.7
        )
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response['content']
        })
        
        return response['content']
    
    def reset(self):
        """Reset conversation history"""
        self.conversation_history = []
    
    def set_system_prompt(self, prompt: str):
        """Set a system prompt"""
        self.conversation_history.insert(0, {
            "role": "system",
            "content": prompt
        })


def example_simple_chatbot():
    """Example: Simple chatbot usage"""
    print("=== Simple Chatbot Example ===\n")
    
    bot = SimpleChatbot()
    bot.set_system_prompt("You are a helpful and friendly assistant.")
    
    # Simulate a conversation
    messages = [
        "Hello! What's your name?",
        "Can you help me write Python code?",
        "Show me how to read a file in Python"
    ]
    
    for msg in messages:
        print(f"User: {msg}")
        response = bot.chat(msg)
        print(f"Bot: {response}\n")