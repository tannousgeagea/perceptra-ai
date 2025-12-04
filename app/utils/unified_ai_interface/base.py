

#!/usr/bin/env python3
"""
Unified AI Provider Interface

A modular system supporting multiple AI providers through a single interface.
Supports: Ollama, OpenAI/ChatGPT, Google Gemini, Hugging Face, and extensible to others.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Generator, Any
from .models import GenerationConfig, GenerationResponse, Message, ModelInfo
from dataclasses import dataclass, asdict
from enum import Enum
import json



# ====================
# BASE PROVIDER INTERFACE
# ====================

class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[GenerationResponse, Generator[str, None, None]]:
        """Generate a completion for the given prompt"""
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[GenerationResponse, Generator[str, None, None]]:
        """Generate a chat completion"""
        pass
    
    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """List available models"""
        pass
    
    @abstractmethod
    def generate_embeddings(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available"""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name"""
        pass
