#!/usr/bin/env python3
"""
Unified AI Provider Interface

A modular system supporting multiple AI providers through a single interface.
Supports: Ollama, OpenAI/ChatGPT, Google Gemini, Hugging Face, and extensible to others.
"""

from typing import Dict, List, Union, Optional, Generator, Any
from .models import GenerationConfig, GenerationResponse, Message, ModelInfo
from .base import AIProvider


# ====================
# UNIFIED AI MANAGER
# ====================

class AIManager:
    """
    Unified manager for multiple AI providers
    Provides a single interface to interact with any provider
    """
    
    def __init__(self):
        self.providers: Dict[str, AIProvider] = {}
        self.default_provider: Optional[str] = None
    
    def register_provider(self, provider: AIProvider, set_as_default: bool = False):
        """Register a new provider"""
        self.providers[provider.provider_name] = provider
        if set_as_default or not self.default_provider:
            self.default_provider = provider.provider_name
    
    def set_default_provider(self, provider_name: str):
        """Set the default provider"""
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not registered")
        self.default_provider = provider_name
    
    def get_provider(self, provider_name: Optional[str] = None) -> AIProvider:
        """Get a specific provider or the default"""
        name = provider_name or self.default_provider
        if not name:
            raise ValueError("No provider specified and no default provider set")
        if name not in self.providers:
            raise ValueError(f"Provider '{name}' not registered")
        return self.providers[name]
    
    def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[GenerationResponse, Generator[str, None, None]]:
        """Generate using specified or default provider"""
        return self.get_provider(provider).generate(prompt, config, stream, **kwargs)
    
    def chat(
        self,
        messages: List[Message],
        provider: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[GenerationResponse, Generator[str, None, None]]:
        """Chat using specified or default provider"""
        return self.get_provider(provider).chat(messages, config, stream, **kwargs)
    
    def list_models(self, provider: Optional[str] = None) -> List[ModelInfo]:
        """List models from specified or default provider"""
        return self.get_provider(provider).list_models()
    
    def list_all_models(self) -> Dict[str, List[ModelInfo]]:
        """List models from all registered providers"""
        return {
            name: provider.list_models()
            for name, provider in self.providers.items()
        }
    
    def generate_embeddings(
        self,
        text: Union[str, List[str]],
        provider: Optional[str] = None,
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings using specified or default provider"""
        return self.get_provider(provider).generate_embeddings(text, **kwargs)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available (reachable) providers"""
        return [
            name for name, provider in self.providers.items()
            if provider.is_available()
        ]