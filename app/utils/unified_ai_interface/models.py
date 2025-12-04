#!/usr/bin/env python3
"""
Unified AI Provider Interface

A modular system supporting multiple AI providers through a single interface.
Supports: Ollama, OpenAI/ChatGPT, Google Gemini, Hugging Face, and extensible to others.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Generator, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json


# ====================
# COMMON DATA STRUCTURES
# ====================

class MessageRole(Enum):
    """Standard message roles across all providers"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """Unified message format"""
    role: MessageRole
    content: str
    images: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "role": self.role.value,
            "content": self.content
        }
        if self.images:
            result["images"] = self.images   # type: ignore
        if self.metadata:
            result["metadata"] = self.metadata  # type: ignore
        return result


@dataclass
class GenerationConfig:
    """Unified generation parameters"""
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    seed: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ModelInfo:
    """Unified model information"""
    name: str
    provider: str
    context_length: Optional[int] = None
    supports_streaming: bool = True
    supports_images: bool = False
    supports_tools: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResponse:
    """Unified response format"""
    content: str
    model: str
    provider: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None