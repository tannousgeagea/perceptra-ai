#!/usr/bin/env python3
"""
Unified AI Provider Interface

A modular system supporting multiple AI providers through a single interface.
Supports: Ollama, OpenAI/ChatGPT, Google Gemini, Hugging Face, and extensible to others.
"""

from typing import Dict, List, Union, Optional, Generator, Any
from .models import GenerationConfig, GenerationResponse, Message, ModelInfo
from .base import AIProvider

class HuggingFaceProvider(AIProvider):
    """Hugging Face provider implementation"""
    
    def __init__(self, api_key: str, default_model: str = "meta-llama/Llama-2-7b-chat-hf"):
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=api_key)
            self.default_model = default_model
        except ImportError:
            raise ImportError("huggingface_hub package not found. Install with: pip install huggingface_hub")
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        model: Optional[str] = None,
        **kwargs
    ) -> Union[GenerationResponse, Generator[str, None, None]]:
        """Generate using Hugging Face"""
        model_name = model or self.default_model
        
        params = self._convert_config_to_hf(config) if config else {}
        params.update(kwargs)
        
        if stream:
            response = self.client.text_generation(
                prompt,
                model=model_name,
                stream=True,
                **params
            )
            return self._stream_hf(response, model_name)
        else:
            response = self.client.text_generation(
                prompt,
                model=model_name,
                **params
            )
            return GenerationResponse(
                content=response,
                model=model_name,
                provider=self.provider_name
            )
    
    def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        model: Optional[str] = None,
        **kwargs
    ) -> Union[GenerationResponse, Generator[str, None, None]]:
        """Generate chat completion using Hugging Face"""
        # Convert messages to a single prompt (simple approach)
        prompt = "\n".join([f"{msg.role.value}: {msg.content}" for msg in messages])
        prompt += "\nassistant:"
        
        return self.generate(prompt, config, stream, model, **kwargs)
    
    def list_models(self) -> List[ModelInfo]:
        """List Hugging Face models (popular ones)"""
        # Note: Full model listing requires API calls, returning popular models
        popular_models = [
            "meta-llama/Llama-2-7b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "google/flan-t5-base",
            "bigscience/bloom-560m"
        ]
        return [
            ModelInfo(
                name=model,
                provider=self.provider_name,
                supports_streaming=True
            )
            for model in popular_models
        ]
    
    def generate_embeddings(
        self,
        text: Union[str, List[str]],
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings using Hugging Face"""
        input_text = [text] if isinstance(text, str) else text
        embeddings = []
        for txt in input_text:
            embedding = self.client.feature_extraction(txt, model=model)
            embeddings.append(embedding)
        return embeddings
    
    def is_available(self) -> bool:
        """Check if Hugging Face is available"""
        try:
            # Simple check - try to generate with a tiny model
            self.client.text_generation("test", model="gpt2", max_new_tokens=1)
            return True
        except:
            return False
    
    @property
    def provider_name(self) -> str:
        return "huggingface"
    
    def _convert_config_to_hf(self, config: GenerationConfig) -> Dict[str, Any]:
        """Convert unified config to Hugging Face parameters"""
        params = {}
        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.top_p is not None:
            params["top_p"] = config.top_p
        if config.top_k is not None:
            params["top_k"] = config.top_k
        if config.max_tokens is not None:
            params["max_new_tokens"] = config.max_tokens
        if config.stop_sequences:
            params["stop_sequences"] = config.stop_sequences
        return params
    
    def _stream_hf(self, response_stream, model_name):
        """Stream Hugging Face responses"""
        for chunk in response_stream:
            yield chunk