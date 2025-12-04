

#!/usr/bin/env python3
"""
Unified AI Provider Interface

A modular system supporting multiple AI providers through a single interface.
Supports: Ollama, OpenAI/ChatGPT, Google Gemini, Hugging Face, and extensible to others.
"""

from typing import Dict, List, Union, Optional, Generator, Any
from .models import GenerationConfig, GenerationResponse, Message, ModelInfo
from .base import AIProvider

class OllamaProvider(AIProvider):
    """Ollama provider implementation"""
    
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "llama3.2"):
        from utils.adapters.ollama import OllamaAdapter, ModelOptions, ChatMessage as OllamaChatMessage
        
        self.adapter = OllamaAdapter(base_url=base_url)
        self.default_model = default_model
        self._ollama_model_options = ModelOptions
        self._ollama_chat_message = OllamaChatMessage
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        model: Optional[str] = None,
        **kwargs
    ) -> Union[GenerationResponse, Generator[str, None, None]]:
        """Generate completion using Ollama"""
        model_name = model or self.default_model
        options = self._convert_config_to_ollama(config) if config else None
        
        response = self.adapter.generate(
            model=model_name,
            prompt=prompt,
            options=options,
            stream=stream,
            **kwargs
        )
        
        if stream:
            return self._stream_ollama_generate(response, model_name)
        else:
            return GenerationResponse(
                content=response["response"],    # type: ignore
                model=model_name,
                provider=self.provider_name,
                finish_reason=response.get("done_reason"),  # type: ignore
                metadata={"context": response.get("context")}  # type: ignore
            )
    
    def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        model: Optional[str] = None,
        **kwargs
    ) -> Union[GenerationResponse, Generator[str, None, None]]:
        """Generate chat completion using Ollama"""
        model_name = model or self.default_model
        options = self._convert_config_to_ollama(config) if config else None
        
        ollama_messages = [
            self._ollama_chat_message(
                role=msg.role.value,
                content=msg.content,
                images=msg.images
            )
            for msg in messages
        ]
        
        response = self.adapter.chat(
            model=model_name,
            messages=ollama_messages,   # type: ignore
            options=options,
            stream=stream,
            **kwargs
        )
        
        if stream:
            return self._stream_ollama_chat(response, model_name)
        else:
            return GenerationResponse(
                content=response["message"]["content"],   # type: ignore
                model=model_name,
                provider=self.provider_name, 
                finish_reason=response.get("done_reason"),   # type: ignore
                metadata=response.get("metadata")   # type: ignore
            )
    
    def list_models(self) -> List[ModelInfo]:
        """List Ollama models"""
        response = self.adapter.list_models()
        return [
            ModelInfo(
                name=model["name"],
                provider=self.provider_name,
                supports_streaming=True,
                metadata=model
            )
            for model in response.get("models", [])
        ]
    
    def generate_embeddings(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings using Ollama"""
        model_name = model or self.default_model
        response = self.adapter.generate_embeddings(
            model=model_name,
            input=text,
            **kwargs
        )
        embeddings = response.get("embeddings", [])
        return embeddings if isinstance(text, list) else [embeddings]
    
    def is_available(self) -> bool:
        """Check if Ollama is available"""
        return self.adapter.is_server_available()
    
    @property
    def provider_name(self) -> str:
        return "ollama"
    
    def _convert_config_to_ollama(self, config: GenerationConfig):
        """Convert unified config to Ollama ModelOptions"""
        return self._ollama_model_options(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            num_predict=config.max_tokens,
            stop=config.stop_sequences,
            seed=config.seed,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty
        )
    
    def _stream_ollama_generate(self, response_gen, model_name):
        """Stream Ollama generate responses"""
        for chunk in response_gen:
            if not chunk.get('done', False):
                yield chunk.get('response', '')
    
    def _stream_ollama_chat(self, response_gen, model_name):
        """Stream Ollama chat responses"""
        for chunk in response_gen:
            if not chunk.get('done', False):
                yield chunk.get('message', {}).get('content', '')