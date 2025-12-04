
#!/usr/bin/env python3
"""
Unified AI Provider Interface

A modular system supporting multiple AI providers through a single interface.
Supports: Ollama, OpenAI/ChatGPT, Google Gemini, Hugging Face, and extensible to others.
"""

from typing import Dict, List, Union, Optional, Generator, Any
from .models import GenerationConfig, GenerationResponse, Message, ModelInfo, MessageRole
from .base import AIProvider



class OpenAIProvider(AIProvider):
    """OpenAI/ChatGPT provider implementation"""
    
    def __init__(self, api_key: str, default_model: str = "gpt-4"):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.default_model = default_model
        except ImportError:
            raise ImportError("openai package not found. Install with: pip install openai")
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        model: Optional[str] = None,
        **kwargs
    ) -> Union[GenerationResponse, Generator[str, None, None]]:
        """Generate using OpenAI (via chat endpoint)"""
        messages = [Message(role=MessageRole.USER, content=prompt)]
        return self.chat(messages, config, stream, model, **kwargs)
    
    def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        model: Optional[str] = None,
        **kwargs
    ) -> Union[GenerationResponse, Generator[str, None, None]]:
        """Generate chat completion using OpenAI"""
        model_name = model or self.default_model
        
        openai_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
        
        params = {
            "model": model_name,
            "messages": openai_messages,
            "stream": stream
        }
        
        if config:
            params.update(self._convert_config_to_openai(config))
        params.update(kwargs)
        
        response = self.client.chat.completions.create(**params)
        
        if stream:
            return self._stream_openai(response, model_name)
        else:
            return GenerationResponse(
                content=response.choices[0].message.content,
                model=model_name,
                provider=self.provider_name,
                finish_reason=response.choices[0].finish_reason,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            )
    
    def list_models(self) -> List[ModelInfo]:
        """List OpenAI models"""
        models = self.client.models.list()
        return [
            ModelInfo(
                name=model.id,
                provider=self.provider_name,
                supports_streaming=True,
                supports_images="vision" in model.id,
                supports_tools=True
            )
            for model in models.data
            if "gpt" in model.id
        ]
    
    def generate_embeddings(
        self,
        text: Union[str, List[str]],
        model: str = "text-embedding-ada-002",
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        input_text = [text] if isinstance(text, str) else text
        response = self.client.embeddings.create(
            model=model,
            input=input_text,
            **kwargs
        )
        return [item.embedding for item in response.data]
    
    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        try:
            self.client.models.list()
            return True
        except:
            return False
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def _convert_config_to_openai(self, config: GenerationConfig) -> Dict[str, Any]:
        """Convert unified config to OpenAI parameters"""
        params = {}
        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.top_p is not None:
            params["top_p"] = config.top_p
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        if config.stop_sequences:
            params["stop"] = config.stop_sequences
        if config.seed is not None:
            params["seed"] = config.seed
        if config.presence_penalty is not None:
            params["presence_penalty"] = config.presence_penalty
        if config.frequency_penalty is not None:
            params["frequency_penalty"] = config.frequency_penalty
        return params
    
    def _stream_openai(self, response_stream, model_name):
        """Stream OpenAI responses"""
        for chunk in response_stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content