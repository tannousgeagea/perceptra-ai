

#!/usr/bin/env python3
"""
Unified AI Provider Interface

A modular system supporting multiple AI providers through a single interface.
Supports: Ollama, OpenAI/ChatGPT, Google Gemini, Hugging Face, and extensible to others.
"""

from typing import Dict, List, Union, Optional, Generator, Any
from .models import GenerationConfig, GenerationResponse, Message, ModelInfo, MessageRole
from .base import AIProvider





class GeminiProvider(AIProvider):
    """Google Gemini provider implementation"""
    
    def __init__(self, api_key: str, default_model: str = "gemini-pro"):
        try:
            from google import genai
            self.genai = genai
            self.client = genai.Client(api_key=api_key)
            self.default_model = default_model
        except ImportError:
            raise ImportError("google-generativeai package not found. Install with: pip install google-generativeai")
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        model: Optional[str] = None,
        **kwargs
    ) -> Union[GenerationResponse, Generator[str, None, None]]:
        """Generate using Gemini"""
        model_name = model or self.default_model
        model_obj = self.client.models
        
        generation_config = self._convert_config_to_gemini(config) if config else None
        
        if stream:
            response = model_obj.generate_content(
                model=model_name,
                contents=prompt,
                config=generation_config,
                stream=True,
                **kwargs
            )
            return self._stream_gemini(response, model_name)
        else:
            response = model_obj.generate_content(
                prompt,
                generation_config=generation_config,
                **kwargs
            )
            return GenerationResponse(
                content=response.text,
                model=model_name,
                provider=self.provider_name,
                finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None
            )
    
    def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        model: Optional[str] = None,
        **kwargs
    ) -> Union[GenerationResponse, Generator[str, None, None]]:
        """Generate chat completion using Gemini"""
        model_name = model or self.default_model
        model_obj = self.genai.GenerativeModel(model_name)
        
        # Convert messages to Gemini format
        history = []
        for i, msg in enumerate(messages[:-1]):  # All except last
            role = "user" if msg.role in [MessageRole.USER, MessageRole.SYSTEM] else "model"
            history.append({"role": role, "parts": [msg.content]})
        
        chat = model_obj.start_chat(history=history)
        generation_config = self._convert_config_to_gemini(config) if config else None
        
        last_message = messages[-1].content
        
        if stream:
            response = chat.send_message(
                last_message,
                generation_config=generation_config,
                stream=True,
                **kwargs
            )
            return self._stream_gemini(response, model_name)
        else:
            response = chat.send_message(
                last_message,
                generation_config=generation_config,
                **kwargs
            )
            return GenerationResponse(
                content=response.text,
                model=model_name,
                provider=self.provider_name
            )
    
    def list_models(self) -> List[ModelInfo]:
        """List Gemini models"""
        models = self.client.list_models()
        return [
            ModelInfo(
                name=model.name.split('/')[-1],
                provider=self.provider_name,
                supports_streaming=True,
                supports_images="vision" in model.name
            )
            for model in models
            if "generateContent" in model.supported_generation_methods
        ]
    
    def generate_embeddings(
        self,
        text: Union[str, List[str]],
        model: str = "models/embedding-001",
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings using Gemini"""
        input_text = [text] if isinstance(text, str) else text
        embeddings = []
        for txt in input_text:
            result = self.genai.embed_content(
                model=model,
                content=txt,
                **kwargs
            )
            embeddings.append(result['embedding'])
        return embeddings
    
    def is_available(self) -> bool:
        """Check if Gemini is available"""
        try:
            list(self.genai.list_models())
            return True
        except:
            return False
    
    @property
    def provider_name(self) -> str:
        return "gemini"
    
    def _convert_config_to_gemini(self, config: GenerationConfig):
        """Convert unified config to Gemini GenerationConfig"""
        params = {}
        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.top_p is not None:
            params["top_p"] = config.top_p
        if config.top_k is not None:
            params["top_k"] = config.top_k
        if config.max_tokens is not None:
            params["max_output_tokens"] = config.max_tokens
        if config.stop_sequences:
            params["stop_sequences"] = config.stop_sequences
        return self.genai.GenerationConfig(**params) if params else None
    
    def _stream_gemini(self, response_stream, model_name):
        """Stream Gemini responses"""
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text