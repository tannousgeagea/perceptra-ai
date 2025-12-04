#!/usr/bin/env python3
"""
Unified AI API Client SDK

A Python client for interacting with the Unified AI API.
Provides a simple, intuitive interface for all API endpoints.
"""

import json
import requests
from typing import List, Optional, Dict, Any, Union, Generator
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ClientConfig:
    """Client configuration"""
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    timeout: int = 30
    verify_ssl: bool = True


class UnifiedAIClient:
    """
    Client for the Unified AI API
    
    Example:
        client = UnifiedAIClient(api_key="your-key")
        response = client.generate("What is AI?")
        print(response['content'])
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        verify_ssl: bool = True
    ):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.config = ClientConfig(
            base_url=base_url.rstrip('/'),
            api_key=api_key,
            timeout=timeout,
            verify_ssl=verify_ssl
        )
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {api_key}"
            })
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> requests.Response:
        """Make an HTTP request"""
        url = f"{self.config.base_url}{endpoint}"
        
        kwargs.setdefault('timeout', self.config.timeout)
        kwargs.setdefault('verify', self.config.verify_ssl)
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise UnifiedAIClientError(f"Request failed: {str(e)}")
    
    def _stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        """Stream response data"""
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data = json.loads(line_str[6:])
                    if 'content' in data:
                        yield data['content']
                    elif 'error' in data:
                        raise UnifiedAIClientError(f"Streaming error: {data['error']}")
    
    # ====================
    # GENERATION METHODS
    # ====================
    
    def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stream: bool = False, 
        **kwargs
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """
        Generate text completion
        
        Args:
            prompt: Input prompt
            provider: AI provider to use (ollama, openai, gemini, huggingface)
            model: Specific model to use
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stream: Whether to stream the response
            **kwargs: Additional parameters
        
        Returns:
            Response dict if not streaming, generator if streaming
        
        Example:
            response = client.generate("Tell me a joke")
            print(response['content'])
        """
        payload = {
            "prompt": prompt,
            "stream": stream
        }
        
        if provider:
            payload["provider"] = provider
        if model:
            payload["model"] = model
        
        # Build config
        config = {}
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        if top_p is not None:
            config["top_p"] = top_p
        if top_k is not None:
            config["top_k"] = top_k
        
        if config:
            payload["config"] = config
        
        # Add any additional kwargs
        payload.update(kwargs)
        
        response = self._request(
            "POST",
            "/api/v1/generate",
            json=payload,
            stream=stream
        )
        
        if stream:
            return self._stream_response(response)
        else:
            return response.json()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """
        Generate chat completion
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            provider: AI provider to use
            model: Specific model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters
        
        Returns:
            Response dict if not streaming, generator if streaming
        
        Example:
            messages = [
                {"role": "user", "content": "Hello!"}
            ]
            response = client.chat(messages)
            print(response['content'])
        """
        payload = {
            "messages": messages,
            "stream": stream
        }
        
        if provider:
            payload["provider"] = provider
        if model:
            payload["model"] = model
        
        # Build config
        config = {}
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        
        if config:
            payload["config"] = config
        
        payload.update(kwargs)
        
        response = self._request(
            "POST",
            "/api/v1/chat",
            json=payload,
            stream=stream
        )
        
        if stream:
            return self._stream_response(response)
        else:
            return response.json()
    
    def generate_with_images(
        self,
        prompt: str,
        images: List[Union[str, Path]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """
        Generate text with image files
        
        Args:
            prompt: Input prompt
            images: List of image file paths
            provider: AI provider to use
            model: Specific model to use
            stream: Whether to stream the response
        
        Returns:
            Response dict if not streaming, generator if streaming
        
        Example:
            response = client.generate_with_images(
                "What's in this image?",
                ["image.jpg"]
            )
            print(response['content'])
        """
        files = []
        for img_path in images:
            path = Path(img_path)
            if not path.exists():
                raise UnifiedAIClientError(f"Image file not found: {img_path}")
            files.append(
                ('files', (path.name, open(path, 'rb'), 'image/jpeg'))
            )
        
        data = {'prompt': prompt}
        if provider:
            data['provider'] = provider
        if model:
            data['model'] = model
        if stream:
            data['stream'] = 'true'
        
        try:
            response = self._request(
                "POST",
                "/api/v1/generate/upload",
                data=data,
                files=files,
                stream=stream
            )
            
            if stream:
                return self._stream_response(response)
            else:
                return response.json()
        finally:
            # Close file handles
            for _, file_tuple in files:
                file_tuple[1].close()
    
    # ====================
    # MODEL METHODS
    # ====================
    
    def list_models(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        List available models
        
        Args:
            provider: Specific provider to list models from (optional)
        
        Returns:
            Dict with models list
        
        Example:
            models = client.list_models()
            for model in models['models']:
                print(model['name'])
        """
        if provider:
            endpoint = f"/api/v1/models/{provider}"
        else:
            endpoint = "/api/v1/models"
        
        response = self._request("GET", endpoint)
        return response.json()
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """
        List available providers
        
        Returns:
            List of provider info dicts
        
        Example:
            providers = client.list_providers()
            for p in providers:
                print(f"{p['name']}: {'✓' if p['available'] else '✗'}")
        """
        response = self._request("GET", "/api/v1/providers")
        return response.json()
    
    # ====================
    # EMBEDDING METHODS
    # ====================
    
    def generate_embeddings(
        self,
        text: Union[str, List[str]],
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate embeddings
        
        Args:
            text: Text or list of texts to embed
            provider: AI provider to use
            model: Specific embedding model to use
        
        Returns:
            Dict with embeddings and metadata
        
        Example:
            result = client.generate_embeddings("Hello world")
            embeddings = result['embeddings']
        """
        payload = {"input": text}
        
        if provider:
            payload["provider"] = provider
        if model:
            payload["model"] = model
        
        response = self._request("POST", "/api/v1/embeddings", json=payload)
        return response.json()
    
    # ====================
    # UTILITY METHODS
    # ====================
    
    def health(self) -> Dict[str, Any]:
        """
        Check API health
        
        Returns:
            Health status dict
        
        Example:
            health = client.health()
            print(health['status'])
        """
        response = self._request("GET", "/health")
        return response.json()
    
    def info(self) -> Dict[str, Any]:
        """
        Get API information
        
        Returns:
            API info dict
        
        Example:
            info = client.info()
            print(info['version'])
        """
        response = self._request("GET", "/info")
        return response.json()
    
    def is_available(self) -> bool:
        """
        Check if API is available
        
        Returns:
            True if API is reachable, False otherwise
        """
        try:
            health = self.health()
            return health['status'] in ['healthy', 'degraded']
        except:
            return False
    
    # ====================
    # CONTEXT MANAGER
    # ====================
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.session.close()
    
    def close(self):
        """Close the session"""
        self.session.close()

    # ====================
    # OLLAMA-SPECIFIC METHODS
    # ====================
    
    def ollama_list_running(self) -> Dict[str, Any]:
        """List currently running Ollama models"""
        response = self._request("GET", "/api/v1/ollama/models/running")
        return response.json()
    
    def ollama_show_model(self, model: str, verbose: bool = False) -> Dict[str, Any]:
        """Show detailed Ollama model information"""
        response = self._request(
            "POST",
            "/api/v1/ollama/models/show",
            params={"model": model, "verbose": verbose}
        )
        return response.json()
    
    def ollama_create_model(
        self,
        name: str,
        modelfile: Optional[str] = None,
        path: Optional[str] = None,
        quantize: Optional[str] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """Create a new Ollama model from Modelfile"""
        params = {"name": name, "stream": stream}
        if modelfile:
            params["modelfile"] = modelfile
        if path:
            params["path"] = path
        if quantize:
            params["quantize"] = quantize
        
        response = self._request(
            "POST",
            "/api/v1/ollama/models/create",
            params=params,
            stream=stream
        )
        
        if stream:
            return self._stream_response(response)
        else:
            return response.json()
    
    def ollama_copy_model(self, source: str, destination: str) -> Dict[str, Any]:
        """Copy an Ollama model"""
        response = self._request(
            "POST",
            "/api/v1/ollama/models/copy",
            params={"source": source, "destination": destination}
        )
        return response.json()
    
    def ollama_delete_model(self, model: str) -> Dict[str, Any]:
        """Delete an Ollama model"""
        response = self._request(
            "DELETE",
            f"/api/v1/ollama/models/{model}"
        )
        return response.json()
    
    def ollama_pull_model(
        self,
        model: str,
        insecure: bool = False,
        stream: bool = False
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """Pull an Ollama model from registry"""
        response = self._request(
            "POST",
            "/api/v1/ollama/models/pull",
            params={"model": model, "insecure": insecure, "stream": stream},
            stream=stream
        )
        
        if stream:
            return self._stream_response(response)
        else:
            return response.json()
    
    def ollama_push_model(
        self,
        model: str,
        insecure: bool = False,
        stream: bool = False
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """Push an Ollama model to registry"""
        response = self._request(
            "POST",
            "/api/v1/ollama/models/push",
            params={"model": model, "insecure": insecure, "stream": stream},
            stream=stream
        )
        
        if stream:
            return self._stream_response(response)
        else:
            return response.json()
    
    def ollama_load_model(self, model: str) -> Dict[str, Any]:
        """Load an Ollama model into memory"""
        response = self._request(
            "POST",
            "/api/v1/ollama/models/load",
            params={"model": model}
        )
        return response.json()
    
    def ollama_unload_model(self, model: str) -> Dict[str, Any]:
        """Unload an Ollama model from memory"""
        response = self._request(
            "POST",
            "/api/v1/ollama/models/unload",
            params={"model": model}
        )
        return response.json()


class UnifiedAIClientError(Exception):
    """Custom exception for client errors"""
    pass


# ====================
# CONVENIENCE FUNCTIONS
# ====================

def quick_generate(
    prompt: str,
    provider: Optional[str] = None,
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None
) -> str:
    """
    Quick one-shot generation
    
    Args:
        prompt: Input prompt
        provider: AI provider to use
        base_url: API base URL
        api_key: Optional API key
    
    Returns:
        Generated text content
    
    Example:
        text = quick_generate("What is AI?")
        print(text)
    """
    with UnifiedAIClient(base_url=base_url, api_key=api_key) as client:
        response = client.generate(prompt, provider=provider)
        return response['content']


def quick_chat(
    message: str,
    provider: Optional[str] = None,
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None
) -> str:
    """
    Quick one-shot chat
    
    Args:
        message: User message
        provider: AI provider to use
        base_url: API base URL
        api_key: Optional API key
    
    Returns:
        Assistant's response content
    
    Example:
        response = quick_chat("Hello!")
        print(response)
    """
    with UnifiedAIClient(base_url=base_url, api_key=api_key) as client:
        messages = [{"role": "user", "content": message}]
        response = client.chat(messages, provider=provider)
        return response['content']


# ====================
# EXAMPLE USAGE
# ====================

if __name__ == "__main__":
    # Initialize client
    client = UnifiedAIClient(
        base_url="http://localhost:8000",
        api_key=None  # Add your API key if needed
    )
    
    try:
        # Check health
        print("=== Health Check ===")
        health = client.health()
        print(f"Status: {health['status']}")
        print(f"Providers: {health['providers']}")
        
        # List providers
        print("\n=== Available Providers ===")
        providers = client.list_providers()
        for provider in providers:
            status = "✓" if provider['available'] else "✗"
            print(f"{status} {provider['name']} - {provider['default_model']}")
        
        # Simple generation
        print("\n=== Simple Generation ===")
        response = client.generate(
            prompt="What is the capital of France?",
            temperature=0.7
        )
        print(f"Provider: {response['provider']}")
        print(f"Model: {response['model']}")
        print(f"Response: {response['content']}")
        
        # Chat
        print("\n=== Chat Example ===")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a short joke"}
        ]
        response = client.chat(messages, temperature=0.8)
        print(f"Response: {response['content']}")
        
        # Streaming
        print("\n=== Streaming Example ===")
        print("Response: ", end="", flush=True)
        for chunk in client.generate(
            prompt="Count from 1 to 5",
            stream=True
        ):
            print(chunk, end="", flush=True)
        print()
        
        # List models
        print("\n=== Available Models ===")
        models = client.list_models()
        print(f"Total models: {models['total']}")
        for model in models['models'][:5]:  # Show first 5
            print(f"  - {model['name']} ({model['provider']})")
        
        # Embeddings
        print("\n=== Embeddings ===")
        embeddings = client.generate_embeddings("Hello world")
        print(f"Provider: {embeddings['provider']}")
        print(f"Dimensions: {embeddings['dimensions']}")
        print(f"First 5 values: {embeddings['embeddings'][0][:5]}")
        
        # Using convenience function
        print("\n=== Quick Generate ===")
        text = quick_generate("What is 2+2?")
        print(f"Response: {text}")
        
        # Ollama-specific features
        print("\n=== Ollama-Specific Features ===")
        try:
            # List running models
            running = client.ollama_list_running()
            print(f"Running models: {running}")
            
            # Show model details
            models_list = client.list_models(provider="ollama")
            if models_list['total'] > 0:
                first_model = models_list['models'][0]['name']
                details = client.ollama_show_model(first_model)
                print(f"Model details for {first_model}: {details.get('modelfile', 'N/A')[:100]}...")
            
            # Load/unload model
            # model_to_test = "llama3.2"
            # print(f"Loading {model_to_test}...")
            # client.ollama_load_model(model_to_test)
            # print("Model loaded")
            
        except Exception as e:
            print(f"Ollama features error: {e}")

        print("\n=== All Examples Completed Successfully ===")
        
    except UnifiedAIClientError as e:
        print(f"Client Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
    finally:
        client.close()