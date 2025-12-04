#!/usr/bin/env python3
"""
Comprehensive Testing Suite for Unified AI API

Tests for both the API endpoints and the client SDK.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

import sys
from pathlib import Path
root = Path(__file__).parent.parent

sys.path.append(f"{root}/app")

# Import the components to test
from sdk.client import UnifiedAIClient, UnifiedAIClientError, quick_generate, quick_chat   # type: ignore
from utils.unified_ai_interface import (                                        # type:ignore
    AIManager, Message, MessageRole, GenerationConfig,
    OllamaProvider, ModelInfo, GenerationResponse
)


# ====================
# FIXTURES
# ====================

@pytest.fixture
def mock_ollama_provider():
    """Mock Ollama provider"""
    provider = Mock(spec=OllamaProvider)
    provider.provider_name = "ollama"
    provider.is_available.return_value = True
    provider.list_models.return_value = [
        ModelInfo(
            name="llama3.2",
            provider="ollama",
            supports_streaming=True,
            supports_images=True
        )
    ]
    provider.generate.return_value = GenerationResponse(
        content="Test response",
        model="llama3.2",
        provider="ollama"
    )
    provider.chat.return_value = GenerationResponse(
        content="Chat response",
        model="llama3.2",
        provider="ollama"
    )
    provider.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
    return provider


@pytest.fixture
def ai_manager(mock_ollama_provider):
    """AI Manager with mock provider"""
    manager = AIManager()
    manager.register_provider(mock_ollama_provider, set_as_default=True)
    return manager


@pytest.fixture
def test_client():
    """Test client instance"""
    return UnifiedAIClient(
        base_url="http://localhost:8000",
        api_key="test-key"
    )


@pytest.fixture
def temp_image():
    """Create a temporary test image"""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        f.write(b'fake image data')
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


# ====================
# AI MANAGER TESTS
# ====================

class TestAIManager:
    """Test suite for AIManager"""
    
    def test_register_provider(self, mock_ollama_provider):
        """Test provider registration"""
        manager = AIManager()
        manager.register_provider(mock_ollama_provider, set_as_default=True)
        
        assert "ollama" in manager.providers
        assert manager.default_provider == "ollama"
    
    def test_get_provider(self, ai_manager):
        """Test getting a provider"""
        provider = ai_manager.get_provider("ollama")
        assert provider.provider_name == "ollama"
    
    def test_get_default_provider(self, ai_manager):
        """Test getting default provider"""
        provider = ai_manager.get_provider()
        assert provider.provider_name == "ollama"
    
    def test_get_nonexistent_provider(self, ai_manager):
        """Test error on nonexistent provider"""
        with pytest.raises(ValueError, match="not registered"):
            ai_manager.get_provider("nonexistent")
    
    def test_generate(self, ai_manager):
        """Test generation through manager"""
        response = ai_manager.generate(
            prompt="Test prompt",
            stream=False
        )
        
        assert response.content == "Test response"
        assert response.provider == "ollama"
    
    def test_chat(self, ai_manager):
        """Test chat through manager"""
        messages = [
            Message(role=MessageRole.USER, content="Hello")
        ]
        
        response = ai_manager.chat(messages=messages, stream=False)
        
        assert response.content == "Chat response"
        assert response.provider == "ollama"
    
    def test_list_models(self, ai_manager):
        """Test listing models"""
        models = ai_manager.list_models()
        
        assert len(models) > 0
        assert models[0].name == "llama3.2"
    
    def test_list_all_models(self, ai_manager):
        """Test listing all models from all providers"""
        all_models = ai_manager.list_all_models()
        
        assert "ollama" in all_models
        assert len(all_models["ollama"]) > 0
    
    def test_generate_embeddings(self, ai_manager):
        """Test embedding generation"""
        embeddings = ai_manager.generate_embeddings("Test text")
        
        assert len(embeddings) > 0
        assert len(embeddings[0]) == 3  # Mock returns 3D vector
    
    def test_get_available_providers(self, ai_manager):
        """Test getting available providers"""
        available = ai_manager.get_available_providers()
        
        assert "ollama" in available


# ====================
# MESSAGE TESTS
# ====================

class TestMessage:
    """Test suite for Message class"""
    
    def test_message_creation(self):
        """Test creating a message"""
        msg = Message(
            role=MessageRole.USER,
            content="Test content"
        )
        
        assert msg.role == MessageRole.USER
        assert msg.content == "Test content"
        assert msg.images is None
    
    def test_message_with_images(self):
        """Test message with images"""
        msg = Message(
            role=MessageRole.USER,
            content="Test content",
            images=["base64image"]
        )
        
        assert len(msg.images) == 1
    
    def test_message_to_dict(self):
        """Test converting message to dict"""
        msg = Message(
            role=MessageRole.USER,
            content="Test"
        )
        
        msg_dict = msg.to_dict()
        
        assert msg_dict["role"] == "user"
        assert msg_dict["content"] == "Test"


# ====================
# GENERATION CONFIG TESTS
# ====================

class TestGenerationConfig:
    """Test suite for GenerationConfig"""
    
    def test_config_creation(self):
        """Test creating config"""
        config = GenerationConfig(
            temperature=0.7,
            max_tokens=100
        )
        
        assert config.temperature == 0.7
        assert config.max_tokens == 100
    
    def test_config_to_dict(self):
        """Test converting config to dict"""
        config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["temperature"] == 0.7
        assert config_dict["top_p"] == 0.9
        assert config_dict["max_tokens"] == 100
        assert "seed" not in config_dict  # None values excluded


# ====================
# CLIENT TESTS
# ====================

class TestUnifiedAIClient:
    """Test suite for UnifiedAIClient"""
    
    def test_client_initialization(self):
        """Test client initialization"""
        client = UnifiedAIClient(
            base_url="http://localhost:8000",
            api_key="test-key"
        )
        
        assert client.config.base_url == "http://localhost:8000"
        assert client.config.api_key == "test-key"
        assert "Authorization" in client.session.headers
    
    def test_client_without_api_key(self):
        """Test client without API key"""
        client = UnifiedAIClient()
        
        assert "Authorization" not in client.session.headers
    
    @patch('requests.Session.request')
    def test_generate(self, mock_request):
        """Test generation via client"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": "Test response",
            "model": "llama3.2",
            "provider": "ollama"
        }
        mock_request.return_value = mock_response
        
        client = UnifiedAIClient()
        response = client.generate("Test prompt")
        
        assert response["content"] == "Test response"
        assert response["provider"] == "ollama"
    
    @patch('requests.Session.request')
    def test_chat(self, mock_request):
        """Test chat via client"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": "Chat response",
            "model": "llama3.2",
            "provider": "ollama"
        }
        mock_request.return_value = mock_response
        
        client = UnifiedAIClient()
        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat(messages)
        
        assert response["content"] == "Chat response"
    
    @patch('requests.Session.request')
    def test_list_models(self, mock_request):
        """Test listing models via client"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "llama3.2",
                    "provider": "ollama"
                }
            ],
            "total": 1
        }
        mock_request.return_value = mock_response
        
        client = UnifiedAIClient()
        models = client.list_models()
        
        assert models["total"] == 1
        assert models["models"][0]["name"] == "llama3.2"
    
    @patch('requests.Session.request')
    def test_list_providers(self, mock_request):
        """Test listing providers via client"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "name": "ollama",
                "enabled": True,
                "available": True
            }
        ]
        mock_request.return_value = mock_response
        
        client = UnifiedAIClient()
        providers = client.list_providers()
        
        assert len(providers) == 1
        assert providers[0]["name"] == "ollama"
    
    @patch('requests.Session.request')
    def test_generate_embeddings(self, mock_request):
        """Test embedding generation via client"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "provider": "ollama",
            "dimensions": 3
        }
        mock_request.return_value = mock_response
        
        client = UnifiedAIClient()
        result = client.generate_embeddings("Test text")
        
        assert result["dimensions"] == 3
        assert len(result["embeddings"]) == 1
    
    @patch('requests.Session.request')
    def test_health_check(self, mock_request):
        """Test health check via client"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "providers": {"ollama": True}
        }
        mock_request.return_value = mock_response
        
        client = UnifiedAIClient()
        health = client.health()
        
        assert health["status"] == "healthy"
    
    @patch('requests.Session.request')
    def test_is_available(self, mock_request):
        """Test availability check"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_request.return_value = mock_response
        
        client = UnifiedAIClient()
        assert client.is_available() is True
    
    def test_context_manager(self):
        """Test using client as context manager"""
        with UnifiedAIClient() as client:
            assert client.session is not None
        
        # Session should be closed after context
        # Note: Can't easily test if session is closed
    
    @patch('requests.Session.request')
    def test_request_error_handling(self, mock_request):
        """Test error handling in requests"""
        mock_request.side_effect = Exception("Network error")
        
        client = UnifiedAIClient()
        
        with pytest.raises(UnifiedAIClientError):
            client.generate("Test")


# ====================
# CONVENIENCE FUNCTION TESTS
# ====================

class TestConvenienceFunctions:
    """Test suite for convenience functions"""
    
    @patch('sdk.client.UnifiedAIClient.generate')
    def test_quick_generate(self, mock_generate):
        """Test quick_generate function"""
        mock_generate.return_value = {"content": "Quick response"}
        
        result = quick_generate("Test prompt")
        
        assert result == "Quick response"
    
    @patch('sdk.client.UnifiedAIClient.chat')
    def test_quick_chat(self, mock_chat):
        """Test quick_chat function"""
        mock_chat.return_value = {"content": "Quick chat response"}
        
        result = quick_chat("Hello")
        
        assert result == "Quick chat response"


# ====================
# INTEGRATION TESTS
# ====================

class TestIntegration:
    """Integration tests (require running API)"""
    
    @pytest.mark.integration
    def test_full_generation_flow(self):
        """Test complete generation flow"""
        client = UnifiedAIClient()
        
        if not client.is_available():
            pytest.skip("API not available")
        
        response = client.generate(
            prompt="Say 'test' and nothing else",
            temperature=0.1,
            max_tokens=10
        )
        
        assert "content" in response
        assert "provider" in response
        assert "model" in response
    
    @pytest.mark.integration
    def test_full_chat_flow(self):
        """Test complete chat flow"""
        client = UnifiedAIClient()
        
        if not client.is_available():
            pytest.skip("API not available")
        
        messages = [
            {"role": "user", "content": "Say 'hello'"}
        ]
        
        response = client.chat(messages, temperature=0.1)
        
        assert "content" in response
    
    @pytest.mark.integration
    def test_streaming(self):
        """Test streaming response"""
        client = UnifiedAIClient()
        
        if not client.is_available():
            pytest.skip("API not available")
        
        chunks = []
        for chunk in client.generate("Count to 3", stream=True):
            chunks.append(chunk)
        
        assert len(chunks) > 0


# ====================
# PERFORMANCE TESTS
# ====================

class TestPerformance:
    """Performance and load tests"""
    
    @pytest.mark.performance
    def test_response_time(self, ai_manager):
        """Test response time is reasonable"""
        start = time.time()
        
        ai_manager.generate("Test prompt", stream=False)
        
        duration = time.time() - start
        
        assert duration < 5.0  # Should respond in under 5 seconds
    
    @pytest.mark.performance
    def test_concurrent_requests(self, ai_manager):
        """Test handling multiple requests"""
        import concurrent.futures
        
        def make_request():
            return ai_manager.generate("Test", stream=False)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in futures]
        
        assert len(results) == 5
        assert all(r.content == "Test response" for r in results)


# ====================
# RUN TESTS
# ====================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])