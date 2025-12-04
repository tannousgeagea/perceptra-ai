#!/usr/bin/env python3
"""
Unified Multi-Provider AI API Service

A comprehensive REST API that supports multiple AI providers (Ollama, OpenAI, Gemini, 
Hugging Face, etc.) through a single unified interface.

Features:
- Support for multiple AI providers with automatic routing
- Unified request/response format across all providers
- File upload support for multimodal models
- Streaming and non-streaming responses
- Provider auto-detection and fallback
- Configuration management
- API documentation with Swagger/OpenAPI
- Authentication support
- CORS configuration
- Health checks
"""

import json
import base64
import tempfile
import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Union, Callable

from fastapi import (
    FastAPI, HTTPException, UploadFile, File, 
    Depends, status, Request, Response, APIRouter
)
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field

# Import our unified AI interface
from utils.unified_ai_interface import (
    AIManager, 
    OllamaProvider, 
    OpenAIProvider, 
    GeminiProvider, 
    HuggingFaceProvider,
    Message, 
    MessageRole, 
    GenerationConfig,
    GenerationResponse,
    ModelInfo
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Global AI Manager instance
ai_manager: Optional[AIManager] = None

# Configuration from environment
API_KEY = os.getenv("API_KEY")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB default

# Provider configurations
PROVIDER_CONFIGS = {
    "ollama": {
        "enabled": os.getenv("OLLAMA_ENABLED", "true").lower() == "true",
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "default_model": os.getenv("OLLAMA_DEFAULT_MODEL", "gemma3:12b-it-qat")
    },
    "openai": {
        "enabled": os.getenv("OPENAI_ENABLED", "false").lower() == "true",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "default_model": os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4")
    },
    "gemini": {
        "enabled": os.getenv("GEMINI_ENABLED", "false").lower() == "true",
        "api_key": os.getenv("GEMINI_API_KEY"),
        "default_model": os.getenv("GEMINI_DEFAULT_MODEL", "gemini-pro")
    },
    "huggingface": {
        "enabled": os.getenv("HUGGINGFACE_ENABLED", "false").lower() == "true",
        "api_key": os.getenv("HUGGINGFACE_API_KEY"),
        "default_model": os.getenv("HUGGINGFACE_DEFAULT_MODEL", "meta-llama/Llama-2-7b-chat-hf")
    }
}


# ===================
# TIMED ROUTE
# ===================

class TimedRoute(APIRoute):
    """Custom route that adds response time header"""
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        
        async def custom_route_handler(request: Request) -> Response:
            before = time.time()
            response: Response = await original_route_handler(request)
            duration = time.time() - before
            response.headers["X-Response-Time"] = str(duration)
            logger.debug(f"Route {request.url.path} took {duration:.3f}s")
            return response
        
        return custom_route_handler




router = APIRouter(
    route_class=TimedRoute,
    responses={404: {"description": "Not found"}},
)

# ===================
# PYDANTIC MODELS
# ===================

class GenerationConfigRequest(BaseModel):
    """Unified generation configuration"""
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=0)
    max_tokens: Optional[int] = Field(None, ge=1)
    stop_sequences: Optional[List[str]] = None
    seed: Optional[int] = None
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    
    def to_generation_config(self) -> GenerationConfig:
        """Convert to GenerationConfig"""
        return GenerationConfig(**self.dict(exclude_none=True))


class MessageRequest(BaseModel):
    """Unified message format"""
    role: str = Field(..., pattern="^(system|user|assistant|tool)$")
    content: str
    images: Optional[List[str]] = Field(None, description="Base64 encoded images")
    
    def to_message(self) -> Message:
        """Convert to Message"""
        role_map = {
            "system": MessageRole.SYSTEM,
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "tool": MessageRole.TOOL
        }
        return Message(
            role=role_map[self.role],
            content=self.content,
            images=self.images
        )


class GenerateRequest(BaseModel):
    """Request for text generation"""
    prompt: str = Field(..., description="Input prompt")
    provider: Optional[str] = Field(None, description="AI provider to use (ollama, openai, gemini, huggingface)")
    model: Optional[str] = Field(None, description="Specific model to use")
    config: Optional[GenerationConfigRequest] = Field(None, description="Generation configuration")
    stream: bool = Field(False, description="Stream response")
    format: Optional[str] = Field(None, pattern="^json$", description="Response format")
    images: Optional[List[str]] = Field(None, description="Base64 encoded images for multimodal models")
    raw: bool = Field(False, description="Raw mode")
    keep_alive: Optional[str] = Field("5m", description="Keep model loaded duration")


class ChatRequest(BaseModel):
    """Request for chat completion"""
    messages: List[MessageRequest] = Field(..., description="Chat messages")
    provider: Optional[str] = Field(None, description="AI provider to use")
    model: Optional[str] = Field(None, description="Specific model to use")
    config: Optional[GenerationConfigRequest] = Field(None, description="Generation configuration")
    stream: bool = Field(False, description="Stream response")
    format: Optional[str] = Field(None, pattern="^json$", description="Response format")
    keep_alive: Optional[str] = Field("5m", description="Keep model loaded duration")


class EmbeddingRequest(BaseModel):
    """Request for embeddings"""
    input: Union[str, List[str]] = Field(..., description="Text to embed")
    provider: Optional[str] = Field(None, description="AI provider to use")
    model: Optional[str] = Field(None, description="Specific embedding model")


class ProviderConfigResponse(BaseModel):
    """Provider configuration response"""
    name: str
    enabled: bool
    available: bool
    default_model: str
    models_count: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str = "2.0.0"
    providers: Dict[str, bool]


class GenerationResponseModel(BaseModel):
    """Unified generation response"""
    content: str
    model: str
    provider: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class EmbeddingResponse(BaseModel):
    """Embedding response"""
    embeddings: List[List[float]]
    model: str
    provider: str
    dimensions: int


class ModelListResponse(BaseModel):
    """Model list response"""
    models: List[Dict[str, Any]]
    total: int
    provider: Optional[str] = None


# ===================
# AUTHENTICATION
# ===================

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token if authentication is enabled"""
    if API_KEY is None:
        return True
    
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    
    return True


# ===================
# STARTUP/SHUTDOWN
# ===================

@router.on_event("startup")
async def startup_event():
    """Initialize AI Manager and providers on startup"""
    global ai_manager
    
    try:
        ai_manager = AIManager()
        logger.info("Initializing AI Manager...")
        
        # Initialize Ollama provider
        if PROVIDER_CONFIGS["ollama"]["enabled"]:
            try:
                ollama = OllamaProvider(
                    base_url=PROVIDER_CONFIGS["ollama"]["base_url"],
                    default_model=PROVIDER_CONFIGS["ollama"]["default_model"]
                )
                ai_manager.register_provider(ollama, set_as_default=True)
                logger.info(f"✓ Ollama provider initialized (available: {ollama.is_available()})")
            except Exception as e:
                logger.error(f"✗ Failed to initialize Ollama provider: {e}")
        
        # Initialize OpenAI provider
        if PROVIDER_CONFIGS["openai"]["enabled"] and PROVIDER_CONFIGS["openai"]["api_key"]:
            try:
                openai = OpenAIProvider(
                    api_key=PROVIDER_CONFIGS["openai"]["api_key"],
                    default_model=PROVIDER_CONFIGS["openai"]["default_model"]
                )
                ai_manager.register_provider(openai)
                logger.info(f"✓ OpenAI provider initialized (available: {openai.is_available()})")
            except Exception as e:
                logger.error(f"✗ Failed to initialize OpenAI provider: {e}")
        
        # Initialize Gemini provider
        if PROVIDER_CONFIGS["gemini"]["enabled"] and PROVIDER_CONFIGS["gemini"]["api_key"]:
            try:
                gemini = GeminiProvider(
                    api_key=PROVIDER_CONFIGS["gemini"]["api_key"],
                    default_model=PROVIDER_CONFIGS["gemini"]["default_model"]
                )
                ai_manager.register_provider(gemini)
                logger.info(f"✓ Gemini provider initialized (available: {gemini.is_available()})")
            except Exception as e:
                logger.error(f"✗ Failed to initialize Gemini provider: {e}")
        
        # Initialize Hugging Face provider
        if PROVIDER_CONFIGS["huggingface"]["enabled"] and PROVIDER_CONFIGS["huggingface"]["api_key"]:
            try:
                hf = HuggingFaceProvider(
                    api_key=PROVIDER_CONFIGS["huggingface"]["api_key"],
                    default_model=PROVIDER_CONFIGS["huggingface"]["default_model"]
                )
                ai_manager.register_provider(hf)
                logger.info(f"✓ Hugging Face provider initialized (available: {hf.is_available()})")
            except Exception as e:
                logger.error(f"✗ Failed to initialize Hugging Face provider: {e}")
        
        available_providers = ai_manager.get_available_providers()
        logger.info(f"Available providers: {', '.join(available_providers)}")
        
        if not available_providers:
            logger.warning("⚠ No providers are available!")
        
    except Exception as e:
        logger.error(f"Failed to initialize AI Manager: {e}")
        raise


@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Unified AI API service")


# ===================
# UTILITY FUNCTIONS
# ===================

def handle_api_error(e: Exception):
    """Convert errors to HTTP exceptions"""
    logger.error(f"API Error: {type(e).__name__}: {str(e)}")
    
    if isinstance(e, HTTPException):
        raise e
    elif isinstance(e, ValueError):
        raise HTTPException(status_code=400, detail=str(e))
    else:
        raise HTTPException(status_code=500, detail="Internal server error")


async def stream_generator(generator):
    """Convert generator to async streaming format"""
    try:
        for chunk in generator:
            if isinstance(chunk, str):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            else:
                yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return base64 encoded content"""
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes"
        )
    
    content = file.file.read()
    
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes"
        )
    
    return base64.b64encode(content).decode('utf-8')


# ===================
# HEALTH & INFO ENDPOINTS
# ===================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    providers_status = {}
    for provider_name in ai_manager.providers.keys():
        try:
            provider = ai_manager.get_provider(provider_name)
            providers_status[provider_name] = provider.is_available()
        except:
            providers_status[provider_name] = False
    
    all_available = any(providers_status.values())
    
    return HealthResponse(
        status="healthy" if all_available else "degraded",
        timestamp=datetime.now(),
        providers=providers_status
    )


@router.get("/info")
async def get_api_info():
    """Get API information"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    available_providers = ai_manager.get_available_providers()
    
    return {
        "name": "Unified AI API",
        "version": "2.0.0",
        "authentication_enabled": API_KEY is not None,
        "max_file_size": MAX_FILE_SIZE,
        "providers": {
            "registered": list(ai_manager.providers.keys()),
            "available": available_providers,
            "default": ai_manager.default_provider
        },
        "endpoints": {
            "generation": ["/generate", "/chat"],
            "models": ["/models", "/models/{provider}"],
            "embeddings": ["/embeddings"],
            "providers": ["/providers"],
            "ollama_specific": [
                "/api/ollama/models/running",
                "/api/ollama/models/show",
                "/api/ollama/models/create",
                "/api/ollama/models/copy",
                "/api/ollama/models/pull",
                "/api/ollama/models/push",
                "/api/ollama/models/load",
                "/api/ollama/models/unload",
                "/api/ollama/models/{model_name} [DELETE]"
            ],
            "health": ["/health", "/info"]
        }
    }


@router.get("/providers", response_model=List[ProviderConfigResponse])
async def list_providers(_: bool = Depends(verify_token)):
    """List all configured providers and their status"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    providers_info = []
    
    for provider_name, config in PROVIDER_CONFIGS.items():
        if not config["enabled"]:
            continue
        
        try:
            provider = ai_manager.get_provider(provider_name)
            available = provider.is_available()
            
            # Try to get model count
            models_count = None
            if available:
                try:
                    models = provider.list_models()
                    models_count = len(models)
                except:
                    pass
            
            providers_info.append(
                ProviderConfigResponse(
                    name=provider_name,
                    enabled=True,
                    available=available,
                    default_model=config.get("default_model", ""),
                    models_count=models_count
                )
            )
        except:
            providers_info.append(
                ProviderConfigResponse(
                    name=provider_name,
                    enabled=True,
                    available=False,
                    default_model=config.get("default_model", "")
                )
            )
    
    return providers_info


# ===================
# GENERATION ENDPOINTS
# ===================

@router.post("/generate", response_model=GenerationResponseModel)
async def generate_text(
    request: GenerateRequest,
    _: bool = Depends(verify_token)
):
    """Generate text completion using any provider"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    try:
        config = request.config.to_generation_config() if request.config else None
        
        # Add images to kwargs if provided
        kwargs = {}
        if request.images:
            kwargs['images'] = request.images
        if request.model:
            kwargs['model'] = request.model
        if request.format:
            kwargs['format'] = request.format
        if request.keep_alive:
            kwargs['keep_alive'] = request.keep_alive
        if request.raw:
            kwargs['raw'] = request.raw
        
        result = ai_manager.generate(
            prompt=request.prompt,
            provider=request.provider,
            config=config,
            stream=request.stream,
            **kwargs
        )
        
        if request.stream:
            return StreamingResponse(
                stream_generator(result),
                media_type="text/event-stream"
            )
        else:
            return GenerationResponseModel(         # type:ignore
                content=result.content,             # type:ignore
                model=result.model,                 # type:ignore
                provider=result.provider,            # type:ignore
                finish_reason=result.finish_reason,   # type:ignore
                usage=result.usage,                    # type:ignore
                metadata=result.metadata               # type:ignore
            )
    
    except Exception as e:
        handle_api_error(e)


@router.post("/generate/upload")
async def generate_with_upload(
    prompt: str,
    files: List[UploadFile] = File(...),
    provider: Optional[str] = None,
    model: Optional[str] = None,
    stream: bool = False,
    _: bool = Depends(verify_token)
):
    """Generate text with uploaded image files"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    try:
        # Process uploaded images
        images = []
        for file in files:
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not an image"
                )
            encoded_image = save_uploaded_file(file)
            images.append(encoded_image)
        
        kwargs = {'images': images}
        if model:
            kwargs['model'] = model   # type:ignore
        
        result = ai_manager.generate(
            prompt=prompt,
            provider=provider,
            stream=stream,
            **kwargs               # type: ignore
        )
        
        if stream:
            return StreamingResponse(
                stream_generator(result),
                media_type="text/event-stream"
            )
        else:
            return GenerationResponseModel(         # type:ignore
                content=result.content,             # type:ignore
                model=result.model,                 # type:ignore
                provider=result.provider,            # type:ignore
                finish_reason=result.finish_reason,   # type:ignore
                usage=result.usage,                    # type:ignore
                metadata=result.metadata               # type:ignore
            )
    
    except Exception as e:
        handle_api_error(e)


@router.post("/chat", response_model=GenerationResponseModel)
async def chat_completion(
    request: ChatRequest,
    _: bool = Depends(verify_token)
):
    """Generate chat completion using any provider"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    try:
        messages = [msg.to_message() for msg in request.messages]
        config = request.config.to_generation_config() if request.config else None
        
        kwargs = {}
        if request.model:
            kwargs['model'] = request.model
        
        result = ai_manager.chat(
            messages=messages,
            provider=request.provider,
            config=config,
            stream=request.stream,
            **kwargs
        )
        
        if request.stream:
            return StreamingResponse(
                stream_generator(result),
                media_type="text/event-stream"
            )
        else:
            return GenerationResponseModel(         # type:ignore
                content=result.content,             # type:ignore
                model=result.model,                 # type:ignore
                provider=result.provider,            # type:ignore
                finish_reason=result.finish_reason,   # type:ignore
                usage=result.usage,                    # type:ignore
                metadata=result.metadata               # type:ignore
            )
    
    except Exception as e:
        handle_api_error(e)


# ===================
# MODEL MANAGEMENT
# ===================

@router.get("/models", response_model=ModelListResponse)
async def list_all_models(_: bool = Depends(verify_token)):
    """List models from all providers"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    try:
        all_models = ai_manager.list_all_models()
        
        models_list = []
        for provider_name, models in all_models.items():
            for model in models:
                models_list.append({
                    "name": model.name,
                    "provider": provider_name,
                    "context_length": model.context_length,
                    "supports_streaming": model.supports_streaming,
                    "supports_images": model.supports_images,
                    "supports_tools": model.supports_tools
                })
        
        return ModelListResponse(
            models=models_list,
            total=len(models_list)
        )
    
    except Exception as e:
        handle_api_error(e)


@router.get("/models/{provider}", response_model=ModelListResponse)
async def list_provider_models(
    provider: str,
    _: bool = Depends(verify_token)
):
    """List models from a specific provider"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    try:
        models = ai_manager.list_models(provider=provider)
        
        models_list = []
        for model in models:
            models_list.append({
                "name": model.name,
                "provider": model.provider,
                "context_length": model.context_length,
                "supports_streaming": model.supports_streaming,
                "supports_images": model.supports_images,
                "supports_tools": model.supports_tools,
                "metadata": model.metadata
            })
        
        return ModelListResponse(
            models=models_list,
            total=len(models_list),
            provider=provider
        )
    
    except Exception as e:
        handle_api_error(e)


# ===================
# EMBEDDING ENDPOINTS
# ===================

@router.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(
    request: EmbeddingRequest,
    _: bool = Depends(verify_token)
):
    """Generate embeddings using any provider"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    try:
        kwargs = {}
        if request.model:
            kwargs['model'] = request.model
        
        embeddings = ai_manager.generate_embeddings(
            text=request.input,
            provider=request.provider,
            **kwargs
        )
        
        # Determine provider and model used
        provider_obj = ai_manager.get_provider(request.provider)
        provider_name = provider_obj.provider_name
        model_name = request.model or "default"
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model_name,
            provider=provider_name,
            dimensions=len(embeddings[0]) if embeddings else 0
        )
    
    except Exception as e:
        handle_api_error(e)


# ===================
# OLLAMA-SPECIFIC ENDPOINTS
# ===================

@router.get("/ollama/models/running")
async def list_running_models(_: bool = Depends(verify_token)):
    """List currently running Ollama models"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    try:
        provider = ai_manager.get_provider("ollama")
        if not hasattr(provider, 'adapter'):
            raise HTTPException(status_code=400, detail="Not an Ollama provider")
        
        return provider.adapter.list_running_models()
    except Exception as e:
        handle_api_error(e)


@router.post("/ollama/models/show")
async def show_ollama_model(
    model: str,
    verbose: bool = False,
    _: bool = Depends(verify_token)
):
    """Show detailed Ollama model information"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    try:
        provider = ai_manager.get_provider("ollama")
        if not hasattr(provider, 'adapter'):
            raise HTTPException(status_code=400, detail="Not an Ollama provider")
        
        return provider.adapter.show_model(model, verbose)
    except Exception as e:
        handle_api_error(e)


@router.post("/ollama/models/create")
async def create_ollama_model(
    name: str,
    modelfile: Optional[str] = None,
    path: Optional[str] = None,
    quantize: Optional[str] = None,
    stream: bool = False,
    _: bool = Depends(verify_token)
):
    """Create a new Ollama model from Modelfile"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    try:
        provider = ai_manager.get_provider("ollama")
        if not hasattr(provider, 'adapter'):
            raise HTTPException(status_code=400, detail="Not an Ollama provider")
        
        result = provider.adapter.create_model(
            name=name,
            modelfile=modelfile,
            path=path,
            quantize=quantize,
            stream=stream
        )
        
        if stream:
            return StreamingResponse(
                stream_generator(result),
                media_type="text/event-stream"
            )
        else:
            return result
    except Exception as e:
        handle_api_error(e)


@router.post("/ollama/models/copy")
async def copy_ollama_model(
    source: str,
    destination: str,
    _: bool = Depends(verify_token)
):
    """Copy an Ollama model"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    try:
        provider = ai_manager.get_provider("ollama")
        if not hasattr(provider, 'adapter'):
            raise HTTPException(status_code=400, detail="Not an Ollama provider")
        
        provider.adapter.copy_model(source, destination)
        return {
            "status": "success",
            "message": f"Model {source} copied to {destination}"
        }
    except Exception as e:
        handle_api_error(e)


@router.delete("/ollama/models/{model_name}")
async def delete_ollama_model(
    model_name: str,
    _: bool = Depends(verify_token)
):
    """Delete an Ollama model"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    try:
        provider = ai_manager.get_provider("ollama")
        if not hasattr(provider, 'adapter'):
            raise HTTPException(status_code=400, detail="Not an Ollama provider")
        
        provider.adapter.delete_model(model_name)
        return {
            "status": "success",
            "message": f"Model {model_name} deleted"
        }
    except Exception as e:
        handle_api_error(e)


@router.post("/ollama/models/pull")
async def pull_ollama_model(
    model: str,
    insecure: bool = False,
    stream: bool = False,
    _: bool = Depends(verify_token)
):
    """Pull an Ollama model from registry"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    try:
        provider = ai_manager.get_provider("ollama")
        if not hasattr(provider, 'adapter'):
            raise HTTPException(status_code=400, detail="Not an Ollama provider")
        
        result = provider.adapter.pull_model(
            name=model,
            insecure=insecure,
            stream=stream
        )
        
        if stream:
            return StreamingResponse(
                stream_generator(result),
                media_type="text/event-stream"
            )
        else:
            return result
    except Exception as e:
        handle_api_error(e)


@router.post("/ollama/models/push")
async def push_ollama_model(
    model: str,
    insecure: bool = False,
    stream: bool = False,
    _: bool = Depends(verify_token)
):
    """Push an Ollama model to registry"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    try:
        provider = ai_manager.get_provider("ollama")
        if not hasattr(provider, 'adapter'):
            raise HTTPException(status_code=400, detail="Not an Ollama provider")
        
        result = provider.adapter.push_model(
            name=model,
            insecure=insecure,
            stream=stream
        )
        
        if stream:
            return StreamingResponse(
                stream_generator(result),
                media_type="text/event-stream"
            )
        else:
            return result
    except Exception as e:
        handle_api_error(e)


@router.post("/ollama/models/load")
async def load_ollama_model(
    model: str,
    _: bool = Depends(verify_token)
):
    """Load an Ollama model into memory"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    try:
        provider = ai_manager.get_provider("ollama")
        if not hasattr(provider, 'adapter'):
            raise HTTPException(status_code=400, detail="Not an Ollama provider")
        
        result = provider.adapter.load_model(model)
        return {
            "status": "success",
            "message": f"Model {model} loaded",
            "details": result
        }
    except Exception as e:
        handle_api_error(e)


@router.post("/ollama/models/unload")
async def unload_ollama_model(
    model: str,
    _: bool = Depends(verify_token)
):
    """Unload an Ollama model from memory"""
    if not ai_manager:
        raise HTTPException(status_code=503, detail="AI Manager not initialized")
    
    try:
        provider = ai_manager.get_provider("ollama")
        if not hasattr(provider, 'adapter'):
            raise HTTPException(status_code=400, detail="Not an Ollama provider")
        
        result = provider.adapter.unload_model(model)
        return {
            "status": "success",
            "message": f"Model {model} unloaded",
            "details": result
        }
    except Exception as e:
        handle_api_error(e)


# ===================
# ROOT ENDPOINT
# ===================

@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Unified AI API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "info": "/info"
    }


# ===================
# RUN SERVER
# ===================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Unified AI API on {host}:{port}")
    
    uvicorn.run(
        "unified_ai_api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )