import os
import uvicorn
import logging
import inspect
import importlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROUTERS_DIR = os.path.dirname(__file__) + "/routers"
ROUTERS = [
    f"api.routers.{f.replace('/', '.')}" 
    for f in os.listdir(ROUTERS_DIR)
    if not f.endswith('__pycache__')
    if not f.endswith('__.py')
    ]

def create_app() -> FastAPI:
    tags_meta = [
        {
            "name": "AI Context Agent",
            "description": "Context Agent"
        }
    ]

    app = FastAPI(
        openapi_tags = tags_meta,
        debug=True,
        description="Multi-provider AI API supporting Ollama, OpenAI, Gemini, Hugging Face, and more",
        docs_url="/docs",
        redoc_url="/redoc",
        summary="",
        version="0.0.1",
        contact={
            "name": "Tannous Geagea",
            "url": "https://wasteant.com",
            "email": "tannous.geagea@wasteant.com",            
        },
        openapi_url="/openapi.json"
    )

    origins = [f'http://localhost:{os.getenv("DATA_API_PORT")}', os.getenv("FRONTEND_ENDPOINT")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["X-Requested-With", "X-Request-ID", "Authorization"],
        expose_headers=["X-Request-ID", "X-Progress-ID", "x-response-time"],
    )

    for R in ROUTERS:
        try:
            module = importlib.import_module(R)
            attr = getattr(module, 'endpoint')
            if inspect.ismodule(attr):
                app.include_router(module.endpoint.router)
        except ImportError as err:
            logging.error(f'Failed to import {R}: {err}')
    
    return app

app = create_app()
