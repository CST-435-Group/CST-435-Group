"""
Unified API Gateway for All Projects
Routes requests to ANN, CNN, and NLP backends
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="CST-435 Projects API Gateway",
    description="Unified API for ANN, CNN, and NLP ML Projects",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://*.vercel.app",
        "*"  # In production, specify exact origins
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers - support running from both root and backend directory
try:
    from routers import ann, cnn, nlp, rnn, docs
except ImportError:
    from launcher.backend.routers import ann, cnn, nlp, rnn, docs

# Lazy import GAN router from GAN project
gan_router = None
GAN_AVAILABLE = False

def load_gan_router():
    """Lazy load GAN router from GAN project"""
    global gan_router, GAN_AVAILABLE
    if gan_router is not None:
        return gan_router

    import sys
    from pathlib import Path

    # Try to find GAN backend
    possible_paths = [
        Path(__file__).parent.parent.parent / "GAN" / "backend",  # launcher/backend -> GAN/backend
        Path(__file__).parent.parent / "GAN" / "backend",  # alternate path
    ]

    for path in possible_paths:
        if (path / "gan_router.py").exists():
            if str(path.parent) not in sys.path:
                sys.path.insert(0, str(path.parent))
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            try:
                from backend.gan_router import router as gan_r
                gan_router = gan_r
                GAN_AVAILABLE = True
                print(f"[OK] GAN router loaded from: {path}")
                return gan_router
            except ImportError:
                try:
                    from gan_router import router as gan_r
                    gan_router = gan_r
                    GAN_AVAILABLE = True
                    print(f"[OK] GAN router loaded from: {path}")
                    return gan_router
                except ImportError as e:
                    print(f"[WARN] Failed to import GAN router: {e}")

    print("[WARN] GAN router not found")
    return None

# Try to load GAN router at startup
load_gan_router()

# Include routers with prefixes
app.include_router(ann.router, prefix="/api/ann", tags=["ANN - NBA Team Selection"])
app.include_router(cnn.router, prefix="/api/cnn", tags=["CNN - Fruit Classification"])
app.include_router(nlp.router, prefix="/api/nlp", tags=["NLP - Sentiment Analysis"])
app.include_router(rnn.router, prefix="/api/rnn", tags=["RNN - Text Generation"])
app.include_router(docs.router, prefix="/api/docs", tags=["Project Docs & Cost Reports"])

# Include GAN router if available
if gan_router is not None:
    app.include_router(gan_router, prefix="/api/gan", tags=["GAN - Military Vehicle Generation"])


# Root endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    projects = {
        "ANN": {
            "name": "NBA Team Selection",
            "endpoint": "/api/ann",
            "description": "Neural network for optimal NBA team composition"
        },
        "CNN": {
            "name": "Fruit Image Classification",
            "endpoint": "/api/cnn",
            "description": "Convolutional neural network for fruit recognition"
        },
        "NLP": {
            "name": "Sentiment Analysis",
            "endpoint": "/api/nlp",
            "description": "7-point scale sentiment analyzer for reviews"
        },
        "RNN": {
            "name": "Text Generation",
            "endpoint": "/api/rnn",
            "description": "LSTM-based recurrent neural network for next-word prediction"
        }
    }

    # Add GAN if available
    if GAN_AVAILABLE:
        projects["GAN"] = {
            "name": "Military Vehicle Generation",
            "endpoint": "/api/gan",
            "description": "Dual conditional GAN for synthetic military vehicle image generation"
        }

    return {
        "message": "CST-435 Machine Learning Projects API Gateway",
        "version": "1.0.0",
        "projects": projects,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check for all services"""
    from routers.ann import ann_model
    from routers.cnn import cnn_model
    from routers.nlp import nlp_analyzer
    from routers.rnn import rnn_generator

    services = {
        "ann": {
            "status": "ready" if ann_model is not None else "not_loaded",
            "model_loaded": ann_model is not None
        },
        "cnn": {
            "status": "ready" if cnn_model is not None else "not_loaded",
            "model_loaded": cnn_model is not None
        },
        "nlp": {
            "status": "ready" if nlp_analyzer is not None else "not_loaded",
            "model_loaded": nlp_analyzer is not None
        },
        "rnn": {
            "status": "ready" if rnn_generator is not None else "not_loaded",
            "model_loaded": rnn_generator is not None
        }
    }

    # Add GAN health if available
    if GAN_AVAILABLE and gan_router is not None:
        try:
            from backend.gan_router import gan_generator
        except ImportError:
            try:
                from gan_router import gan_generator
            except ImportError:
                gan_generator = None
        services["gan"] = {
            "status": "ready" if gan_generator is not None else "not_loaded",
            "model_loaded": gan_generator is not None,
            "available": True
        }
    else:
        services["gan"] = {
            "status": "unavailable",
            "model_loaded": False,
            "available": False
        }

    return {
        "status": "healthy",
        "services": services
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
