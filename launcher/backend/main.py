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

# Import routers
from routers import ann, cnn, nlp, rnn

# Include routers with prefixes
app.include_router(ann.router, prefix="/api/ann", tags=["ANN - NBA Team Selection"])
app.include_router(cnn.router, prefix="/api/cnn", tags=["CNN - Fruit Classification"])
app.include_router(nlp.router, prefix="/api/nlp", tags=["NLP - Sentiment Analysis"])
app.include_router(rnn.router, prefix="/api/rnn", tags=["RNN - Text Generation"])


# Root endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "CST-435 Machine Learning Projects API Gateway",
        "version": "1.0.0",
        "projects": {
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
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check for all services"""
    from routers.ann import ann_model
    from routers.cnn import cnn_model
    from routers.nlp import nlp_analyzer
    from routers.rnn import rnn_generator

    return {
        "status": "healthy",
        "services": {
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
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
