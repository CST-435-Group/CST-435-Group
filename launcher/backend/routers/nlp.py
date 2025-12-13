"""
FastAPI Router for NLP Project (Sentiment Analysis)
Wraps existing NLP backend functionality
"""

# Force transformers to use PyTorch only (disable TensorFlow) - must be set before any imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['USE_TF'] = '0'  # Disable TensorFlow in transformers
os.environ['USE_TORCH'] = '1'  # Enable PyTorch in transformers

# Import the mock torchvision module BEFORE anything else to prevent import errors
try:
    from . import torchvision_mock
except ImportError:
    pass

import sys
import warnings
warnings.filterwarnings('ignore')

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from pathlib import Path

# Add NLP project to path
nlp_project_path = Path(__file__).parent.parent.parent.parent / "NLP" / "nlp-react" / "backend"
sys.path.insert(0, str(nlp_project_path))

try:
    from sentiment_model import SentimentAnalyzer
except ImportError:
    print("Warning: Could not import NLP model. NLP endpoints will not work.")
    SentimentAnalyzer = None

# Create router
router = APIRouter()

# Global analyzer storage (lazy loading)
nlp_analyzer = None


# Request/Response Models
class TextInput(BaseModel):
    """Single text input for sentiment analysis"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")


class BatchTextInput(BaseModel):
    """Batch text input for sentiment analysis"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")


class SentimentResult(BaseModel):
    """Sentiment analysis result"""
    text: str
    sentiment_score: int
    sentiment_label: str
    emoji: str
    confidence: float
    probabilities: Dict[str, float]


def load_nlp_analyzer():
    """Lazy load NLP sentiment analyzer"""
    global nlp_analyzer

    if nlp_analyzer is not None:
        return nlp_analyzer

    try:
        if SentimentAnalyzer is None:
            raise ImportError("SentimentAnalyzer not available")

        analyzer = SentimentAnalyzer()
        nlp_analyzer = analyzer
        print("[OK] NLP sentiment analyzer loaded successfully!")
        return analyzer

    except Exception as e:
        print(f"[ERROR] Error loading NLP analyzer: {e}")
        return None


@router.get("/")
async def nlp_info():
    """Get NLP project information"""
    return {
        "project": "Multi-Scale Sentiment Analysis",
        "description": "3-point scale sentiment analyzer for hospital reviews and healthcare feedback",
        "endpoints": {
            "/health": "Check if model is loaded",
            "/analyze": "Analyze sentiment of single text",
            "/analyze/batch": "Analyze sentiment of multiple texts",
            "/examples": "Get example reviews by sentiment",
            "/sentiment-scale": "Get sentiment scale information"
        }
    }


@router.get("/health")
async def health_check():
    """Check if NLP analyzer is loaded"""
    return {
        "status": "ready" if nlp_analyzer is not None else "not_loaded",
        "model_loaded": nlp_analyzer is not None
    }


@router.post("/preload")
async def preload_model():
    """Preload the NLP model into memory"""
    global nlp_analyzer

    if nlp_analyzer is not None:
        return {
            "status": "already_loaded",
            "message": "NLP model is already loaded"
        }

    print("[LOADING] Preloading NLP model on user request...")
    analyzer = load_nlp_analyzer()

    if analyzer is None:
        raise HTTPException(status_code=500, detail="Failed to load NLP model")

    return {
        "status": "loaded",
        "message": "NLP model loaded successfully"
    }


@router.post("/unload")
async def unload_model():
    """Unload the NLP model from memory"""
    global nlp_analyzer

    if nlp_analyzer is None:
        return {
            "status": "not_loaded",
            "message": "NLP model was not loaded"
        }

    print("[UNLOADING] Unloading NLP model to free memory...")
    nlp_analyzer = None

    # Force garbage collection to free memory immediately
    import gc
    gc.collect()

    return {
        "status": "unloaded",
        "message": "NLP model unloaded successfully"
    }


@router.post("/analyze", response_model=SentimentResult)
async def analyze_sentiment(input_data: TextInput):
    """
    Analyze sentiment of a single text

    **Parameters:**
    - text: The text to analyze (hospital review, patient feedback, etc.)

    **Returns:**
    - sentiment_score: Integer from 1 to 3
    - sentiment_label: Descriptive label (e.g., "Positive")
    - emoji: Emoji representation
    - confidence: Model confidence (0-1)
    - probabilities: Probability distribution across all classes
    """
    analyzer = load_nlp_analyzer()

    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = analyzer.analyze(input_data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")


@router.post("/analyze/batch", response_model=List[SentimentResult])
async def analyze_batch(input_data: BatchTextInput):
    """
    Analyze sentiment of multiple texts in batch

    **Parameters:**
    - texts: List of texts to analyze (max 100)

    **Returns:**
    - List of sentiment analysis results
    """
    analyzer = load_nlp_analyzer()

    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = [analyzer.analyze(text) for text in input_data.texts]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing batch: {str(e)}")


@router.get("/examples", response_model=Dict[str, List[str]])
async def get_examples():
    """
    Get example reviews for each sentiment level

    **Returns:**
    - Dictionary mapping sentiment levels to example reviews
    """
    examples = {
        "1": [
            "Terrible experience. Very disappointed with the care I received.",
            "Hopeless hospital. They have no value for patients or their time.",
            "Very poor service. Long wait times and unprofessional staff.",
            "Not good at all. Would not recommend this hospital.",
        ],
        "2": [
            "It was okay. Nothing particularly special about the service.",
            "Average hospital. Neither exceptionally good nor bad.",
            "The service was fine but could be improved in some areas.",
            "Decent care but the wait times could be better.",
        ],
        "3": [
            "Good and clean hospital. Great team of doctors and medical facilities.",
            "Really great experience! The staff were professional and caring.",
            "Excellent hospital with strong medical care and friendly nurses.",
            "Over all experience was good, from reception to doctor consultation.",
            "Absolutely amazing! Best care I've ever received at a hospital.",
            "Outstanding service! The doctors are extremely skilled and compassionate.",
        ]
    }
    return examples


@router.get("/sentiment-scale", response_model=Dict[int, Dict[str, str]])
async def get_sentiment_scale():
    """
    Get information about the 3-point sentiment scale

    **Returns:**
    - Dictionary mapping scores to labels and emojis
    """
    analyzer = load_nlp_analyzer()

    if analyzer is None:
        # Return default scale if analyzer not loaded
        return {
            1: {"label": "Negative", "emoji": "😞"},
            2: {"label": "Neutral", "emoji": "😐"},
            3: {"label": "Positive", "emoji": "😊"}
        }

    return analyzer.get_sentiment_scale()


# Preload model on startup (optional)
@router.on_event("startup")
async def startup_event():
    """Preload model on startup"""
    # Uncomment to preload (uses more RAM)
    # load_nlp_analyzer()
    pass
