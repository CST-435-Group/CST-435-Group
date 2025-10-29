"""
FastAPI Router for NLP Project (Sentiment Analysis)
Wraps existing NLP backend functionality
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import sys
from pathlib import Path

# Add NLP project to path
nlp_project_path = Path(__file__).parent.parent.parent.parent / "NLP" / "nlp-react" / "backend"
sys.path.insert(0, str(nlp_project_path))

try:
    from model import SentimentAnalyzer
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
        print("✅ NLP sentiment analyzer loaded successfully!")
        return analyzer

    except Exception as e:
        print(f"❌ Error loading NLP analyzer: {e}")
        return None


@router.get("/")
async def nlp_info():
    """Get NLP project information"""
    return {
        "project": "Multi-Scale Sentiment Analysis",
        "description": "7-point scale sentiment analyzer for movie reviews and text",
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
    analyzer = load_nlp_analyzer()
    return {
        "status": "ready" if analyzer is not None else "not_loaded",
        "model_loaded": analyzer is not None
    }


@router.post("/analyze", response_model=SentimentResult)
async def analyze_sentiment(input_data: TextInput):
    """
    Analyze sentiment of a single text

    **Parameters:**
    - text: The text to analyze (movie review, comment, etc.)

    **Returns:**
    - sentiment_score: Integer from -3 to +3
    - sentiment_label: Descriptive label (e.g., "Very Positive")
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
        "-3": [
            "This movie was absolutely terrible! Worst film I've ever seen.",
            "Complete waste of time and money. Awful in every way.",
        ],
        "-2": [
            "Very disappointing. Poor acting and weak plot.",
            "Not good at all. Would not recommend.",
        ],
        "-1": [
            "The movie had potential but didn't deliver.",
            "Below average. Some moments but mostly forgettable.",
        ],
        "0": [
            "It was okay. Nothing particularly special.",
            "Average film. Neither good nor bad.",
        ],
        "1": [
            "Pretty decent movie. I enjoyed parts of it.",
            "Good film with some nice moments.",
        ],
        "2": [
            "Really great movie! Thoroughly enjoyed it.",
            "Excellent film with strong performances.",
        ],
        "3": [
            "Absolutely amazing! Best movie I've seen this year!",
            "Masterpiece! Incredible in every way!",
        ]
    }
    return examples


@router.get("/sentiment-scale", response_model=Dict[int, Dict[str, str]])
async def get_sentiment_scale():
    """
    Get information about the 7-point sentiment scale

    **Returns:**
    - Dictionary mapping scores to labels and emojis
    """
    analyzer = load_nlp_analyzer()

    if analyzer is None:
        # Return default scale if analyzer not loaded
        return {
            -3: {"label": "Very Negative", "emoji": "😡"},
            -2: {"label": "Negative", "emoji": "😞"},
            -1: {"label": "Slightly Negative", "emoji": "😕"},
            0: {"label": "Neutral", "emoji": "😐"},
            1: {"label": "Slightly Positive", "emoji": "🙂"},
            2: {"label": "Positive", "emoji": "😊"},
            3: {"label": "Very Positive", "emoji": "🤩"}
        }

    return analyzer.get_sentiment_scale()


# Preload model on startup (optional)
@router.on_event("startup")
async def startup_event():
    """Preload model on startup"""
    # Uncomment to preload (uses more RAM)
    # load_nlp_analyzer()
    pass
