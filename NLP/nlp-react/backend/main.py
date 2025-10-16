"""
FastAPI Backend for Multi-Scale Sentiment Analyzer
Author: AIT-204 Course
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import uvicorn

from model import SentimentAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Scale Sentiment Analyzer API",
    description="Analyze movie reviews on a 7-point sentiment scale (-3 to +3)",
    version="1.0.0"
)

# Configure CORS for React frontend
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

# Initialize sentiment analyzer
analyzer = None


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


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load the model on startup"""
    global analyzer
    try:
        analyzer = SentimentAnalyzer()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # In production, you might want to fail gracefully


# Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Scale Sentiment Analyzer API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": analyzer is not None
    }


@app.post("/analyze", response_model=SentimentResult)
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
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = analyzer.analyze(input_data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")


@app.post("/analyze/batch", response_model=List[SentimentResult])
async def analyze_batch(input_data: BatchTextInput):
    """
    Analyze sentiment of multiple texts in batch

    **Parameters:**
    - texts: List of texts to analyze (max 100)

    **Returns:**
    - List of sentiment analysis results
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = [analyzer.analyze(text) for text in input_data.texts]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing batch: {str(e)}")


@app.get("/examples", response_model=Dict[str, List[str]])
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


@app.get("/sentiment-scale", response_model=Dict[int, Dict[str, str]])
async def get_sentiment_scale():
    """
    Get information about the 7-point sentiment scale

    **Returns:**
    - Dictionary mapping scores to labels and emojis
    """
    return analyzer.get_sentiment_scale() if analyzer else {}


# Run the app
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
