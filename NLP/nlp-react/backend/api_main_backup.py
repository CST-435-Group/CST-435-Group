"""
FastAPI Backend for Hospital Review Sentiment Analyzer
Author: AIT-204 Course
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import uvicorn
import os

from sentiment_model import SentimentAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="Hospital Review Sentiment Analyzer API",
    description="Analyze hospital reviews on a 3-point sentiment scale",
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
    sentiment: str
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
        
        # Try to load pre-trained model
        if os.path.exists('./saved_model/sentiment_model.pkl'):
            analyzer.load_model(model_dir='./saved_model')
            print("✅ Model loaded successfully!")
        else:
            print("⚠️ No pre-trained model found. Please train a model first.")
            analyzer = None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        analyzer = None


# Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Hospital Review Sentiment Analyzer API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": analyzer is not None and analyzer.model is not None
    }


@app.post("/analyze", response_model=SentimentResult)
async def analyze_sentiment(input_data: TextInput):
    """
    Analyze sentiment of a single hospital review

    **Parameters:**
    - text: The review text to analyze

    **Returns:**
    - sentiment: 'positive', 'neutral', or 'negative'
    - confidence: Model confidence (0-1)
    - probabilities: Probability distribution across all classes
    """
    if analyzer is None or analyzer.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")

    try:
        result = analyzer.predict_single(input_data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")


@app.post("/analyze/batch", response_model=List[SentimentResult])
async def analyze_batch(input_data: BatchTextInput):
    """
    Analyze sentiment of multiple reviews in batch

    **Parameters:**
    - texts: List of review texts to analyze (max 100)

    **Returns:**
    - List of sentiment analysis results
    """
    if analyzer is None or analyzer.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = [analyzer.predict_single(text) for text in input_data.texts]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing batch: {str(e)}")


@app.get("/examples", response_model=Dict[str, List[str]])
async def get_examples():
    """
    Get example hospital reviews for each sentiment level

    **Returns:**
    - Dictionary mapping sentiment levels to example reviews
    """
    examples = {
        "positive": [
            "The hospital staff was amazing and very caring. Best experience ever!",
            "Excellent care and clean facilities. Highly recommend this hospital."
        ],
        "neutral": [
            "It was okay. Nothing special but not terrible either.",
            "Average experience. Could be better but could be worse."
        ],
        "negative": [
            "Terrible service. Long wait times and rude staff. Very disappointed.",
            "Worst hospital ever! I will never come back here again!"
        ]
    }
    return examples


@app.get("/statistics")
async def get_statistics():
    """
    Get simple statistics about the dataset
    
    **Returns:**
    - Basic dataset statistics
    """
    if analyzer is None or analyzer.df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded. Train a model first.")
    
    try:
        stats = {
            "total_reviews": len(analyzer.df),
            "sentiment_distribution": analyzer.df['sentiment'].value_counts().to_dict() if 'sentiment' in analyzer.df.columns else {}
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")


# Run the app
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )