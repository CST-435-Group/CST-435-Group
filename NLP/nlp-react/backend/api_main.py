"""
FastAPI Backend for Hospital Review Sentiment Analyzer
Updated for Transformer Model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
from typing import List, Dict, Optional
import uvicorn

from sentiment_model import SentimentAnalyzer

app = FastAPI(
    title="Hospital Review Sentiment Analyzer API",
    description="Analyze hospital reviews using Fine-tuned RoBERTa Transformer",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = None

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)

class SentimentResult(BaseModel):
    text: str
    sentiment_score: int
    sentiment_label: str
    sentiment_label_verbose: Optional[str] = None
    emoji: str
    confidence: float
    probabilities: Dict[str, float]

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool

@app.on_event("startup")
async def startup_event():
    """Load the transformer model on startup"""
    global analyzer
    try:
        print("Loading sentiment analyzer...")
        analyzer = SentimentAnalyzer(
            model_path='../saved_model',
            data_path='../../data/hospital_cleaned.csv'
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        analyzer = None

@app.get("/")
async def root():
    return {
        "message": "Hospital Review Sentiment Analyzer API",
        "version": "2.0.0",
        "model": "Fine-tuned RoBERTa Transformer",
        "sentiment_scale": "-3 (Very Negative) to +3 (Very Positive)",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": analyzer is not None
    }

@app.post("/analyze", response_model=SentimentResult)
async def analyze_sentiment(input_data: TextInput):
    """
    Analyze sentiment of a single text
    
    Returns sentiment on a 7-point scale:
    -3: Very Negative
    -2: Negative
    -1: Slightly Negative
    0: Neutral
    +1: Slightly Positive
    +2: Positive
    +3: Very Positive
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = analyzer.analyze(input_data.text)
        
        # Convert to response format
        return {
            "text": result['text'],
            "sentiment_score": result['sentiment_score'],
            "sentiment_label": result['sentiment_label'],
            "sentiment_label_verbose": result.get('sentiment_label_verbose'),
            "emoji": result['emoji'],
            "confidence": float(result['confidence']),
            "probabilities": {k: float(v) for k, v in result['probabilities'].items()}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")

@app.post("/analyze/batch", response_model=List[SentimentResult])
async def analyze_batch(input_data: BatchTextInput):
    """Analyze sentiment of multiple texts in batch"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = [analyzer.analyze(text) for text in input_data.texts]
        
        # Convert to response format
        return [
            {
                "text": r['text'],
                "sentiment_score": r['sentiment_score'],
                "sentiment_label": r['sentiment_label'],
                "sentiment_label_verbose": r.get('sentiment_label_verbose'),
                "emoji": r['emoji'],
                "confidence": float(r['confidence']),
                "probabilities": {k: float(v) for k, v in r['probabilities'].items()}
            }
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing batch: {str(e)}")

@app.get("/examples")
async def get_examples():
    """Get hospital review examples for each sentiment level"""
    # Aggregate original 7-point examples into three buckets
    raw = {
        "-3": ["Terrible experience! Worst hospital I've ever been to!", "Absolutely horrible! Never coming back here again!"],
        "-2": ["Poor service. Long wait times and unprofessional staff.", "Not satisfied with the care. Would not recommend."],
        "-1": ["Not very impressed. Could have been better.", "Some issues with wait times and communication."],
        "0": ["It was okay, nothing special.", "Average experience. Neither particularly good nor bad."],
        "1": ["Nice service. Generally satisfied.", "The care was good, though there were a few delays."],
        "2": ["Good experience overall. The staff was friendly and helpful.", "The facility was clean and the doctors were knowledgeable."],
        "3": ["This hospital was absolutely amazing! Best care I've ever received!", "Exceptional service! The doctors and nurses were incredibly caring and professional."]
    }

    # Simplify to three buckets: positive (1,2,3), neutral (0), negative (-1,-2,-3)
    positive = []
    for k in ["1", "2", "3"]:
        positive.extend(raw.get(k, []))

    neutral = raw.get("0", [])

    negative = []
    for k in ["-1", "-2", "-3"]:
        negative.extend(raw.get(k, []))

    return {
        "positive": positive,
        "neutral": neutral,
        "negative": negative
    }


@app.get("/statistics")
async def get_statistics():
    """
    Return simple dataset statistics: total_reviews and sentiment_distribution
    """
    import pandas as pd
    # Try to locate the dataset used by the app
    possible_paths = [
        './saved_model',
        '../saved_model',
        '../../data/hospital_cleaned.csv',
        '../data/hospital_cleaned.csv',
        './data/hospital_cleaned.csv'
    ]

    data_path = '../../data/hospital_cleaned.csv'
    # prefer the path used at startup if analyzer was initialized with a data_path attribute
    try:
        if analyzer is not None and hasattr(analyzer, 'data_path'):
            data_path = analyzer.data_path
    except Exception:
        pass

    # If the resolved path doesn't exist, try alternatives
    import os
    if not os.path.exists(data_path):
        for p in possible_paths:
            if os.path.exists(p) and p.endswith('.csv'):
                data_path = p
                break

    if not os.path.exists(data_path):
        # No dataset found
        return {"total_reviews": 0, "sentiment_distribution": {}}

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return {"total_reviews": 0, "sentiment_distribution": {}}

    total = len(df)

    # Determine sentiment column if present
    sentiment_dist = {"positive": 0, "neutral": 0, "negative": 0}

    # Helper: map numeric score to bucket
    def score_to_bucket(score):
        try:
            s = int(score)
        except Exception:
            return None
        if s < 0:
            return 'negative'
        if s > 0:
            return 'positive'
        return 'neutral'

    # 1) If there's a 'sentiment' or 'sentiment_label' column with text labels
    if 'sentiment' in df.columns:
        vals = df['sentiment'].astype(str).str.lower()
        sentiment_dist['positive'] = int((vals == 'positive').sum())
        sentiment_dist['neutral'] = int((vals == 'neutral').sum())
        sentiment_dist['negative'] = int((vals == 'negative').sum())
    elif 'sentiment_label' in df.columns or 'Sentiment Label' in df.columns:
        col = 'sentiment_label' if 'sentiment_label' in df.columns else 'Sentiment Label'
        # If it's numeric 0/1 (0 negative, 1 positive)
        if pd.api.types.is_numeric_dtype(df[col]):
            vals = df[col].fillna(-1).astype(int)
            sentiment_dist['positive'] = int((vals == 1).sum())
            sentiment_dist['negative'] = int((vals == 0).sum())
            # neutral remains 0
        else:
            vals = df[col].astype(str).str.lower()
            sentiment_dist['positive'] = int((vals == 'positive').sum())
            sentiment_dist['neutral'] = int((vals == 'neutral').sum())
            sentiment_dist['negative'] = int((vals == 'negative').sum())
    elif 'Ratings' in df.columns:
        # Map ratings to score then bucket (similar to older logic)
        def rating_to_score(rating):
            try:
                r = float(rating)
            except Exception:
                return 0
            if r <= 1:
                return -3
            elif r <= 2:
                return -2
            elif r <= 3:
                return -1
            elif r <= 4:
                return 0
            elif r <= 5:
                return 1
            elif r <= 7:
                return 2
            else:
                return 3

        scores = df['Ratings'].apply(rating_to_score)
        sentiment_dist = {'positive': int((scores > 0).sum()), 'neutral': int((scores == 0).sum()), 'negative': int((scores < 0).sum())}
    else:
        # Try to detect a numeric score column (e.g., sentiment_score)
        candidates = [c for c in df.columns if 'score' in c.lower() or c.lower() in ['sentiment_score', 'score']]
        found = False
        for c in candidates:
            if pd.api.types.is_numeric_dtype(df[c]):
                buckets = df[c].fillna(0).astype(int).apply(score_to_bucket)
                sentiment_dist = {'positive': int((buckets == 'positive').sum()), 'neutral': int((buckets == 'neutral').sum()), 'negative': int((buckets == 'negative').sum())}
                found = True
                break
        if not found:
            # Could not determine sentiment column; return totals only
            return {"total_reviews": total, "sentiment_distribution": {}}

    return {"total_reviews": int(total), "sentiment_distribution": sentiment_dist}

if __name__ == "__main__":
    uvicorn.run(
        "api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )