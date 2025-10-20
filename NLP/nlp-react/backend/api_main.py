"""
FastAPI Backend for Hospital Review Sentiment Analyzer
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import uvicorn
import os

from sentiment_model import SentimentAnalyzer

app = FastAPI(
    title="Hospital Review Sentiment Analyzer API",
    description="Analyze hospital reviews using NLTK",
    version="1.0.0"
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
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool

@app.on_event("startup")
async def startup_event():
    global analyzer
    try:
        analyzer = SentimentAnalyzer(data_path='../../data/hospital.csv')
        
        if os.path.exists('./saved_model/sentiment_model.pkl'):
            analyzer.load_model(model_dir='./saved_model')
            
            # Load the dataset for statistics
            try:
                analyzer.load_data(text_column='Feedback', rating_column='Ratings')
                analyzer.df['sentiment'] = analyzer.df['Ratings'].apply(
                    lambda r: 'negative' if r <= 2 else ('neutral' if r == 3 else 'positive')
                )
                print("✅ Model and dataset loaded successfully!")
            except Exception as e:
                print(f"⚠️ Model loaded but couldn't load dataset: {e}")
                print("   Statistics will not be available.")
        else:
            print("⚠️ No model found. Train first: python main_sentiment.py")
            analyzer = None
    except Exception as e:
        print(f"❌ Error: {e}")
        analyzer = None

@app.get("/")
async def root():
    return {
        "message": "Hospital Review Sentiment Analyzer API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": analyzer is not None and analyzer.model is not None
    }

@app.post("/analyze", response_model=SentimentResult)
async def analyze_sentiment(input_data: TextInput):
    if analyzer is None or analyzer.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = analyzer.predict_single(input_data.text)
        
        # Convert numpy floats to Python floats
        clean_probabilities = {
            k: float(v) for k, v in result['probabilities'].items()
        }
        
        return {
            "text": result['text'],
            "sentiment": result['sentiment'],
            "confidence": float(result['confidence']),
            "probabilities": clean_probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch", response_model=List[SentimentResult])
async def analyze_batch(input_data: BatchTextInput):
    if analyzer is None or analyzer.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = [analyzer.predict_single(text) for text in input_data.texts]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/examples")
async def get_examples():
    """Get hospital review examples"""
    return {
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

@app.get("/statistics")
async def get_statistics():
    """Get detailed dataset statistics"""
    if analyzer is None or analyzer.df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded. Train model first.")
    
    try:
        # Basic stats
        sentiment_dist = analyzer.df['sentiment'].value_counts().to_dict()
        
        # Get common words from cleaned text
        from collections import Counter
        all_words = []
        if 'cleaned_text' in analyzer.df.columns:
            for text in analyzer.df['cleaned_text'].dropna():
                all_words.extend(text.split())
        else:
            # Clean on the fly if not available
            for text in analyzer.df[analyzer.text_column].dropna()[:100]:
                cleaned = analyzer.clean_text(text)
                all_words.extend(cleaned.split())
        
        word_freq = Counter(all_words)
        top_words = dict(word_freq.most_common(20))
        
        # Words by sentiment
        positive_words = []
        negative_words = []
        neutral_words = []
        
        for idx, row in analyzer.df.iterrows():
            text = row.get('cleaned_text', analyzer.clean_text(row[analyzer.text_column]))
            sentiment = row['sentiment']
            words = text.split()
            
            if sentiment == 'positive':
                positive_words.extend(words)
            elif sentiment == 'negative':
                negative_words.extend(words)
            else:
                neutral_words.extend(words)
        
        stats = {
            "total_reviews": len(analyzer.df),
            "sentiment_distribution": sentiment_dist,
            "average_review_length": analyzer.df['text_length'].mean() if 'text_length' in analyzer.df.columns else None,
            "rating_distribution": analyzer.df[analyzer.rating_column].value_counts().to_dict() if hasattr(analyzer, 'rating_column') else None,
            "top_words": top_words,
            "positive_words": dict(Counter(positive_words).most_common(15)),
            "negative_words": dict(Counter(negative_words).most_common(15)),
            "neutral_words": dict(Counter(neutral_words).most_common(15))
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )