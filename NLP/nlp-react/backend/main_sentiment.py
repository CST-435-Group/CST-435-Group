"""
Sentiment Analysis Model - Trained on Hospital Reviews
Handles model loading, inference, and result formatting
"""

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


class SentimentAnalyzer:
    """Multi-scale sentiment analyzer using trained ML model"""

    def __init__(self, model_path: str = '../saved_model', data_path: str = '../../data/hospital.csv'):
        """
        Initialize the sentiment analyzer
        
        Args:
            model_path: Path to saved model files (relative to backend folder)
            data_path: Path to training data (relative to backend folder)
        """
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        
        # Download NLTK resources
        self._download_nltk_resources()
        
        # Setup stop words (keep sentiment-important words)
        self.stop_words = set(stopwords.words('english'))
        self.stop_words = self.stop_words - {
            'not', 'no', 'nor', 'neither', 'never', 'none',
            'good', 'bad', 'best', 'worst', 'great', 'terrible',
            'very', 'too', 'more', 'most', 'less', 'least'
        }
        
        # Try to load existing model, otherwise train new one
        if os.path.exists(os.path.join(model_path, 'sentiment_model.pkl')):
            self._load_model()
        else:
            print("No saved model found. Training new model...")
            self._train_model()

    @staticmethod
    def _download_nltk_resources():
        """Download required NLTK resources"""
        for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
            try:
                nltk.download(resource, quiet=True)
            except:
                pass

    def clean_text(self, text):
        """Clean and preprocess text"""
        text = str(text).lower()
        
        # Remove HTML
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove URLs and emails
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Handle negations
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in self.stop_words and len(word) > 1
        ]
        
        return ' '.join(tokens)

    def _train_model(self):
        """Train the sentiment model on hospital data"""
        print("Training model on hospital data...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        df = df.dropna(subset=['Feedback', 'Ratings'])
        
        # Convert ratings to sentiment scores (-3 to +3)
        # Rating 1 -> -3, Rating 2 -> -2, ..., Rating 5 -> +2, etc.
        def rating_to_score(rating):
            if rating <= 1:
                return -3
            elif rating <= 2:
                return -2
            elif rating <= 3:
                return -1
            elif rating <= 4:
                return 0
            elif rating <= 5:
                return 1
            elif rating <= 7:
                return 2
            else:
                return 3
        
        df['sentiment_score'] = df['Ratings'].apply(rating_to_score)
        
        print(f"Loaded {len(df)} reviews")
        print(f"Score distribution:\n{df['sentiment_score'].value_counts().sort_index()}")
        
        # Clean text
        print("Cleaning text...")
        df['cleaned_text'] = df['Feedback'].apply(self.clean_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], 
            df['sentiment_score'],
            test_size=0.2,
            random_state=42,
            stratify=df['sentiment_score']
        )
        
        # Vectorize
        print("Vectorizing text...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train model
        print("Training Logistic Regression model...")
        self.model = LogisticRegression(
            C=5.0,
            solver='saga',
            max_iter=2000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, self.model.predict(X_train_tfidf))
        test_acc = accuracy_score(y_test, self.model.predict(X_test_tfidf))
        
        print(f"\n✅ Training complete!")
        print(f"   Train accuracy: {train_acc:.4f}")
        print(f"   Test accuracy: {test_acc:.4f}")
        
        # Save model
        self._save_model()

    def _save_model(self):
        """Save model and vectorizer"""
        os.makedirs(self.model_path, exist_ok=True)
        
        with open(os.path.join(self.model_path, 'sentiment_model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(os.path.join(self.model_path, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"✅ Model saved to {self.model_path}")

    def _load_model(self):
        """Load saved model and vectorizer"""
        with open(os.path.join(self.model_path, 'sentiment_model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        
        with open(os.path.join(self.model_path, 'vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        print(f"✅ Model loaded from {self.model_path}")

    @staticmethod
    def get_verbose_label(score: int) -> str:
        """Get descriptive (verbose) label for sentiment score"""
        labels = {
            -3: "Negative",
            -2: "Negative",
            -1: "Slightly Negative",
            0: "Neutral",
            1: "Slightly Positive",
            2: "Positive",
            3: "Positive"
        }
        return labels.get(score, "Unknown")

    @staticmethod
    def get_sentiment_label(score: int) -> str:
        """Map a score (-3..+3) to one of three condensed labels"""
        if score < 0:
            return "negative"
        if score > 0:
            return "positive"
        return "neutral"

    @staticmethod
    def get_sentiment_emoji(score: int) -> str:
        """Get emoji for sentiment score"""
        emojis = {
            -3: "😢",
            -2: "😞",
            -1: "😐",
            0: "😶",
            1: "🙂",
            2: "😊",
            3: "🤩"
        }
        return emojis.get(score, "❓")

    def get_sentiment_scale(self) -> dict:
        """Get the complete sentiment scale information"""
        scale = {}
        for i in range(-3, 4):
            scale[i] = {
                "label": self.get_sentiment_label(i),
                "emoji": self.get_sentiment_emoji(i)
            }
        return scale

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not loaded or trained")
        
        # Clean and vectorize text
        cleaned = self.clean_text(text)
        vectorized = self.vectorizer.transform([cleaned])
        
        # Get prediction
        sentiment_score = int(self.model.predict(vectorized)[0])
        probabilities = self.model.predict_proba(vectorized)[0]
        
        # Get label and emoji
        sentiment_label = self.get_sentiment_label(sentiment_score)
        emoji = self.get_sentiment_emoji(sentiment_score)
        
        # Get confidence (probability of predicted class)
        class_index = list(self.model.classes_).index(sentiment_score)
        confidence = float(probabilities[class_index])
        
        # Aggregate probabilities into three buckets based on numeric class labels
        neg_prob = 0.0
        neu_prob = 0.0
        pos_prob = 0.0
        for i, class_label in enumerate(self.model.classes_):
            c = int(class_label)
            p = float(probabilities[i])
            if c < 0:
                neg_prob += p
            elif c == 0:
                neu_prob += p
            else:
                pos_prob += p
        prob_dict = {
            "negative": neg_prob,
            "neutral": neu_prob,
            "positive": pos_prob
        }
        
        # Provide condensed label and keep verbose label too
        sentiment_label_verbose = self.get_verbose_label(sentiment_score)

        return {
            "text": text,
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "sentiment_label_verbose": sentiment_label_verbose,
            "emoji": emoji,
            "confidence": confidence,
            "probabilities": prob_dict
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize and train (or load) model
    analyzer = SentimentAnalyzer()
    
    # Test with sample reviews
    test_texts = [
        "This hospital was absolutely amazing! Best care I've ever received!",
        "The service was good and staff were friendly.",
        "It was okay, nothing special.",
        "Not very satisfied with the long wait times.",
        "Terrible experience. Very disappointed with everything."
    ]
    
    print("\n" + "="*80)
    print("Testing Sentiment Analyzer")
    print("="*80 + "\n")
    
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"Score: {result['sentiment_score']:+d}/3")
        print(f"Label: {result['sentiment_label']} {result['emoji']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print("-" * 80)