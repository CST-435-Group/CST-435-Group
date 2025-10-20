"""
Optimized Sentiment Analysis Model for Maximum Accuracy
Handles data preprocessing, model training, and predictions
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support
)
import pickle
import os

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class SentimentAnalyzer:
    """Optimized sentiment analyzer for hospital reviews"""
    
    def __init__(self, data_path='../data/hospital.csv'):
        self.data_path = data_path
        self.df = None
        self.vectorizer = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self._download_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Keep important sentiment words
        self.stop_words = self.stop_words - {
            'not', 'no', 'nor', 'neither', 'never', 'none',
            'good', 'bad', 'best', 'worst', 'great', 'terrible',
            'very', 'too', 'more', 'most', 'less', 'least',
            'really', 'quite', 'rather', 'somewhat'
        }
    
    @staticmethod
    def _download_nltk_resources():
        print("Downloading NLTK resources...")
        for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']:
            try:
                nltk.download(resource, quiet=True)
            except:
                pass
        print("✅ NLTK resources ready\n")
    
    def load_data(self, text_column='Feedback', rating_column='Rating'):
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"\n✅ Loaded {len(self.df)} reviews")
        
        self.text_column = text_column
        self.rating_column = rating_column
        
        return self.df
    
    def preprocess_and_visualize(self):
        print("\n" + "="*80)
        print("PREPROCESSING")
        print("="*80)
        
        # Handle missing values
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=[self.text_column, self.rating_column])
        removed = initial_count - len(self.df)
        if removed > 0:
            print(f"Removed {removed} rows with missing values")
        
        # Create sentiment labels
        self.df['sentiment'] = self.df[self.rating_column].apply(self._rating_to_sentiment)
        
        sentiment_counts = self.df['sentiment'].value_counts()
        print(f"\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            pct = (count / len(self.df)) * 100
            print(f"  {sentiment}: {count} ({pct:.1f}%)")
        
        return self.df
    
    @staticmethod
    def _rating_to_sentiment(rating):
        """Convert rating to sentiment - more balanced thresholds"""
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:
            return 'positive'
    
    def clean_text(self, text):
        """Advanced text cleaning that preserves sentiment"""
        text = str(text).lower()
        
        # Remove HTML
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove URLs and emails
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Handle negations (CRITICAL for sentiment)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        
        # Keep exclamation/question marks as features
        text = re.sub(r'!+', ' EXCLAMATION ', text)
        text = re.sub(r'\?+', ' QUESTION ', text)
        
        # Remove punctuation (except spaces)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Lemmatize but keep important words
        tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if (word not in self.stop_words or word in ['not', 'no', 'never']) and len(word) > 1
        ]
        
        return ' '.join(tokens)
    
    def build_model(self, test_size=0.2, random_state=42):
        """Build optimized model"""
        print("\n" + "="*80)
        print("BUILDING MODEL")
        print("="*80)
        
        # Clean text
        print("\n--- Cleaning Text ---")
        self.df['cleaned_text'] = self.df[self.text_column].apply(self.clean_text)
        
        # Show example
        if len(self.df) > 0:
            print(f"\nExample cleaning:")
            print(f"Original: {self.df[self.text_column].iloc[0][:100]}")
            print(f"Cleaned:  {self.df['cleaned_text'].iloc[0][:100]}")
        
        # Prepare data
        X = self.df['cleaned_text']
        y = self.df['sentiment']
        
        # Split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"\nTrain: {len(self.X_train)} | Test: {len(self.X_test)}")
        
        # Optimized TF-IDF
        print("\n--- TF-IDF Vectorization ---")
        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True,
            norm='l2'
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        X_test_tfidf = self.vectorizer.transform(self.X_test)
        
        print(f"Features: {X_train_tfidf.shape[1]}")
        
        # Test multiple models
        print("\n--- Testing Models ---")
        
        models = {
            'Logistic Regression': LogisticRegression(
                C=5.0,
                solver='saga',
                max_iter=2000,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            ),
            'Linear SVM': LinearSVC(
                C=1.0,
                max_iter=3000,
                class_weight='balanced',
                random_state=random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=50,
                min_samples_split=2,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            )
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        for name, model in models.items():
            print(f"\n{name}...")
            model.fit(X_train_tfidf, self.y_train)
            
            train_score = model.score(X_train_tfidf, self.y_train)
            test_score = model.score(X_test_tfidf, self.y_test)
            
            print(f"  Train: {train_score:.4f} | Test: {test_score:.4f}")
            
            if test_score > best_score:
                best_score = test_score
                best_model = model
                best_name = name
        
        self.model = best_model
        
        print("\n" + "="*80)
        print(f"🏆 BEST MODEL: {best_name}")
        print(f"   Testing Accuracy: {best_score:.4f} ({best_score*100:.2f}%)")
        print("="*80)
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80)
        
        X_test_tfidf = self.vectorizer.transform(self.X_test)
        y_pred = self.model.predict(X_test_tfidf)
        
        print("\n" + classification_report(self.y_test, y_pred, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred, labels=['negative', 'neutral', 'positive'])
        print("\nConfusion Matrix:")
        print(cm)
        
        return cm
    
    def make_predictions(self, custom_texts=None):
        """Test predictions"""
        if custom_texts is None:
            custom_texts = [
                "The hospital staff was amazing and very caring. Best experience ever!",
                "Terrible service. Long wait times and rude staff. Very disappointed.",
                "It was okay. Nothing special but not terrible either.",
                "The doctor was professional and the facility was clean.",
                "Worst hospital ever! I will never come back here again!",
                "Average experience. Could be better but could be worse."
            ]
        
        print("\n" + "="*80)
        print("TESTING PREDICTIONS")
        print("="*80)
        
        for text in custom_texts:
            result = self.predict_single(text)
            print(f"\nText: {text}")
            print(f"→ {result['sentiment'].upper()} ({result['confidence']:.1%})")
    
    def predict_single(self, text):
        """Predict sentiment for single text"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained")
        
        cleaned = self.clean_text(text)
        vectorized = self.vectorizer.transform([cleaned])
        
        prediction = self.model.predict(vectorized)[0]
        
        # Handle models with/without predict_proba
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(vectorized)[0]
            prob_dict = dict(zip(self.model.classes_, probabilities))
            confidence = float(max(probabilities))
        else:
            # LinearSVC doesn't have predict_proba
            prob_dict = {prediction: 1.0}
            confidence = 1.0
        
        return {
            'text': text,
            'sentiment': prediction,
            'probabilities': prob_dict,
            'confidence': confidence
        }
    
    def save_model(self, model_dir='./saved_model'):
        """Save model"""
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, 'sentiment_model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ Model saved: {model_dir}/sentiment_model.pkl")
        
        with open(os.path.join(model_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"✅ Vectorizer saved: {model_dir}/vectorizer.pkl")
    
    def load_model(self, model_dir='./saved_model'):
        """Load model"""
        with open(os.path.join(model_dir, 'sentiment_model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        print(f"✅ Model loaded: {model_dir}/sentiment_model.pkl")
        
        with open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        print(f"✅ Vectorizer loaded: {model_dir}/vectorizer.pkl")
        