"""
Sentiment Analysis Model
Includes: Better preprocessing, advanced feature engineering, model optimization
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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
    """Optimized sentiment analyzer with advanced techniques"""
    
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
        
        # Remove sentiment-bearing words from stopwords
        self.stop_words = self.stop_words - {
            'not', 'no', 'nor', 'neither', 'never', 'none',
            'good', 'bad', 'best', 'worst', 'great', 'terrible',
            'very', 'too', 'more', 'most', 'less', 'least'
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
        print(f"Removed {initial_count - len(self.df)} rows with missing values")
        
        # Create sentiment labels
        self.df['sentiment'] = self.df[self.rating_column].apply(self._rating_to_sentiment)
        
        sentiment_counts = self.df['sentiment'].value_counts()
        print(f"\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment}: {count}")
        
        return self.df
    
    @staticmethod
    def _rating_to_sentiment(rating):
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:
            return 'positive'
    
    def clean_text(self, text):
        """Advanced text cleaning"""
        text = str(text).lower()
        
        # Remove HTML
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove URLs and emails
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Handle negations (critical for sentiment)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        
        # Remove punctuation but keep important sentiment markers
        text = text.translate(str.maketrans('', '', string.punctuation.replace('!', '').replace('?', '')))
        
        # Replace multiple exclamation/question marks
        text = re.sub(r'!+', ' exclamation ', text)
        text = re.sub(r'\?+', ' question ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize (but keep sentiment words)
        tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return ' '.join(tokens)
    
    def build_model(self, test_size=0.2, random_state=42):
        """Build optimized model with best parameters"""
        print("\n" + "="*80)
        print("BUILDING OPTIMIZED MODEL")
        print("="*80)
        
        # Clean text
        print("\n--- Cleaning Text ---")
        self.df['cleaned_text'] = self.df[self.text_column].apply(self.clean_text)
        
        # Prepare data
        X = self.df['cleaned_text']
        y = self.df['sentiment']
        
        # Split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Training: {len(self.X_train)} | Testing: {len(self.X_test)}")
        
        # Advanced TF-IDF with optimal parameters
        print("\n--- Advanced TF-IDF Vectorization ---")
        self.vectorizer = TfidfVectorizer(
            max_features=15000,        # More features
            ngram_range=(1, 3),        # Unigrams, bigrams, trigrams
            min_df=2,                  # Ignore very rare terms
            max_df=0.85,               # Ignore very common terms
            sublinear_tf=True,         # Apply sublinear scaling
            use_idf=True,              # Use inverse document frequency
            smooth_idf=True,           # Smooth IDF weights
            norm='l2'                  # L2 normalization
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        X_test_tfidf = self.vectorizer.transform(self.X_test)
        
        print(f"Feature matrix: {X_train_tfidf.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        # Try multiple models and pick the best
        print("\n--- Testing Multiple Models ---")
        
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
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            )
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
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
        print(f"   Training Accuracy: {self.model.score(X_train_tfidf, self.y_train):.4f}")
        print(f"   Testing Accuracy: {best_score:.4f} ({best_score*100:.2f}%)")
        print("="*80)
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        X_test_tfidf = self.vectorizer.transform(self.X_test)
        y_pred = self.model.predict(X_test_tfidf)
        
        print("\n--- Classification Report ---")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred, labels=['negative', 'neutral', 'positive'])
        self._plot_confusion_matrix(cm, ['negative', 'neutral', 'positive'])
        
        return cm
    
    def _plot_confusion_matrix(self, cm, labels):
        """Plot confusion matrix"""
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Sentiment Classification', fontsize=16, fontweight='bold')
        plt.ylabel('Actual Sentiment', fontsize=12)
        plt.xlabel('Predicted Sentiment', fontsize=12)
        plt.tight_layout()
        # plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✅ Generated confusion matrix")
        plt.close()
    
    def predict_single(self, text):
        """Predict sentiment for a single text"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call build_model() first.")
        
        cleaned = self.clean_text(text)
        vectorized = self.vectorizer.transform([cleaned])
        
        prediction = self.model.predict(vectorized)[0]
        probabilities = self.model.predict_proba(vectorized)[0] if hasattr(self.model, 'predict_proba') else None
        
        if probabilities is not None:
            prob_dict = dict(zip(self.model.classes_, probabilities))
            confidence = max(probabilities)
        else:
            # For LinearSVC which doesn't have predict_proba
            prob_dict = {prediction: 1.0}
            confidence = 1.0
        
        return {
            'text': text,
            'sentiment': prediction,
            'probabilities': prob_dict,
            'confidence': confidence
        }
    
    def save_model(self, model_dir='./saved_model'):
        """Save trained model and vectorizer"""
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'sentiment_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ Model saved: {model_path}")
        
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"✅ Vectorizer saved: {vectorizer_path}")
    
    def load_model(self, model_dir='./saved_model'):
        """Load trained model and vectorizer"""
        model_path = os.path.join(model_dir, 'sentiment_model.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✅ Model loaded: {model_path}")
        
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print(f"✅ Vectorizer loaded: {vectorizer_path}")


# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer(data_path='../data/hospital.csv')
    analyzer.load_data(text_column='Feedback', rating_column='Rating')
    analyzer.preprocess_and_visualize()
    analyzer.build_model()
    analyzer.evaluate_model()
    analyzer.save_model()