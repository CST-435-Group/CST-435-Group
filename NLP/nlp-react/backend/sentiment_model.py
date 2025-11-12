"""
Sentiment Analysis Model using Fine-tuned Transformer
Academic approach: Transfer learning + domain-specific fine-tuning
Properly handles: Very Negative, Negative, Neutral, Positive, Very Positive
"""

# Force transformers to use PyTorch only (disable TensorFlow)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['USE_TF'] = '0'  # Disable TensorFlow in transformers
os.environ['USE_TORCH'] = '1'  # Enable PyTorch in transformers

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment analysis"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


class SentimentAnalyzer:
    """Multi-scale sentiment analyzer using fine-tuned transformer"""

    def __init__(self,
                 model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
                 data_path: str = None,
                 model_path: str = None,
                 use_pretrained_only: bool = True):
        """
        Initialize the sentiment analyzer

        Uses transfer learning:
        1. Starts with pre-trained RoBERTa sentiment model
        2. Optionally fine-tunes on hospital review data to learn domain-specific patterns

        Args:
            model_name: Base pre-trained model
            data_path: Path to hospital CSV data (auto-resolved if None)
            model_path: Path to save fine-tuned model (auto-resolved if None)
            use_pretrained_only: If True, skip fine-tuning and use pre-trained model directly (default: True)
        """
        self.model_name = model_name

        # Auto-resolve paths relative to this file's location
        current_dir = Path(__file__).parent

        # Default data_path: look for data in NLP/data/hospital_cleaned.csv
        if data_path is None:
            self.data_path = str(current_dir.parent.parent / "data" / "hospital_cleaned.csv")
        else:
            self.data_path = data_path

        # Default model_path: save in NLP/saved_model
        if model_path is None:
            self.model_path = str(current_dir.parent / "saved_model")
        else:
            self.model_path = model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Debug: Print resolved paths
        print(f"[DATA] Data path resolved to: {self.data_path}")
        print(f"[DATA] Model path resolved to: {self.model_path}")
        print(f"[DATA] Data file exists: {os.path.exists(self.data_path)}")

        # Check if fine-tuned model exists
        if os.path.exists(os.path.join(self.model_path, 'pytorch_model.bin')) or \
           os.path.exists(os.path.join(self.model_path, 'model.safetensors')):
            print(f"[OK] Loading fine-tuned model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        elif use_pretrained_only:
            # Use pre-trained model directly without fine-tuning
            print(f"[OK] Using pre-trained model: {self.model_name} (no fine-tuning)")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        else:
            # Attempt to fine-tune (requires accelerate package)
            print(f"No fine-tuned model found. Training on hospital data...")
            try:
                self._train_model()
            except Exception as e:
                print(f"[WARN] Fine-tuning failed: {e}")
                print(f"[OK] Falling back to pre-trained model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        self.model.to(self.device)
        self.model.eval()
        print(f"[OK] Model ready on device: {self.device}\n")

    def _train_model(self):
        """Fine-tune the model on hospital data"""
        print("\n" + "="*80)
        print("FINE-TUNING TRANSFORMER ON HOSPITAL DATA")
        print("="*80 + "\n")
        
        # Load data
        print("Loading hospital data...")
        df = pd.read_csv(self.data_path)
        df = df.dropna(subset=['Feedback', 'Sentiment Label'])
        
        # Use Sentiment Label: 0=negative, 1=positive
        # Model expects 3 classes: 0=neg, 1=neu, 2=pos
        # Since we only have binary labels, we'll use: 0=neg, 2=pos
        df['label'] = df['Sentiment Label'].apply(lambda x: 2 if x == 1 else 0)
        
        print(f"Loaded {len(df)} reviews")
        print(f"Label distribution:\n{df['label'].value_counts().sort_index()}\n")
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['Feedback'].tolist(),
            df['label'].tolist(),
            test_size=0.2,
            random_state=42,
            stratify=df['label']
        )
        
        print(f"Train samples: {len(train_texts)}")
        print(f"Val samples: {len(val_texts)}\n")
        
        # Load base model and tokenizer
        print(f"Loading base model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,  # negative, neutral, positive
            ignore_mismatched_sizes=True
        )
        
        # Create datasets
        print("Preparing datasets...")
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer)
        
        # Lazy import Trainer and TrainingArguments to avoid heavy TF imports at module import time
        from transformers import Trainer, TrainingArguments

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.model_path,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=os.path.join(self.model_path, 'logs'),
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train
        print("\n" + "="*80)
        print("TRAINING...")
        print("="*80 + "\n")
        trainer.train()

        # Save
        print(f"\n[OK] Saving fine-tuned model to {self.model_path}")
        trainer.save_model(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        
        # Evaluate
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80)
        results = trainer.evaluate()
        print(f"Validation Loss: {results['eval_loss']:.4f}")
        print("="*80 + "\n")

    @staticmethod
    def get_verbose_label(score: int) -> str:
        """Get descriptive (verbose) label for sentiment score"""
        labels = {
            1: "Negative",
            2: "Neutral",
            3: "Positive"
        }
        return labels.get(score, "Unknown")

    @staticmethod
    def get_sentiment_label(score: int) -> str:
        """Map a score (1-3) to one of three condensed labels"""
        if score == 1:
            return "negative"
        if score == 3:
            return "positive"
        return "neutral"

    @staticmethod
    def get_sentiment_emoji(score: int) -> str:
        """Get emoji for sentiment score"""
        emojis = {
            1: "😞",
            2: "😐",
            3: "😊"
        }
        return emojis.get(score, "❓")

    def get_sentiment_scale(self) -> dict:
        """Get the complete sentiment scale information"""
        scale = {}
        for i in range(1, 4):
            scale[i] = {
                "label": self.get_sentiment_label(i),
                "emoji": self.get_sentiment_emoji(i)
            }
        return scale

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment using fine-tuned transformer

        Maps to 3-point scale: 1 (Negative), 2 (Neutral), 3 (Positive)

        Args:
            text: Text to analyze

        Returns:
            Dictionary containing sentiment analysis results
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = probabilities.argmax(dim=-1).item()
            model_confidence = probabilities[0][predicted_class].item()

        # Get probabilities for the 3-class model (neg, neu, pos)
        neg_prob = float(probabilities[0][0].item())
        neu_prob = float(probabilities[0][1].item())
        pos_prob = float(probabilities[0][2].item())

        # Step 1: Check for neutral indicators (highest priority)
        text_lower = text.lower()
        neutral_words = ['okay', 'ok', 'fine', 'average', 'decent', 'nothing special',
                        'so-so', 'alright', 'fair', 'neither good nor bad', 'mixed',
                        'acceptable', 'satisfactory', 'moderate', 'mediocre']
        has_neutral = any(word in text_lower for word in neutral_words)

        # Step 2: Determine sentiment score (1-3 scale)
        if has_neutral:
            # Force neutral if neutral indicators present
            sentiment_score = 2  # Neutral
            confidence = 0.65
        elif predicted_class == 1:
            # Model predicted neutral
            sentiment_score = 2  # Neutral
            confidence = model_confidence
        elif predicted_class == 0:
            # NEGATIVE
            sentiment_score = 1  # Negative
            confidence = model_confidence
        else:  # predicted_class == 2
            # POSITIVE
            sentiment_score = 3  # Positive
            confidence = model_confidence

        # Get condensed label and verbose label
        sentiment_label = self.get_sentiment_label(sentiment_score)
        sentiment_label_verbose = self.get_verbose_label(sentiment_score)
        emoji = self.get_sentiment_emoji(sentiment_score)

        # Create probability distribution
        # Aggregate into three-class probabilities for the frontend
        prob_dict = {
            "negative": neg_prob,
            "neutral": neu_prob,
            "positive": pos_prob,
        }

        return {
            "text": text,
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "sentiment_label_verbose": sentiment_label_verbose,
            "emoji": emoji,
            "confidence": float(confidence),
            "probabilities": prob_dict
        }


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Fine-tuned Transformer Sentiment Analyzer")
    print("Transfer Learning: Pre-trained RoBERTa + Hospital Domain Fine-tuning")
    print("="*80 + "\n")
    
    # Initialize (will train if needed)
    analyzer = SentimentAnalyzer()

    # Test with diverse examples
    test_texts = [
        "This hospital was absolutely amazing! Best care I've ever received!",
        "The service was good and staff were friendly.",
        "It was okay, nothing special.",
        "Not very satisfied with the long wait times.",
        "Terrible experience. Very disappointed with everything.",
        "Horrible! Worst hospital ever! Never going back!",
        "The facility was clean and staff were professional.",
        "Average experience, nothing to complain about."
    ]

    print("="*80)
    print("Testing Sentiment Analyzer - 7-Point Scale")
    print("="*80 + "\n")

    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"Score: {result['sentiment_score']:+d} ({result['sentiment_label']}) {result['emoji']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print("-" * 80)