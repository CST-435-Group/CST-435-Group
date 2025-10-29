"""
Sentiment Analysis Model
Handles model loading, inference, and result formatting
"""

import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """Multi-scale sentiment analyzer using transformer models"""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize the sentiment analyzer

        Args:
            model_name: Name of the pre-trained model to use, or path to local model
        """
        import os

        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prefer a local `saved_model/` in the backend folder if present.
        # This lets you drop a HuggingFace-format model into `backend/saved_model/`
        # and have the app load it automatically.
        backend_dir = Path(__file__).resolve().parent
        local_saved = backend_dir / "saved_model"

        if local_saved.exists() and (local_saved / "config.json").exists():
            model_source = str(local_saved)
            print(f"Loading model from local path: {model_source}")
        else:
            model_source = model_name
            print(f"Loading model: {model_source}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=7,  # -3 to +3 scale = 7 classes
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded successfully on device: {self.device}")


    @staticmethod
    def class_to_score(class_id: int) -> int:
        """Convert class ID (0-6) to sentiment score (-3 to +3)"""
        return class_id - 3

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
            Dictionary containing:
                - text: Original input text
                - sentiment_score: Integer from -3 to +3
                - sentiment_label: Descriptive label
                - emoji: Emoji representation
                - confidence: Model confidence (0-1)
                - probabilities: Dict of all class probabilities
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
            confidence = probabilities[0][predicted_class].item()

        # Convert to sentiment score
        sentiment_score = self.class_to_score(predicted_class)

        # condensed label for frontend/backend (negative/neutral/positive)
        sentiment_label = self.get_sentiment_label(sentiment_score)
        # also keep a verbose label if needed (collapsed -3/3 -> Negative/Positive)
        sentiment_label_verbose = self.get_verbose_label(sentiment_score)
        emoji = self.get_sentiment_emoji(sentiment_score)

        # Create probability distribution dictionary
        # Aggregate probabilities into three buckets: negative, neutral, positive
        neg_prob = float(sum(probabilities[0][i].item() for i in range(0, 3)))
        neu_prob = float(probabilities[0][3].item())
        pos_prob = float(sum(probabilities[0][i].item() for i in range(4, 7)))
        prob_dict = {
            "negative": neg_prob,
            "neutral": neu_prob,
            "positive": pos_prob
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
    # Test the analyzer
    analyzer = SentimentAnalyzer()

    test_texts = [
        "This movie was absolutely phenomenal! Best film of the decade!",
        "It was okay, nothing special.",
        "Terrible movie. Complete waste of time."
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
