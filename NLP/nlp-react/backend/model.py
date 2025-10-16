"""
Sentiment Analysis Model
Handles model loading, inference, and result formatting
"""

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
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=7,  # -3 to +3 scale = 7 classes
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on device: {self.device}")

    @staticmethod
    def class_to_score(class_id: int) -> int:
        """Convert class ID (0-6) to sentiment score (-3 to +3)"""
        return class_id - 3

    @staticmethod
    def get_sentiment_label(score: int) -> str:
        """Get descriptive label for sentiment score"""
        labels = {
            -3: "Very Negative",
            -2: "Negative",
            -1: "Slightly Negative",
            0: "Neutral",
            1: "Slightly Positive",
            2: "Positive",
            3: "Very Positive"
        }
        return labels.get(score, "Unknown")

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
        sentiment_label = self.get_sentiment_label(sentiment_score)
        emoji = self.get_sentiment_emoji(sentiment_score)

        # Create probability distribution dictionary
        prob_dict = {}
        for i in range(7):
            score = self.class_to_score(i)
            label = self.get_sentiment_label(score)
            prob_dict[f"{score:+d} ({label})"] = float(probabilities[0][i].item())

        return {
            "text": text,
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
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
