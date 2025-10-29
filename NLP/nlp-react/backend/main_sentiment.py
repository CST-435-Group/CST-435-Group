"""
Sentiment Analysis Model - HuggingFace version
Handles model loading, inference, and retraining
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class SentimentAnalyzer:
    """HuggingFace-based sentiment analyzer"""

    def __init__(self, model_path: str = "./saved_model", default_model: str = "distilbert-base-uncased"):
        """
        Initialize sentiment analyzer.
        Loads retrained model if available; otherwise downloads default.
        """
        self.model_path = model_path

        if os.path.exists(os.path.join(model_path, "config.json")):
            print(f"✅ Loading model from local path: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            print(f"⚠️ No local model found. Loading base model: {default_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(default_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(default_model)

        # Create a sentiment analysis pipeline
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True
        )

    def analyze(self, text: str):
        """
        Analyze sentiment of input text.
        Returns dict with label, confidence, and probabilities.
        """
        results = self.pipeline(text)[0]  # list of dicts
        best = max(results, key=lambda r: r['score'])

        # Map to three-class system (Positive / Neutral / Negative)
        label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
        label = label_map.get(best['label'], best['label'].lower())

        probabilities = {label_map.get(r['label'], r['label'].lower()): r['score'] for r in results}

        return {
            "text": text,
            "sentiment_label": label,
            "confidence": best['score'],
            "probabilities": probabilities
        }


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    test_texts = [
        "This hospital was absolutely amazing! Best care I've ever received!",
        "The service was good and staff were friendly.",
        "It was okay, nothing special.",
        "Not very satisfied with the long wait times.",
        "Terrible experience. Very disappointed with everything."
    ]

    for text in test_texts:
        result = analyzer.analyze(text)
        print("\nText:", text)
        print("Label:", result['sentiment_label'])
        print("Confidence:", f"{result['confidence']:.2%}")
        print("Probabilities:", result['probabilities'])
