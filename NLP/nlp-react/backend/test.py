from sentiment_model import SentimentAnalyzer

analyzer = SentimentAnalyzer()
analyzer.load_model(model_dir='./saved_model')

# Test predictions
test_texts = [
    "Amazing hospital! Best care ever!",
    "Terrible experience. Worst hospital!",
    "It was okay, nothing special."
]

for text in test_texts:
    result = analyzer.predict_single(text)
    print(f"Text: {text}")
    print(f"Prediction: {result['sentiment']}")
    print(f"Probabilities: {result['probabilities']}")
    print()