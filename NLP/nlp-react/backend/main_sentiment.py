"""
Simple Training Script
"""

from sentiment_model import SentimentAnalyzer

print("\n" + "="*80)
print("TRAINING MODEL")
print("="*80 + "\n")

analyzer = SentimentAnalyzer(data_path='../../data/hospital.csv')

print("Loading data...")
analyzer.load_data(text_column='Feedback', rating_column='Ratings')

print("Preprocessing...")
analyzer.preprocess_and_visualize()

print("Training...")
analyzer.build_model(test_size=0.2)

print("Evaluating...")
analyzer.evaluate_model()

print("Saving...")
analyzer.save_model(model_dir='./saved_model')

print("\n✅ DONE! Model saved to ./saved_model/")
print("Start API with: python main.py")
