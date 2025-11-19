"""
Generate Images using Trained GAN
CST-435 Neural Networks Assignment

Load a trained GAN and generate new 128x128 grayscale fruit images
Can optionally classify them with the CNN model
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torchvision.utils import save_image
from pathlib import Path
import argparse

# Import Generator from train_gan
from train_gan import Generator

# Import model utilities
from model_utils import load_model_chunked

# Add CNN_Project to path
sys.path.append(str(Path(__file__).parent.parent / 'CNN_Project'))

def load_generator(model_path, device):
    """Load trained generator from checkpoint (supports chunked models)"""
    checkpoint = load_model_chunked(model_path, device=str(device))

    if 'noise_dim' in checkpoint:
        noise_dim = checkpoint['noise_dim']
    else:
        noise_dim = 100  # Default

    generator = Generator(noise_dim).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    return generator, noise_dim

def generate_images(generator, noise_dim, num_images, device, output_dir):
    """Generate images using the trained generator"""
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        noise = torch.randn(num_images, noise_dim, device=device)
        fake_images = generator(noise)

        # Denormalize from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2

        # Save individual images
        for i in range(num_images):
            save_path = Path(output_dir) / f"generated_{i:04d}.png"
            save_image(fake_images[i], save_path)

        # Save grid if more than 1 image
        if num_images > 1:
            grid_size = min(8, num_images)
            save_image(fake_images[:min(64, num_images)], Path(output_dir) / 'grid.png', nrow=grid_size)

    return fake_images

def classify_with_cnn(images, device):
    """Classify generated images using the CNN model"""
    try:
        CNN_PROJECT_DIR = Path(__file__).parent.parent / 'CNN_Project'
        CNN_MODEL_PATH = CNN_PROJECT_DIR / 'models' / 'best_model.pth'

        if not CNN_MODEL_PATH.exists():
            print("CNN model not found. Skipping classification.")
            return None

        # Import CNN model
        from train_model import FruitCNN

        # Load model metadata
        with open(CNN_PROJECT_DIR / 'models' / 'model_metadata.json', 'r') as f:
            cnn_metadata = json.load(f)

        fruit_names = cnn_metadata['fruit_names']
        num_classes = cnn_metadata['num_classes']

        # Load CNN model
        cnn_model = FruitCNN(num_classes=num_classes).to(device)
        checkpoint = torch.load(CNN_MODEL_PATH, map_location=device)
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
        cnn_model.eval()

        print(f"\nCNN Classifier loaded ({num_classes} classes)")
        print(f"Classes: {', '.join(fruit_names)}")

        # Classify
        with torch.no_grad():
            # Images are already in [-1, 1] range (before denormalization)
            # Need to renormalize back
            images_normalized = images * 2 - 1
            outputs = cnn_model(images_normalized)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)

            print("\nClassification Results:")
            print("-" * 60)

            # Count predictions
            pred_counts = {}
            for i in range(len(predictions)):
                pred_class = predictions[i].item()
                confidence = confidences[i].item()
                fruit_name = fruit_names[pred_class]

                print(f"  Image {i+1:3d}: {fruit_name:15s} (confidence: {confidence*100:.1f}%)")

                if fruit_name not in pred_counts:
                    pred_counts[fruit_name] = 0
                pred_counts[fruit_name] += 1

            print("-" * 60)
            print("\nPrediction Distribution:")
            for fruit, count in sorted(pred_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {fruit:15s}: {count:3d} images ({count/len(predictions)*100:.1f}%)")

        return predictions, confidences, fruit_names

    except Exception as e:
        print(f"Error during classification: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate images using trained GAN')
    parser.add_argument('--model', type=str, default='GAN/models/best_model.pth',
                        help='Path to trained GAN model (default: best_model.pth)')
    parser.add_argument('--num', type=int, default=50,
                        help='Number of images to generate')
    parser.add_argument('--output', type=str, default='GAN/generated_images',
                        help='Output directory for generated images')
    parser.add_argument('--classify', action='store_true',
                        help='Classify generated images with CNN')

    args = parser.parse_args()

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("GAN IMAGE GENERATION")
    print("=" * 80)
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load generator
    print(f"\nLoading generator from: {args.model}")
    if not Path(args.model).exists():
        print(f"ERROR: Model file not found: {args.model}")
        print("Please train the GAN first using: python GAN/train_gan.py")
        sys.exit(1)

    generator, noise_dim = load_generator(args.model, device)
    print(f"Generator loaded successfully (noise_dim={noise_dim})")

    # Generate images
    print(f"\nGenerating {args.num} images...")
    fake_images = generate_images(generator, noise_dim, args.num, device, args.output)
    print(f"\n✅ Generated {args.num} images in {args.output}/")
    if args.num > 1:
        print(f"✅ Grid saved to {args.output}/grid.png")

    # Optional: Classify with CNN
    if args.classify:
        print("\n" + "=" * 80)
        print("CLASSIFYING GENERATED IMAGES WITH CNN")
        print("=" * 80)
        classify_with_cnn(fake_images, device)

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated images are 128x128 grayscale, compatible with CNN classifier")

if __name__ == "__main__":
    main()
