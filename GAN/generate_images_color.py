"""
Generate Color Images using Trained GAN
200x200 RGB Image Generation

NO CNN classifier dependency - standalone color image generator
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torchvision.utils import save_image
from pathlib import Path
import argparse

# Import model utilities
from model_utils import load_model_chunked

# --------------------------
# Generator Architecture - MUST MATCH TRAINING
# --------------------------
class Generator(nn.Module):
    """
    Generator: Transforms random noise (100D) into 200x200 RGB images
    """
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()

        # Initial dense layer: 100 -> 13x13x512
        self.fc = nn.Linear(noise_dim, 13 * 13 * 512)
        self.bn0 = nn.BatchNorm1d(13 * 13 * 512)

        # Upsampling blocks
        # 13x13x512 -> 25x25x256
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # 25x25x256 -> 50x50x128
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        # 50x50x128 -> 100x100x64
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        # 100x100x64 -> 200x200x3 (RGB)
        self.conv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Dense + reshape
        x = self.fc(x)
        x = self.bn0(x)
        x = self.leaky_relu(x)
        x = x.view(-1, 512, 13, 13)

        # Upsampling blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)

        x = self.conv4(x)
        x = self.tanh(x)  # Output range: [-1, 1]

        return x

def load_generator(model_path, device):
    """Load trained generator from checkpoint (supports chunked models)"""
    print(f"Loading model from: {model_path}")
    checkpoint = load_model_chunked(model_path, device=str(device))

    # Get configuration
    if 'noise_dim' in checkpoint:
        noise_dim = checkpoint['noise_dim']
    else:
        noise_dim = 100  # Default

    if 'image_size' in checkpoint:
        image_size = checkpoint['image_size']
    else:
        image_size = 200  # Default for color

    if 'channels' in checkpoint:
        channels = checkpoint['channels']
    else:
        channels = 3  # RGB

    print(f"Model configuration:")
    print(f"  Noise dim: {noise_dim}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Channels: {channels} ({'RGB' if channels == 3 else 'Grayscale'})")

    # Load generator
    generator = Generator(noise_dim).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    return generator, noise_dim, image_size, channels

def generate_images(generator, noise_dim, num_images, device, output_dir):
    """Generate images using the trained generator"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating {num_images} images...")

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
            save_image(fake_images[:min(64, num_images)],
                      Path(output_dir) / 'grid.png',
                      nrow=grid_size)

    return fake_images

def main():
    parser = argparse.ArgumentParser(description='Generate color images using trained GAN')
    parser.add_argument('--model', type=str, default='GAN/models_color/best_model.pth',
                        help='Path to trained GAN model (default: models_color/best_model.pth)')
    parser.add_argument('--num', type=int, default=50,
                        help='Number of images to generate')
    parser.add_argument('--output', type=str, default='GAN/generated_images_color',
                        help='Output directory for generated images')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("COLOR GAN IMAGE GENERATION")
    print("=" * 80)
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Check model exists
    if not Path(args.model).exists():
        print(f"\n[ERROR] Model file not found: {args.model}")
        print("\nAvailable model locations:")
        print("  - GAN/models_color/best_model.pth (recommended)")
        print("  - GAN/models_color/latest_model.pth")
        print("  - GAN/models_color/checkpoint_epoch_XXX.pth")
        print("\nTrain a model first:")
        print("  python GAN/train_gan_color.py")
        sys.exit(1)

    # Load generator
    print(f"\nLoading generator...")
    try:
        generator, noise_dim, image_size, channels = load_generator(args.model, device)
        print(f"✅ Generator loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # Generate images
    fake_images = generate_images(generator, noise_dim, args.num, device, args.output)

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\n✅ Generated {args.num} color images")
    print(f"📁 Location: {args.output}/")
    if args.num > 1:
        print(f"🖼️  Grid: {args.output}/grid.png")

    print(f"\nGenerated images are {image_size}x{image_size} RGB color images")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
