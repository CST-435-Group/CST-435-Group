"""
Generate Specific Image Types using Conditional GAN
YOU CONTROL what vehicle type to generate!

Usage:
  python GAN/generate_images_conditional.py --class tank --num 10
  python GAN/generate_images_conditional.py --class jet --num 5
  python GAN/generate_images_conditional.py --all-classes --num 3
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torchvision.utils import save_image
from pathlib import Path
import argparse

from model_utils import load_model_chunked

# --------------------------
# Conditional Generator (must match training)
# --------------------------
class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10, embed_dim=50):
        super(ConditionalGenerator, self).__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        combined_dim = noise_dim + embed_dim

        self.fc = nn.Linear(combined_dim, 13 * 13 * 512)
        self.bn0 = nn.BatchNorm1d(13 * 13 * 512)

        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, noise, labels):
        label_embed = self.label_embedding(labels)
        x = torch.cat([noise, label_embed], dim=1)

        x = self.fc(x)
        x = self.bn0(x)
        x = self.leaky_relu(x)
        x = x.view(-1, 512, 13, 13)

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
        x = self.tanh(x)

        return x

def load_conditional_generator(model_path, device):
    """Load trained conditional generator"""
    print(f"Loading conditional GAN from: {model_path}")
    checkpoint = load_model_chunked(model_path, device=str(device))

    # Get configuration
    noise_dim = checkpoint.get('noise_dim', 100)
    num_classes = checkpoint.get('num_classes', 8)
    class_names = checkpoint.get('class_names', [f"class_{i}" for i in range(num_classes)])
    image_size = checkpoint.get('image_size', 200)
    channels = checkpoint.get('channels', 3)

    print(f"\nModel configuration:")
    print(f"  Noise dim: {noise_dim}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Channels: {channels}")
    print(f"  Number of classes: {num_classes}")
    print(f"\nAvailable classes:")
    for i, class_name in enumerate(class_names):
        print(f"  [{i}] {class_name}")

    # Load generator
    generator = ConditionalGenerator(noise_dim, num_classes).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    return generator, noise_dim, num_classes, class_names

def generate_specific_class(generator, noise_dim, class_idx, num_images, device, output_dir, class_name):
    """Generate images of a specific class"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating {num_images} images of class: {class_name}")

    with torch.no_grad():
        noise = torch.randn(num_images, noise_dim, device=device)
        labels = torch.full((num_images,), class_idx, dtype=torch.long, device=device)

        fake_images = generator(noise, labels)
        fake_images = (fake_images + 1) / 2  # Denormalize

        # Save individual images
        for i in range(num_images):
            save_path = Path(output_dir) / f"{class_name}_{i:04d}.png"
            save_image(fake_images[i], save_path)

        # Save grid
        if num_images > 1:
            grid_size = min(8, num_images)
            save_image(fake_images[:min(64, num_images)],
                      Path(output_dir) / f'{class_name}_grid.png',
                      nrow=grid_size)

    return fake_images

def generate_all_classes(generator, noise_dim, num_classes, class_names, num_per_class, device, output_dir):
    """Generate images for all classes"""
    print(f"\nGenerating {num_per_class} images for each of {num_classes} classes...")

    all_images = []
    all_class_names = []

    for class_idx, class_name in enumerate(class_names):
        print(f"  Generating {class_name}...")

        with torch.no_grad():
            noise = torch.randn(num_per_class, noise_dim, device=device)
            labels = torch.full((num_per_class,), class_idx, dtype=torch.long, device=device)

            fake_images = generator(noise, labels)
            fake_images = (fake_images + 1) / 2

            all_images.append(fake_images)
            all_class_names.extend([class_name] * num_per_class)

            # Save individual class images
            class_output_dir = Path(output_dir) / class_name
            class_output_dir.mkdir(parents=True, exist_ok=True)

            for i in range(num_per_class):
                save_path = class_output_dir / f"{class_name}_{i:04d}.png"
                save_image(fake_images[i], save_path)

    # Create combined grid showing all classes
    all_images = torch.cat(all_images, dim=0)
    save_image(all_images, Path(output_dir) / 'all_classes_grid.png',
              nrow=num_per_class, normalize=False)

    print(f"\n✅ Generated grid showing all classes: {output_dir}/all_classes_grid.png")
    print(f"   Each row is a different class")

    return all_images

def main():
    parser = argparse.ArgumentParser(description='Generate specific image types using Conditional GAN')
    parser.add_argument('--model', type=str, default='GAN/models_conditional/best_model.pth',
                        help='Path to trained conditional GAN model')
    parser.add_argument('--class', type=str, dest='class_name', default=None,
                        help='Class name to generate (e.g., "tank", "jet", "ship")')
    parser.add_argument('--class-idx', type=int, default=None,
                        help='Class index to generate (e.g., 0, 1, 2)')
    parser.add_argument('--all-classes', action='store_true',
                        help='Generate images for all classes')
    parser.add_argument('--num', type=int, default=10,
                        help='Number of images to generate (per class if --all-classes)')
    parser.add_argument('--output', type=str, default='GAN/generated_images_conditional',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("CONDITIONAL GAN IMAGE GENERATION")
    print("CONTROL WHAT YOU GENERATE!")
    print("=" * 80)
    print(f"\nDevice: {device}")

    # Check model exists
    if not Path(args.model).exists():
        print(f"\n[ERROR] Model not found: {args.model}")
        print("\nTrain a conditional GAN first:")
        print("  python GAN/train_gan_conditional.py")
        sys.exit(1)

    # Load generator
    try:
        generator, noise_dim, num_classes, class_names = load_conditional_generator(args.model, device)
        print(f"\n✅ Conditional generator loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # Generate images
    if args.all_classes:
        # Generate for all classes
        generate_all_classes(generator, noise_dim, num_classes, class_names,
                           args.num, device, args.output)

        print("\n" + "=" * 80)
        print("GENERATION COMPLETE")
        print("=" * 80)
        print(f"\n✅ Generated {args.num} images per class ({num_classes} classes)")
        print(f"📁 Location: {args.output}/")
        print(f"🖼️  Combined grid: {args.output}/all_classes_grid.png")

    else:
        # Generate specific class
        if args.class_name is not None:
            # Find class by name
            class_name_lower = args.class_name.lower()
            class_idx = None

            for idx, name in enumerate(class_names):
                if class_name_lower in name.lower():
                    class_idx = idx
                    break

            if class_idx is None:
                print(f"\n[ERROR] Class '{args.class_name}' not found!")
                print(f"\nAvailable classes: {', '.join(class_names)}")
                sys.exit(1)

            class_name = class_names[class_idx]

        elif args.class_idx is not None:
            # Use class index directly
            if args.class_idx < 0 or args.class_idx >= num_classes:
                print(f"\n[ERROR] Class index {args.class_idx} out of range [0-{num_classes-1}]")
                sys.exit(1)

            class_idx = args.class_idx
            class_name = class_names[class_idx]

        else:
            # No class specified - show menu
            print("\n" + "=" * 80)
            print("SELECT A CLASS TO GENERATE")
            print("=" * 80)
            for idx, name in enumerate(class_names):
                print(f"  [{idx}] {name}")

            choice = input("\nEnter class number: ").strip()
            try:
                class_idx = int(choice)
                if class_idx < 0 or class_idx >= num_classes:
                    raise ValueError
                class_name = class_names[class_idx]
            except:
                print("[ERROR] Invalid choice")
                sys.exit(1)

        # Generate
        generate_specific_class(generator, noise_dim, class_idx, args.num,
                               device, args.output, class_name)

        print("\n" + "=" * 80)
        print("GENERATION COMPLETE")
        print("=" * 80)
        print(f"\n✅ Generated {args.num} images of class: {class_name}")
        print(f"📁 Location: {args.output}/")
        print(f"🖼️  Grid: {args.output}/{class_name}_grid.png")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
