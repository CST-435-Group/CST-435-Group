"""
GAN Training Script for SINGLE Fruit Type (128x128 Grayscale)
CST-435 Neural Networks Assignment

Trains GAN to generate ONE specific fruit type
Much better results than training on all fruits at once!

Usage:
    python train_gan_single_fruit.py --fruit Apple --epochs 50
    python train_gan_single_fruit.py --fruit Pear --epochs 100
"""

import os
import sys
import json
import time
from datetime import timedelta
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib
# Try to use interactive backend, fall back to Agg if not available
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add CNN_Project to path for model loading
sys.path.append(str(Path(__file__).parent.parent / 'CNN_Project'))

# Import GAN models from main training script
from train_gan import Generator, Discriminator

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Parse arguments
parser = argparse.ArgumentParser(description='Train GAN for specific fruit type')
parser.add_argument('--fruit', type=str, required=True,
                   choices=['Pear', 'Apple', 'Tomato', 'Peach', 'Cucumber'],
                   help='Fruit type to generate (Pear, Apple, Tomato, Peach, or Cucumber)')
parser.add_argument('--epochs', type=int, default=100,
                   help='Number of training epochs (default: 100)')
parser.add_argument('--batch-size', type=int, default=64,
                   help='Batch size (default: 64)')
parser.add_argument('--lr', type=float, default=0.0002,
                   help='Learning rate (default: 0.0002)')

args = parser.parse_args()

FRUIT_TYPE = args.fruit
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr

# Create fruit-specific directories
FRUIT_DIR = f'GAN/models/{FRUIT_TYPE}'
os.makedirs(f'{FRUIT_DIR}', exist_ok=True)
os.makedirs(f'GAN/generated_images/{FRUIT_TYPE}', exist_ok=True)
os.makedirs(f'GAN/training_progress/{FRUIT_TYPE}', exist_ok=True)

print("=" * 80)
print(f"GAN TRAINING FOR {FRUIT_TYPE.upper()} (128x128 Grayscale)")
print("=" * 80)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("WARNING: No GPU available, training will be slow!")

# --------------------------
# Load Images for Specific Fruit
# --------------------------
class SingleFruitDataset(Dataset):
    """Load images for ONE specific fruit type only"""
    def __init__(self, fruit_name, image_dir, transform=None):
        self.fruit_name = fruit_name
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Find all subdirectories matching the fruit name
        self.image_paths = []
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']

        # Search for folders matching fruit name
        for folder in self.image_dir.iterdir():
            if folder.is_dir() and fruit_name.lower() in folder.name.lower():
                for ext in image_extensions:
                    self.image_paths.extend(list(folder.glob(f'**/*{ext}')))

        print(f"\n[INFO] Loading {fruit_name} images...")
        print(f"  Found {len(self.image_paths)} {fruit_name} images")

        if len(self.image_paths) == 0:
            print(f"\n[ERROR] No {fruit_name} images found!")
            print(f"  Looking in: {image_dir}")
            print(f"  Available folders:")
            for folder in self.image_dir.iterdir():
                if folder.is_dir():
                    print(f"    - {folder.name}")
            sys.exit(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)

        return img

print(f"\n[STEP 1] Loading {FRUIT_TYPE} images from CNN_Project...")

CNN_PROJECT_DIR = Path(__file__).parent.parent / 'CNN_Project'
PREPROCESSED_IMAGES_DIR = CNN_PROJECT_DIR / 'preprocessed_images'

if not PREPROCESSED_IMAGES_DIR.exists():
    print(f"ERROR: {PREPROCESSED_IMAGES_DIR} does not exist!")
    print("Please run CNN_Project/train_model.py first.")
    sys.exit(1)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = SingleFruitDataset(FRUIT_TYPE, PREPROCESSED_IMAGES_DIR, transform=transform)

num_workers = 0 if os.name == 'nt' else 2
pin_memory = torch.cuda.is_available()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_memory)

print(f"  Dataset size: {len(dataset)} images")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Batches per epoch: {len(dataloader)}")

# --------------------------
# Initialize Models
# --------------------------
print(f"\n[STEP 2] Initializing GAN models for {FRUIT_TYPE}...")

noise_dim = 100
generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)

gen_params = sum(p.numel() for p in generator.parameters())
disc_params = sum(p.numel() for p in discriminator.parameters())

print(f"\nGenerator: {gen_params:,} parameters")
print(f"Discriminator: {disc_params:,} parameters")

# --------------------------
# Training Setup
# --------------------------
print(f"\n[STEP 3] Setting up training...")

criterion = nn.BCEWithLogitsLoss()
beta1 = 0.5
beta2 = 0.999

optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(beta1, beta2))

print(f"Loss: BCEWithLogitsLoss")
print(f"Optimizer: Adam (lr={LR}, beta1={beta1}, beta2={beta2})")
print(f"Epochs: {NUM_EPOCHS}")

# --------------------------
# Live Plotting Function
# --------------------------
def update_live_plot(history, epoch, num_epochs, fruit_name):
    """Update live training plot"""
    try:
        plt.figure(1)
        plt.clf()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), num=1)

        epochs_range = range(1, len(history['D_loss']) + 1)

        # Plot losses
        ax1.plot(epochs_range, history['D_loss'], label='Discriminator Loss',
                linewidth=2, color='#e74c3c', marker='o', markersize=3)
        ax1.plot(epochs_range, history['G_loss'], label='Generator Loss',
                linewidth=2, color='#3498db', marker='o', markersize=3)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'{fruit_name} GAN Training Losses (Epoch {epoch}/{num_epochs})',
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot discriminator outputs
        ax2.plot(epochs_range, history['D_real'], label='D(real) - target: ~1.0',
                linewidth=2, color='#27ae60', marker='o', markersize=3)
        ax2.plot(epochs_range, history['D_fake'], label='D(fake) - target: ~0.5',
                linewidth=2, color='#e67e22', marker='o', markersize=3)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Ideal: 0.5')
        ax2.axhline(y=1.0, color='lightgray', linestyle='--', alpha=0.5, label='Ideal: 1.0')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Discriminator Output', fontsize=12)
        ax2.set_title(f'{fruit_name} Discriminator Performance (Epoch {epoch}/{num_epochs})',
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.1])

        plt.tight_layout()
        plt.savefig(f'{FRUIT_DIR}/training_history_live.png', dpi=150, bbox_inches='tight')

        try:
            plt.pause(0.001)
            plt.show(block=False)
        except:
            pass
    except:
        pass

# --------------------------
# Training Loop
# --------------------------
print(f"\n[STEP 4] Training {FRUIT_TYPE} GAN...")

fixed_noise = torch.randn(64, noise_dim, device=device)

history = {
    'D_loss': [],
    'G_loss': [],
    'D_real': [],
    'D_fake': []
}

plt.ion()
fig = plt.figure(figsize=(14, 5))

print(f"\nTraining for {NUM_EPOCHS} epochs...")
print("=" * 80)

training_start_time = time.time()
epoch_times = []

best_d_fake_score = float('inf')
best_epoch = 0

print(f"\n[INFO] Model saving strategy:")
print(f"  - Models saved to: {FRUIT_DIR}/")
print(f"  - latest_model.pth: Saved EVERY epoch")
print(f"  - best_model.pth: Best D(fake) score (closest to 0.5)")
print(f"  - checkpoint_epoch_XXX.pth: Every 10 epochs")

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()

    generator.train()
    discriminator.train()

    epoch_D_loss = 0
    epoch_G_loss = 0
    epoch_D_real = 0
    epoch_D_fake = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for i, real_images in enumerate(pbar):
        batch_size_curr = real_images.size(0)
        real_images = real_images.to(device)

        real_labels = torch.ones(batch_size_curr, 1, device=device)
        fake_labels = torch.zeros(batch_size_curr, 1, device=device)

        # Train Discriminator
        optimizer_D.zero_grad()

        D_real_output = discriminator(real_images)
        D_real_loss = criterion(D_real_output, real_labels)

        noise = torch.randn(batch_size_curr, noise_dim, device=device)
        fake_images = generator(noise)
        D_fake_output = discriminator(fake_images.detach())
        D_fake_loss = criterion(D_fake_output, fake_labels)

        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        D_fake_output = discriminator(fake_images)
        G_loss = criterion(D_fake_output, real_labels)

        G_loss.backward()
        optimizer_G.step()

        # Track metrics
        epoch_D_loss += D_loss.item()
        epoch_G_loss += G_loss.item()
        epoch_D_real += torch.sigmoid(D_real_output).mean().item()
        epoch_D_fake += torch.sigmoid(D_fake_output).mean().item()

        batch_progress = ((i + 1) / len(dataloader)) * 100

        pbar.set_postfix({
            'Progress': f"{batch_progress:.1f}%",
            'D_loss': f"{D_loss.item():.4f}",
            'G_loss': f"{G_loss.item():.4f}",
            'D(fake)': f"{torch.sigmoid(D_fake_output).mean().item():.3f}"
        })

    # Average metrics
    avg_D_loss = epoch_D_loss / len(dataloader)
    avg_G_loss = epoch_G_loss / len(dataloader)
    avg_D_real = epoch_D_real / len(dataloader)
    avg_D_fake = epoch_D_fake / len(dataloader)

    history['D_loss'].append(avg_D_loss)
    history['G_loss'].append(avg_G_loss)
    history['D_real'].append(avg_D_real)
    history['D_fake'].append(avg_D_fake)

    # Calculate time stats
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    remaining_epochs = NUM_EPOCHS - (epoch + 1)
    eta_seconds = avg_epoch_time * remaining_epochs
    eta = str(timedelta(seconds=int(eta_seconds)))

    elapsed_time = time.time() - training_start_time
    elapsed = str(timedelta(seconds=int(elapsed_time)))

    progress_pct = ((epoch + 1) / NUM_EPOCHS) * 100

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} ({progress_pct:.1f}% complete)")
    print(f"  D_loss: {avg_D_loss:.4f} | G_loss: {avg_G_loss:.4f}")
    print(f"  D(real): {avg_D_real:.3f} | D(fake): {avg_D_fake:.3f}")
    print(f"  Epoch Time: {epoch_time:.1f}s | Avg: {avg_epoch_time:.1f}s/epoch")
    print(f"  Elapsed: {elapsed} | ETA: {eta}")

    # Update live plot
    update_live_plot(history, epoch+1, NUM_EPOCHS, FRUIT_TYPE)

    # Save models
    model_data = {
        'epoch': epoch + 1,
        'fruit_type': FRUIT_TYPE,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'noise_dim': noise_dim,
        'history': history,
        'avg_D_loss': avg_D_loss,
        'avg_G_loss': avg_G_loss,
        'avg_D_real': avg_D_real,
        'avg_D_fake': avg_D_fake
    }

    # Save latest
    torch.save(model_data, f'{FRUIT_DIR}/latest_model.pth')

    metadata = {
        'epoch': epoch + 1,
        'fruit_type': FRUIT_TYPE,
        'noise_dim': noise_dim,
        'avg_D_loss': float(avg_D_loss),
        'avg_G_loss': float(avg_G_loss),
        'avg_D_real': float(avg_D_real),
        'avg_D_fake': float(avg_D_fake),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(f'{FRUIT_DIR}/latest_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Track best model
    d_fake_distance = abs(avg_D_fake - 0.5)

    if d_fake_distance < best_d_fake_score:
        best_d_fake_score = d_fake_distance
        best_epoch = epoch + 1

        torch.save(model_data, f'{FRUIT_DIR}/best_model.pth')

        best_metadata = metadata.copy()
        best_metadata['best_epoch'] = best_epoch
        best_metadata['best_d_fake_score'] = float(best_d_fake_score)
        with open(f'{FRUIT_DIR}/best_metadata.json', 'w') as f:
            json.dump(best_metadata, f, indent=2)

        print(f"  [BEST] New best model! D(fake)={avg_D_fake:.3f} (distance: {d_fake_distance:.3f})")

    # Save progress images
    if (epoch + 1) % 5 == 0:
        generator.eval()
        with torch.no_grad():
            fake_images = generator(fixed_noise)
            fake_images = (fake_images + 1) / 2
            save_image(fake_images, f'GAN/training_progress/{FRUIT_TYPE}/epoch_{epoch+1:03d}.png', nrow=8)
        generator.train()
        print(f"  [SAVED] Progress images to GAN/training_progress/{FRUIT_TYPE}/epoch_{epoch+1:03d}.png")

    # Save checkpoints
    if (epoch + 1) % 10 == 0:
        torch.save(model_data, f'{FRUIT_DIR}/checkpoint_epoch_{epoch+1:03d}.pth')
        print(f"  [CHECKPOINT] Saved to {FRUIT_DIR}/checkpoint_epoch_{epoch+1:03d}.pth")

plt.ioff()

total_training_time = time.time() - training_start_time
total_time_str = str(timedelta(seconds=int(total_training_time)))

print("\n" + "=" * 80)
print(f"{FRUIT_TYPE.upper()} GAN TRAINING COMPLETE!")
print("=" * 80)
print(f"\nTotal Time: {total_time_str}")
print(f"Best Epoch: {best_epoch}")
print(f"Best D(fake) score: {best_d_fake_score:.4f}")

# Generate sample images
print(f"\n[STEP 5] Generating {FRUIT_TYPE} sample images...")

generator.eval()
num_samples = 100

with torch.no_grad():
    noise = torch.randn(num_samples, noise_dim, device=device)
    generated_images = generator(noise)
    generated_images = (generated_images + 1) / 2

    for i in range(num_samples):
        save_image(generated_images[i], f'GAN/generated_images/{FRUIT_TYPE}/sample_{i:03d}.png')

    save_image(generated_images[:64], f'GAN/generated_images/{FRUIT_TYPE}/grid_64.png', nrow=8)

print(f"Generated {num_samples} {FRUIT_TYPE} images in GAN/generated_images/{FRUIT_TYPE}/")

# Test with CNN
print(f"\n[STEP 6] Testing {FRUIT_TYPE} images with CNN classifier...")

try:
    CNN_MODEL_PATH = CNN_PROJECT_DIR / 'models' / 'best_model.pth'

    if CNN_MODEL_PATH.exists():
        from train_model import FruitCNN

        with open(CNN_PROJECT_DIR / 'models' / 'model_metadata.json', 'r') as f:
            cnn_metadata = json.load(f)

        fruit_names = cnn_metadata['fruit_names']
        num_classes = cnn_metadata['num_classes']

        # Find target fruit index
        if FRUIT_TYPE in fruit_names:
            target_idx = fruit_names.index(FRUIT_TYPE)

            cnn_model = FruitCNN(num_classes=num_classes).to(device)
            checkpoint = torch.load(CNN_MODEL_PATH, map_location=device)
            cnn_model.load_state_dict(checkpoint['model_state_dict'])
            cnn_model.eval()

            print(f"\nCNN Classifier loaded")
            print(f"Target fruit: {FRUIT_TYPE} (class {target_idx})")

            # Classify generated images
            with torch.no_grad():
                noise = torch.randn(50, noise_dim, device=device)
                fake_images = generator(noise)

                outputs = cnn_model(fake_images)
                _, predictions = torch.max(outputs, 1)

                # Count how many predicted as target fruit
                correct = (predictions == target_idx).sum().item()
                total = len(predictions)
                accuracy = (correct / total) * 100

                print(f"\n{FRUIT_TYPE} Generation Accuracy:")
                print(f"  {correct}/{total} images classified as {FRUIT_TYPE} ({accuracy:.1f}%)")

                # Show sample predictions
                print(f"\nSample predictions:")
                for i in range(min(10, len(predictions))):
                    pred_class = predictions[i].item()
                    is_correct = "✅" if pred_class == target_idx else "❌"
                    print(f"  Image {i+1}: {fruit_names[pred_class]} {is_correct}")
        else:
            print(f"Warning: {FRUIT_TYPE} not in CNN training classes: {fruit_names}")

except Exception as e:
    print(f"Could not test with CNN: {e}")

print("\n" + "=" * 80)
print(f"{FRUIT_TYPE.upper()} GAN COMPLETE - SUMMARY")
print("=" * 80)
print(f"\nModels saved to: {FRUIT_DIR}/")
print(f"  - best_model.pth (epoch {best_epoch})")
print(f"  - latest_model.pth (epoch {NUM_EPOCHS})")
print(f"\nGenerated images: GAN/generated_images/{FRUIT_TYPE}/")
print(f"Training progress: GAN/training_progress/{FRUIT_TYPE}/")

print(f"\n✅ {FRUIT_TYPE} GAN training complete!")
print("\n" + "=" * 80)
