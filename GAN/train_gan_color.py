"""
GAN Training Script for 200x200 RGB Color Images
CST-435 Neural Networks Assignment

Generates high-quality color images (military vehicles, custom datasets)
Uses PyTorch with GPU acceleration
"""

import os
import sys
import json
import time
from datetime import timedelta
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
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import model utilities for chunked saving/loading
from model_utils import save_model_chunked, load_model_chunked

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Configuration
IMAGE_SIZE = 200
CHANNELS = 3  # RGB
NOISE_DIM = 100
BATCH_SIZE = 32  # Smaller batch for larger images
NUM_EPOCHS = 200  # More epochs for better quality
LR = 0.0002

# Dataset path - UPDATE THIS
DATASET_PATH = 'GAN/datasets/military_vehicles_processed'

# Create directories
os.makedirs('GAN/generated_images_color', exist_ok=True)
os.makedirs('GAN/models_color', exist_ok=True)
os.makedirs('GAN/training_progress_color', exist_ok=True)

print("=" * 80)
print("GAN TRAINING FOR 200x200 RGB COLOR IMAGES")
print("=" * 80)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("WARNING: No GPU available, training will be VERY slow!")

# --------------------------
# Generator Architecture - 200x200 RGB
# --------------------------
class Generator(nn.Module):
    """
    Generator: Transforms random noise (100D) into 200x200 RGB images

    Architecture progression:
    [100] -> Dense -> [13x13x512] -> ConvTranspose -> [25x25x256] ->
    ConvTranspose -> [50x50x128] -> ConvTranspose -> [100x100x64] ->
    ConvTranspose -> [200x200x3]
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

# --------------------------
# Discriminator Architecture - 200x200 RGB
# --------------------------
class Discriminator(nn.Module):
    """
    Discriminator: Classifies 200x200 RGB images as real or fake

    Architecture progression:
    [200x200x3] -> Conv2D -> [100x100x64] -> Conv2D -> [50x50x128] ->
    Conv2D -> [25x25x256] -> Conv2D -> [13x13x512] -> Flatten -> [1]
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        # Downsampling blocks
        # 200x200x3 -> 100x100x64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.3)

        # 100x100x64 -> 50x50x128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.3)

        # 50x50x128 -> 25x25x256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.3)

        # 25x25x256 -> 13x13x512
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout4 = nn.Dropout(0.3)

        # Classifier
        self.fc = nn.Linear(13 * 13 * 512, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # Downsampling blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.dropout4(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# --------------------------
# Dataset for Color Images
# --------------------------
class ColorImageDataset(Dataset):
    """Load preprocessed 200x200 RGB images"""
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Collect all image paths
        self.image_paths = []
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']

        for ext in image_extensions:
            self.image_paths.extend(list(self.image_dir.glob(f'**/*{ext}')))

        print(f"Found {len(self.image_paths)} images in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')  # Ensure RGB

        if self.transform:
            img = self.transform(img)

        return img

# --------------------------
# Initialize Models
# --------------------------
print("\n[STEP 1] Initializing GAN models...")

generator = Generator(NOISE_DIM).to(device)
discriminator = Discriminator().to(device)

# Count parameters
gen_params = sum(p.numel() for p in generator.parameters())
disc_params = sum(p.numel() for p in discriminator.parameters())

print(f"\nGenerator:")
print(f"  Parameters: {gen_params:,}")
print(f"  Architecture: [100] -> [13x13x512] -> ... -> [200x200x3]")

print(f"\nDiscriminator:")
print(f"  Parameters: {disc_params:,}")
print(f"  Architecture: [200x200x3] -> ... -> [13x13x512] -> [1]")

# --------------------------
# Load Dataset
# --------------------------
print("\n[STEP 2] Loading dataset...")

if not Path(DATASET_PATH).exists():
    print(f"ERROR: {DATASET_PATH} does not exist!")
    print("\nPlease download and preprocess images first:")
    print("  1. python GAN/download_dataset.py")
    print("  2. python GAN/preprocess_dataset.py")
    sys.exit(1)

# Transform: Resize to 200x200, normalize to [-1, 1]
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # RGB normalization
])

dataset = ColorImageDataset(DATASET_PATH, transform=transform)

if len(dataset) == 0:
    print("ERROR: No images found in dataset!")
    sys.exit(1)

if len(dataset) < 200:
    print(f"WARNING: Dataset is small ({len(dataset)} images). Recommended: >500 images")

num_workers = 0 if os.name == 'nt' else 2
pin_memory = torch.cuda.is_available()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_memory)

print(f"Dataset size: {len(dataset)} images")
print(f"Batch size: {BATCH_SIZE}")
print(f"Batches per epoch: {len(dataloader)}")

# --------------------------
# Training Setup
# --------------------------
print("\n[STEP 3] Setting up training...")

criterion = nn.BCEWithLogitsLoss()
beta1 = 0.5
beta2 = 0.999

optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(beta1, beta2))

print(f"Loss: BCEWithLogitsLoss")
print(f"Optimizer: Adam (lr={LR}, beta1={beta1}, beta2={beta2})")
print(f"Epochs: {NUM_EPOCHS}")

# --------------------------
# Live Plotting Setup
# --------------------------
def update_live_plot(history, epoch, num_epochs):
    """Update live training plot"""
    try:
        plt.figure(1)
        plt.clf()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), num=1)

        epochs_range = range(1, len(history['D_loss']) + 1)

        ax1.plot(epochs_range, history['D_loss'], label='Discriminator Loss', linewidth=2, color='#e74c3c', marker='o', markersize=3)
        ax1.plot(epochs_range, history['G_loss'], label='Generator Loss', linewidth=2, color='#3498db', marker='o', markersize=3)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'Color GAN Training Losses (Epoch {epoch}/{num_epochs})', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs_range, history['D_real'], label='D(real) - target: ~1.0', linewidth=2, color='#27ae60', marker='o', markersize=3)
        ax2.plot(epochs_range, history['D_fake'], label='D(fake) - target: ~0.5', linewidth=2, color='#e67e22', marker='o', markersize=3)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Ideal: 0.5')
        ax2.axhline(y=1.0, color='lightgray', linestyle='--', alpha=0.5, label='Ideal: 1.0')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Discriminator Output', fontsize=12)
        ax2.set_title(f'Discriminator Performance (Epoch {epoch}/{num_epochs})', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.1])

        plt.tight_layout()
        plt.savefig('GAN/training_history_color_live.png', dpi=150, bbox_inches='tight')

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
print("\n[STEP 4] Starting training...")

fixed_noise = torch.randn(min(64, BATCH_SIZE*2), NOISE_DIM, device=device)

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

print("\n[INFO] Model saving strategy:")
print("  - Models saved to: GAN/models_color/")
print("  - latest_model.pth: Saved EVERY epoch")
print("  - best_model.pth: Best D(fake) score (closest to 0.5)")
print("  - checkpoint_epoch_XXX.pth: Every 10 epochs")
print("  - Models automatically chunked if >90MB")

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()

    generator.train()
    discriminator.train()

    epoch_D_loss = 0
    epoch_G_loss = 0
    epoch_D_real = 0
    epoch_D_fake = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for i, real_images in enumerate(pbar):
        batch_size_curr = real_images.size(0)
        real_images = real_images.to(device)

        real_labels = torch.ones(batch_size_curr, 1, device=device)
        fake_labels = torch.zeros(batch_size_curr, 1, device=device)

        # Train Discriminator
        optimizer_D.zero_grad()

        D_real_output = discriminator(real_images)
        D_real_loss = criterion(D_real_output, real_labels)

        noise = torch.randn(batch_size_curr, NOISE_DIM, device=device)
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

        pbar.set_postfix({
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

    # Time tracking
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

    # Update plot
    update_live_plot(history, epoch+1, NUM_EPOCHS)

    # Save models (with automatic chunking)
    model_data = {
        'epoch': epoch + 1,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'noise_dim': NOISE_DIM,
        'image_size': IMAGE_SIZE,
        'channels': CHANNELS,
        'history': history,
        'avg_D_loss': avg_D_loss,
        'avg_G_loss': avg_G_loss,
        'avg_D_real': avg_D_real,
        'avg_D_fake': avg_D_fake
    }

    # Save latest
    save_model_chunked(model_data, 'GAN/models_color/latest_model.pth')

    # Track best model
    d_fake_distance = abs(avg_D_fake - 0.5)

    if d_fake_distance < best_d_fake_score:
        best_d_fake_score = d_fake_distance
        best_epoch = epoch + 1

        save_model_chunked(model_data, 'GAN/models_color/best_model.pth')
        print(f"  [BEST] New best model! D(fake)={avg_D_fake:.3f} (distance: {d_fake_distance:.3f})")

    # Save progress images
    if (epoch + 1) % 5 == 0:
        generator.eval()
        with torch.no_grad():
            fake_images = generator(fixed_noise)
            fake_images = (fake_images + 1) / 2  # Denormalize
            save_image(fake_images, f'GAN/training_progress_color/epoch_{epoch+1:03d}.png', nrow=8)
        generator.train()
        print(f"  [SAVED] Progress images")

    # Save checkpoints
    if (epoch + 1) % 10 == 0:
        save_model_chunked(model_data, f'GAN/models_color/checkpoint_epoch_{epoch+1:03d}.pth')
        print(f"  [CHECKPOINT] Saved")

plt.ioff()

total_training_time = time.time() - training_start_time
total_time_str = str(timedelta(seconds=int(total_training_time)))

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nTotal Time: {total_time_str}")
print(f"Best Epoch: {best_epoch}")
print(f"Models saved to: GAN/models_color/")
print("\nGenerate images:")
print("  python GAN/generate_images.py --model GAN/models_color/best_model.pth --num 50")
