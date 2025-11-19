"""
GAN Training Script for 128x128 Grayscale Fruit Images
CST-435 Neural Networks Assignment

Generates synthetic fruit images compatible with the CNN classifier from CNN_Project
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
from torchvision.utils import save_image, make_grid
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

# Import model utilities for chunked saving/loading
from model_utils import save_model_chunked, load_model_chunked

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Create directories
os.makedirs('GAN/generated_images', exist_ok=True)
os.makedirs('GAN/models', exist_ok=True)
os.makedirs('GAN/training_progress', exist_ok=True)

print("=" * 80)
print("GAN TRAINING FOR 128x128 GRAYSCALE FRUIT IMAGES")
print("=" * 80)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("WARNING: No GPU available, training will be slow!")

# --------------------------
# Generator Architecture
# --------------------------
class Generator(nn.Module):
    """
    Generator: Transforms random noise (100D) into 128x128 grayscale images

    Architecture progression:
    [100] -> Dense -> [8x8x512] -> Conv2DTranspose -> [16x16x256] ->
    Conv2DTranspose -> [32x32x128] -> Conv2DTranspose -> [64x64x64] ->
    Conv2DTranspose -> [128x128x1]
    """
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()

        # Initial dense layer: 100 -> 8x8x512
        self.fc = nn.Linear(noise_dim, 8 * 8 * 512)
        self.bn0 = nn.BatchNorm1d(8 * 8 * 512)

        # Upsampling blocks
        # 8x8x512 -> 16x16x256
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # 16x16x256 -> 32x32x128
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        # 32x32x128 -> 64x64x64
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        # 64x64x64 -> 128x128x1
        self.conv4 = nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Dense + reshape
        x = self.fc(x)
        x = self.bn0(x)
        x = self.leaky_relu(x)
        x = x.view(-1, 512, 8, 8)

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
# Discriminator Architecture
# --------------------------
class Discriminator(nn.Module):
    """
    Discriminator: Classifies 128x128 images as real or fake

    Architecture progression:
    [128x128x1] -> Conv2D -> [64x64x64] -> Conv2D -> [32x32x128] ->
    Conv2D -> [16x16x256] -> Conv2D -> [8x8x512] -> Flatten -> [1]
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        # Downsampling blocks
        # 128x128x1 -> 64x64x64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2)
        self.dropout1 = nn.Dropout(0.3)

        # 64x64x64 -> 32x32x128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.dropout2 = nn.Dropout(0.3)

        # 32x32x128 -> 16x16x256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.dropout3 = nn.Dropout(0.3)

        # 16x16x256 -> 8x8x512
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.dropout4 = nn.Dropout(0.3)

        # Classifier
        self.fc = nn.Linear(8 * 8 * 512, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # Downsampling blocks
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.dropout4(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x  # Output: logit (no sigmoid, using BCEWithLogitsLoss)

# --------------------------
# Dataset for CNN Project Images
# --------------------------
class FruitImageDataset(Dataset):
    """Load preprocessed fruit images from CNN_Project"""
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
        img = Image.open(img_path).convert('L')  # Grayscale

        if self.transform:
            img = self.transform(img)

        return img

# --------------------------
# Initialize Models
# --------------------------
print("\n[STEP 1] Initializing GAN models...")

noise_dim = 100
generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)

# Count parameters
gen_params = sum(p.numel() for p in generator.parameters())
disc_params = sum(p.numel() for p in discriminator.parameters())

print(f"\nGenerator:")
print(f"  Parameters: {gen_params:,}")
print(f"  Architecture: [100] -> [8x8x512] -> ... -> [128x128x1]")

print(f"\nDiscriminator:")
print(f"  Parameters: {disc_params:,}")
print(f"  Architecture: [128x128x1] -> ... -> [8x8x512] -> [1]")

# --------------------------
# Load Real Images from CNN_Project
# --------------------------
print("\n[STEP 2] Loading real images from CNN_Project...")

CNN_PROJECT_DIR = Path(__file__).parent.parent / 'CNN_Project'
PREPROCESSED_IMAGES_DIR = CNN_PROJECT_DIR / 'preprocessed_images'

if not PREPROCESSED_IMAGES_DIR.exists():
    print(f"ERROR: {PREPROCESSED_IMAGES_DIR} does not exist!")
    print("Please run CNN_Project/train_model.py first to preprocess images.")
    sys.exit(1)

# Transform: images are already 128x128 grayscale, just normalize to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # [0, 1] -> [-1, 1]
])

dataset = FruitImageDataset(PREPROCESSED_IMAGES_DIR, transform=transform)

if len(dataset) == 0:
    print("ERROR: No images found in preprocessed_images!")
    print("Please run CNN_Project/train_model.py first.")
    sys.exit(1)

batch_size = 64
# Set num_workers=0 for Windows to avoid multiprocessing issues
num_workers = 0 if os.name == 'nt' else 2
pin_memory = torch.cuda.is_available()  # Only use pin_memory if CUDA available
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

print(f"Dataset size: {len(dataset)} images")
print(f"Batch size: {batch_size}")
print(f"Batches per epoch: {len(dataloader)}")

# --------------------------
# Training Setup
# --------------------------
print("\n[STEP 3] Setting up training...")

# Loss and optimizers
criterion = nn.BCEWithLogitsLoss()
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

print(f"Loss: BCEWithLogitsLoss")
print(f"Optimizer: Adam (lr={lr}, beta1={beta1}, beta2={beta2})")

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

        # Plot losses
        ax1.plot(epochs_range, history['D_loss'], label='Discriminator Loss', linewidth=2, color='#e74c3c', marker='o', markersize=3)
        ax1.plot(epochs_range, history['G_loss'], label='Generator Loss', linewidth=2, color='#3498db', marker='o', markersize=3)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'GAN Training Losses (Epoch {epoch}/{num_epochs})', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot discriminator outputs
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
        plt.savefig('GAN/training_history_live.png', dpi=150, bbox_inches='tight')

        # Try to show plot if interactive mode works
        try:
            plt.pause(0.001)
            plt.show(block=False)
        except:
            pass
    except Exception as e:
        pass  # Skip if display not available

# --------------------------
# Training Loop
# --------------------------
print("\n[STEP 4] Starting training...")

num_epochs = 100
fixed_noise = torch.randn(64, noise_dim, device=device)  # For tracking progress

history = {
    'D_loss': [],
    'G_loss': [],
    'D_real': [],
    'D_fake': []
}

# Setup live plotting
plt.ion()  # Turn on interactive mode
fig = plt.figure(figsize=(14, 5))

print(f"\nTraining for {num_epochs} epochs...")
print("=" * 80)

# Time tracking
training_start_time = time.time()
epoch_times = []

# Best model tracking (based on D(fake) being close to 0.5)
best_d_fake_score = float('inf')  # Distance from 0.5
best_epoch = 0

print("\n[INFO] Model saving strategy:")
print("  - latest_model.pth: Saved EVERY epoch (always usable)")
print("  - best_model.pth: Saved when D(fake) closest to 0.5 (best quality)")
print("  - checkpoint_epoch_XXX.pth: Saved every 10 epochs (for resume)")
print("  All models are immediately usable with generate_images.py")

for epoch in range(num_epochs):
    epoch_start_time = time.time()

    generator.train()
    discriminator.train()

    epoch_D_loss = 0
    epoch_G_loss = 0
    epoch_D_real = 0
    epoch_D_fake = 0

    # Progress bar with percentage
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for i, real_images in enumerate(pbar):
        batch_size_curr = real_images.size(0)
        real_images = real_images.to(device)

        # Labels
        real_labels = torch.ones(batch_size_curr, 1, device=device)
        fake_labels = torch.zeros(batch_size_curr, 1, device=device)

        # --------------------------
        # Train Discriminator
        # --------------------------
        optimizer_D.zero_grad()

        # Real images
        D_real_output = discriminator(real_images)
        D_real_loss = criterion(D_real_output, real_labels)

        # Fake images
        noise = torch.randn(batch_size_curr, noise_dim, device=device)
        fake_images = generator(noise)
        D_fake_output = discriminator(fake_images.detach())
        D_fake_loss = criterion(D_fake_output, fake_labels)

        # Total discriminator loss
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        optimizer_D.step()

        # --------------------------
        # Train Generator
        # --------------------------
        optimizer_G.zero_grad()

        # Generate fake images and try to fool discriminator
        D_fake_output = discriminator(fake_images)
        G_loss = criterion(D_fake_output, real_labels)  # Want discriminator to think they're real

        G_loss.backward()
        optimizer_G.step()

        # Track metrics
        epoch_D_loss += D_loss.item()
        epoch_G_loss += G_loss.item()
        epoch_D_real += torch.sigmoid(D_real_output).mean().item()
        epoch_D_fake += torch.sigmoid(D_fake_output).mean().item()

        # Calculate batch progress percentage
        batch_progress = ((i + 1) / len(dataloader)) * 100

        pbar.set_postfix({
            'Progress': f"{batch_progress:.1f}%",
            'D_loss': f"{D_loss.item():.4f}",
            'G_loss': f"{G_loss.item():.4f}",
            'D_real': f"{torch.sigmoid(D_real_output).mean().item():.3f}",
            'D_fake': f"{torch.sigmoid(D_fake_output).mean().item():.3f}"
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

    # Calculate epoch time and ETA
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    remaining_epochs = num_epochs - (epoch + 1)
    eta_seconds = avg_epoch_time * remaining_epochs
    eta = str(timedelta(seconds=int(eta_seconds)))

    elapsed_time = time.time() - training_start_time
    elapsed = str(timedelta(seconds=int(elapsed_time)))

    progress_pct = ((epoch + 1) / num_epochs) * 100

    print(f"\nEpoch {epoch+1}/{num_epochs} ({progress_pct:.1f}% complete)")
    print(f"  D_loss: {avg_D_loss:.4f} | G_loss: {avg_G_loss:.4f}")
    print(f"  D(real): {avg_D_real:.3f} | D(fake): {avg_D_fake:.3f}")
    print(f"  Epoch Time: {epoch_time:.1f}s | Avg: {avg_epoch_time:.1f}s/epoch")
    print(f"  Elapsed: {elapsed} | ETA: {eta}")

    # Update live plot
    update_live_plot(history, epoch+1, num_epochs)

    # --------------------------
    # Save Latest Model (EVERY EPOCH - Always Usable!)
    # --------------------------
    model_data = {
        'epoch': epoch + 1,
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

    # Save latest model (with automatic chunking if needed)
    save_model_chunked(model_data, 'GAN/models/latest_model.pth')

    # Save metadata for easy inspection
    metadata = {
        'epoch': epoch + 1,
        'noise_dim': noise_dim,
        'avg_D_loss': float(avg_D_loss),
        'avg_G_loss': float(avg_G_loss),
        'avg_D_real': float(avg_D_real),
        'avg_D_fake': float(avg_D_fake),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open('GAN/models/latest_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # --------------------------
    # Track and Save Best Model
    # --------------------------
    # Best model = D(fake) closest to 0.5 (generator fooling discriminator)
    d_fake_distance = abs(avg_D_fake - 0.5)

    if d_fake_distance < best_d_fake_score:
        best_d_fake_score = d_fake_distance
        best_epoch = epoch + 1

        # Save best model (with automatic chunking if needed)
        save_model_chunked(model_data, 'GAN/models/best_model.pth')

        # Save best metadata
        best_metadata = metadata.copy()
        best_metadata['best_epoch'] = best_epoch
        best_metadata['best_d_fake_score'] = float(best_d_fake_score)
        with open('GAN/models/best_metadata.json', 'w') as f:
            json.dump(best_metadata, f, indent=2)

        print(f"  [BEST] New best model! D(fake)={avg_D_fake:.3f} (distance from 0.5: {d_fake_distance:.3f})")

    # Save generated images every 5 epochs
    if (epoch + 1) % 5 == 0:
        generator.eval()
        with torch.no_grad():
            fake_images = generator(fixed_noise)
            # Denormalize from [-1, 1] to [0, 1]
            fake_images = (fake_images + 1) / 2
            save_image(fake_images, f'GAN/training_progress/epoch_{epoch+1:03d}.png', nrow=8)
        generator.train()
        print(f"  [SAVED] Generated images to GAN/training_progress/epoch_{epoch+1:03d}.png")

    # Save numbered checkpoints every 10 epochs (for resume training)
    if (epoch + 1) % 10 == 0:
        save_model_chunked(model_data, f'GAN/models/checkpoint_epoch_{epoch+1:03d}.pth')

        checkpoint_metadata = metadata.copy()
        with open(f'GAN/models/checkpoint_epoch_{epoch+1:03d}_metadata.json', 'w') as f:
            json.dump(checkpoint_metadata, f, indent=2)

        print(f"  [CHECKPOINT] Saved to GAN/models/checkpoint_epoch_{epoch+1:03d}.pth")

plt.ioff()  # Turn off interactive mode

total_training_time = time.time() - training_start_time
total_time_str = str(timedelta(seconds=int(total_training_time)))

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nTotal Training Time: {total_time_str}")
print(f"Average Time per Epoch: {avg_epoch_time:.1f}s")

# --------------------------
# Save Final Models and Summary
# --------------------------
print("\n[STEP 5] Saving final models and summary...")

# Final model (same format as others for consistency)
final_model_data = {
    'epoch': num_epochs,
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
    'noise_dim': noise_dim,
    'history': history,
    'avg_D_loss': history['D_loss'][-1],
    'avg_G_loss': history['G_loss'][-1],
    'avg_D_real': history['D_real'][-1],
    'avg_D_fake': history['D_fake'][-1],
    'total_training_time': total_training_time,
    'best_epoch': best_epoch,
    'best_d_fake_score': best_d_fake_score
}

save_model_chunked(final_model_data, 'GAN/models/final_gan.pth')

# Final metadata
final_metadata = {
    'epoch': num_epochs,
    'noise_dim': noise_dim,
    'total_epochs': num_epochs,
    'total_training_time': total_time_str,
    'avg_epoch_time': float(avg_epoch_time),
    'best_epoch': best_epoch,
    'best_d_fake_score': float(best_d_fake_score),
    'final_D_loss': float(history['D_loss'][-1]),
    'final_G_loss': float(history['G_loss'][-1]),
    'final_D_real': float(history['D_real'][-1]),
    'final_D_fake': float(history['D_fake'][-1]),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
}
with open('GAN/models/final_metadata.json', 'w') as f:
    json.dump(final_metadata, f, indent=2)

print("Saved: GAN/models/final_gan.pth")

# Create models index file
models_index = {
    'training_completed': True,
    'total_epochs': num_epochs,
    'total_training_time': total_time_str,
    'best_epoch': best_epoch,
    'best_d_fake_score': float(best_d_fake_score),
    'available_models': {
        'best_model.pth': {
            'description': 'Best quality model (D(fake) closest to 0.5)',
            'recommended': True,
            'epoch': best_epoch,
            'usable_with_generate_images': True
        },
        'latest_model.pth': {
            'description': 'Most recent model from last completed epoch',
            'recommended': False,
            'epoch': num_epochs,
            'usable_with_generate_images': True
        },
        'final_gan.pth': {
            'description': 'Final model after all epochs',
            'recommended': False,
            'epoch': num_epochs,
            'usable_with_generate_images': True
        }
    },
    'checkpoints': [f'checkpoint_epoch_{e:03d}.pth' for e in range(10, num_epochs+1, 10)],
    'usage': 'python GAN/generate_images.py --model GAN/models/best_model.pth --num 50'
}
with open('GAN/models/models_index.json', 'w') as f:
    json.dump(models_index, f, indent=2)

print("Saved: GAN/models/models_index.json (summary of all available models)")

# --------------------------
# Generate Sample Images
# --------------------------
print("\n[STEP 6] Generating sample images...")

generator.eval()
num_samples = 100

with torch.no_grad():
    noise = torch.randn(num_samples, noise_dim, device=device)
    generated_images = generator(noise)
    # Denormalize
    generated_images = (generated_images + 1) / 2

    # Save individual images
    for i in range(num_samples):
        save_image(generated_images[i], f'GAN/generated_images/sample_{i:03d}.png')

    # Save grid
    save_image(generated_images[:64], 'GAN/generated_images/grid_64.png', nrow=8)

print(f"Generated {num_samples} sample images in GAN/generated_images/")
print("Saved grid: GAN/generated_images/grid_64.png")

# --------------------------
# Plot Training History
# --------------------------
print("\n[STEP 7] Plotting training history...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot losses
ax1.plot(history['D_loss'], label='Discriminator Loss', linewidth=2, color='#e74c3c')
ax1.plot(history['G_loss'], label='Generator Loss', linewidth=2, color='#3498db')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('GAN Training Losses', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot discriminator outputs
ax2.plot(history['D_real'], label='D(real) - should be ~1', linewidth=2, color='#27ae60')
ax2.plot(history['D_fake'], label='D(fake) - should be ~0.5', linewidth=2, color='#e67e22')
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Target: 0.5')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Discriminator Output', fontsize=12)
ax2.set_title('Discriminator Performance', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('GAN/training_history.png', dpi=300, bbox_inches='tight')
print("Saved: GAN/training_history.png")

# --------------------------
# Test with CNN Classifier (Optional)
# --------------------------
print("\n[STEP 8] Testing generated images with CNN classifier...")

try:
    # Try to load CNN model
    CNN_MODEL_PATH = CNN_PROJECT_DIR / 'models' / 'best_model.pth'

    if CNN_MODEL_PATH.exists():
        # Import CNN model
        from train_model import FruitCNN

        # Load model metadata
        with open(CNN_PROJECT_DIR / 'models' / 'model_metadata.json', 'r') as f:
            cnn_metadata = json.load(f)

        fruit_names = cnn_metadata['fruit_names']
        num_classes = cnn_metadata['num_classes']

        # Load CNN model
        cnn_model = FruitCNN(num_classes=num_classes).to(device)
        checkpoint = torch.load(CNN_MODEL_PATH)
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
        cnn_model.eval()

        print(f"\nLoaded CNN model (trained on {num_classes} fruit classes)")
        print(f"Classes: {', '.join(fruit_names)}")

        # Classify generated images
        with torch.no_grad():
            noise = torch.randn(16, noise_dim, device=device)
            fake_images = generator(noise)

            # CNN expects normalized images [-1, 1] which is what generator outputs
            outputs = cnn_model(fake_images)
            _, predictions = torch.max(outputs, 1)

            print("\nSample predictions on generated images:")
            for i in range(min(16, len(predictions))):
                pred_class = predictions[i].item()
                print(f"  Image {i+1}: {fruit_names[pred_class]}")

        print("\n✅ Generated images are compatible with CNN classifier!")
    else:
        print(f"CNN model not found at {CNN_MODEL_PATH}")
        print("Run CNN_Project/train_model.py first to train the classifier.")

except Exception as e:
    print(f"Could not load CNN model: {e}")
    print("Generated images are still saved and can be used later.")

# --------------------------
# Final Summary
# --------------------------
print("\n" + "=" * 80)
print("GAN TRAINING COMPLETE - SUMMARY")
print("=" * 80)
print(f"\nTraining Statistics:")
print(f"  Total Time: {total_time_str}")
print(f"  Average Epoch Time: {avg_epoch_time:.1f}s")
print(f"  Total Epochs: {num_epochs}")

print(f"\n{'='*80}")
print("AVAILABLE MODELS (ALL IMMEDIATELY USABLE!)")
print(f"{'='*80}")
print(f"\n✅ best_model.pth (RECOMMENDED)")
print(f"   - Epoch: {best_epoch}")
print(f"   - D(fake) score: {best_d_fake_score:.4f} (closest to ideal 0.5)")
print(f"   - Best quality generator")
print(f"\n✅ latest_model.pth")
print(f"   - Epoch: {num_epochs}")
print(f"   - Most recent checkpoint")
print(f"   - Updated every epoch (safe to kill training anytime)")
print(f"\n✅ final_gan.pth")
print(f"   - Final model after all {num_epochs} epochs")
print(f"\n✅ checkpoint_epoch_*.pth")
print(f"   - Saved every 10 epochs for resume training")

print(f"\n{'='*80}")
print("METADATA FILES (Human Readable)")
print(f"{'='*80}")
print(f"\n📄 models_index.json - Summary of all models")
print(f"📄 best_metadata.json - Best model info")
print(f"📄 latest_metadata.json - Latest model info")
print(f"📄 final_metadata.json - Final training stats")

print(f"\n{'='*80}")
print("GENERATED FILES")
print(f"{'='*80}")
print(f"\n🖼️  Generated Images:")
print(f"   - GAN/generated_images/ (100 samples)")
print(f"   - GAN/generated_images/grid_64.png")
print(f"\n📊 Training Progress:")
print(f"   - GAN/training_progress/epoch_*.png")
print(f"   - GAN/training_history.png")
print(f"   - GAN/training_history_live.png")

print(f"\n{'='*80}")
print("FINAL METRICS")
print(f"{'='*80}")
print(f"\nGenerator Loss: {history['G_loss'][-1]:.4f}")
print(f"Discriminator Loss: {history['D_loss'][-1]:.4f}")
print(f"D(real): {history['D_real'][-1]:.3f} (target: ~1.0)")
print(f"D(fake): {history['D_fake'][-1]:.3f} (target: ~0.5)")

print(f"\n{'='*80}")
print("USAGE")
print(f"{'='*80}")
print(f"\nGenerate images with best model:")
print(f"  python GAN/generate_images.py --model GAN/models/best_model.pth --num 50")
print(f"\nGenerate and classify:")
print(f"  python GAN/generate_images.py --model GAN/models/best_model.pth --num 50 --classify")
print(f"\nUse latest model:")
print(f"  python GAN/generate_images.py --model GAN/models/latest_model.pth --num 50")

print(f"\n{'='*80}")
print("⚠️  IMPORTANT: Training was safely interrupted-compatible!")
print(f"{'='*80}")
print(f"All models saved every epoch. You can Ctrl+C anytime and use latest_model.pth")
print(f"or best_model.pth immediately. No data loss!")

print("\n" + "=" * 80)
