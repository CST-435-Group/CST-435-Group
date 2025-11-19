"""
Conditional GAN Training Script for 200x200 RGB Color Images
CST-435 Neural Networks Assignment

Conditional GAN - YOU CONTROL what type of image to generate!
Perfect for multi-class datasets (tanks, jets, ships, etc.)

Usage:
  python GAN/train_gan_conditional.py
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

# Import model utilities
from model_utils import save_model_chunked

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Configuration
IMAGE_SIZE = 200
CHANNELS = 3  # RGB
NOISE_DIM = 100
BATCH_SIZE = 32
NUM_EPOCHS = 200
LR = 0.0002

# Dataset path
DATASET_PATH = 'GAN/datasets/military_vehicles_processed'

# Create directories
os.makedirs('GAN/generated_images_conditional', exist_ok=True)
os.makedirs('GAN/models_conditional', exist_ok=True)
os.makedirs('GAN/training_progress_conditional', exist_ok=True)

print("=" * 80)
print("CONDITIONAL GAN TRAINING - 200x200 RGB COLOR IMAGES")
print("CONTROL WHAT YOU GENERATE!")
print("=" * 80)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# --------------------------
# Conditional Generator
# --------------------------
class ConditionalGenerator(nn.Module):
    """
    Conditional Generator: Takes noise + class label -> generates specific image type

    Input:
      - noise: [batch, 100]
      - labels: [batch, num_classes] (one-hot encoded)
    Output:
      - images: [batch, 3, 200, 200]
    """
    def __init__(self, noise_dim=100, num_classes=10, embed_dim=50):
        super(ConditionalGenerator, self).__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Combine noise + embedded label
        combined_dim = noise_dim + embed_dim

        # Initial dense layer
        self.fc = nn.Linear(combined_dim, 13 * 13 * 512)
        self.bn0 = nn.BatchNorm1d(13 * 13 * 512)

        # Upsampling blocks (same as before)
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
        # Embed labels
        label_embed = self.label_embedding(labels)  # [batch, embed_dim]

        # Concatenate noise and label embedding
        x = torch.cat([noise, label_embed], dim=1)  # [batch, noise_dim + embed_dim]

        # Generate image
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

# --------------------------
# Conditional Discriminator
# --------------------------
class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator: Takes image + class label -> Real/Fake + Correct class?

    Input:
      - images: [batch, 3, 200, 200]
      - labels: [batch, num_classes] (one-hot encoded)
    Output:
      - validity: [batch, 1] (real or fake)
    """
    def __init__(self, num_classes=10, embed_dim=50):
        super(ConditionalDiscriminator, self).__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Project embedding to image size to concatenate
        self.label_projection = nn.Linear(embed_dim, IMAGE_SIZE * IMAGE_SIZE)

        # First conv takes 4 channels (3 RGB + 1 label channel)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout4 = nn.Dropout(0.3)

        self.fc = nn.Linear(13 * 13 * 512, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, images, labels):
        # Embed and project labels to image size
        label_embed = self.label_embedding(labels)  # [batch, embed_dim]
        label_proj = self.label_projection(label_embed)  # [batch, 200*200]
        label_proj = label_proj.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)  # [batch, 1, 200, 200]

        # Concatenate image and label channel
        x = torch.cat([images, label_proj], dim=1)  # [batch, 4, 200, 200]

        # Discriminate
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

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# --------------------------
# Dataset with Class Labels
# --------------------------
class ConditionalImageDataset(Dataset):
    """
    Load images and automatically assign class labels based on folder/filename

    Expected structure:
      datasets/military_vehicles_processed/
        - Files with 'tank' in name -> class 0
        - Files with 'jet' or 'aircraft' -> class 1
        - Files with 'ship' or 'warship' -> class 2
        - etc.

    Or organized in folders:
      datasets/military_vehicles_processed/
        tanks/
        jets/
        ships/
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.image_paths = []
        self.labels = []
        self.class_names = []

        # Define class keywords for automatic labeling
        self.class_keywords = {
            'tank': ['tank', 'armor'],
            'jet': ['jet', 'fighter', 'aircraft'],
            'ship': ['ship', 'warship', 'navy', 'vessel'],
            'helicopter': ['helicopter', 'heli', 'chopper'],
            'submarine': ['submarine', 'sub'],
            'armored_vehicle': ['armored', 'apc', 'ifv'],
            'drone': ['drone', 'uav'],
            'other': ['military', 'vehicle']
        }

        # Scan directory
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        all_files = []
        for ext in image_extensions:
            all_files.extend(list(self.image_dir.rglob(f'*{ext}')))

        print(f"\nScanning {len(all_files)} images for class labels...")

        # Assign classes
        class_counts = {}

        for img_path in all_files:
            # Check parent folder name and filename
            full_path = str(img_path).lower()

            assigned_class = None
            for class_name, keywords in self.class_keywords.items():
                if any(keyword in full_path for keyword in keywords):
                    assigned_class = class_name
                    break

            if assigned_class is None:
                assigned_class = 'other'

            self.image_paths.append(img_path)
            self.labels.append(assigned_class)

            class_counts[assigned_class] = class_counts.get(assigned_class, 0) + 1

        # Create class name to index mapping
        self.class_names = sorted(set(self.labels))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        # Convert labels to indices
        self.label_indices = [self.class_to_idx[label] for label in self.labels]

        print(f"\nDataset Statistics:")
        print(f"  Total images: {len(self.image_paths)}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"\nClass distribution:")
        for class_name in self.class_names:
            count = class_counts.get(class_name, 0)
            pct = (count / len(self.image_paths)) * 100
            print(f"    {class_name}: {count} images ({pct:.1f}%)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = self.label_indices[idx]

        return img, label

# --------------------------
# Load Dataset
# --------------------------
print("\n[STEP 1] Loading dataset...")

if not Path(DATASET_PATH).exists():
    print(f"ERROR: {DATASET_PATH} does not exist!")
    print("\nDownload and preprocess images first:")
    print("  python GAN/download_dataset.py")
    print("  python GAN/preprocess_dataset.py")
    sys.exit(1)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = ConditionalImageDataset(DATASET_PATH, transform=transform)

if len(dataset) == 0:
    print("ERROR: No images found!")
    sys.exit(1)

NUM_CLASSES = dataset.num_classes
CLASS_NAMES = dataset.class_names

print(f"\n{'='*80}")
print(f"CLASS LABELS: {', '.join(CLASS_NAMES)}")
print(f"{'='*80}")

num_workers = 0 if os.name == 'nt' else 2
pin_memory = torch.cuda.is_available()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_memory)

# --------------------------
# Initialize Models
# --------------------------
print("\n[STEP 2] Initializing Conditional GAN...")

generator = ConditionalGenerator(NOISE_DIM, NUM_CLASSES).to(device)
discriminator = ConditionalDiscriminator(NUM_CLASSES).to(device)

gen_params = sum(p.numel() for p in generator.parameters())
disc_params = sum(p.numel() for p in discriminator.parameters())

print(f"\nConditional Generator:")
print(f"  Parameters: {gen_params:,}")
print(f"  Input: Noise[{NOISE_DIM}] + Class Label")
print(f"  Output: {IMAGE_SIZE}x{IMAGE_SIZE} RGB image")

print(f"\nConditional Discriminator:")
print(f"  Parameters: {disc_params:,}")
print(f"  Input: Image + Class Label")
print(f"  Output: Real/Fake prediction")

# --------------------------
# Training Setup
# --------------------------
print("\n[STEP 3] Setting up training...")

criterion = nn.BCEWithLogitsLoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# --------------------------
# Training Loop
# --------------------------
print("\n[STEP 4] Starting training...")

# Fixed noise for each class (for visualization)
fixed_noise = torch.randn(NUM_CLASSES * 4, NOISE_DIM, device=device)
fixed_labels = torch.tensor([i for i in range(NUM_CLASSES) for _ in range(4)], device=device)

history = {'D_loss': [], 'G_loss': [], 'D_real': [], 'D_fake': []}

training_start_time = time.time()
epoch_times = []
best_d_fake_score = float('inf')
best_epoch = 0

print(f"\nTraining for {NUM_EPOCHS} epochs...")
print("=" * 80)

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()

    generator.train()
    discriminator.train()

    epoch_D_loss = 0
    epoch_G_loss = 0
    epoch_D_real = 0
    epoch_D_fake = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for i, (real_images, real_labels) in enumerate(pbar):
        batch_size_curr = real_images.size(0)
        real_images = real_images.to(device)
        real_labels = real_labels.to(device)

        valid = torch.ones(batch_size_curr, 1, device=device)
        fake = torch.zeros(batch_size_curr, 1, device=device)

        # Train Discriminator
        optimizer_D.zero_grad()

        D_real_output = discriminator(real_images, real_labels)
        D_real_loss = criterion(D_real_output, valid)

        noise = torch.randn(batch_size_curr, NOISE_DIM, device=device)
        fake_labels = torch.randint(0, NUM_CLASSES, (batch_size_curr,), device=device)
        fake_images = generator(noise, fake_labels)

        D_fake_output = discriminator(fake_images.detach(), fake_labels)
        D_fake_loss = criterion(D_fake_output, fake)

        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        D_fake_output = discriminator(fake_images, fake_labels)
        G_loss = criterion(D_fake_output, valid)

        G_loss.backward()
        optimizer_G.step()

        # Track metrics
        epoch_D_loss += D_loss.item()
        epoch_G_loss += G_loss.item()
        epoch_D_real += torch.sigmoid(D_real_output).mean().item()
        epoch_D_fake += torch.sigmoid(D_fake_output).mean().item()

        pbar.set_postfix({
            'D_loss': f"{D_loss.item():.4f}",
            'G_loss': f"{G_loss.item():.4f}"
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

    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print(f"  D_loss: {avg_D_loss:.4f} | G_loss: {avg_G_loss:.4f}")
    print(f"  D(real): {avg_D_real:.3f} | D(fake): {avg_D_fake:.3f}")
    print(f"  Time: {epoch_time:.1f}s")

    # Save models
    model_data = {
        'epoch': epoch + 1,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'noise_dim': NOISE_DIM,
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'image_size': IMAGE_SIZE,
        'channels': CHANNELS,
        'history': history
    }

    save_model_chunked(model_data, 'GAN/models_conditional/latest_model.pth')

    # Track best
    d_fake_distance = abs(avg_D_fake - 0.5)
    if d_fake_distance < best_d_fake_score:
        best_d_fake_score = d_fake_distance
        best_epoch = epoch + 1
        save_model_chunked(model_data, 'GAN/models_conditional/best_model.pth')
        print(f"  [BEST] Saved best model!")

    # Generate samples for each class
    if (epoch + 1) % 5 == 0:
        generator.eval()
        with torch.no_grad():
            samples = generator(fixed_noise, fixed_labels)
            samples = (samples + 1) / 2
            save_image(samples, f'GAN/training_progress_conditional/epoch_{epoch+1:03d}.png',
                      nrow=4, normalize=False)
        generator.train()
        print(f"  [SAVED] Progress images (one row per class)")

    if (epoch + 1) % 10 == 0:
        save_model_chunked(model_data, f'GAN/models_conditional/checkpoint_epoch_{epoch+1:03d}.pth')

total_time = time.time() - training_start_time
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nTotal Time: {str(timedelta(seconds=int(total_time)))}")
print(f"Best Epoch: {best_epoch}")
print(f"\nClass labels: {', '.join(CLASS_NAMES)}")
print(f"\nGenerate specific classes:")
print("  python GAN/generate_images_conditional.py --model GAN/models_conditional/best_model.pth --class tank --num 10")
