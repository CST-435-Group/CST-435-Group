"""
Fruit Image Classification - Training Script (FIXED)
CST-435 Neural Networks Assignment
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('preprocessed_images', exist_ok=True)

print("=" * 70)
print("FRUIT IMAGE CLASSIFICATION - TRAINING")
print("=" * 70)

# --------------------------
# STEP 1: Download Dataset
# --------------------------
print("\n[STEP 1] Downloading Fruits dataset from Kaggle...")
try:
    import kagglehub
    path = kagglehub.dataset_download("moltean/fruits")
    print(f"[OK] Dataset downloaded to: {path}")
except Exception as e:
    print(f"[ERROR] Error downloading dataset: {e}")
    sys.exit(1)

# --------------------------
# STEP 2: Explore Dataset Structure
# --------------------------
print("\n[STEP 2] Exploring dataset structure...")

dataset_path = Path(path)
print(f"Dataset path: {dataset_path}")

def find_training_folder(base_path):
    """Recursively find the Training folder"""
    for root, dirs, files in os.walk(base_path):
        if 'Training' in dirs:
            return Path(root) / 'Training'
    return None

# Try fruits-360_100x100 first (best for training)
main_folder = dataset_path / 'fruits-360_100x100'
if main_folder.exists():
    training_folder = find_training_folder(main_folder)
else:
    # Try other folders
    for folder in dataset_path.iterdir():
        if folder.is_dir() and 'fruits-360' in folder.name:
            training_folder = find_training_folder(folder)
            if training_folder:
                break

if training_folder is None:
    print("[ERROR] Could not find Training folder")
    sys.exit(1)

print(f"[OK] Found training folder: {training_folder}")

# Get all fruit categories (immediate subdirectories)
fruit_folders = [f for f in training_folder.iterdir() if f.is_dir()]
print(f"\nFound {len(fruit_folders)} fruit categories")

# Count images and aggregate by base fruit name
print("\nCounting images in each category...")
fruit_counts = {}
fruit_folder_map = {}  # Map base name to list of variety folders
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']

for fruit_folder in tqdm(fruit_folders, desc="Scanning fruits"):
    fruit_name = fruit_folder.name
    
    # Extract base fruit name (remove variety numbers)
    # "Grape Blue 1" -> "Grape Blue", "Pear 9" -> "Pear"
    parts = fruit_name.split()
    base_name_parts = []
    for part in parts:
        if part.isdigit():  # Stop when we hit a number
            break
        base_name_parts.append(part)
    
    base_fruit_name = ' '.join(base_name_parts) if base_name_parts else fruit_name
    
    # Count all images in all subfolders
    image_count = 0
    for ext in image_extensions:
        image_count += len(list(fruit_folder.glob(f'**/*{ext}')))
    
    if image_count > 0:
        # Aggregate counts by base fruit name
        if base_fruit_name in fruit_counts:
            fruit_counts[base_fruit_name] += image_count
            fruit_folder_map[base_fruit_name].append(fruit_folder)
        else:
            fruit_counts[base_fruit_name] = image_count
            fruit_folder_map[base_fruit_name] = [fruit_folder]

# Sort by count
sorted_fruits = sorted(fruit_counts.items(), key=lambda x: x[1], reverse=True)

print("\n[STATS] Top 20 fruits by image count:")
for i, (fruit, count) in enumerate(sorted_fruits[:20], 1):
    print(f"  {i:2d}. {fruit:30s}: {count:5d} images")

# --------------------------
# STEP 3: Select 5 Fruits
# --------------------------
print("\n[STEP 3] Selecting 5 fruit categories...")

MIN_IMAGES = 500
SELECTED_FRUITS = []

for fruit, count in sorted_fruits:
    if count >= MIN_IMAGES and len(SELECTED_FRUITS) < 5:
        SELECTED_FRUITS.append(fruit)

if len(SELECTED_FRUITS) < 5:
    print("[WARNING] Not enough fruits with 500+ images. Using top 5...")
    SELECTED_FRUITS = [fruit for fruit, _ in sorted_fruits[:5]]

print("\n[SELECTED] Selected fruits for classification:")
for i, fruit in enumerate(SELECTED_FRUITS, 1):
    count = fruit_counts.get(fruit, 0)
    print(f"  {i}. {fruit:30s}: {count} images")

# --------------------------
# STEP 4: Collect and Preprocess Images
# --------------------------
print("\n[STEP 4] Collecting and preprocessing images...")
print("Converting to grayscale 128x128...")

MAX_IMAGES_PER_FRUIT = 1000

def preprocess_and_save_image(img_path, save_path):
    """Load image, convert to grayscale 128x128, save"""
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_gray = img.convert('L')
        img_resized = img_gray.resize((128, 128), Image.Resampling.LANCZOS)
        img_resized.save(save_path)
        return True
    except Exception as e:
        return False

image_paths = []
labels = []

for fruit_idx, base_fruit_name in enumerate(SELECTED_FRUITS):
    print(f"\nProcessing '{base_fruit_name}'...")
    
    # Create output directory
    safe_fruit_name = base_fruit_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    output_dir = Path('preprocessed_images') / safe_fruit_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all variety folders for this base fruit
    variety_folders = fruit_folder_map.get(base_fruit_name, [])
    
    # Collect all image files from all varieties
    image_files = []
    for variety_folder in variety_folders:
        for ext in image_extensions:
            image_files.extend(list(variety_folder.glob(f'**/*{ext}')))
    
    print(f"  Found {len(image_files)} total images across {len(variety_folders)} varieties")
    
    # Limit to MAX_IMAGES_PER_FRUIT
    if len(image_files) > MAX_IMAGES_PER_FRUIT:
        import random
        random.seed(42)
        image_files = random.sample(image_files, MAX_IMAGES_PER_FRUIT)
        print(f"  Sampling {MAX_IMAGES_PER_FRUIT} images")
    
    # Process each image
    successful = 0
    for img_path in tqdm(image_files, desc=f"  {base_fruit_name}"):
        # Create unique save path
        save_path = output_dir / f"{img_path.parent.name}_{img_path.stem}.png"
        
        # Skip if already processed
        if save_path.exists():
            image_paths.append(str(save_path))
            labels.append(fruit_idx)
            successful += 1
            continue
        
        # Preprocess and save
        if preprocess_and_save_image(img_path, save_path):
            image_paths.append(str(save_path))
            labels.append(fruit_idx)
            successful += 1
    
    print(f"  [OK] Successfully processed {successful} images")

print(f"\n[OK] Total images processed: {len(image_paths)}")

if len(image_paths) < 100:
    print("\n[ERROR] Not enough images processed.")
    sys.exit(1)

# Save metadata
metadata = {
    'image_paths': image_paths,
    'labels': labels,
    'fruit_names': SELECTED_FRUITS,
    'num_classes': len(SELECTED_FRUITS)
}

with open('data/dataset_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("[OK] Saved dataset metadata")

# --------------------------
# STEP 5: Create PyTorch Dataset
# --------------------------
print("\n[STEP 5] Creating PyTorch dataset...")

class FruitImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')
        
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        label = self.labels[idx]
        return img, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

full_dataset = FruitImageDataset(image_paths, labels, transform=transform)

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Dataset splits:")
print(f"  Training: {len(train_dataset)}")
print(f"  Validation: {len(val_dataset)}")
print(f"  Test: {len(test_dataset)}")

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print("[OK] DataLoaders created")

# --------------------------
# STEP 6: Define CNN Model
# --------------------------
print("\n[STEP 6] Defining CNN architecture...")

class FruitCNN(nn.Module):
    def __init__(self, num_classes):
        super(FruitCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FruitCNN(num_classes=len(SELECTED_FRUITS)).to(device)

print(f"Model created on: {device}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# --------------------------
# STEP 7: Training Setup
# --------------------------
print("\n[STEP 7] Setting up training...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
# Note: verbose parameter removed (not supported in all PyTorch versions)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

print("[OK] Loss: CrossEntropyLoss")
print("[OK] Optimizer: AdamW (lr=0.0005, weight_decay=0.01)")
print("[OK] Scheduler: ReduceLROnPlateau")

# --------------------------
# STEP 8: Training Loop
# --------------------------
print("\n[STEP 8] Training model...")

num_epochs = 50
best_val_acc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print(f"Training for {num_epochs} epochs on {device}...")
print("=" * 70)

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100*correct/total:.2f}%"})
    
    train_loss = running_loss / len(train_dataset)
    train_acc = correct / total
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_dataset)
    val_acc = correct / total
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    scheduler.step(val_loss)
    
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'fruit_names': SELECTED_FRUITS,
            'num_classes': len(SELECTED_FRUITS)
        }, 'models/best_model.pth')
        print(f"  [SAVED] Saved best model (val_acc: {val_acc*100:.2f}%)")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)

# --------------------------
# STEP 9: Final Evaluation
# --------------------------
print("\n[STEP 9] Evaluating on test set...")

checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_loss = 0.0
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss = test_loss / len(test_dataset)
test_acc = correct / total

print(f"\n[RESULTS] Final Test Results:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_acc*100:.2f}%")

from sklearn.metrics import classification_report
print(f"\n[RESULTS] Per-Class Performance:")
print(classification_report(all_labels, all_preds, target_names=SELECTED_FRUITS))

with open('models/training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

final_metadata = {
    'fruit_names': SELECTED_FRUITS,
    'num_classes': len(SELECTED_FRUITS),
    'test_accuracy': test_acc,
    'best_val_accuracy': best_val_acc,
    'total_images': len(image_paths),
    'train_size': len(train_dataset),
    'val_size': len(val_dataset),
    'test_size': len(test_dataset)
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(final_metadata, f, indent=2)

print("\n[OK] All files saved!")
print("\n[COMPLETE] Training complete! Ready for Streamlit deployment!")
print("Run: streamlit run streamlit_app.py")
