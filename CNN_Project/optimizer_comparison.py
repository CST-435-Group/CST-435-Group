"""
Optimizer Comparison: Adam vs AdamW
CST-435 Neural Networks Optimization Project
"""

import os
import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# Load Dataset Metadata
# --------------------------
print("=" * 70)
print("OPTIMIZER COMPARISON: Adam vs AdamW")
print("=" * 70)

# Check if dataset exists
if not os.path.exists('data/dataset_metadata.json'):
    print("\n❌ Dataset not found. Please run train_model.py first to prepare the dataset.")
    sys.exit(1)

with open('data/dataset_metadata.json', 'r') as f:
    metadata = json.load(f)

image_paths = metadata['image_paths']
labels = metadata['labels']
SELECTED_FRUITS = metadata['fruit_names']
num_classes = metadata['num_classes']

print(f"\n[OK] Loaded dataset: {len(image_paths)} images, {num_classes} classes")
print(f"Classes: {', '.join(SELECTED_FRUITS)}")

# --------------------------
# Dataset Class
# --------------------------
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

# --------------------------
# CNN Model
# --------------------------
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

# --------------------------
# Prepare Data
# --------------------------
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

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"\nDataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --------------------------
# Training Function
# --------------------------
def train_with_optimizer(optimizer_name, optimizer, model, num_epochs=10):
    """Train model and return history"""
    print(f"\n{'='*70}")
    print(f"Training with {optimizer_name}")
    print(f"{'='*70}")

    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels_batch in pbar:
            images, labels_batch = images.to(device), labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100*correct/total:.2f}%"})

        train_loss = running_loss / len(train_dataset)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels_batch in val_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels_batch)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()

        val_loss = val_loss / len(val_dataset)
        val_acc = correct / total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

    return history

# --------------------------
# Comparison Experiment
# --------------------------
NUM_EPOCHS = 10
results = {}

# Test 1: Adam (Original)
print("\n" + "="*70)
print("EXPERIMENT 1: Original Adam Optimizer")
print("="*70)
torch.manual_seed(42)
np.random.seed(42)
model_adam = FruitCNN(num_classes=num_classes).to(device)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
history_adam = train_with_optimizer("Adam (lr=0.001)", optimizer_adam, model_adam, NUM_EPOCHS)
results['adam'] = history_adam

# Test 2: AdamW (Improved)
print("\n" + "="*70)
print("EXPERIMENT 2: Improved AdamW Optimizer")
print("="*70)
torch.manual_seed(42)
np.random.seed(42)
model_adamw = FruitCNN(num_classes=num_classes).to(device)
optimizer_adamw = optim.AdamW(model_adamw.parameters(), lr=0.0005, weight_decay=0.01)
history_adamw = train_with_optimizer("AdamW (lr=0.0005, weight_decay=0.01)", optimizer_adamw, model_adamw, NUM_EPOCHS)
results['adamw'] = history_adamw

# --------------------------
# Generate Comparison Report
# --------------------------
print("\n" + "="*70)
print("COMPARISON RESULTS")
print("="*70)

print("\n[RESULTS] Final Performance Metrics:")
print(f"\nAdam (Original):")
print(f"  Final Train Accuracy: {history_adam['train_acc'][-1]*100:.2f}%")
print(f"  Final Val Accuracy:   {history_adam['val_acc'][-1]*100:.2f}%")
print(f"  Final Train Loss:     {history_adam['train_loss'][-1]:.4f}")
print(f"  Final Val Loss:       {history_adam['val_loss'][-1]:.4f}")

print(f"\nAdamW (Improved):")
print(f"  Final Train Accuracy: {history_adamw['train_acc'][-1]*100:.2f}%")
print(f"  Final Val Accuracy:   {history_adamw['val_acc'][-1]*100:.2f}%")
print(f"  Final Train Loss:     {history_adamw['train_loss'][-1]:.4f}")
print(f"  Final Val Loss:       {history_adamw['val_loss'][-1]:.4f}")

print(f"\n[IMPROVEMENT]:")
val_acc_improvement = (history_adamw['val_acc'][-1] - history_adam['val_acc'][-1]) * 100
val_loss_improvement = history_adam['val_loss'][-1] - history_adamw['val_loss'][-1]
print(f"  Validation Accuracy: {'+' if val_acc_improvement > 0 else ''}{val_acc_improvement:.2f}%")
print(f"  Validation Loss:     {'+' if val_loss_improvement > 0 else ''}{val_loss_improvement:.4f}")

# --------------------------
# Plot Comparison
# --------------------------
print("\n[PLOT] Generating comparison plots...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

epochs_range = range(1, NUM_EPOCHS + 1)

# Training Loss
ax1.plot(epochs_range, history_adam['train_loss'], 'b-', label='Adam', linewidth=2)
ax1.plot(epochs_range, history_adamw['train_loss'], 'r-', label='AdamW', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Validation Loss
ax2.plot(epochs_range, history_adam['val_loss'], 'b-', label='Adam', linewidth=2)
ax2.plot(epochs_range, history_adamw['val_loss'], 'r-', label='AdamW', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Validation Loss Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Training Accuracy
ax3.plot(epochs_range, [acc*100 for acc in history_adam['train_acc']], 'b-', label='Adam', linewidth=2)
ax3.plot(epochs_range, [acc*100 for acc in history_adamw['train_acc']], 'r-', label='AdamW', linewidth=2)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Accuracy (%)')
ax3.set_title('Training Accuracy Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Validation Accuracy
ax4.plot(epochs_range, [acc*100 for acc in history_adam['val_acc']], 'b-', label='Adam', linewidth=2)
ax4.plot(epochs_range, [acc*100 for acc in history_adamw['val_acc']], 'r-', label='AdamW', linewidth=2)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Accuracy (%)')
ax4.set_title('Validation Accuracy Comparison')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Saved comparison plot: optimizer_comparison.png")

# Save results
with open('optimizer_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("[OK] Saved results: optimizer_comparison_results.json")

print("\n" + "="*70)
print("COMPARISON COMPLETE!")
print("="*70)
