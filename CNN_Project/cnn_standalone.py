"""
CNN Image Classification - Standalone Python Script (PyTorch Version)
CST-435 Neural Networks Assignment

This script can be run directly without Jupyter Notebook:
    python cnn_standalone.py

For the full notebook experience with visualizations, use:
    jupyter notebook CNN_Image_Classification.ipynb

Requires: PyTorch with CUDA support for GPU acceleration on Windows
"""

import os
import sys

# Import required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Class names for CIFAR-10
CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")

class CIFAR10_CNN(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10 Classification
    """
    
    def __init__(self, num_classes=10):
        super(CIFAR10_CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(4 * 4 * 128, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Third conv block
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def load_and_preprocess_data():
    """Load and preprocess CIFAR-10 dataset"""
    print_section("LOADING AND PREPROCESSING DATA")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    print("Loading CIFAR-10 dataset...")
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Number of classes: {len(CLASS_NAMES)}")
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
    
    print(f"✅ DataLoaders created (batch size: {batch_size})")
    
    return train_loader, test_loader, train_dataset, test_dataset

def build_cnn_model():
    """Build the CNN model"""
    print_section("BUILDING CNN ARCHITECTURE")
    
    model = CIFAR10_CNN(num_classes=10).to(device)
    
    print("Model Architecture:")
    print("-" * 70)
    print(model)
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✅ Model created and moved to: {device}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print("\n   Pooling Type: MAX POOLING (nn.MaxPool2d)")
    print("   - Extracts maximum value from each 2x2 region")
    print("   - Optimized for CUDA acceleration")
    
    return model

def compile_model(model):
    """Define loss function and optimizer"""
    print_section("DEFINING LOSS AND OPTIMIZER")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("✅ Configuration complete!")
    print("   Loss function: CrossEntropyLoss")
    print("   Optimizer: Adam (lr=0.001)")
    print("   Metrics: Accuracy")
    print(f"   Device: {device}")
    
    return criterion, optimizer

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    avg_loss = running_loss / len(train_loader.dataset)
    accuracy = correct / total
    
    return avg_loss, accuracy

def validate_epoch(model, test_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(test_loader.dataset)
    accuracy = correct / total
    
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=50):
    """Train the CNN model"""
    print_section(f"TRAINING MODEL ({epochs} EPOCHS)")
    
    print("Starting training...")
    if torch.cuda.is_available():
        print(f"   Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Expected time: ~5-10 minutes")
    else:
        print(f"   Using CPU (no GPU available)")
        print(f"   Expected time: ~15-20 minutes")
    print("-" * 70 + "\n")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    
    print("\n✅ Training complete!")
    
    return history

def evaluate_model(model, test_loader, history):
    """Evaluate the trained model"""
    print_section("EVALUATING MODEL PERFORMANCE")
    
    final_train_loss = history['train_loss'][-1]
    final_train_acc = history['train_acc'][-1]
    final_val_loss = history['val_loss'][-1]
    final_val_acc = history['val_acc'][-1]
    
    print("Final Performance Metrics:")
    print("-" * 70)
    print(f"\nTraining Set:")
    print(f"   Loss:     {final_train_loss:.4f}")
    print(f"   Accuracy: {final_train_acc*100:.2f}%")
    print(f"\nTest Set:")
    print(f"   Loss:     {final_val_loss:.4f}")
    print(f"   Accuracy: {final_val_acc*100:.2f}%")
    
    accuracy_diff = (final_train_acc - final_val_acc) * 100
    print(f"\nGeneralization Gap: {accuracy_diff:.2f}%")
    if accuracy_diff > 10:
        print("   ⚠️  Model shows signs of overfitting")
    else:
        print("   ✅ Model generalizes well")
    
    if torch.cuda.is_available():
        print(f"\n🚀 GPU Acceleration: ENABLED")
    
    return final_val_loss, final_val_acc

def plot_training_history(history, save_path='training_plots.png'):
    """Plot training history"""
    print_section("GENERATING VISUALIZATIONS")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss', linewidth=2, color='#e74c3c')
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2, color='#3498db')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Model Loss Over Training Epochs', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot([acc*100 for acc in history['train_acc']], label='Training Accuracy', 
             linewidth=2, color='#e74c3c')
    ax2.plot([acc*100 for acc in history['val_acc']], label='Validation Accuracy', 
             linewidth=2, color='#3498db')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Model Accuracy Over Training Epochs', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training plots saved to: {save_path}")
    
    try:
        plt.show(block=False)
        print("   (Close the plot window to continue)")
        plt.pause(3)
    except:
        print("   (Plot display not available in this environment)")
    
    plt.close()

def analyze_predictions(model, test_loader, test_dataset):
    """Analyze model predictions"""
    print_section("ANALYZING PREDICTIONS")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    print("Per-Class Accuracy:")
    print("-" * 50)
    for i, name in enumerate(CLASS_NAMES):
        print(f"   {name:12s}: {class_accuracy[i]*100:5.2f}%")
    print("-" * 50)
    print(f"   {'Mean':12s}: {class_accuracy.mean()*100:5.2f}%")
    print(f"   {'Best':12s}: {CLASS_NAMES[class_accuracy.argmax()]} ({class_accuracy.max()*100:.2f}%)")
    print(f"   {'Worst':12s}: {CLASS_NAMES[class_accuracy.argmin()]} ({class_accuracy.min()*100:.2f}%)")
    
    # Misclassification
    misclassified = np.sum(all_preds != all_labels)
    print(f"\nMisclassification Summary:")
    print(f"   Total misclassified: {misclassified} out of {len(all_labels)}")
    print(f"   Misclassification rate: {misclassified/len(all_labels)*100:.2f}%")
    
    return all_preds, all_labels

def save_model(model, model_path='cifar10_cnn_pytorch.pth'):
    """Save the trained model"""
    print_section("SAVING MODEL")
    
    torch.save({
        'model_state_dict': model.state_dict(),
    }, model_path)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"\nTo load the model later:")
    print(f"   model = CIFAR10_CNN()")
    print(f"   checkpoint = torch.load('{model_path}')")
    print(f"   model.load_state_dict(checkpoint['model_state_dict'])")
    print(f"   model.eval()")

def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print(" CNN IMAGE CLASSIFICATION PROJECT (PyTorch + CUDA)")
    print(" CST-435 Neural Networks Assignment")
    print("=" * 70)
    
    # Check PyTorch and CUDA
    print(f"\nPyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CUDA Available: No (using CPU)")
    print(f"Device: {device}")
    
    # Step 1: Load data
    train_loader, test_loader, train_dataset, test_dataset = load_and_preprocess_data()
    
    # Step 2: Build model
    model = build_cnn_model()
    
    # Step 3: Define loss and optimizer
    criterion, optimizer = compile_model(model)
    
    # Step 4: Train model
    history = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=50)
    
    # Step 5: Evaluate model
    test_loss, test_accuracy = evaluate_model(model, test_loader, history)
    
    # Step 6: Plot history
    plot_training_history(history)
    
    # Step 7: Analyze predictions
    all_preds, all_labels = analyze_predictions(model, test_loader, test_dataset)
    
    # Step 8: Save model
    save_model(model)
    
    # Final summary
    print_section("PROJECT COMPLETE")
    print("✅ All tasks completed successfully!")
    print(f"\nFinal Test Accuracy: {test_accuracy*100:.2f}%")
    print("\nGenerated Files:")
    print("   - cifar10_cnn_pytorch.pth (trained model)")
    print("   - training_plots.png (loss and accuracy graphs)")
    print("\nPowered by PyTorch with CUDA support!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
