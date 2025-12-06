"""
Behavioral Cloning Training Script
Trains an agent using human gameplay data collected from the frontend.
Uses supervised learning instead of reinforcement learning.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


def check_cuda():
    """Check if CUDA GPU is available"""
    if torch.cuda.is_available():
        print(f"[GPU] CUDA is available!")
        print(f"[GPU] Device: {torch.cuda.get_device_name(0)}")
        print(f"[GPU] CUDA Version: {torch.version.cuda}")
        print(f"[GPU] Device Count: {torch.cuda.device_count()}")
        return "cuda"
    else:
        print("[CPU] CUDA not available, training will use CPU")
        return "cpu"


class PolicyNetwork(nn.Module):
    """
    Multi-Layer Perceptron for behavioral cloning
    Maps state features to action probabilities
    """
    def __init__(self, input_size, hidden_sizes=[256, 256], num_actions=4):
        super(PolicyNetwork, self).__init__()

        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        # Output layer for each action (independent binary classification)
        layers.append(nn.Linear(prev_size, num_actions))
        layers.append(nn.Sigmoid())  # Independent probabilities for each action

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class HumanGameplayDataset(Dataset):
    """
    Dataset of state-action pairs from human gameplay
    """
    def __init__(self, data_points, normalize=True):
        """
        Args:
            data_points: List of dicts with state features and actions
            normalize: Whether to normalize state features
        """
        self.data_points = data_points
        self.normalize = normalize

        # Extract features and actions
        self.states = []
        self.actions = []

        for point in data_points:
            # State features (16 total - added ice platform flags)
            state = [
                point['player_x'],
                point['player_y'],
                point['player_vx'],
                point['player_vy'],
                float(point['player_on_ground']),
                point['platform_below_x'] if point['platform_below_x'] is not None else 0.0,
                point['platform_below_y'] if point['platform_below_y'] is not None else 0.0,
                float(point.get('platform_below_is_ice', False)),  # NEW: ice flag
                point['platform_ahead_x'] if point['platform_ahead_x'] is not None else 0.0,
                point['platform_ahead_y'] if point['platform_ahead_y'] is not None else 0.0,
                float(point.get('platform_ahead_is_ice', False)),  # NEW: ice flag
                point.get('enemy_x', 0.0) if point.get('enemy_x') is not None else 0.0,
                point.get('enemy_y', 0.0) if point.get('enemy_y') is not None else 0.0,
                point['goal_x'],
                point['goal_y'],
                # Encode difficulty as number: easy=0, medium=1, hard=2
                {'easy': 0.0, 'medium': 1.0, 'hard': 2.0}.get(point.get('difficulty', 'easy'), 0.0)
            ]

            # Actions (4 binary values)
            action = [
                float(point['action_left']),
                float(point['action_right']),
                float(point['action_jump']),
                float(point['action_sprint'])
            ]

            self.states.append(state)
            self.actions.append(action)

        self.states = np.array(self.states, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.float32)

        # Normalize states (important for neural networks)
        if self.normalize:
            self.mean = np.mean(self.states, axis=0)
            self.std = np.std(self.states, axis=0) + 1e-8  # Avoid division by zero
            self.states = (self.states - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'state': torch.FloatTensor(self.states[idx]),
            'action': torch.FloatTensor(self.actions[idx])
        }


def load_training_data(data_dir, max_files=None, filter_difficulty=None):
    """
    Load all training data from JSON files

    Args:
        data_dir: Path to training_data directory
        max_files: Maximum number of files to load (for testing)
        filter_difficulty: Only load data from specific difficulty

    Returns:
        List of all training data points
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Training data directory not found: {data_dir}")

    all_data_points = []
    file_count = 0

    print(f"[DATA] Loading training data from: {data_dir}")

    # Walk through all date directories
    for date_dir in sorted(data_dir.iterdir()):
        if not date_dir.is_dir():
            continue

        # Process all JSON files in this date directory
        for json_file in sorted(date_dir.glob("*.json")):
            if max_files and file_count >= max_files:
                break

            try:
                with open(json_file, 'r') as f:
                    batch = json.load(f)

                # Filter by difficulty if specified
                if filter_difficulty and batch.get('difficulty') != filter_difficulty:
                    continue

                # Add all data points from this batch
                data_points = batch.get('data_points', [])
                all_data_points.extend(data_points)
                file_count += 1

                if file_count % 10 == 0:
                    print(f"[DATA] Loaded {file_count} files, {len(all_data_points)} data points...")

            except Exception as e:
                print(f"[WARNING] Failed to load {json_file}: {e}")
                continue

        if max_files and file_count >= max_files:
            break

    print(f"[DATA] Loaded {len(all_data_points)} total data points from {file_count} files")

    # Print action distribution
    action_counts = defaultdict(int)
    for point in all_data_points:
        if point['action_left']:
            action_counts['left'] += 1
        if point['action_right']:
            action_counts['right'] += 1
        if point['action_jump']:
            action_counts['jump'] += 1
        if point['action_sprint']:
            action_counts['sprint'] += 1

    print(f"[DATA] Action distribution:")
    print(f"  Left: {action_counts['left']:,} ({100*action_counts['left']/len(all_data_points):.1f}%)")
    print(f"  Right: {action_counts['right']:,} ({100*action_counts['right']/len(all_data_points):.1f}%)")
    print(f"  Jump: {action_counts['jump']:,} ({100*action_counts['jump']/len(all_data_points):.1f}%)")
    print(f"  Sprint: {action_counts['sprint']:,} ({100*action_counts['sprint']/len(all_data_points):.1f}%)")

    return all_data_points


def train_behavioral_cloning(
    data_dir,
    epochs=100,
    batch_size=256,
    learning_rate=0.001,
    val_split=0.2,
    hidden_sizes=[256, 256],
    save_path="models/bc_agent",
    device="cuda"
):
    """
    Train behavioral cloning agent

    Args:
        data_dir: Path to training_data directory
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        val_split: Validation split ratio
        hidden_sizes: Hidden layer sizes for MLP
        save_path: Where to save trained model
        device: Device to train on ('cuda' or 'cpu')

    Returns:
        Trained model, training stats
    """
    print("=" * 60)
    print("BEHAVIORAL CLONING TRAINING")
    print("=" * 60)
    print(f"[CONFIG] Epochs: {epochs}")
    print(f"[CONFIG] Batch size: {batch_size}")
    print(f"[CONFIG] Learning rate: {learning_rate}")
    print(f"[CONFIG] Validation split: {val_split}")
    print(f"[CONFIG] Hidden layers: {hidden_sizes}")
    print(f"[CONFIG] Device: {device}")
    print()

    # Load training data
    all_data_points = load_training_data(data_dir)

    if len(all_data_points) < 100:
        raise ValueError(f"Not enough training data! Only {len(all_data_points)} data points. Need at least 100.")

    # Create dataset
    dataset = HumanGameplayDataset(all_data_points, normalize=True)

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"[DATA] Training samples: {len(train_dataset):,}")
    print(f"[DATA] Validation samples: {len(val_dataset):,}")
    print()

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    input_size = 16  # State features (added enemy positions + ice platform flags)
    num_actions = 4  # left, right, jump, sprint
    model = PolicyNetwork(input_size, hidden_sizes, num_actions).to(device)

    print(f"[MODEL] Created policy network with {sum(p.numel() for p in model.parameters()):,} parameters")
    print()

    # Loss and optimizer (Binary Cross-Entropy for multi-label classification)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop
    best_val_loss = float('inf')
    training_stats = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': [],
        'epochs': []
    }

    print("[TRAINING] Starting training...")
    print()

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            states = batch['state'].to(device)
            actions = batch['action'].to(device)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(states)
            loss = criterion(predictions, actions)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy (predict action if probability > 0.5)
            pred_actions = (predictions > 0.5).float()
            train_correct += (pred_actions == actions).sum().item()
            train_total += actions.numel()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                states = batch['state'].to(device)
                actions = batch['action'].to(device)

                predictions = model(states)
                loss = criterion(predictions, actions)

                val_loss += loss.item()

                pred_actions = (predictions > 0.5).float()
                val_correct += (pred_actions == actions).sum().item()
                val_total += actions.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

            # Save model state dict and normalization parameters
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': input_size,
                'hidden_sizes': hidden_sizes,
                'num_actions': num_actions,
                'normalize_mean': dataset.dataset.mean if hasattr(dataset, 'dataset') else dataset.mean,
                'normalize_std': dataset.dataset.std if hasattr(dataset, 'dataset') else dataset.std,
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy
            }, f"{save_path}.pth")

        # Update stats
        training_stats['train_losses'].append(avg_train_loss)
        training_stats['val_losses'].append(avg_val_loss)
        training_stats['train_accs'].append(train_accuracy)
        training_stats['val_accs'].append(val_accuracy)
        training_stats['epochs'].append(epoch)

        # Update status.json for frontend
        status = {
            'is_training': True,
            'current_epoch': epoch + 1,
            'total_epochs': epochs,
            'progress': (epoch + 1) / epochs,
            'train_loss': float(avg_train_loss),
            'val_loss': float(avg_val_loss),
            'train_accuracy': float(train_accuracy),
            'val_accuracy': float(val_accuracy),
            'best_val_loss': float(best_val_loss),
            'timestamp': time.time(),
            'training_mode': 'behavioral_cloning'
        }

        with open('status.json', 'w') as f:
            json.dump(status, f, indent=2)

        epoch_time = time.time() - epoch_start_time

        # Print progress
        print(f"[EPOCH {epoch+1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.3f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.3f} | "
              f"Time: {epoch_time:.1f}s")

    # Mark training as complete
    status['is_training'] = False
    status['completed'] = True
    status['completed_at'] = time.time()
    status['message'] = 'Behavioral cloning training completed successfully'

    with open('status.json', 'w') as f:
        json.dump(status, f, indent=2)

    print()
    print("=" * 60)
    print("[SUCCESS] Training complete!")
    print(f"[SUCCESS] Best validation loss: {best_val_loss:.4f}")
    print(f"[SUCCESS] Model saved to: {save_path}.pth")
    print("=" * 60)

    return model, training_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train agent using behavioral cloning")
    parser.add_argument("--data-dir", type=str,
                       default="../../launcher/backend/data/training_data",
                       help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation split ratio")
    parser.add_argument("--hidden-sizes", type=int, nargs='+', default=[256, 256],
                       help="Hidden layer sizes")
    parser.add_argument("--save-path", type=str, default="models/bc_agent",
                       help="Path to save trained model")

    args = parser.parse_args()

    # Check CUDA availability
    device = check_cuda()

    try:
        model, stats = train_behavioral_cloning(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_split=args.val_split,
            hidden_sizes=args.hidden_sizes,
            save_path=args.save_path,
            device=device
        )

        print("\n[NEXT STEPS]:")
        print("  1. Model is saved in PyTorch format (.pth)")
        print("  2. To use in the game, you'll need to create an inference wrapper")
        print("  3. The model takes 14 numerical features and outputs 4 action probabilities")
        print("     Features: player state (5) + platforms (4) + enemy (2) + goal (2) + difficulty (1)")
        print()

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()

        # Mark as failed in status
        status = {
            'is_training': False,
            'completed': False,
            'error': str(e),
            'timestamp': time.time()
        }
        with open('status.json', 'w') as f:
            json.dump(status, f, indent=2)

        sys.exit(1)
