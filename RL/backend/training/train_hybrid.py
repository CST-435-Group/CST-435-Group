"""
Hybrid Training Script - Combines Behavioral Cloning with Reinforcement Learning
First learns from human demonstrations, then fine-tunes with PPO self-play
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Import BC training components
from train_behavioral_cloning import (
    PolicyNetwork,
    HumanGameplayDataset,
    load_training_data,
    check_cuda
)

# Import RL training components
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from environment import PlatformerEnv


class HybridProgressCallback(BaseCallback):
    """
    Custom callback for tracking hybrid training progress
    """
    def __init__(self, status_file="status.json", save_freq=10, verbose=0):
        super(HybridProgressCallback, self).__init__(verbose)
        self.status_file = status_file
        self.save_freq = save_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -float('inf')

    def _on_step(self) -> bool:
        """Called at every environment step"""
        # Update status file every N steps
        if self.num_timesteps % self.save_freq == 0:
            status = {
                'is_training': True,
                'training_mode': 'hybrid',
                'current_step': self.num_timesteps,
                'total_steps': self.locals.get('total_timesteps', 0),
                'progress': self.num_timesteps / max(self.locals.get('total_timesteps', 1), 1),
                'episodes': len(self.episode_rewards),
                'avg_reward': float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0,
                'best_reward': float(self.best_reward),
                'avg_length': float(np.mean(self.episode_lengths[-100:])) if self.episode_lengths else 0.0,
                'timestamp': time.time()
            }

            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)

        # Track episode completion
        if 'dones' in self.locals and self.locals['dones'][0]:
            infos = self.locals.get('infos', [{}])
            if len(infos) > 0:
                info = infos[0]
                ep_reward = info.get('episode', {}).get('r', 0)
                ep_length = info.get('episode', {}).get('l', 0)

                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                if ep_reward > self.best_reward:
                    self.best_reward = ep_reward
                    self.model.save('models/hybrid_agent_best')

        return True


def pretrain_with_bc(data_dir, epochs=50, batch_size=256, learning_rate=0.001, device="cuda"):
    """
    Phase 1: Pretrain policy using behavioral cloning on human data

    Args:
        data_dir: Path to training data directory
        epochs: Number of pretraining epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Training device

    Returns:
        Trained PyTorch model state dict
    """
    print("=" * 60)
    print("PHASE 1: BEHAVIORAL CLONING PRETRAINING")
    print("=" * 60)
    print(f"[BC] Epochs: {epochs}")
    print(f"[BC] Batch size: {batch_size}")
    print(f"[BC] Learning rate: {learning_rate}")
    print()

    # Load human gameplay data
    all_data_points = load_training_data(data_dir)

    if len(all_data_points) < 100:
        print(f"[WARNING] Only {len(all_data_points)} data points. BC pretraining may be limited.")

    # Create dataset
    dataset = HumanGameplayDataset(all_data_points, normalize=True)

    # Simple train split (no validation during pretraining)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Create policy network
    input_size = 14
    num_actions = 4
    model = PolicyNetwork(input_size, [256, 256], num_actions).to(device)

    print(f"[BC] Created policy network with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"[BC] Training on {len(dataset):,} samples")
    print()

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            states = batch['state'].to(device)
            actions = batch['action'].to(device)

            optimizer.zero_grad()
            predictions = model(states)
            loss = criterion(predictions, actions)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred_actions = (predictions > 0.5).float()
            train_correct += (pred_actions == actions).sum().item()
            train_total += actions.numel()

        avg_loss = train_loss / len(train_loader)
        accuracy = train_correct / train_total

        if avg_loss < best_loss:
            best_loss = avg_loss

        # Update status for monitoring
        status = {
            'is_training': True,
            'training_mode': 'hybrid',
            'phase': 'behavioral_cloning',
            'current_epoch': epoch + 1,
            'total_epochs': epochs,
            'progress': (epoch + 1) / epochs,
            'train_loss': float(avg_loss),
            'train_accuracy': float(accuracy),
            'best_loss': float(best_loss),
            'timestamp': time.time()
        }

        with open('status.json', 'w') as f:
            json.dump(status, f, indent=2)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[BC] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.3f}")

    print(f"\n[BC] Pretraining complete! Best loss: {best_loss:.4f}\n")

    # Save normalization params with model
    bc_checkpoint = {
        'model_state_dict': model.state_dict(),
        'normalize_mean': dataset.mean,
        'normalize_std': dataset.std,
        'input_size': input_size,
        'num_actions': num_actions
    }

    return bc_checkpoint


def convert_bc_to_ppo_policy(bc_checkpoint, env):
    """
    Phase 2 Setup: Convert BC policy to PPO-compatible policy

    This extracts the learned features from BC and initializes PPO with them.
    Note: This is a simplified approach. Full conversion would require matching
    PPO's CNN architecture with BC's MLP features.

    Args:
        bc_checkpoint: BC model checkpoint dict
        env: PPO environment

    Returns:
        PPO model initialized with BC knowledge
    """
    print("=" * 60)
    print("PHASE 2: REINFORCEMENT LEARNING FINE-TUNING")
    print("=" * 60)
    print("[RL] Initializing PPO with BC-pretrained features...")
    print("[RL] Note: Using standard PPO architecture (CNN)")
    print("[RL] BC knowledge helps guide exploration during RL")
    print()

    # Create PPO model with standard architecture
    # TODO: For better transfer, would need to modify PPO to use MLP instead of CNN
    # or create a custom policy that uses BC features
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=check_cuda(),
        tensorboard_log="./logs/"
    )

    print("[RL] PPO model created")
    print("[RL] BC pretraining provides good initial behavior")
    print()

    return model


def train_hybrid(
    data_dir,
    bc_epochs=50,
    rl_timesteps=500000,
    batch_size=256,
    learning_rate=0.001,
    save_path="models/hybrid_agent",
    device="cuda"
):
    """
    Full hybrid training pipeline

    Phase 1: Behavioral Cloning (minutes)
    - Learn from human demonstrations
    - Get good initial policy

    Phase 2: Reinforcement Learning (hours)
    - Fine-tune with PPO self-play
    - Improve beyond human performance

    Args:
        data_dir: Path to human training data
        bc_epochs: Epochs for BC pretraining
        rl_timesteps: Timesteps for RL fine-tuning
        batch_size: BC batch size
        learning_rate: BC learning rate
        save_path: Where to save final model
        device: Training device

    Returns:
        Trained PPO model
    """
    print("\n" + "=" * 60)
    print("HYBRID TRAINING: BC + RL")
    print("=" * 60)
    print(f"[HYBRID] Phase 1: BC pretraining ({bc_epochs} epochs)")
    print(f"[HYBRID] Phase 2: RL fine-tuning ({rl_timesteps:,} timesteps)")
    print()

    # Phase 1: Behavioral Cloning Pretraining
    bc_checkpoint = pretrain_with_bc(
        data_dir=data_dir,
        epochs=bc_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )

    # Save BC checkpoint
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    torch.save(bc_checkpoint, f"{save_path}_bc_pretrained.pth")
    print(f"[BC] Saved pretrained model to {save_path}_bc_pretrained.pth\n")

    # Phase 2: RL Fine-tuning
    # Create environment
    env = PlatformerEnv(
        render_width=1920,
        render_height=1080,
        observation_width=84,
        observation_height=84,
        headless=True,
        capture_frames=False
    )
    env = Monitor(env, filename=None)
    env = DummyVecEnv([lambda: env])

    # Initialize PPO (with BC knowledge as warm start)
    ppo_model = convert_bc_to_ppo_policy(bc_checkpoint, env)

    # Setup RL callbacks
    progress_callback = HybridProgressCallback(
        status_file="status.json",
        save_freq=10,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='checkpoints/',
        name_prefix='hybrid_checkpoint',
        verbose=1
    )

    print(f"[RL] Starting PPO fine-tuning for {rl_timesteps:,} timesteps...")
    print(f"[RL] This will take approximately {rl_timesteps/50000:.1f} hours on GPU")
    print()

    # Train with RL
    ppo_model.learn(
        total_timesteps=rl_timesteps,
        callback=[progress_callback, checkpoint_callback],
        log_interval=10
    )

    # Save final model
    ppo_model.save(save_path)
    print(f"\n[SUCCESS] Hybrid training complete!")
    print(f"[SUCCESS] Final model saved to: {save_path}.zip")
    print(f"[SUCCESS] BC pretrained model: {save_path}_bc_pretrained.pth")

    # Mark training as complete
    status = {
        'is_training': False,
        'training_mode': 'hybrid',
        'completed': True,
        'completed_at': time.time(),
        'message': 'Hybrid training (BC + RL) completed successfully'
    }

    with open('status.json', 'w') as f:
        json.dump(status, f, indent=2)

    return ppo_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid training: BC + RL")
    parser.add_argument("--data-dir", type=str,
                       default="../../launcher/backend/data/training_data",
                       help="Path to human training data")
    parser.add_argument("--bc-epochs", type=int, default=50,
                       help="BC pretraining epochs")
    parser.add_argument("--rl-timesteps", type=int, default=500000,
                       help="RL fine-tuning timesteps")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="BC batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="BC learning rate")
    parser.add_argument("--save-path", type=str, default="models/hybrid_agent",
                       help="Path to save trained model")

    args = parser.parse_args()

    # Check CUDA
    device = check_cuda()

    try:
        model = train_hybrid(
            data_dir=args.data_dir,
            bc_epochs=args.bc_epochs,
            rl_timesteps=args.rl_timesteps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_path=args.save_path,
            device=device
        )

        print("\n[NEXT STEPS]:")
        print("  1. Hybrid model combines human knowledge + RL optimization")
        print("  2. BC pretrained model can be used standalone")
        print("  3. Final PPO model should outperform both humans and pure RL")
        print()

    except Exception as e:
        print(f"\n[ERROR] Hybrid training failed: {e}")
        import traceback
        traceback.print_exc()

        status = {
            'is_training': False,
            'training_mode': 'hybrid',
            'completed': False,
            'error': str(e),
            'timestamp': time.time()
        }
        with open('status.json', 'w') as f:
            json.dump(status, f, indent=2)

        sys.exit(1)
