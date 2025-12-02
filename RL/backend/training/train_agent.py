"""
Training Script for RL Agent
Uses Stable-Baselines3 to train agent on randomly generated levels
Automatically uses GPU if available via PyTorch CUDA
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from environment import PlatformerEnv
import torch
import argparse
import os


# Check CUDA availability
def check_cuda():
    """Check if CUDA GPU is available"""
    if torch.cuda.is_available():
        print(f"[GPU] CUDA is available!")
        print(f"[GPU] Device: {torch.cuda.get_device_name(0)}")
        print(f"[GPU] CUDA Version: {torch.version.cuda}")
        print(f"[GPU] Device Count: {torch.cuda.device_count()}")
        return "cuda"
    else:
        print("[CPU] CUDA not available, training will use CPU (slower)")
        return "cpu"


def create_env():
    """
    Create and wrap the environment.

    Returns:
        Wrapped environment ready for training
    """
    pass


def train_agent(total_timesteps=1_000_000, save_path="models/platformer_agent"):
    """
    Train the RL agent using PPO with GPU acceleration.

    Args:
        total_timesteps: Number of training steps (1M = ~4-8 hours on GPU)
        save_path: Where to save trained model

    Returns:
        Trained model
    """
    # Check CUDA availability
    device = check_cuda()

    print(f"\n[TRAINING] Starting RL agent training")
    print(f"[TRAINING] Total timesteps: {total_timesteps:,}")
    print(f"[TRAINING] Device: {device}")
    print(f"[TRAINING] Algorithm: PPO (Proximal Policy Optimization)")

    # Create environment
    env = create_env()

    # Setup callbacks
    callbacks = setup_callbacks()

    # Create PPO model with GPU support
    # policy_kwargs can be customized for deeper networks
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Larger networks for complex visual input
    )

    model = PPO(
        "CnnPolicy",  # CNN policy for visual observations
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
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,  # Use CUDA if available!
        tensorboard_log="./logs/"
    )

    print(f"\n[TRAINING] Model created. Training starting...")
    print(f"[TRAINING] You can monitor progress in TensorBoard:")
    print(f"[TRAINING]   tensorboard --logdir ./logs/")

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10
    )

    # Save final model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)

    print(f"\n[SUCCESS] Training complete!")
    print(f"[SUCCESS] Model saved to: {save_path}.zip")

    return model


def evaluate_agent(model, num_episodes=10):
    """
    Evaluate trained agent performance.

    Args:
        model: Trained model
        num_episodes: Number of test episodes

    Returns:
        dict: Evaluation metrics (avg_reward, success_rate, avg_distance)
    """
    pass


def visualize_agent(model, num_episodes=5):
    """
    Run agent and render gameplay for visualization.

    Args:
        model: Trained model
        num_episodes: Number of episodes to show
    """
    pass


def setup_callbacks(save_freq=10000):
    """
    Setup training callbacks for checkpointing and logging.

    Args:
        save_freq: Save checkpoint every N steps

    Returns:
        list: Callback objects
    """
    pass


if __name__ == "__main__":
    """
    Main training loop.
    Run: python train_agent.py
    Or: python train_agent.py --timesteps 500000
    """
    parser = argparse.ArgumentParser(description="Train RL platformer agent")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1M)"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/platformer_agent",
        help="Path to save trained model"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RL PLATFORMER AGENT TRAINING")
    print("=" * 60)

    # Train the agent
    model = train_agent(
        total_timesteps=args.timesteps,
        save_path=args.save_path
    )

    # Evaluate trained model
    print("\n[EVAL] Evaluating trained model...")
    eval_results = evaluate_agent(model, num_episodes=10)

    print("\n[EVAL] Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value}")

    print("\n[NEXT STEPS]:")
    print("  1. Export model: python training/export_model.py")
    print("  2. Copy model to frontend: cp -r models/tfjs_model frontend/public/models/")
    print("  3. Start frontend: cd frontend && npm start")
    print("\n" + "=" * 60)
