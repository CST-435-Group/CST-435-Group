"""
Training Script for RL Agent
Uses Stable-Baselines3 to train agent on randomly generated levels
Automatically uses GPU if available via PyTorch CUDA
"""

# IMPORTANT: Set SDL to headless mode BEFORE importing pygame or environment
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Headless mode for training
os.environ['SDL_AUDIODRIVER'] = 'dummy'  # No audio needed

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from environment import PlatformerEnv
import torch
import argparse
import json
import time
import numpy as np


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
    # Create the platformer environment
    env = PlatformerEnv(
        render_width=1920,
        render_height=1080,
        observation_width=84,
        observation_height=84,
        headless=True,  # No GUI window for training
        capture_frames=False  # Frame capture handled by callback
    )

    # Wrap in DummyVecEnv for Stable-Baselines3 compatibility
    env = DummyVecEnv([lambda: env])

    return env


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
        net_arch=dict(pi=[256, 256], vf=[256, 256])  # Larger networks for complex visual input
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


class ProgressCallback(BaseCallback):
    """
    Custom callback to track and save training progress.
    Writes status.json and captures frames for intelligent visualization.
    """

    def __init__(self, status_file="status.json",
                 frame_dir="frames",
                 log_file="training.log",
                 save_freq=10, frame_save_freq=100,
                 verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.status_file = status_file
        self.frame_dir = frame_dir
        self.log_file = log_file
        self.save_freq = save_freq
        self.frame_save_freq = frame_save_freq

        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -float('inf')

        # Create directories
        status_dir = os.path.dirname(status_file)
        if status_dir:  # Only create if there's a directory component
            os.makedirs(status_dir, exist_ok=True)
        os.makedirs(frame_dir, exist_ok=True)

    def _on_step(self) -> bool:
        """Called at every environment step"""

        # Update status file every N steps
        if self.num_timesteps % self.save_freq == 0:
            status = {
                'is_training': True,
                'current_step': self.num_timesteps,
                'total_steps': self.locals.get('total_timesteps', 0),
                'progress': self.num_timesteps / max(self.locals.get('total_timesteps', 1), 1),
                'episodes': len(self.episode_rewards),
                'avg_reward': float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0,
                'best_reward': float(self.best_reward),
                'avg_length': float(np.mean(self.episode_lengths[-100:])) if self.episode_lengths else 0.0,
                'fps': 0,  # Would need timing to calculate
                'timestamp': time.time()
            }

            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)

        # Check for episode end
        if 'dones' in self.locals and self.locals['dones'][0]:
            infos = self.locals.get('infos', [{}])
            if len(infos) > 0:
                info = infos[0]

                # Track episode metrics
                ep_reward = info.get('episode', {}).get('r', 0)
                ep_length = info.get('episode', {}).get('l', 0)

                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                # Update best reward
                if ep_reward > self.best_reward:
                    self.best_reward = ep_reward
                    # Save best model
                    self.model.save('models/platformer_agent_best')

                # Intelligent frame capture (save interesting episodes)
                should_save = self._should_save_frame(
                    len(self.episode_rewards),
                    ep_reward
                )

                if should_save:
                    # Capture frame from environment
                    try:
                        frame_filename = f"{self.frame_dir}/episode_{len(self.episode_rewards):05d}_reward_{int(ep_reward)}.png"
                        # Note: Frame capture would need environment access
                        # For now, just log that we would save
                        if self.verbose > 0:
                            print(f"[FRAME] Would save frame: {frame_filename}")
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"[FRAME] Error saving frame: {e}")

        return True  # Continue training

    def _should_save_frame(self, episode_num, episode_reward):
        """
        Intelligent frame skipping logic.
        Save frames for:
        - First 10 episodes (for debugging)
        - Every 100th episode
        - New best reward
        - Reward > 80% of best (interesting episodes)
        """
        if episode_num <= 10:
            return True
        if episode_num % self.frame_save_freq == 0:
            return True
        if episode_reward >= self.best_reward:
            return True
        if self.best_reward > 0 and episode_reward >= self.best_reward * 0.8:
            return True
        return False


def evaluate_agent(model, num_episodes=10):
    """
    Evaluate trained agent performance.

    Args:
        model: Trained model
        num_episodes: Number of test episodes

    Returns:
        dict: Evaluation metrics (avg_reward, success_rate, avg_distance)
    """
    env = create_env()

    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            done = dones[0]
            episode_reward += reward[0]
            steps += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(steps)
                if info[0].get('goal_reached', False):
                    success_count += 1

    env.close()

    return {
        'avg_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'avg_length': float(np.mean(episode_lengths)),
        'success_rate': success_count / num_episodes,
        'num_episodes': num_episodes
    }


def visualize_agent(model, num_episodes=5):
    """
    Run agent and render gameplay for visualization.

    Args:
        model: Trained model
        num_episodes: Number of episodes to show
    """
    env = create_env()

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        print(f"\n[VIZ] Episode {episode + 1}/{num_episodes}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            done = dones[0]
            episode_reward += reward[0]

            if done:
                print(f"[VIZ] Reward: {episode_reward:.2f}, Steps: {info[0].get('episode_length', 0)}")
                if info[0].get('goal_reached', False):
                    print("[VIZ] Goal reached!")
                elif not info[0].get('died', True):
                    print("[VIZ] Timeout")
                else:
                    print("[VIZ] Died")

    env.close()


def setup_callbacks(save_freq=10000):
    """
    Setup training callbacks for checkpointing and logging.

    Args:
        save_freq: Save checkpoint every N steps

    Returns:
        list: Callback objects
    """
    # Progress tracking callback
    progress_callback = ProgressCallback(
        status_file="status.json",
        frame_dir="frames",
        log_file="training.log",
        save_freq=10,
        frame_save_freq=100,
        verbose=1
    )

    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path='checkpoints/',
        name_prefix='checkpoint',
        verbose=1
    )

    return [progress_callback, checkpoint_callback]


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
