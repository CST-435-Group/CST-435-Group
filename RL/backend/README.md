# RL Platformer - Backend Training

## Setup

```bash
cd backend
pip install -r requirements.txt
```

## Training the Agent

```bash
python training/train_agent.py
```

This will:
1. Create the custom Gym environment
2. Generate random maps for each episode
3. Train the agent using PPO (Proximal Policy Optimization)
4. Save checkpoints every 10,000 steps
5. Save final model to `models/platformer_agent.zip`

**Training time:** 4-8 hours on GPU, 12-24 hours on CPU for 1M steps

## Exporting for Web

```bash
python training/export_model.py
```

This converts the trained PyTorch model to TensorFlow.js format for the frontend.

Output: `models/tfjs_model/` directory with model.json and weight files

## File Structure

- `training/environment.py` - Custom Gym environment with visual observations
- `training/map_generator.py` - Procedural level generation
- `training/train_agent.py` - Training script
- `training/export_model.py` - Model export to TensorFlow.js
- `utils/config.py` - Game constants and hyperparameters
- `models/` - Saved models (created during training)
