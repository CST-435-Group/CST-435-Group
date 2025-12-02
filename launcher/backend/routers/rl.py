"""
RL Platformer Router
Proxies requests to RL backend
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import subprocess
import os
import sys
from pathlib import Path

router = APIRouter(prefix="/rl", tags=["RL Platformer"])

# Find RL backend path
RL_BACKEND_PATH = None
for possible_path in [
    Path(__file__).parent.parent.parent.parent / "RL" / "backend",
    Path(__file__).parent.parent.parent / "RL" / "backend",
]:
    if possible_path.exists():
        RL_BACKEND_PATH = possible_path
        break

if RL_BACKEND_PATH:
    sys.path.insert(0, str(RL_BACKEND_PATH))


class TrainingStatus(BaseModel):
    """Training status response"""
    is_training: bool
    current_step: int
    total_steps: int
    best_reward: float
    avg_reward: float


class ModelInfo(BaseModel):
    """Model information"""
    is_loaded: bool
    model_path: Optional[str]
    input_shape: Optional[tuple]
    num_actions: int


@router.get("/", summary="RL API Info")
def rl_info():
    """Get RL API information"""
    return {
        "project": "RL Platformer",
        "description": "Reinforcement Learning side-scrolling platformer - Human vs AI racing",
        "features": [
            "Procedurally generated levels",
            "Visual-based RL agent",
            "Human vs AI racing",
            "1920x1080 HD gameplay",
            "PyTorch GPU training"
        ],
        "endpoints": [
            "/rl/status - Check if model exists",
            "/rl/training/status - Get training status",
            "/rl/training/start - Start training",
            "/rl/training/stop - Stop training",
            "/rl/model/info - Get model information",
            "/rl/model/export - Export model to TensorFlow.js"
        ]
    }


@router.get("/status", summary="Check RL Status")
def check_status():
    """Check if RL backend is available and model exists"""
    model_path = RL_BACKEND_PATH / "models" / "platformer_agent.zip" if RL_BACKEND_PATH else None
    tfjs_model = RL_BACKEND_PATH / "models" / "tfjs_model" / "model.json" if RL_BACKEND_PATH else None

    return {
        "backend_available": RL_BACKEND_PATH is not None,
        "backend_path": str(RL_BACKEND_PATH) if RL_BACKEND_PATH else None,
        "model_exists": model_path.exists() if model_path else False,
        "tfjs_model_exists": tfjs_model.exists() if tfjs_model else False,
        "gpu_available": check_gpu_available(),
        "ready_for_training": RL_BACKEND_PATH is not None,
        "ready_for_gameplay": tfjs_model.exists() if tfjs_model else False
    }


@router.get("/training/status", summary="Get Training Status")
def get_training_status():
    """Get current training status"""
    # TODO: Implement training status tracking
    # This would check a shared status file or database
    return {
        "is_training": False,
        "current_step": 0,
        "total_steps": 1000000,
        "best_reward": 0.0,
        "avg_reward": 0.0,
        "message": "No training in progress"
    }


@router.post("/training/start", summary="Start Training")
def start_training(timesteps: int = 1000000):
    """
    Start RL agent training
    This launches the training script as a background process
    """
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    train_script = RL_BACKEND_PATH / "training" / "train_agent.py"
    if not train_script.exists():
        raise HTTPException(status_code=500, detail="Training script not found")

    try:
        # Launch training in background
        # On Windows, use pythonw to avoid popup window
        python_cmd = sys.executable

        # Start process in background
        subprocess.Popen(
            [python_cmd, str(train_script), "--timesteps", str(timesteps)],
            cwd=str(RL_BACKEND_PATH),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        return {
            "status": "started",
            "message": f"Training started with {timesteps} timesteps",
            "estimated_time_hours": timesteps / 50000,  # Rough estimate
            "note": "Training is running in the background. Check status endpoint for progress."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@router.post("/training/stop", summary="Stop Training")
def stop_training():
    """Stop current training"""
    # TODO: Implement graceful training stop
    return {
        "status": "stopped",
        "message": "Training stop requested"
    }


@router.get("/model/info", summary="Get Model Info")
def get_model_info():
    """Get information about trained model"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    model_path = RL_BACKEND_PATH / "models" / "platformer_agent.zip"
    tfjs_path = RL_BACKEND_PATH / "models" / "tfjs_model" / "model.json"

    return {
        "pytorch_model": {
            "exists": model_path.exists(),
            "path": str(model_path) if model_path.exists() else None
        },
        "tfjs_model": {
            "exists": tfjs_path.exists(),
            "path": str(tfjs_path) if tfjs_path.exists() else None
        },
        "input_shape": [84, 84, 3],  # Observation shape
        "num_actions": 6,  # Left, Right, Jump, Sprint, Duck, Idle
        "algorithm": "PPO (Proximal Policy Optimization)"
    }


@router.post("/model/export", summary="Export Model")
def export_model():
    """Export PyTorch model to TensorFlow.js format"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    export_script = RL_BACKEND_PATH / "training" / "export_model.py"
    if not export_script.exists():
        raise HTTPException(status_code=500, detail="Export script not found")

    model_path = RL_BACKEND_PATH / "models" / "platformer_agent.zip"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="No trained model found. Train a model first.")

    try:
        # Run export script
        result = subprocess.run(
            [sys.executable, str(export_script)],
            cwd=str(RL_BACKEND_PATH),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Model exported to TensorFlow.js format",
                "output_path": str(RL_BACKEND_PATH / "models" / "tfjs_model"),
                "details": result.stdout
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Export failed: {result.stderr}"
            )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Export timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


def check_gpu_available() -> bool:
    """Check if CUDA GPU is available for training"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@router.get("/gpu/info", summary="Get GPU Information")
def get_gpu_info():
    """Get GPU information for training"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "cuda_available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "device_capability": torch.cuda.get_device_capability(0),
                "memory_allocated": torch.cuda.memory_allocated(0),
                "memory_reserved": torch.cuda.memory_reserved(0)
            }
        else:
            return {
                "cuda_available": False,
                "message": "No CUDA GPU detected. Training will use CPU (slower)."
            }
    except ImportError:
        return {
            "cuda_available": False,
            "message": "PyTorch not installed"
        }
