"""
RL Platformer Router
Proxies requests to RL backend
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import subprocess
import os
import sys
import json
import asyncio
import time
import signal
from pathlib import Path

router = APIRouter(prefix="/rl", tags=["RL Platformer"])

# Global training process tracking
training_process: Optional[subprocess.Popen] = None
training_pid: Optional[int] = None

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
    """Get current training status from status.json file"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    status_file = RL_BACKEND_PATH / "training" / "status.json"

    # Check if process is still running
    global training_process, training_pid
    is_process_alive = False
    if training_process and training_process.poll() is None:
        is_process_alive = True
    elif training_pid:
        # Check if PID exists
        try:
            os.kill(training_pid, 0)
            is_process_alive = True
        except (OSError, ProcessLookupError):
            is_process_alive = False

    # Read status file if it exists
    if status_file.exists():
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
                status['process_alive'] = is_process_alive
                status['pid'] = training_pid
                return status
        except Exception as e:
            return {
                "is_training": is_process_alive,
                "error": f"Failed to read status file: {str(e)}",
                "process_alive": is_process_alive,
                "pid": training_pid
            }
    else:
        return {
            "is_training": is_process_alive,
            "current_step": 0,
            "total_steps": 0,
            "progress": 0.0,
            "episodes": 0,
            "avg_reward": 0.0,
            "best_reward": 0.0,
            "message": "Training not started or status file not created yet",
            "process_alive": is_process_alive,
            "pid": training_pid
        }


@router.post("/training/start", summary="Start Training")
def start_training(timesteps: int = 1000000):
    """
    Start RL agent training
    This launches the training script as a background process
    """
    global training_process, training_pid

    # Check if training is already running
    if training_process and training_process.poll() is None:
        raise HTTPException(status_code=400, detail="Training is already running")

    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    train_script = RL_BACKEND_PATH / "training" / "train_agent.py"
    if not train_script.exists():
        raise HTTPException(status_code=500, detail="Training script not found")

    try:
        # Clear old status file
        status_file = RL_BACKEND_PATH / "training" / "status.json"
        if status_file.exists():
            status_file.unlink()

        # Launch training in background
        python_cmd = sys.executable

        # Start process in background
        training_process = subprocess.Popen(
            [python_cmd, str(train_script), "--timesteps", str(timesteps)],
            cwd=str(RL_BACKEND_PATH / "training"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )

        training_pid = training_process.pid

        return {
            "status": "started",
            "pid": training_pid,
            "message": f"Training started with {timesteps} timesteps",
            "estimated_time_hours": timesteps / 50000,  # Rough estimate
            "note": "Training is running in the background. Check /training/status or /training/stream for progress."
        }
    except Exception as e:
        training_process = None
        training_pid = None
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@router.post("/training/stop", summary="Stop Training")
def stop_training():
    """Stop current training process gracefully"""
    global training_process, training_pid

    if not training_process and not training_pid:
        raise HTTPException(status_code=400, detail="No training process is running")

    try:
        # Try to terminate gracefully
        if training_process and training_process.poll() is None:
            # On Windows, send CTRL_BREAK_EVENT
            if os.name == 'nt':
                training_process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                training_process.terminate()

            # Wait up to 10 seconds for graceful shutdown
            try:
                training_process.wait(timeout=10)
                message = "Training stopped gracefully"
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop
                training_process.kill()
                training_process.wait()
                message = "Training forcefully terminated (did not stop gracefully)"
        elif training_pid:
            # Try to kill by PID
            try:
                if os.name == 'nt':
                    os.kill(training_pid, signal.CTRL_BREAK_EVENT)
                else:
                    os.kill(training_pid, signal.SIGTERM)
                message = "Training stop signal sent to PID"
            except (OSError, ProcessLookupError):
                message = "Training process not found (may have already stopped)"

        # Clear tracking variables
        training_process = None
        training_pid = None

        return {
            "status": "stopped",
            "message": message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop training: {str(e)}")


@router.get("/training/stream", summary="Stream Training Status (SSE)")
async def stream_training_status():
    """Stream training status updates using Server-Sent Events"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    status_file = RL_BACKEND_PATH / "training" / "status.json"

    async def event_generator():
        """Generate SSE events with training status updates"""
        last_status = None
        no_file_count = 0

        try:
            while True:
                # Check if training process is still alive
                global training_process, training_pid
                is_process_alive = False
                if training_process and training_process.poll() is None:
                    is_process_alive = True
                elif training_pid:
                    try:
                        os.kill(training_pid, 0)
                        is_process_alive = True
                    except (OSError, ProcessLookupError):
                        is_process_alive = False

                # Read status file
                if status_file.exists():
                    try:
                        with open(status_file, 'r') as f:
                            status = json.load(f)
                            status['process_alive'] = is_process_alive
                            status['pid'] = training_pid

                            # Only send if status changed
                            if status != last_status:
                                yield f"data: {json.dumps(status)}\n\n"
                                last_status = status
                                no_file_count = 0

                    except Exception as e:
                        yield f"data: {json.dumps({'error': f'Failed to read status: {str(e)}', 'process_alive': is_process_alive})}\n\n"
                else:
                    no_file_count += 1
                    # Send a waiting message every 5 seconds if no status file
                    if no_file_count % 5 == 0:
                        yield f"data: {json.dumps({'is_training': is_process_alive, 'message': 'Waiting for training to start...', 'process_alive': is_process_alive})}\n\n"

                # If process died and no more updates, break
                if not is_process_alive and last_status and last_status.get('is_training') == False:
                    yield f"data: {json.dumps({'is_training': False, 'message': 'Training completed', 'process_alive': False})}\n\n"
                    break

                await asyncio.sleep(1)  # Update every second

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/training/frames", summary="Get Training Frames")
def get_training_frames(limit: int = 10):
    """Get list of captured training frames"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    frames_dir = RL_BACKEND_PATH / "training" / "frames"

    if not frames_dir.exists():
        return {"frames": [], "message": "No frames directory found"}

    try:
        # Get all frame files
        frame_files = sorted(frames_dir.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)

        # Limit number of frames
        frame_files = frame_files[:limit]

        frames = []
        for frame_file in frame_files:
            frames.append({
                "filename": frame_file.name,
                "url": f"/rl/training/frame/{frame_file.name}",
                "timestamp": frame_file.stat().st_mtime
            })

        return {
            "frames": frames,
            "total_count": len(list(frames_dir.glob("*.png"))),
            "returned_count": len(frames)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list frames: {str(e)}")


@router.get("/training/frame/{filename}", summary="Get Specific Frame")
def get_training_frame(filename: str):
    """Get a specific training frame image"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    # Security: only allow alphanumeric and underscore/dash/dot in filename
    if not filename.replace("_", "").replace("-", "").replace(".", "").replace("step", "").replace("episode", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid filename")

    frame_path = RL_BACKEND_PATH / "training" / "frames" / filename

    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")

    return FileResponse(frame_path, media_type="image/png")


@router.get("/training/logs", summary="Get Training Logs")
def get_training_logs(lines: int = 100):
    """Get recent training logs"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    # Look for log files in logs directory
    logs_dir = RL_BACKEND_PATH / "logs"

    if not logs_dir.exists():
        return {"logs": "", "message": "No logs directory found"}

    try:
        # Find most recent log file
        log_files = sorted(logs_dir.rglob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)

        if not log_files:
            # Try tensorboard event files
            event_files = sorted(logs_dir.rglob("events.out.tfevents.*"), key=lambda x: x.stat().st_mtime, reverse=True)
            if event_files:
                return {
                    "logs": "",
                    "message": "Training logs are in TensorBoard format. Use TensorBoard to view.",
                    "tensorboard_command": f"tensorboard --logdir {logs_dir}",
                    "event_files": [str(f.name) for f in event_files[:5]]
                }
            return {"logs": "", "message": "No log files found"}

        # Read the most recent log file
        log_file = log_files[0]
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return {
            "logs": "".join(recent_lines),
            "log_file": str(log_file.name),
            "total_lines": len(all_lines),
            "returned_lines": len(recent_lines)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")


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
