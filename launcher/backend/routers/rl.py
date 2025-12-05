"""
RL Platformer Router
Proxies requests to RL backend
"""

from fastapi import APIRouter, HTTPException, Header
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
import bcrypt
import jwt
from datetime import datetime, timedelta
import uuid

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


class ScoreEntry(BaseModel):
    """Score entry for leaderboard"""
    name: str
    time: float
    score: int
    distance: int
    difficulty: str = 'easy'  # easy, medium, hard
    timestamp: Optional[str] = None
    date: Optional[str] = None


class UserRegister(BaseModel):
    """User registration request"""
    username: str
    password: str


class UserLogin(BaseModel):
    """User login request"""
    username: str
    password: str


class GameMetrics(BaseModel):
    """Game metrics for a single session"""
    jumps: int
    points: int
    distance: int
    time_played: float


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


@router.get("/debug/paths", summary="Debug Path Detection")
def debug_paths():
    """Debug endpoint to show path detection"""
    file_path = Path(__file__)
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "RL" / "backend",
        Path(__file__).parent.parent.parent / "RL" / "backend",
    ]

    return {
        "current_file": str(file_path),
        "parent": str(file_path.parent),
        "parent.parent": str(file_path.parent.parent),
        "parent.parent.parent": str(file_path.parent.parent.parent),
        "parent.parent.parent.parent": str(file_path.parent.parent.parent.parent),
        "RL_BACKEND_PATH": str(RL_BACKEND_PATH) if RL_BACKEND_PATH else None,
        "RL_BACKEND_PATH_exists": RL_BACKEND_PATH.exists() if RL_BACKEND_PATH else False,
        "possible_paths_checked": [
            {
                "path": str(p),
                "exists": p.exists()
            }
            for p in possible_paths
        ]
    }


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@router.post("/auth/register", summary="Register New User")
def register_user(user_data: UserRegister):
    """
    Register a new user account
    Returns JWT token on success
    """
    users = load_users()

    # Check if username already exists
    if any(u['username'].lower() == user_data.username.lower() for u in users):
        raise HTTPException(status_code=400, detail="Username already exists")

    # Validate username (3-20 chars, alphanumeric + underscore)
    if not user_data.username or len(user_data.username) < 3 or len(user_data.username) > 20:
        raise HTTPException(status_code=400, detail="Username must be 3-20 characters")

    # Validate password (minimum 6 characters)
    if not user_data.password or len(user_data.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    # Create new user
    user_id = str(uuid.uuid4())
    new_user = {
        "user_id": user_id,
        "username": user_data.username,
        "password_hash": hash_password(user_data.password),
        "created_at": datetime.utcnow().isoformat(),
        "total_games": 0,
        "total_jumps": 0,
        "total_points": 0,
        "total_distance": 0,
        "total_playtime": 0.0
    }

    users.append(new_user)
    save_users(users)

    # Generate JWT token
    token = create_jwt_token(user_id, user_data.username)

    return {
        "status": "success",
        "message": f"User {user_data.username} registered successfully",
        "token": token,
        "user": {
            "user_id": user_id,
            "username": user_data.username,
            "created_at": new_user["created_at"]
        }
    }


@router.post("/auth/login", summary="User Login")
def login_user(user_data: UserLogin):
    """
    Login with username and password
    Returns JWT token on success
    """
    users = load_users()

    # Find user
    user = next((u for u in users if u['username'].lower() == user_data.username.lower()), None)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Verify password
    if not verify_password(user_data.password, user['password_hash']):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Generate JWT token
    token = create_jwt_token(user['user_id'], user['username'])

    return {
        "status": "success",
        "message": f"Logged in as {user['username']}",
        "token": token,
        "user": {
            "user_id": user['user_id'],
            "username": user['username'],
            "created_at": user['created_at']
        }
    }


@router.get("/auth/verify", summary="Verify Token")
def verify_token(authorization: Optional[str] = Header(None)):
    """
    Verify JWT token and get current user info
    """
    user = get_current_user(authorization)

    return {
        "status": "valid",
        "user": {
            "user_id": user['user_id'],
            "username": user['username'],
            "created_at": user['created_at'],
            "stats": {
                "total_games": user.get('total_games', 0),
                "total_jumps": user.get('total_jumps', 0),
                "total_points": user.get('total_points', 0),
                "total_distance": user.get('total_distance', 0),
                "total_playtime": user.get('total_playtime', 0.0)
            }
        }
    }


@router.get("/status", summary="Check RL Status")
def check_status():
    """Check if RL backend is available and model exists"""
    model_path = RL_BACKEND_PATH / "models" / "platformer_agent.zip" if RL_BACKEND_PATH else None
    tfjs_model = RL_BACKEND_PATH / "models" / "tfjs_model" / "model.json" if RL_BACKEND_PATH else None

    # Check for any ONNX models
    onnx_models_exist = False
    if RL_BACKEND_PATH:
        models_dir = RL_BACKEND_PATH / "models"
        if models_dir.exists():
            # Check for any episode_*_onnx directories with model.onnx
            onnx_models_exist = any((models_dir / d / "model.onnx").exists()
                                   for d in models_dir.glob("episode_*_onnx"))

    any_model_exists = (tfjs_model.exists() if tfjs_model else False) or onnx_models_exist

    return {
        "backend_available": RL_BACKEND_PATH is not None,
        "backend_path": str(RL_BACKEND_PATH) if RL_BACKEND_PATH else None,
        "model_exists": model_path.exists() if model_path else False,
        "tfjs_model_exists": tfjs_model.exists() if tfjs_model else False,
        "onnx_model_exists": onnx_models_exist,
        "any_model_exists": any_model_exists,
        "gpu_available": check_gpu_available(),
        "ready_for_training": RL_BACKEND_PATH is not None,
        "ready_for_gameplay": any_model_exists
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
        # Use DEVNULL instead of PIPE to avoid Windows asyncio connection errors
        training_process = subprocess.Popen(
            [python_cmd, str(train_script), "--timesteps", str(timesteps)],
            cwd=str(RL_BACKEND_PATH / "training"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
                try:
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

                except (asyncio.CancelledError, GeneratorExit):
                    # Client disconnected - exit gracefully
                    break
                except Exception as e:
                    # Log other errors but continue
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    await asyncio.sleep(1)

        except (asyncio.CancelledError, GeneratorExit, ConnectionResetError):
            # Client disconnected - exit gracefully without error
            pass
        except Exception as e:
            # Log unexpected errors
            try:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            except:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
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
        "num_actions": 9,  # 0=idle, 1=left, 2=right, 3=jump, 4=sprint+right, 5=duck, 6=jump+left, 7=jump+right, 8=sprint+jump+right
        "algorithm": "PPO (Proximal Policy Optimization)"
    }


@router.get("/models/available", summary="List Available Models")
def get_available_models():
    """Get list of all available models for gameplay (both TensorFlow.js and ONNX)"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    models_dir = RL_BACKEND_PATH / "models"
    available_models = []

    # Check for main trained model (TensorFlow.js)
    main_tfjs = models_dir / "tfjs_model" / "model.json"
    if main_tfjs.exists():
        available_models.append({
            "id": "main_tfjs",
            "name": "Main Model (TensorFlow.js)",
            "path": "/models/rl/tfjs_model/model.json",
            "description": "Primary trained model",
            "type": "main",
            "format": "tfjs"
        })

    # Check for main trained model (ONNX)
    main_onnx = models_dir / "exported_onnx" / "model.onnx"
    if main_onnx.exists():
        available_models.append({
            "id": "main_onnx",
            "name": "Main Model (ONNX)",
            "path": str(main_onnx),
            "description": "Primary trained model (ONNX format)",
            "type": "main",
            "format": "onnx"
        })

    # Check for episode-specific models
    if models_dir.exists():
        # TensorFlow.js episode models
        for episode_dir in models_dir.glob("episode_*_tfjs"):
            tfjs_model = episode_dir / "tfjs_model" / "model.json"
            if tfjs_model.exists():
                try:
                    episode_num = int(episode_dir.name.split("_")[1])
                    available_models.append({
                        "id": f"episode_{episode_num}_tfjs",
                        "name": f"Episode {episode_num} (TensorFlow.js)",
                        "path": f"/models/{episode_dir.name}/tfjs_model/model.json",
                        "description": f"Episode {episode_num} checkpoint",
                        "type": "episode",
                        "episode": episode_num,
                        "format": "tfjs"
                    })
                except (IndexError, ValueError):
                    pass

        # ONNX episode models
        for episode_dir in models_dir.glob("episode_*_onnx"):
            onnx_model = episode_dir / "model.onnx"
            if onnx_model.exists():
                try:
                    episode_num = int(episode_dir.name.split("_")[1])

                    # Try to read reward from checkpoint manifest
                    manifest_path = RL_BACKEND_PATH / "training" / "episode_checkpoints" / "manifest.json"
                    reward_info = ""
                    if manifest_path.exists():
                        try:
                            with open(manifest_path, 'r') as f:
                                checkpoints = json.load(f)
                                checkpoint = next((c for c in checkpoints if c['episode'] == episode_num), None)
                                if checkpoint:
                                    reward_info = f" (Reward: {checkpoint['reward']:.1f})"
                        except:
                            pass

                    available_models.append({
                        "id": f"episode_{episode_num}_onnx",
                        "name": f"Episode {episode_num} (ONNX){reward_info}",
                        "path": f"/api/rl/models/onnx/{episode_num}/model.onnx",
                        "description": f"Episode {episode_num} checkpoint",
                        "type": "episode",
                        "episode": episode_num,
                        "format": "onnx"
                    })
                except (IndexError, ValueError):
                    pass

    # Sort by episode number (newest first), with main models at top
    available_models.sort(key=lambda m: (
        0 if m['type'] == 'main' else 1,  # Main models first
        -m.get('episode', 0)  # Then by episode (descending)
    ))

    return {
        "models": available_models,
        "count": len(available_models)
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
        # Check if export virtual environment exists
        export_env_python = RL_BACKEND_PATH / "export_env" / "Scripts" / "python.exe"
        if export_env_python.exists():
            python_cmd = str(export_env_python)
            print(f"[EXPORT] Using export environment: {python_cmd}")
        else:
            python_cmd = sys.executable
            print(f"[EXPORT] WARNING: Export environment not found")
            print(f"[EXPORT] Run setup_export_env.bat in RL/backend to create it")

        # Run export script
        result = subprocess.run(
            [python_cmd, str(export_script)],
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


@router.get("/checkpoints/episodes", summary="Get Episode Checkpoints")
def get_episode_checkpoints():
    """Get list of recent episode checkpoints (last 10)"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    manifest_path = RL_BACKEND_PATH / "training" / "episode_checkpoints" / "manifest.json"

    if not manifest_path.exists():
        return {
            "checkpoints": [],
            "message": "No episode checkpoints found. Start training to create checkpoints."
        }

    try:
        with open(manifest_path, 'r') as f:
            checkpoints = json.load(f)

        # Find the best model (highest reward)
        best_model = max(checkpoints, key=lambda x: x['reward']) if checkpoints else None

        # Get 10 most recent episodes sorted by episode number (newest first)
        recent_checkpoints = sorted(checkpoints, key=lambda x: x['episode'], reverse=True)[:10]

        return {
            "best_model": best_model,
            "recent_checkpoints": recent_checkpoints,
            "total_count": len(checkpoints)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read checkpoints: {str(e)}")


@router.post("/checkpoints/export-onnx/{episode}", summary="Export Episode Checkpoint to ONNX")
def export_episode_checkpoint_onnx(episode: int):
    """
    Export a specific episode checkpoint to ONNX format for ONNX Runtime Web
    This is the recommended export method - avoids TensorFlow conversion issues
    """
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    # Read manifest to find checkpoint
    manifest_path = RL_BACKEND_PATH / "training" / "episode_checkpoints" / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="No episode checkpoints found")

    try:
        with open(manifest_path, 'r') as f:
            checkpoints = json.load(f)

        # Find the checkpoint
        checkpoint = next((c for c in checkpoints if c['episode'] == episode), None)
        if not checkpoint:
            raise HTTPException(status_code=404, detail=f"Checkpoint for episode {episode} not found")

        model_path = RL_BACKEND_PATH / "training" / (checkpoint['path'] + '.zip')
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint file not found: {model_path}")

        # Export to ONNX using new export script
        export_script = RL_BACKEND_PATH / "training" / "export_model_onnx.py"
        if not export_script.exists():
            raise HTTPException(status_code=500, detail="ONNX export script not found")

        # Output directory for this episode
        output_dir = RL_BACKEND_PATH / "models" / f"episode_{episode}_onnx"

        # Check if export virtual environment exists
        export_env_python = RL_BACKEND_PATH / "export_env" / "Scripts" / "python.exe"
        if export_env_python.exists():
            python_cmd = str(export_env_python)
            print(f"[EXPORT] Using export environment: {python_cmd}")
        else:
            python_cmd = sys.executable
            print(f"[EXPORT] WARNING: Export environment not found")
            print(f"[EXPORT] Using system Python (may have dependency issues)")

        cmd = [
            python_cmd,
            str(export_script),
            "--model-path", str(model_path),
            "--output-dir", str(output_dir)
        ]

        print(f"[EXPORT] Running ONNX export: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        print(f"[EXPORT] Return code: {result.returncode}")
        if result.stdout:
            print(f"[EXPORT] stdout: {result.stdout}")
        if result.stderr:
            print(f"[EXPORT] stderr: {result.stderr}")

        if result.returncode == 0:
            return {
                "status": "success",
                "episode": episode,
                "reward": checkpoint['reward'],
                "model_path": str(output_dir / "model.onnx"),
                "format": "ONNX",
                "runtime": "ONNX Runtime Web",
                "message": f"Episode {episode} exported to ONNX successfully",
                "usage_guide": str(output_dir / "USAGE.md")
            }
        else:
            error_msg = f"ONNX export failed: {result.stderr}"
            print(f"[EXPORT ERROR] {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Export timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export to ONNX: {str(e)}")


@router.post("/checkpoints/export/{episode}", summary="Export Episode Checkpoint")
def export_episode_checkpoint(episode: int):
    """
    Export a specific episode checkpoint to TensorFlow.js format
    This allows playing against that specific episode's AI
    NOTE: This endpoint has known issues with Conv layer conversion.
    Use /checkpoints/export-onnx/{episode} instead for ONNX Runtime Web (recommended)
    """
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    # Read manifest to find checkpoint
    manifest_path = RL_BACKEND_PATH / "training" / "episode_checkpoints" / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="No episode checkpoints found")

    try:
        with open(manifest_path, 'r') as f:
            checkpoints = json.load(f)

        # Find the checkpoint
        checkpoint = next((c for c in checkpoints if c['episode'] == episode), None)
        if not checkpoint:
            raise HTTPException(status_code=404, detail=f"Checkpoint for episode {episode} not found")

        model_path = RL_BACKEND_PATH / "training" / (checkpoint['path'] + '.zip')
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint file not found: {model_path}")

        # Export to TensorFlow.js
        export_script = RL_BACKEND_PATH / "training" / "export_model.py"
        if not export_script.exists():
            raise HTTPException(status_code=500, detail="Export script not found")

        # Run export with specific output directory
        output_dir = RL_BACKEND_PATH / "models" / f"episode_{episode}_tfjs"

        # Check if export virtual environment exists
        export_env_python = RL_BACKEND_PATH / "export_env" / "Scripts" / "python.exe"
        if export_env_python.exists():
            python_cmd = str(export_env_python)
            print(f"[EXPORT] Using export environment: {python_cmd}")
        else:
            python_cmd = sys.executable
            print(f"[EXPORT] WARNING: Export environment not found at {export_env_python}")
            print(f"[EXPORT] Using system Python: {python_cmd}")
            print(f"[EXPORT] Run setup_export_env.bat in RL/backend to create export environment")

        cmd = [
            python_cmd,
            str(export_script),
            "--model-path", str(model_path),
            "--output-dir", str(output_dir)
        ]

        print(f"[EXPORT] Running command: {' '.join(cmd)}")
        print(f"[EXPORT] Model path: {model_path}")
        print(f"[EXPORT] Output dir: {output_dir}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        print(f"[EXPORT] Return code: {result.returncode}")
        if result.stdout:
            print(f"[EXPORT] stdout: {result.stdout}")
        if result.stderr:
            print(f"[EXPORT] stderr: {result.stderr}")

        if result.returncode == 0:
            return {
                "status": "success",
                "episode": episode,
                "reward": checkpoint['reward'],
                "model_path": f"/models/episode_{episode}_tfjs/tfjs_model/model.json",
                "message": f"Episode {episode} exported successfully"
            }
        else:
            error_msg = f"Export failed with code {result.returncode}. stderr: {result.stderr}"
            print(f"[EXPORT ERROR] {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Export timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export checkpoint: {str(e)}")


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


@router.get("/models/onnx/{episode}/model.onnx", summary="Get ONNX Model File")
def get_onnx_model(episode: int):
    """Serve ONNX model file for browser loading"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    model_path = RL_BACKEND_PATH / "models" / f"episode_{episode}_onnx" / "model.onnx"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"ONNX model for episode {episode} not found")

    return FileResponse(
        model_path,
        media_type="application/octet-stream",
        filename=f"episode_{episode}_model.onnx"
    )


# Database Paths
DATA_DIR = Path(__file__).parent.parent / "data"
SCORES_DB_PATH = DATA_DIR / "rl_scores.json"
USERS_DB_PATH = DATA_DIR / "rl_users.json"
METRICS_DB_PATH = DATA_DIR / "rl_metrics.json"

# JWT Secret (in production, use environment variable)
JWT_SECRET = "your-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_DAYS = 30


# ============================================================================
# USER AUTHENTICATION FUNCTIONS
# ============================================================================

def load_users() -> List[Dict[str, Any]]:
    """Load users from JSON file"""
    if not USERS_DB_PATH.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return []

    try:
        with open(USERS_DB_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading users: {e}")
        return []


def save_users(users: List[Dict[str, Any]]):
    """Save users to JSON file"""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(USERS_DB_PATH, 'w') as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        print(f"Error saving users: {e}")
        raise


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def create_jwt_token(user_id: str, username: str) -> str:
    """Create a JWT token for authentication"""
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(days=JWT_EXPIRATION_DAYS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Get current user from Authorization header"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization.replace("Bearer ", "")
    payload = verify_jwt_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    users = load_users()
    user = next((u for u in users if u['user_id'] == payload['user_id']), None)

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


# ============================================================================
# USER METRICS FUNCTIONS
# ============================================================================

def update_user_metrics(user_id: str, metrics: GameMetrics):
    """Update user's cumulative metrics"""
    users = load_users()

    user = next((u for u in users if u['user_id'] == user_id), None)
    if not user:
        return False

    # Update cumulative stats
    user['total_games'] = user.get('total_games', 0) + 1
    user['total_jumps'] = user.get('total_jumps', 0) + metrics.jumps
    user['total_points'] = user.get('total_points', 0) + metrics.points
    user['total_distance'] = user.get('total_distance', 0) + metrics.distance
    user['total_playtime'] = user.get('total_playtime', 0.0) + metrics.time_played

    save_users(users)
    return True


# ============================================================================
# SCOREBOARD FUNCTIONS
# ============================================================================

def load_scores() -> List[Dict[str, Any]]:
    """Load scores from JSON file"""
    if not SCORES_DB_PATH.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return []

    try:
        with open(SCORES_DB_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading scores: {e}")
        return []


def save_scores(scores: List[Dict[str, Any]]):
    """Save scores to JSON file"""
    try:
        SCORES_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SCORES_DB_PATH, 'w') as f:
            json.dump(scores, f, indent=2)
    except Exception as e:
        print(f"Error saving scores: {e}")
        raise


@router.get("/scores", summary="Get Leaderboard Scores")
def get_scores(limit: int = 10, difficulty: str = 'easy'):
    """
    Get top scores from the global leaderboard
    Returns scores sorted by completion time (fastest first)
    Filters by difficulty level (easy, medium, hard)
    """
    scores = load_scores()

    # Filter by difficulty
    filtered_scores = [s for s in scores if s.get('difficulty', 'easy') == difficulty]

    # Sort by time (fastest first) and limit results
    sorted_scores = sorted(filtered_scores, key=lambda x: x['time'])[:limit]

    return {
        "scores": sorted_scores,
        "total_count": len(filtered_scores),
        "returned_count": len(sorted_scores),
        "difficulty": difficulty
    }


@router.post("/scores", summary="Submit Score to Leaderboard")
def submit_score(score_entry: ScoreEntry, authorization: Optional[str] = Header(None)):
    """
    Submit a new score to the global leaderboard
    If logged in, score is linked to user_id
    If player already has a score for this difficulty, only updates if new time is better (faster)
    """
    from datetime import datetime

    # Get user if authenticated
    user = None
    user_id = None
    try:
        if authorization:
            user = get_current_user(authorization)
            user_id = user['user_id']
    except:
        pass  # Allow unauthenticated submissions

    # Load existing scores
    scores = load_scores()

    # Create new score entry
    new_score = {
        "name": score_entry.name.strip() or "Anonymous",
        "time": round(score_entry.time, 1),
        "score": score_entry.score,
        "distance": score_entry.distance,
        "difficulty": score_entry.difficulty,
        "timestamp": datetime.utcnow().isoformat(),
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "user_id": user_id  # Link to user if authenticated
    }

    # Check if player already exists for this difficulty
    existing_index = None
    for i, s in enumerate(scores):
        # Match by user_id if authenticated, otherwise by name
        if user_id:
            if s.get('user_id') == user_id and s.get('difficulty', 'easy') == new_score['difficulty']:
                existing_index = i
                break
        else:
            if (s['name'].lower() == new_score['name'].lower() and
                s.get('difficulty', 'easy') == new_score['difficulty'] and
                not s.get('user_id')):  # Only match unlinked scores
                existing_index = i
                break

    if existing_index is not None:
        # Only update if new time is better (faster)
        if new_score['time'] < scores[existing_index]['time']:
            previous_time = scores[existing_index]['time']
            scores[existing_index] = new_score
            save_scores(scores)
            return {
                "status": "updated",
                "message": f"New best time for {new_score['name']} on {new_score['difficulty']}!",
                "score": new_score,
                "previous_time": previous_time
            }
        else:
            return {
                "status": "not_updated",
                "message": f"Previous time was better",
                "current_best": scores[existing_index]['time'],
                "submitted_time": new_score['time']
            }
    else:
        # Add new player for this difficulty
        scores.append(new_score)
        save_scores(scores)
        return {
            "status": "created",
            "message": f"Score added for {new_score['name']} on {new_score['difficulty']}",
            "score": new_score
        }


@router.delete("/scores", summary="Clear All Scores")
def clear_scores():
    """
    Clear all scores from the leaderboard
    WARNING: This is irreversible!
    """
    try:
        if SCORES_DB_PATH.exists():
            SCORES_DB_PATH.unlink()
        return {
            "status": "success",
            "message": "All scores have been cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear scores: {str(e)}")


@router.delete("/scores/{player_name}/{difficulty}", summary="Delete Individual Score")
def delete_score(player_name: str, difficulty: str):
    """
    Delete a specific player's score from a specific difficulty
    Used to remove leaderboard abuse entries
    """
    try:
        scores = load_scores()

        # Find and remove the score
        initial_count = len(scores)
        scores = [s for s in scores if not (
            s['name'].lower() == player_name.lower() and
            s.get('difficulty', 'easy') == difficulty
        )]

        if len(scores) == initial_count:
            raise HTTPException(status_code=404, detail=f"Score not found for {player_name} on {difficulty}")

        save_scores(scores)

        return {
            "status": "success",
            "message": f"Deleted score for {player_name} on {difficulty}",
            "remaining_scores": len(scores)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete score: {str(e)}")


# ============================================================================
# USER METRICS ENDPOINTS
# ============================================================================

@router.post("/metrics/submit", summary="Submit Game Metrics")
def submit_metrics(metrics: GameMetrics, authorization: Optional[str] = Header(None)):
    """
    Submit game session metrics (jumps, points, distance, time)
    Updates user's cumulative statistics
    Requires authentication
    """
    user = get_current_user(authorization)

    success = update_user_metrics(user['user_id'], metrics)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update metrics")

    # Get updated user stats
    users = load_users()
    updated_user = next((u for u in users if u['user_id'] == user['user_id']), None)

    return {
        "status": "success",
        "message": "Metrics updated successfully",
        "stats": {
            "total_games": updated_user.get('total_games', 0),
            "total_jumps": updated_user.get('total_jumps', 0),
            "total_points": updated_user.get('total_points', 0),
            "total_distance": updated_user.get('total_distance', 0),
            "total_playtime": updated_user.get('total_playtime', 0.0)
        }
    }


@router.get("/metrics/stats", summary="Get User Stats")
def get_user_stats(authorization: Optional[str] = Header(None)):
    """
    Get current user's statistics
    Requires authentication
    """
    user = get_current_user(authorization)

    return {
        "status": "success",
        "username": user['username'],
        "stats": {
            "total_games": user.get('total_games', 0),
            "total_jumps": user.get('total_jumps', 0),
            "total_points": user.get('total_points', 0),
            "total_distance": user.get('total_distance', 0),
            "total_playtime": user.get('total_playtime', 0.0)
        },
        "created_at": user['created_at']
    }
