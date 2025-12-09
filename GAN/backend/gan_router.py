"""
FastAPI Router for GAN Project (Military Vehicle Image Generation)
Dual Conditional GAN for generating synthetic tank images with specific type and view angle
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
import json
import io
import base64
from pathlib import Path
import sys

# Try to use torchvision if available, fall back to mock
try:
    from torchvision.transforms import ToPILImage
except ImportError:
    # Use mock if torchvision not available
    try:
        from torchvision_mock import ToPILImageMock as ToPILImage
    except ImportError:
        # Create a simple fallback
        from PIL import Image
        class ToPILImage:
            def __init__(self, mode='RGB'):
                self.mode = mode
            
            def __call__(self, tensor):
                if tensor.dim() == 3 and tensor.shape[0] == 3:
                    tensor = tensor.permute(1, 2, 0)
                np_array = tensor.cpu().detach().numpy()
                if np_array.max() <= 1.0:
                    np_array = (np_array * 255).astype('uint8')
                else:
                    np_array = np_array.astype('uint8')
                return Image.fromarray(np_array, mode=self.mode)

# Create router
router = APIRouter()

# Global model storage (lazy loading)
gan_generator = None
gan_mappings = None

# Constants
LATENT_DIM = 100
EMBED_DIM = 50

# Try to import GAN modules
GAN_AVAILABLE = False
GAN_PATH = None

def setup_gan_path():
    """Setup path to GAN modules"""
    global GAN_AVAILABLE, GAN_PATH

    possible_paths = [
        Path(__file__).parent.parent,  # GAN/backend -> GAN
        Path(__file__).parent.parent.parent / "GAN",  # launcher/backend/routers -> GAN
    ]

    for path in possible_paths:
        if (path / "models_dual_conditional.py").exists():
            GAN_PATH = path.resolve()
            if str(GAN_PATH) not in sys.path:
                sys.path.insert(0, str(GAN_PATH))
            GAN_AVAILABLE = True
            print(f"[GAN] Found GAN module at: {GAN_PATH}")
            return True

    print("[GAN] Warning: GAN modules not found")
    return False

setup_gan_path()


# Request/Response Models
class GenerateRequest(BaseModel):
    """Request to generate images"""
    tank_type: str = Field(..., description="Tank type (e.g., M1A1_Abrams, Leopard2, T90)")
    view_angle: str = Field(..., description="View angle (e.g., front, side, back)")
    num_images: int = Field(default=1, ge=1, le=16, description="Number of images to generate (1-16)")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class GeneratedImage(BaseModel):
    """A single generated image"""
    image_base64: str
    tank_type: str
    view_angle: str
    index: int


class GenerateResponse(BaseModel):
    """Response with generated images"""
    images: List[GeneratedImage]
    tank_type: str
    view_angle: str
    num_generated: int


class ModelInfo(BaseModel):
    """GAN model information"""
    available_tanks: List[str]
    available_views: List[str]
    available_models: List[str]
    current_model: Optional[str]
    latent_dim: int
    image_size: int


class AvailableModelsResponse(BaseModel):
    """Available model checkpoints"""
    models: List[Dict[str, str]]
    current_model: Optional[str]


def get_model_dir() -> Path:
    """Get the models directory path"""
    if GAN_PATH:
        return GAN_PATH / "models_dual_conditional"
    return Path(__file__).parent.parent / "models_dual_conditional"


def load_model_chunked(model_path: Path):
    """Load model from chunks if manifest exists, otherwise load directly"""
    manifest_path = Path(str(model_path) + '.manifest.json')

    if manifest_path.exists():
        # Load from chunks
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        state_dict = {}
        base_dir = model_path.parent

        for chunk_name in manifest['chunks']:
            chunk_path = base_dir / chunk_name
            chunk_dict = torch.load(chunk_path, map_location='cpu', weights_only=True)
            state_dict.update(chunk_dict)

        print(f"  [GAN] Loaded {len(manifest['chunks'])} chunks from {manifest_path.name}")
        return state_dict
    elif model_path.exists():
        # Load directly
        print(f"  [GAN] Loaded {model_path.name}")
        return torch.load(model_path, map_location='cpu', weights_only=True)
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")


def load_gan_generator(model_name: Optional[str] = None):
    """Lazy load GAN generator model"""
    global gan_generator, gan_mappings

    if not GAN_AVAILABLE:
        raise RuntimeError("GAN modules not available")

    # Import here to avoid import errors if GAN not available
    from models_dual_conditional import DualConditionalGenerator

    model_dir = get_model_dir()

    # Load label mappings
    mappings_path = model_dir / "label_mappings.json"
    if not mappings_path.exists():
        raise FileNotFoundError(f"Label mappings not found: {mappings_path}")

    with open(mappings_path) as f:
        gan_mappings = json.load(f)

    num_tanks = len(gan_mappings['tank_to_idx'])
    num_views = len(gan_mappings['view_to_idx'])

    # Determine which model to load
    if model_name:
        model_path = model_dir / model_name
    else:
        model_path = model_dir / "latest_generator.pth"
        if not model_path.exists():
            # Find the latest epoch model
            epoch_models = sorted(model_dir.glob("generator_epoch_*.pth"))
            if epoch_models:
                model_path = epoch_models[-1]

    # Check if model exists (either as file or chunked with manifest)
    manifest_path = Path(str(model_path) + '.manifest.json')
    if not model_path.exists() and not manifest_path.exists():
        raise FileNotFoundError(f"Generator model not found: {model_path}")

    print(f"[GAN] Loading generator from: {model_path}")

    # Create and load generator
    generator = DualConditionalGenerator(
        latent_dim=LATENT_DIM,
        num_tanks=num_tanks,
        num_views=num_views,
        embed_dim=EMBED_DIM
    )

    # Use chunked loader (handles both chunked and regular models)
    state_dict = load_model_chunked(model_path)
    generator.load_state_dict(state_dict)
    generator.eval()

    gan_generator = generator
    print(f"[GAN] Generator loaded successfully!")

    return generator, gan_mappings


def generate_images_internal(tank_idx: int, view_idx: int, num_images: int, seed: Optional[int] = None):
    """Generate images using the loaded generator"""
    global gan_generator

    if gan_generator is None:
        load_gan_generator()

    device = torch.device('cpu')  # Use CPU for server deployment

    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        noise = torch.randn(num_images, LATENT_DIM, device=device)
        tank_labels = torch.full((num_images,), tank_idx, dtype=torch.long, device=device)
        view_labels = torch.full((num_images,), view_idx, dtype=torch.long, device=device)

        fake_images = gan_generator(noise, tank_labels, view_labels)
        # Denormalize from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2
        fake_images = fake_images.clamp(0, 1)

    return fake_images


def tensor_to_base64(tensor):
    """Convert a tensor image to base64 PNG"""
    to_pil = ToPILImage()
    img = to_pil(tensor)

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@router.get("/")
async def gan_info():
    """Get GAN project information"""
    return {
        "project": "Dual Conditional GAN - Military Vehicle Image Generation",
        "description": "Generate synthetic military vehicle images conditioned on tank type and view angle",
        "endpoints": {
            "/health": "Check if model is loaded",
            "/info": "Get model information (available tanks, views, models)",
            "/models": "Get available model checkpoints",
            "/generate": "Generate images with specific tank type and view angle",
            "/preload": "Preload the model into memory",
            "/unload": "Unload the model from memory"
        }
    }


@router.get("/health")
async def health_check():
    """Check if GAN model is loaded"""
    return {
        "status": "ready" if gan_generator is not None else "not_loaded",
        "model_loaded": gan_generator is not None,
        "gan_available": GAN_AVAILABLE
    }


@router.post("/preload")
async def preload_model():
    """Preload the GAN generator into memory"""
    global gan_generator

    if gan_generator is not None:
        return {
            "status": "already_loaded",
            "message": "GAN generator is already loaded"
        }

    if not GAN_AVAILABLE:
        raise HTTPException(status_code=503, detail="GAN modules not available")

    print("[GAN] Preloading generator on user request...")
    try:
        load_gan_generator()
        return {
            "status": "loaded",
            "message": "GAN generator loaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load GAN generator: {str(e)}")


@router.post("/unload")
async def unload_model():
    """Unload the GAN generator from memory"""
    global gan_generator, gan_mappings

    if gan_generator is None:
        return {
            "status": "not_loaded",
            "message": "GAN generator was not loaded"
        }

    print("[GAN] Unloading generator to free memory...")
    gan_generator = None
    gan_mappings = None

    # Force garbage collection
    import gc
    gc.collect()

    return {
        "status": "unloaded",
        "message": "GAN generator unloaded successfully"
    }


@router.get("/info", response_model=ModelInfo)
async def get_model_info():
    """Get GAN model information"""
    global gan_mappings

    if gan_mappings is None:
        # Load mappings without loading the full model
        model_dir = get_model_dir()
        mappings_path = model_dir / "label_mappings.json"

        if not mappings_path.exists():
            raise HTTPException(status_code=503, detail="Model mappings not found")

        with open(mappings_path) as f:
            gan_mappings = json.load(f)

    # Get available models
    model_dir = get_model_dir()
    available_models = []
    for model_file in sorted(model_dir.glob("generator_epoch_*.pth")):
        available_models.append(model_file.name)
    # Check for latest_generator (either as file or chunked)
    latest_path = model_dir / "latest_generator.pth"
    latest_manifest = model_dir / "latest_generator.pth.manifest.json"
    if latest_path.exists() or latest_manifest.exists():
        available_models.append("latest_generator.pth")

    return {
        "available_tanks": list(gan_mappings['tank_to_idx'].keys()),
        "available_views": list(gan_mappings['view_to_idx'].keys()),
        "available_models": available_models,
        "current_model": "latest_generator.pth" if gan_generator else None,
        "latent_dim": LATENT_DIM,
        "image_size": 200
    }


@router.get("/models", response_model=AvailableModelsResponse)
async def get_available_models():
    """Get available model checkpoints"""
    model_dir = get_model_dir()

    models = []

    # Add epoch models
    for model_file in sorted(model_dir.glob("generator_epoch_*.pth")):
        epoch_num = model_file.stem.split("_")[-1]
        models.append({
            "name": model_file.name,
            "display_name": f"Epoch {epoch_num}",
            "type": "checkpoint"
        })

    # Add latest model (check for both file and chunked versions)
    latest_path = model_dir / "latest_generator.pth"
    latest_manifest = model_dir / "latest_generator.pth.manifest.json"
    if latest_path.exists() or latest_manifest.exists():
        models.append({
            "name": "latest_generator.pth",
            "display_name": "Latest (Best)",
            "type": "latest"
        })

    return {
        "models": models,
        "current_model": "latest_generator.pth" if gan_generator else None
    }


@router.post("/models/switch")
async def switch_model(model_name: str):
    """Switch to a different model checkpoint"""
    global gan_generator, gan_mappings

    model_dir = get_model_dir()
    model_path = model_dir / model_name

    # Check if model exists (either as file or chunked with manifest)
    manifest_path = Path(str(model_path) + '.manifest.json')
    if not model_path.exists() and not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

    # Unload current model
    gan_generator = None

    # Load new model
    try:
        load_gan_generator(model_name)
        return {
            "status": "switched",
            "model": model_name,
            "message": f"Switched to model: {model_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.post("/generate", response_model=GenerateResponse)
async def generate_images(request: GenerateRequest):
    """
    Generate synthetic military vehicle images

    **Parameters:**
    - tank_type: Tank type (M1A1_Abrams, Leopard2, T90)
    - view_angle: View angle (front, side, back)
    - num_images: Number of images to generate (1-16)
    - seed: Optional random seed for reproducibility

    **Returns:**
    - List of generated images as base64-encoded PNGs
    """
    global gan_mappings

    if not GAN_AVAILABLE:
        raise HTTPException(status_code=503, detail="GAN modules not available")

    # Ensure model is loaded
    if gan_generator is None:
        try:
            load_gan_generator()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to load model: {str(e)}")

    # Validate tank type
    if request.tank_type not in gan_mappings['tank_to_idx']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tank type: {request.tank_type}. Available: {list(gan_mappings['tank_to_idx'].keys())}"
        )

    # Validate view angle
    if request.view_angle not in gan_mappings['view_to_idx']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid view angle: {request.view_angle}. Available: {list(gan_mappings['view_to_idx'].keys())}"
        )

    tank_idx = gan_mappings['tank_to_idx'][request.tank_type]
    view_idx = gan_mappings['view_to_idx'][request.view_angle]

    try:
        # Generate images
        images_tensor = generate_images_internal(
            tank_idx, view_idx, request.num_images, request.seed
        )

        # Convert to base64
        generated_images = []
        for i in range(request.num_images):
            img_base64 = tensor_to_base64(images_tensor[i])
            generated_images.append(GeneratedImage(
                image_base64=img_base64,
                tank_type=request.tank_type,
                view_angle=request.view_angle,
                index=i
            ))

        return GenerateResponse(
            images=generated_images,
            tank_type=request.tank_type,
            view_angle=request.view_angle,
            num_generated=len(generated_images)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating images: {str(e)}")


@router.get("/tanks")
async def get_tank_types():
    """Get available tank types"""
    global gan_mappings

    if gan_mappings is None:
        model_dir = get_model_dir()
        mappings_path = model_dir / "label_mappings.json"

        if not mappings_path.exists():
            raise HTTPException(status_code=503, detail="Model mappings not found")

        with open(mappings_path) as f:
            gan_mappings = json.load(f)

    tanks = []
    for name, idx in sorted(gan_mappings['tank_to_idx'].items(), key=lambda x: x[1]):
        tanks.append({"name": name, "index": idx})

    return {"tanks": tanks, "count": len(tanks)}


@router.get("/views")
async def get_view_angles():
    """Get available view angles"""
    global gan_mappings

    if gan_mappings is None:
        model_dir = get_model_dir()
        mappings_path = model_dir / "label_mappings.json"

        if not mappings_path.exists():
            raise HTTPException(status_code=503, detail="Model mappings not found")

        with open(mappings_path) as f:
            gan_mappings = json.load(f)

    views = []
    for name, idx in sorted(gan_mappings['view_to_idx'].items(), key=lambda x: x[1]):
        views.append({"name": name, "index": idx})

    return {"views": views, "count": len(views)}
