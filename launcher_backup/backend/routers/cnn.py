"""
FastAPI Router for CNN Project (Fruit Classification)
Converts Streamlit functionality to REST API endpoints
"""

from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import io
import base64
from pathlib import Path

# Create router
router = APIRouter()

# Global model storage (lazy loading)
cnn_model = None
cnn_metadata = None


# Define Model Architecture (must match training)
class FruitCNN(nn.Module):
    def __init__(self, num_classes):
        super(FruitCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


# Request/Response Models
class PredictionResult(BaseModel):
    """Prediction result for a single image"""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]


class ModelInfo(BaseModel):
    """CNN model information"""
    num_classes: int
    fruit_names: List[str]
    test_accuracy: float
    total_images: int


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model input"""
    # Convert to RGB first (handles various formats)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to grayscale
    img_gray = image.convert('L')

    # Resize to 128x128
    img_resized = img_gray.resize((128, 128), Image.Resampling.LANCZOS)

    # Normalize and standardize
    img_array = np.array(img_resized) / 255.0
    img_array = (img_array - 0.5) / 0.5
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)

    return img_tensor


def load_cnn_model():
    """Lazy load CNN model"""
    global cnn_model, cnn_metadata

    if cnn_model is not None:
        return cnn_model, cnn_metadata

    try:
        # Try multiple paths for model files
        cnn_project_path = Path(__file__).parent.parent.parent.parent / "CNN_Project"

        model_paths = [
            cnn_project_path / "models" / "best_model.pth",
            cnn_project_path / "best_model.pth",
            cnn_project_path / "data" / "best_model.pth"
        ]

        metadata_paths = [
            cnn_project_path / "models" / "model_metadata.json",
            cnn_project_path / "model_metadata.json",
            cnn_project_path / "data" / "model_metadata.json"
        ]

        # Find existing model file
        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = path
                break

        # Find existing metadata file
        metadata_path = None
        for path in metadata_paths:
            if path.exists():
                metadata_path = path
                break

        if model_path is None or metadata_path is None:
            raise FileNotFoundError(
                f"Model files not found. Checked: {[str(p) for p in model_paths + metadata_paths]}"
            )

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')

        model = FruitCNN(num_classes=metadata['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        cnn_model = model
        cnn_metadata = metadata

        print("✅ CNN model loaded successfully!")
        return model, metadata

    except Exception as e:
        print(f"❌ Error loading CNN model: {e}")
        return None, None


@router.get("/")
async def cnn_info():
    """Get CNN project information"""
    return {
        "project": "Fruit Image Classification using Convolutional Neural Networks",
        "description": "CNN model for recognizing different types of fruits from images",
        "endpoints": {
            "/health": "Check if model is loaded",
            "/info": "Get model information",
            "/predict": "Classify uploaded fruit image",
            "/predict-url": "Classify fruit image from URL",
            "/fruit-list": "Get list of recognizable fruits"
        }
    }


@router.get("/health")
async def health_check():
    """Check if CNN model is loaded"""
    model, _ = load_cnn_model()
    return {
        "status": "ready" if model is not None else "not_loaded",
        "model_loaded": model is not None
    }


@router.get("/info", response_model=ModelInfo)
async def get_model_info():
    """Get CNN model information"""
    model, metadata = load_cnn_model()

    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "num_classes": metadata['num_classes'],
        "fruit_names": metadata['fruit_names'],
        "test_accuracy": metadata.get('test_accuracy', 0.0),
        "total_images": metadata.get('total_images', 0)
    }


@router.get("/fruit-list")
async def get_fruit_list():
    """Get list of recognizable fruits"""
    model, metadata = load_cnn_model()

    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "fruits": metadata['fruit_names'],
        "count": len(metadata['fruit_names'])
    }


@router.post("/predict", response_model=PredictionResult)
async def predict_image(file: UploadFile = File(...)):
    """
    Classify uploaded fruit image

    **Parameters:**
    - file: Image file (JPEG, PNG, etc.)

    **Returns:**
    - predicted_class: Name of the predicted fruit
    - confidence: Model confidence (0-1)
    - probabilities: Probability distribution across all fruit classes
    """
    model, metadata = load_cnn_model()

    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        img_tensor = preprocess_image(image)

        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            pred_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class_idx].item()

        # Get fruit names
        fruit_names = metadata['fruit_names']
        predicted_fruit = fruit_names[pred_class_idx]

        # Create probability distribution
        prob_dict = {
            fruit_names[i]: float(probabilities[0][i].item())
            for i in range(len(fruit_names))
        }

        return {
            "predicted_class": predicted_fruit,
            "confidence": confidence,
            "probabilities": prob_dict
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


class ImageBase64Request(BaseModel):
    """Request with base64-encoded image"""
    image_base64: str = Field(..., description="Base64-encoded image data")


@router.post("/predict-base64", response_model=PredictionResult)
async def predict_base64(request: ImageBase64Request):
    """
    Classify fruit image from base64-encoded data

    **Parameters:**
    - image_base64: Base64-encoded image string

    **Returns:**
    - predicted_class: Name of the predicted fruit
    - confidence: Model confidence (0-1)
    - probabilities: Probability distribution across all fruit classes
    """
    model, metadata = load_cnn_model()

    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))

        img_tensor = preprocess_image(image)

        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            pred_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class_idx].item()

        # Get fruit names
        fruit_names = metadata['fruit_names']
        predicted_fruit = fruit_names[pred_class_idx]

        # Create probability distribution
        prob_dict = {
            fruit_names[i]: float(probabilities[0][i].item())
            for i in range(len(fruit_names))
        }

        return {
            "predicted_class": predicted_fruit,
            "confidence": confidence,
            "probabilities": prob_dict
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# Preload model on startup (optional)
@router.on_event("startup")
async def startup_event():
    """Preload model on startup"""
    # Uncomment to preload (uses more RAM)
    # load_cnn_model()
    pass
