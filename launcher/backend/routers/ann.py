"""
FastAPI Router for ANN Project (NBA Team Selection)
Converts Streamlit functionality to REST API endpoints
"""

from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add ANN_Project to path
ann_project_path = Path(__file__).parent.parent.parent.parent / "ANN_Project"
sys.path.insert(0, str(ann_project_path))

# Import ANN modules
from src.model import create_model
from src.select_team import TeamSelector
from src.preprocess import NBADataPreprocessor
from src.load_data import load_nba_data, get_feature_columns, create_position_labels
from src.utils import get_device

# Create router
router = APIRouter()

# Global model storage (lazy loading)
ann_model = None
ann_preprocessor = None
ann_data = None


# Request/Response Models
class TeamSelectionRequest(BaseModel):
    """Request for team selection"""
    method: str = Field("balanced", description="Selection method: greedy, balanced, or exhaustive")
    start_year: str = Field("1996-97", description="Start year for data")
    end_year: str = Field("2019-20", description="End year for data")
    n_players: int = Field(100, description="Number of players to analyze")


class PlayerEvaluation(BaseModel):
    """Player evaluation result"""
    player_name: str
    predicted_position: str
    guard_prob: float
    forward_prob: float
    center_prob: float
    team_fit_score: float
    position_confidence: float
    overall_score: float


class TeamSelectionResult(BaseModel):
    """Team selection result"""
    method: str
    players: List[Dict]
    position_distribution: Dict[str, int]
    team_metrics: Dict[str, float]
    composition_analysis: str


class DataInfo(BaseModel):
    """Data information"""
    total_players: int
    avg_points: float
    avg_rebounds: float
    avg_assists: float
    position_distribution: Dict[str, int]


def load_ann_model():
    """Lazy load ANN model and data"""
    global ann_model, ann_preprocessor, ann_data

    if ann_model is not None:
        return ann_model, ann_preprocessor, ann_data

    try:
        # Load pre-trained model if it exists
        model_path = ann_project_path / "best_model.pth"
        data_path = ann_project_path / "data" / "nba_players.csv"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        if not data_path.exists():
            raise FileNotFoundError(f"Data not found at {data_path}")

        # Load data
        df = load_nba_data(str(data_path), "1996-97", "2019-20", 100)
        df = create_position_labels(df)

        # Create preprocessor
        numerical_features, categorical_features, _ = get_feature_columns()
        preprocessor = NBADataPreprocessor()
        features, _ = preprocessor.fit_transform(df, numerical_features, categorical_features)

        # Load model
        device = get_device()
        checkpoint = torch.load(model_path, map_location=device)

        model = create_model(
            input_dim=features.shape[1],
            config={
                'hidden_dims': [256, 128, 64, 32],
                'dropout_rate': 0.25,
                'activation': 'relu',
                'use_batch_norm': True
            }
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        ann_model = model
        ann_preprocessor = preprocessor
        ann_data = df

        print("✅ ANN model loaded successfully!")
        return model, preprocessor, df

    except Exception as e:
        print(f"❌ Error loading ANN model: {e}")
        return None, None, None


@router.get("/")
async def ann_info():
    """Get ANN project information"""
    return {
        "project": "NBA Team Selection using Artificial Neural Networks",
        "description": "Multi-layer perceptron for player position classification and team optimization",
        "endpoints": {
            "/health": "Check if model is loaded",
            "/data-info": "Get dataset information",
            "/select-team": "Select optimal 5-player team",
            "/evaluate-player": "Get evaluation for specific player"
        }
    }


@router.get("/health")
async def health_check():
    """Check if ANN model is loaded"""
    model, _, _ = load_ann_model()
    return {
        "status": "ready" if model is not None else "not_loaded",
        "model_loaded": model is not None
    }


@router.get("/data-info", response_model=DataInfo)
async def get_data_info():
    """Get dataset information"""
    model, preprocessor, data = load_ann_model()

    if data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    position_names = {0: 'Guard', 1: 'Forward', 2: 'Center'}
    position_counts = data['primary_position'].map(position_names).value_counts().to_dict()

    return {
        "total_players": len(data),
        "avg_points": float(data['pts'].mean()),
        "avg_rebounds": float(data['reb'].mean()),
        "avg_assists": float(data['ast'].mean()),
        "position_distribution": position_counts
    }


@router.post("/select-team", response_model=TeamSelectionResult)
async def select_team(request: TeamSelectionRequest):
    """
    Select optimal 5-player NBA team

    **Parameters:**
    - method: Selection method (greedy, balanced, or exhaustive)
    - start_year: Start year for data analysis
    - end_year: End year for data analysis
    - n_players: Number of players to analyze

    **Returns:**
    - Selected team with player details and metrics
    """
    model, preprocessor, data = load_ann_model()

    if model is None or data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare features
        numerical_features, categorical_features, _ = get_feature_columns()
        features, _ = preprocessor.transform(data, numerical_features, categorical_features)

        # Create team selector
        device = get_device()
        selector = TeamSelector(model, device)

        # Evaluate all players
        evaluations = selector.evaluate_players(
            features,
            data['player_name'].tolist(),
            data
        )

        # Select optimal team
        team = selector.select_optimal_team(evaluations, method=request.method)

        return team

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error selecting team: {str(e)}")


@router.get("/players", response_model=List[PlayerEvaluation])
async def get_all_players(limit: int = 20):
    """
    Get evaluations for all players (limited to top N)

    **Parameters:**
    - limit: Maximum number of players to return

    **Returns:**
    - List of player evaluations
    """
    model, preprocessor, data = load_ann_model()

    if model is None or data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare features
        numerical_features, categorical_features, _ = get_feature_columns()
        features, _ = preprocessor.transform(data, numerical_features, categorical_features)

        # Create team selector
        device = get_device()
        selector = TeamSelector(model, device)

        # Evaluate all players
        evaluations = selector.evaluate_players(
            features,
            data['player_name'].tolist(),
            data
        )

        # Return top N players
        top_players = evaluations.nlargest(limit, 'overall_score')

        return top_players.to_dict('records')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating players: {str(e)}")


# Preload model on startup (optional - for faster first request)
@router.on_event("startup")
async def startup_event():
    """Preload model on startup"""
    # Uncomment to preload (uses more RAM)
    # load_ann_model()
    pass
