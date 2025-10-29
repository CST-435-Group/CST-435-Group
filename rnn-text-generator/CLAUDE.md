# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an RNN-based text generator built with PyTorch (backend) and React/TypeScript (frontend). The system trains LSTM models on classical literature to generate text predictions using word-level tokenization.

## Architecture

### Backend (Python/PyTorch)
- **Framework**: FastAPI for REST API
- **ML Library**: PyTorch with LSTM neural networks
- **Model Architecture**: Word-level LSTM with embeddings, multiple layers, and dropout
- **Key Components**:
  - `backend/app/text_generator.py`: Core LSTM model (`LSTMModel`) and training logic (`TextGenerator`)
  - `backend/app/text_generator_limited_vocab.py`: Variant with vocabulary limiting for better performance
  - `backend/app/main.py`: FastAPI server with endpoints for generation, model info, testing, and model switching
  - `backend/app/models.py`: Pydantic models for API requests/responses
  - `backend/app/train.py`: Basic training script
  - `backend/app/train_optimal.py`: Optimized training with limited vocabulary (10k words)
  - `backend/app/train_improved.py`: Larger model variant
  - `backend/app/train_hybrid.py`: Hybrid training approach

### Frontend (React/TypeScript)
- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS
- **HTTP Client**: Axios
- **Charts**: Recharts for visualizations
- **Structure**:
  - `frontend/src/App.tsx`: Main application component
  - `frontend/src/components/`: UI components
  - `frontend/src/services/`: API communication layer

### Data Pipeline
- Training data stored in `backend/data/` (Bible translations, Shakespeare, classic literature)
- Scripts in `backend/scripts/`:
  - `download_data.py`: Downloads training texts
  - `download_training_books.py`: Downloads additional literature
  - `combine_bibles.py`: Merges Bible translations
  - `extract_vocabulary.py`: Pre-extracts vocabulary for limited vocab models

## Common Commands

### Backend Development

**Install dependencies:**
```bash
cd backend
python -m pip install -r requirements.txt
```

**Start FastAPI server:**
```bash
cd backend
python app/main.py
# Or with uvicorn directly:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Train models:**
```bash
cd backend
# Basic training (150 units, 2 layers, full vocab)
python app/train.py

# Optimized training (256 units, 2 layers, 10k vocab) - RECOMMENDED
python app/train_optimal.py

# Improved training (256 units, 3 layers, full vocab)
python app/train_improved.py
```

**Download training data:**
```bash
cd backend
python scripts/download_data.py
```

**Extract vocabulary (for optimal training):**
```bash
cd backend
python scripts/extract_vocabulary.py
```

### Frontend Development

**Install dependencies:**
```bash
cd frontend
npm install
```

**Start development server:**
```bash
cd frontend
npm start
# Opens at http://localhost:3000
```

**Build for production:**
```bash
cd frontend
npm run build
```

**Run tests:**
```bash
cd frontend
npm test
```

## Model Architecture Details

### TextGenerator Class (text_generator.py)
- **Tokenization**: Word-level with SimpleTokenizer class
- **Preprocessing**: Regex-based (`r'\b[a-zA-Z]+(?:\'[a-z]+)?\b'`) - preserves contractions
- **LSTM Architecture**:
  - Embedding layer with dropout
  - Multi-layer LSTM with dropout between layers
  - Layer normalization before final dense layer
  - Xavier/Glorot weight initialization
- **Training**: AdamW optimizer with step LR decay, gradient clipping, early stopping
- **Generation Methods**:
  - **Sampling** (default): Temperature-based random sampling for creative, varied output
  - **Beam Search**: Deterministic search for most probable sequences (more coherent, consistent)

### Model Files Structure
```
backend/saved_models/
├── model.pt                    # Default model weights
├── tokenizer.pkl               # Pickled tokenizer with vocab
├── model_config.json           # Model hyperparameters
├── model_best.pt               # Best checkpoint during training
├── model_optimal_*.pt          # Optimal models with vocab suffix
└── tokenizer_optimal_*.pkl     # Corresponding tokenizers
```

### API Endpoints
- `GET /` - Health check
- `GET /model/info` - Get current model info (vocab size, architecture)
- `POST /generate` - Generate text from seed
- `GET /model/test` - Evaluate model on test data
- `GET /models/available` - List available trained models
- `POST /models/switch` - Switch to different model
- `GET /visualizations/architecture` - Model architecture diagram
- `GET /visualizations/training` - Training history plot

## Development Notes

### Training Considerations
- **Vocabulary Size**: Full vocab (~50k words) vs limited vocab (5k-10k words)
  - Limited vocab provides better accuracy (40-50% vs 28-32%)
  - Use `train_optimal.py` for best results
- **Model Size**: Balance between capacity and overfitting
  - Baseline: 150 units, 2 layers (~8.4M params)
  - Optimal: 256 units, 2-3 layers with limited vocab (~4-5M params)
- **Training Time**: ~20-30 min/epoch on CPU, much faster on GPU
- **Early Stopping**: Patience of 3 epochs to prevent overfitting

### Text Preprocessing Critical Rule
**IMPORTANT**: Preprocessing in `text_generator.py` MUST match `extract_vocabulary.py` exactly:
- Use regex: `r'\b[a-zA-Z]+(?:\'[a-z]+)?\b'`
- Keep contractions intact (don't → don't, not don ' t)
- Lowercase all text
- This prevents unknown token issues during generation

### Model Loading
The FastAPI server auto-loads models on startup:
1. Looks for `model.pt` + `tokenizer.pkl` + `model_config.json`
2. Falls back to first available model in `saved_models/`
3. Supports switching between multiple trained models via API

### Windows-Specific Notes
- DataLoader uses `num_workers=0` to avoid multiprocessing issues
- Use `python -m pip install` for package management
- PowerShell requires escaping or quotes for some commands

### File Paths
- Always use relative paths from `backend/` directory when running training scripts
- Model paths: `saved_models/model.pt`
- Data paths: `data/training_text.txt`
- Visualization paths: `visualizations/training_history.png`

## Testing the Application

1. **Ensure model is trained**: Check for `model.pt`, `tokenizer.pkl`, and `model_config.json` in `backend/saved_models/`
2. **Start backend**: `cd backend && python app/main.py`
3. **Start frontend**: `cd frontend && npm start`
4. **Open browser**: Navigate to http://localhost:3000
5. **Generate text**: Enter seed text, adjust temperature (0.5-1.5), and click Generate

## Quick Reference

### Port Usage
- Backend API: 8000
- Frontend Dev: 3000

### Python Version
- Requires Python 3.8+
- PyTorch 2.0+ for best performance

### Key Hyperparameters

**Model Architecture:**
- `sequence_length`: Context window (default: 50 words)
- `embedding_dim`: Word embedding size (100-128)
- `lstm_units`: LSTM hidden size (150-256)
- `num_layers`: LSTM layers (2-3)
- `dropout_rate`: Regularization (0.2-0.3)

**Generation (Sampling):**
- `temperature`: Randomness (0.5=conservative, 1.5=creative)

**Generation (Beam Search):**
- `use_beam_search`: Enable beam search (default: false)
- `beam_width`: Number of beams (1-10, default: 5)
- `length_penalty`: Sequence length preference (0.5=shorter, 2.0=longer, default: 1.0)
