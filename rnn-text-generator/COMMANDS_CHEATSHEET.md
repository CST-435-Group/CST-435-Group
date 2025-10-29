# Commands Cheat Sheet

## Quick Start (Testing Current Model)

### Option A: If you have model_best.pt but missing tokenizer

```bash
# Terminal 1: Prepare and start backend
cd backend
pip install tqdm
python app/prepare_model_for_testing.py
python app/main.py
```

```bash
# Terminal 2: Start frontend
cd frontend
npm install
npm start
```

### Option B: Train from scratch with improvements

```bash
# Terminal 1: Train and start backend
cd backend
pip install tqdm
python app/train.py          # Current architecture with progress bars
# OR
python app/train_improved.py  # Larger model (better accuracy)
# OR
python app/train_optimal.py   # Best accuracy (limited vocab)

# After training completes or you stop it (Ctrl+C after a few epochs):
python app/main.py
```

```bash
# Terminal 2: Start frontend
cd frontend
npm install
npm start
```

---

## Backend Commands

### Start Backend Server
```bash
cd backend
python app/main.py
```
Server runs at: `http://localhost:8000`

### Training Scripts
```bash
# Original model with progress bars and timing
python app/train.py

# Improved model (256 units, 3 layers)
python app/train_improved.py

# Optimal model (limited vocab + larger model)
python app/train_optimal.py
```

### Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Check if Model is Ready
```bash
ls saved_models/
# Should see: model.pt, tokenizer.pkl, model_config.json
```

---

## Frontend Commands

### Start Development Server
```bash
cd frontend
npm start
```
Opens at: `http://localhost:3000`

### Install Dependencies
```bash
cd frontend
npm install
```

### Build for Production
```bash
cd frontend
npm run build
```

---

## Testing Commands

### Check Backend Health
```bash
# Backend must be running first (python app/main.py)

# Windows:
curl http://localhost:8000/

# Or open in browser:
# http://localhost:8000/docs
```

### Test Text Generation (API)
```bash
# Backend must be running first

curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d "{\"seed_text\":\"The quick brown\",\"num_words\":20,\"temperature\":1.0}"
```

---

## Common Issues & Fixes

### "Model not loaded"
```bash
cd backend
python app/prepare_model_for_testing.py
# OR
python app/train.py  # Let at least 1 epoch complete
```

### "No module named 'tqdm'"
```bash
pip install tqdm
```

### "No module named 'text_generator'"
```bash
# Make sure you're in the backend directory
cd backend
python app/main.py  # NOT: python main.py
```

### Frontend can't connect to backend
```bash
# Check backend is running:
curl http://localhost:8000/

# If not running:
cd backend
python app/main.py
```

### Port already in use
```bash
# Backend (port 8000):
uvicorn app.main:app --port 8001  # Use different port

# Frontend (port 3000):
# When prompted, press 'y' to use different port
```

---

## File Paths Reference

| File | Purpose | Created By |
|------|---------|------------|
| `backend/saved_models/model.pt` | Trained model weights | Training script |
| `backend/saved_models/model_best.pt` | Best checkpoint | Training (auto) |
| `backend/saved_models/tokenizer.pkl` | Word vocabulary | Training script |
| `backend/saved_models/model_config.json` | Model architecture | Training script |
| `backend/data/training_text.txt` | Training data | You/download script |
| `backend/app/main.py` | API server | Existing |
| `frontend/build/` | Production build | `npm run build` |

---

## Ports Used

| Service | Port | URL |
|---------|------|-----|
| Backend API | 8000 | http://localhost:8000 |
| API Docs | 8000 | http://localhost:8000/docs |
| Frontend Dev | 3000 | http://localhost:3000 |

---

## Training Time Estimates

| Configuration | Time per Epoch | 100 Epochs | Expected Val Acc |
|---------------|----------------|------------|------------------|
| Current (150u, 2L) | ~20 min | ~33 hours | 30-32% |
| Improved (256u, 3L) | ~30 min | ~50 hours | 32-37% |
| Optimal (256u, 3L, 20k vocab) | ~15 min | ~25 hours | 40-50% |

*Based on your hardware (10 hours for 30 epochs = ~20 min/epoch)*

---

## Quick Reference

### Test the trained model NOW:
```bash
# Terminal 1
cd backend
python app/prepare_model_for_testing.py && python app/main.py

# Terminal 2
cd frontend
npm start
```

### Train with improvements FIRST:
```bash
# Terminal 1
cd backend
pip install tqdm
python app/train_optimal.py  # Best option
# Press Ctrl+C after several epochs
python app/main.py

# Terminal 2
cd frontend
npm start
```
