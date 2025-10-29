# Quick Start Guide - Testing Your Trained Model

## Current Status

Your training created:
- ✓ `model_best.pt` (53 MB) - Your trained model from 30 epochs
- ✗ `tokenizer.pkl` - Missing or empty (0 bytes) - **PROBLEM!**
- ✗ `model_best_config.json` - Missing - **PROBLEM!**

## Problem: Incomplete Save

Your training was interrupted (KeyboardInterrupt at epoch 30), so the final model save didn't complete. You have the checkpoint but not the tokenizer or config.

---

## Solution: Two Options

### Option 1: Complete the Training (Recommended)

This will properly save everything you need:

```bash
# Navigate to backend
cd backend

# Install tqdm (for progress bars)
pip install tqdm

# Run training - it will resume and save properly
python app/train.py
```

The training will:
- Load your data
- Train the model (with progress bars now!)
- Save `model.pt`, `tokenizer.pkl`, and `model_config.json` properly
- Let it run for at least a few epochs, then you can stop it safely

### Option 2: Quick Test with Manual Setup

If you want to test immediately, we need to recreate the tokenizer and config from your training data:

```bash
# Navigate to backend
cd backend

# Run this helper script (I'll create it below)
python app/prepare_model_for_testing.py
```

---

## After Model is Ready: Start Backend & Frontend

### Terminal 1: Start Backend (FastAPI)

```bash
# Navigate to backend directory
cd backend

# Activate virtual environment if you have one
# (Optional: depends on your setup)

# Install dependencies if needed
pip install -r requirements.txt

# Start the FastAPI server
python app/main.py

# OR use uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be at: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/`

### Terminal 2: Start Frontend (React)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start the development server
npm start
```

Frontend will open at: `http://localhost:3000`

---

## Full Command Summary

### If Training Completed Successfully:

**Terminal 1 (Backend):**
```bash
cd backend
pip install -r requirements.txt
python app/main.py
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm install
npm start
```

### If Tokenizer is Missing (Need to Train First):

**Step 1: Complete Training**
```bash
cd backend
pip install tqdm
python app/train.py
# Wait for at least 1 epoch to complete
# Press Ctrl+C to stop
```

**Step 2: Start Backend**
```bash
python app/main.py
```

**Step 3: Start Frontend (new terminal)**
```bash
cd frontend
npm start
```

---

## Expected Startup Messages

### Backend Startup (Success):
```
[OK] Model loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Backend Startup (No Model):
```
[WARNING] Model files not found. Train the model first.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Frontend Startup:
```
Compiled successfully!

You can now view rnn-text-generator-frontend in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000
```

---

## Troubleshooting

### Issue: "Model not loaded" error in backend
**Solution**: The tokenizer.pkl or model_config.json is missing. Run training again:
```bash
cd backend
python app/train.py
# Let at least 1 epoch complete, then Ctrl+C
```

### Issue: Backend won't start - "No module named 'text_generator'"
**Solution**: Make sure you're in the backend directory and app is in the path:
```bash
cd backend
python app/main.py
```

### Issue: Frontend shows connection error
**Solution**: Make sure backend is running on port 8000:
```bash
# In backend directory
python app/main.py
# Should show "Uvicorn running on http://0.0.0.0:8000"
```

### Issue: "No module named 'tqdm'"
**Solution**: Install the new dependency:
```bash
pip install tqdm
```

### Issue: Frontend npm packages missing
**Solution**: Install dependencies:
```bash
cd frontend
npm install
```

---

## Testing the Model

Once both are running:

1. Open `http://localhost:3000` in your browser
2. Enter seed text (e.g., "The quick brown fox")
3. Adjust settings:
   - **Temperature**: 0.5 (conservative) to 1.5 (creative)
   - **Words to generate**: 50-100
4. Click "Generate"
5. View your generated text!

---

## Quick Health Check

Before starting frontend, check if backend is working:

```bash
# Backend should be running (python app/main.py)

# In a new terminal:
curl http://localhost:8000/

# Should return: {"status":"healthy","model_loaded":true}
```

If `model_loaded: false`, you need to complete the training first.

---

## File Structure Reference

```
rnn-text-generator/
├── backend/
│   ├── app/
│   │   ├── main.py              ← FastAPI server
│   │   ├── train.py             ← Training script
│   │   ├── text_generator.py   ← Model code
│   │   └── models.py            ← API models
│   ├── saved_models/
│   │   ├── model_best.pt        ← Your checkpoint (exists)
│   │   ├── model.pt             ← Expected by API
│   │   ├── tokenizer.pkl        ← Missing! (needs training)
│   │   └── model_config.json    ← Missing! (needs training)
│   └── requirements.txt
│
└── frontend/
    ├── src/
    ├── public/
    ├── package.json
    └── node_modules/
```

---

## Next Steps After Testing

1. Let training complete (full 100 epochs with early stopping)
2. Try the improved models:
   - `python app/train_improved.py` - Larger model
   - `python app/train_optimal.py` - Best accuracy
3. Compare text generation quality
4. Adjust hyperparameters based on results
