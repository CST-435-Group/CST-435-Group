# NLP Model Update Guide

## Current Situation

You have **two NLP project locations**:
1. **Old NLP** at: `NLP/nlp-react/backend/` (current working version)
2. **New/Other NLP** at: `nlp-react/backend/` (different location)

The launcher's NLP router currently points to: `nlp-react/backend/model.py`

## Files You Need to Understand

### Current Model File (What the Launcher Uses)
**Location**: `nlp-react/backend/model.py`

**What it does**:
- Uses DistilBERT transformer model (`distilbert-base-uncased`)
- 7-point sentiment scale (-3 to +3)
- Downloads model from HuggingFace on first run (~250MB)
- Returns: score, label, emoji, confidence, probabilities

**Key class**: `SentimentAnalyzer`
- `__init__()` - Loads the model
- `analyze(text)` - Analyzes sentiment and returns dict

### Launcher Router (What Calls the Model)
**Location**: `launcher/backend/routers/nlp.py`

**Line 13**: Points to NLP project location
```python
nlp_project_path = Path(__file__).parent.parent.parent.parent / "nlp-react" / "backend"
```

This resolves to: `CST-435-Group/nlp-react/backend/`

---

## How to Switch to Different NLP Model

### Option 1: Update the Path (Use Different NLP Project)

If you want to use the NLP from `NLP/nlp-react/backend/` instead:

**Edit**: `launcher/backend/routers/nlp.py` line 13

**Change from**:
```python
nlp_project_path = Path(__file__).parent.parent.parent.parent / "nlp-react" / "backend"
```

**Change to**:
```python
nlp_project_path = Path(__file__).parent.parent.parent.parent / "NLP" / "nlp-react" / "backend"
```

### Option 2: Replace the Model File

If you want to replace the model code entirely:

1. **Copy your new model file** to: `nlp-react/backend/model.py`
2. **Requirements**:
   - Must have a class named `SentimentAnalyzer`
   - Must have a method `analyze(text: str) -> dict`
   - Return dict must include these keys:
     - `text` (str)
     - `sentiment_score` (int)
     - `sentiment_label` (str)
     - `emoji` (str)
     - `confidence` (float)
     - `probabilities` (dict)

### Option 3: Use a Completely Different Model

If you want to use a different transformer model (e.g., RoBERTa, BERT-large, etc.):

**Edit**: `nlp-react/backend/model.py` line 15

**Change from**:
```python
def __init__(self, model_name: str = "distilbert-base-uncased"):
```

**Change to** (example with RoBERTa):
```python
def __init__(self, model_name: str = "roberta-base"):
```

Or pass it when creating the analyzer in your code.

---

## Files That Need to Match

For the NLP endpoint to work, you need:

1. **Model file**: `nlp-react/backend/model.py` (or `NLP/nlp-react/backend/model.py`)
   - Contains `SentimentAnalyzer` class

2. **Router file**: `launcher/backend/routers/nlp.py`
   - Imports from the model file
   - Path on line 13 must point to correct location

3. **Dependencies**: Make sure `requirements.txt` has:
   ```
   transformers>=4.30.0
   torch>=2.0.0
   ```

---

## What Each File Does

### `nlp-react/backend/model.py`
- **Purpose**: Defines the ML model and inference logic
- **Key Class**: `SentimentAnalyzer`
- **Model**: DistilBERT fine-tuned for 7-class sentiment
- **Output**: Sentiment score, label, emoji, confidence

### `launcher/backend/routers/nlp.py`
- **Purpose**: FastAPI endpoints for the launcher
- **What it does**:
  - Imports `SentimentAnalyzer` from model.py
  - Creates API endpoints: `/analyze`, `/analyze/batch`, `/examples`, `/health`
  - Lazy loads model (doesn't load until first request)
  - Wraps model output in FastAPI responses

### `nlp-react/backend/main.py`
- **Purpose**: Standalone FastAPI app for the original NLP project
- **Note**: NOT used by the launcher (launcher has its own main.py)
- **Use case**: Run NLP project independently

---

## Directory Structure

```
CST-435-Group/
├── nlp-react/                    ← Currently used by launcher
│   └── backend/
│       ├── model.py              ← Model implementation
│       ├── main.py               ← Standalone API (not used by launcher)
│       └── requirements.txt
│
├── NLP/                          ← Alternative NLP project
│   └── nlp-react/
│       └── backend/
│           ├── model.py
│           └── ...
│
└── launcher/
    └── backend/
        ├── main.py               ← Launcher main API
        └── routers/
            └── nlp.py            ← Imports from nlp-react/backend/model.py
```

---

## To Update the Model

### Step 1: Choose Your Approach
- **A**: Change path to point to different NLP project
- **B**: Replace model.py file
- **C**: Modify model parameters

### Step 2: Make Changes
Edit the appropriate files based on your choice above

### Step 3: Test Locally
```bash
# Start backend
cd launcher/backend
uvicorn main:app --reload --port 8000

# Test endpoint
curl -X POST http://localhost:8000/api/nlp/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I loved it!"}'
```

### Step 4: Deploy
```bash
git add .
git commit -m "Update NLP model"
git push
```

Railway will automatically redeploy.

---

## Common Issues

### Issue: "Could not import NLP model"
**Fix**: Check that the path in `nlp.py` line 13 points to a directory with `model.py`

### Issue: "SentimentAnalyzer not available"
**Fix**: Make sure `model.py` has a class named `SentimentAnalyzer`

### Issue: Model returns different format
**Fix**: Update the response model in `nlp.py` or modify your model's `analyze()` method to match the expected format

---

## Summary

**To switch NLP models, you have 3 options**:

1. **Change the import path** (line 13 of `launcher/backend/routers/nlp.py`)
2. **Replace the model file** (`nlp-react/backend/model.py`)
3. **Modify model parameters** (change model_name in `model.py`)

**Key requirement**: Whatever model you use must have:
- A `SentimentAnalyzer` class
- An `analyze(text)` method
- Return format matching the API response model

That's it! Choose your approach and update the files accordingly.
