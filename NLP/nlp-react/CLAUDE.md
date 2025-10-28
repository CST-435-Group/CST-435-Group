# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a full-stack sentiment analysis web application designed as an educational project for the AIT-204 course. It analyzes text (originally movie reviews, now adapted for hospital reviews) and determines sentiment on a 7-point scale (-3 to +3).

**Tech Stack:**
- Frontend: React 18.2 + Vite 7.1 + Tailwind CSS
- Backend: FastAPI (Python) + PyTorch + Transformers (DistilBERT)
- Deployment: Frontend on Vercel, Backend on Render/Railway

## Common Development Commands

### Frontend Development
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server (runs on port 3000)
npm run dev

# Build for production
npm run build

# Preview production build
npm preview

# Lint code
npm run lint
```

### Backend Development
```bash
# Navigate to backend directory
cd backend

# Create virtual environment (first time only)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run development server (runs on port 8000)
python main.py

# Test the model directly
python model.py
```

### Full Stack Development
```bash
# From root directory - run both frontend and backend
npm run dev

# Install all dependencies
npm run install:all

# Frontend only
npm run dev:frontend

# Backend only
npm run dev:backend

# Build frontend
npm run build:frontend
```

## Architecture

### Frontend Component Structure

The React application follows a component-based architecture:

```
App.jsx (Main Container)
├── Header.jsx - App title, status indicator, sentiment scale
├── Tab Navigation - Switches between "Analysis Tool" and "About NLP"
├── Analysis Tab:
│   ├── InputSection.jsx - Text input area with character counter
│   ├── ResultsSection.jsx - Displays analysis results with charts
│   ├── ExamplesSection.jsx - Quick example buttons for testing
│   ├── InfoSection.jsx - Information panel about the app
│   ├── SimpleStats.jsx - Statistical information display
│   └── ModelPerformance.jsx - Model performance metrics
└── About Tab:
    └── AboutNLP.jsx - Educational content about NLP
```

**State Management:** Uses React hooks (useState, useEffect) with state lifted to App.jsx for shared data (result, loading, apiStatus). Local component state is used for component-specific data.

**API Communication:** Centralized in `frontend/src/services/api.js` using axios for HTTP requests.

### Backend Architecture

The FastAPI backend is structured as:

```
main.py - FastAPI endpoints and route handlers
├── POST /analyze - Single text analysis
├── POST /analyze/batch - Batch text analysis
├── GET /examples - Example reviews by sentiment level
├── GET /sentiment-scale - Sentiment scale metadata
├── GET /health - Health check endpoint
└── GET / - API info

model.py - Sentiment analysis model wrapper
├── SentimentAnalyzer class
│   ├── Loads DistilBERT model
│   ├── Handles tokenization
│   ├── Runs inference
│   └── Formats results with 7-class sentiment scale
```

**Model Details:**
- Uses pre-trained DistilBERT (distilbert-base-uncased)
- Fine-tuned for 7-class sentiment classification (-3 to +3)
- Outputs include: score, label, emoji, confidence, probability distribution

### Data Flow

1. User enters text in `InputSection.jsx`
2. Frontend calls `api.analyzeSentiment(text)` from `services/api.js`
3. Axios sends POST request to backend `/analyze` endpoint
4. `main.py` receives request, validates with Pydantic models
5. `model.py` SentimentAnalyzer tokenizes text and runs through DistilBERT
6. Results formatted and returned as JSON
7. `ResultsSection.jsx` receives data and displays with Recharts visualizations

## Environment Configuration

### Development (Local)

**Frontend:** Uses Vite proxy configuration in `vite.config.js` to proxy `/api` requests to `http://localhost:8000`. No environment variables needed for local development.

**Backend:** Runs directly on port 8000, CORS configured to allow localhost:3000 and localhost:5173.

### Production (Deployed)

**Frontend (.env):**
```bash
VITE_API_URL=https://your-backend-url.onrender.com
```
Must be set in Vercel environment variables (no trailing slash).

**Backend:** CORS configured to allow all origins with `"*"` (should be restricted to specific Vercel domain in production).

## Deployment

### Frontend (Vercel)
- Root Directory: `frontend`
- Framework Preset: Vite
- Build Command: `npm run build` (or use vercel-build)
- Output Directory: `dist`
- Environment Variables: Set `VITE_API_URL` to backend URL

The `vercel.json` at root handles build configuration with `@vercel/static-build`.

### Backend (Render/Railway)
- Root Directory: `backend`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- First deployment takes 10-15 minutes (model download ~250MB)
- Free tier: Service sleeps after inactivity, takes 30-60 seconds to wake

## Important Implementation Details

### Sentiment Scale Mapping
The model outputs 7 classes (0-6) which map to sentiment scores:
- Class 0 → Score -3 (Very Negative) 😢
- Class 1 → Score -2 (Negative) 😞
- Class 2 → Score -1 (Slightly Negative) 😐
- Class 3 → Score 0 (Neutral) 😶
- Class 4 → Score +1 (Slightly Positive) 🙂
- Class 5 → Score +2 (Positive) 😊
- Class 6 → Score +3 (Very Positive) 🤩

This conversion happens in `model.py` via `class_to_score()` method.

### API Response Format
```json
{
  "text": "Original input text",
  "sentiment_score": 2,
  "sentiment_label": "Positive",
  "emoji": "😊",
  "confidence": 0.89,
  "probabilities": {
    "-3 (Very Negative)": 0.01,
    "-2 (Negative)": 0.02,
    "-1 (Slightly Negative)": 0.03,
    "0 (Neutral)": 0.04,
    "+1 (Slightly Positive)": 0.10,
    "+2 (Positive)": 0.65,
    "+3 (Very Positive)": 0.15
  }
}
```

### CORS Configuration
Backend allows multiple origins for flexibility during development:
- `http://localhost:3000` (Vite dev server)
- `http://localhost:5173` (Alternate Vite port)
- `https://*.vercel.app` (Vercel deployments)
- `*` (Wildcard - should be restricted in production)

### Build Considerations
- Frontend uses `--legacy-peer-deps` flag in vercel.json due to dependency conflicts
- Optional dependency `@rollup/rollup-linux-x64-gnu` included for Vercel Linux builds
- Backend model downloads automatically on first startup (cached thereafter)

## Key Files Reference

**Critical Frontend Files:**
- `frontend/src/App.jsx` - Main application component with tab navigation
- `frontend/src/services/api.js` - API communication layer
- `frontend/src/components/InputSection.jsx` - User input handling
- `frontend/src/components/ResultsSection.jsx` - Results visualization
- `frontend/vite.config.js` - Development proxy configuration

**Critical Backend Files:**
- `backend/main.py` - FastAPI app initialization and endpoints
- `backend/model.py` - SentimentAnalyzer class and model logic
- `backend/requirements.txt` - Python dependencies

**Configuration Files:**
- `vercel.json` - Vercel deployment configuration
- `frontend/tailwind.config.js` - Tailwind CSS customization
- `package.json` (root) - Workspace configuration and combined scripts

## Testing Strategy

When making modifications:
1. Test locally first (both frontend and backend running)
2. Check browser console for JavaScript errors (F12)
3. Check Network tab for API request/response details
4. Verify backend logs in terminal for Python errors
5. Test with various input lengths and edge cases
6. Verify responsive design at different screen sizes

## Common Modification Points

**Styling Changes:**
- Colors: `frontend/tailwind.config.js` and component className props
- Layout: Component JSX files with Tailwind grid/flex utilities

**Sentiment Scale Adjustments:**
- Labels/emojis: `backend/model.py` static methods
- Number of classes: Change `num_labels` in model initialization

**API Endpoints:**
- Add new routes in `backend/main.py` following FastAPI patterns
- Update `frontend/src/services/api.js` with corresponding client functions

**UI Components:**
- All components in `frontend/src/components/`
- Each is self-contained and can be modified independently
- Props are passed from App.jsx downward
