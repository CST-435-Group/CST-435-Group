# System Architecture

## Overview

The unified launcher provides a single entry point for three distinct machine learning projects. The system uses a modern web architecture with separate frontend and backend deployments.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         USER BROWSER                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    VERCEL (Frontend)                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │           React + Vite + Tailwind                     │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │  │
│  │  │  Home   │  │  ANN    │  │  CNN    │  ┌─────────┐ │  │
│  │  │  Page   │  │  Page   │  │  Page   │  │  NLP    │ │  │
│  │  │         │  │         │  │         │  │  Page   │ │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │  │
│  │                                                        │  │
│  │  React Router + Lazy Loading                          │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                         HTTPS/REST API
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   RAILWAY (Backend)                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              FastAPI Gateway                          │  │
│  │                   (main.py)                           │  │
│  └───────────────────────────────────────────────────────┘  │
│                              │                              │
│         ┌────────────────────┼────────────────────┐         │
│         ▼                    ▼                    ▼         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐  │
│  │   ANN       │     │   CNN       │     │   NLP       │  │
│  │   Router    │     │   Router    │     │   Router    │  │
│  │  (ann.py)   │     │  (cnn.py)   │     │  (nlp.py)   │  │
│  └─────────────┘     └─────────────┘     └─────────────┘  │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐  │
│  │ ANN Model   │     │ CNN Model   │     │ NLP Model   │  │
│  │ (PyTorch)   │     │ (PyTorch)   │     │(Transformers)│  │
│  │ Lazy Load   │     │ Lazy Load   │     │ Lazy Load   │  │
│  └─────────────┘     └─────────────┘     └─────────────┘  │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Existing Project Code                      │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │  │
│  │  │ ANN_     │  │ CNN_     │  │ nlp-react/       │   │  │
│  │  │ Project/ │  │ Project/ │  │ backend/         │   │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Frontend Architecture (Vercel)

### Technology Stack
- **Framework**: React 18.2
- **Build Tool**: Vite 5.0
- **Styling**: Tailwind CSS 3.4
- **Routing**: React Router 6.20
- **HTTP Client**: Axios 1.6

### Component Structure

```
frontend/
├── src/
│   ├── main.jsx              # App entry point
│   ├── App.jsx               # Root component with routing
│   ├── index.css             # Global styles
│   │
│   ├── pages/                # Lazy-loaded pages
│   │   ├── Home.jsx          # Project selection (loaded on demand)
│   │   ├── ANNProject.jsx    # ANN interface (lazy)
│   │   ├── CNNProject.jsx    # CNN interface (lazy)
│   │   └── NLPProject.jsx    # NLP interface (lazy)
│   │
│   ├── components/           # Shared components
│   │   ├── Header.jsx        # Navigation header
│   │   └── LoadingSpinner.jsx
│   │
│   └── services/             # API integration
│       └── api.js            # Axios client + endpoints
│
├── public/                   # Static assets
├── package.json              # Dependencies
├── vite.config.js            # Vite configuration
└── tailwind.config.js        # Tailwind configuration
```

### Routing Strategy

```javascript
/ (Home)
├── /ann    → ANNProject component (lazy)
├── /cnn    → CNNProject component (lazy)
└── /nlp    → NLPProject component (lazy)
```

**Lazy Loading Benefits**:
- Smaller initial bundle size
- Faster page load
- Code splitting per route
- Only loads what user needs

### API Communication

```javascript
// API Base URL (from environment)
VITE_API_URL → Railway backend URL

// Proxy configuration (development)
/api/* → http://localhost:8000/api/*

// Example API call
axios.post('/api/ann/select-team', {
  method: 'balanced',
  start_year: '1996-97',
  end_year: '2019-20'
})
```

## Backend Architecture (Railway)

### Technology Stack
- **Framework**: FastAPI 0.109
- **Server**: Uvicorn with ASGI
- **ML Libraries**: PyTorch 2.0+, Transformers 4.30+
- **Data Processing**: NumPy, Pandas, Scikit-learn

### API Gateway Pattern

```python
FastAPI App (main.py)
├── CORS Middleware (cross-origin requests)
├── Router: /api/ann/* → ANN endpoints
├── Router: /api/cnn/* → CNN endpoints
└── Router: /api/nlp/* → NLP endpoints
```

### Router Structure

Each router is independent and follows the same pattern:

```python
router = APIRouter()

# Global model storage (lazy loading)
model = None

def load_model():
    """Load model only when first requested"""
    global model
    if model is None:
        # Load from disk
        model = torch.load(...)
    return model

@router.get("/health")
async def health():
    """Check if model is loaded"""
    return {"status": "ready" if model else "not_loaded"}

@router.post("/predict")
async def predict(data):
    """Make prediction with lazy-loaded model"""
    model = load_model()
    return model.predict(data)
```

### Lazy Loading Strategy

**Why Lazy Loading?**
- Railway free tier has limited RAM (512MB - 8GB depending on plan)
- Each model is ~100-500MB
- Only load models when actually needed
- Prevents out-of-memory errors

**How It Works**:
1. Models start as `None`
2. First API request triggers model loading
3. Model stays in memory for subsequent requests
4. If Railway restarts, models reload on next request

### Project Integration

The backend doesn't modify existing project code, it wraps it:

```python
# ANN Router
import sys
sys.path.insert(0, '../ANN_Project')
from src.model import create_model
from src.select_team import TeamSelector

# CNN Router
class FruitCNN(nn.Module):
    # Copy of model architecture from CNN_Project

# NLP Router
sys.path.insert(0, '../nlp-react/backend')
from model import SentimentAnalyzer
```

## Data Flow

### Example: ANN Team Selection

```
User clicks "Select Team" button
    │
    ▼
Frontend sends POST to /api/ann/select-team
    │
    ▼
API Gateway routes to ANN router
    │
    ▼
ANN router checks if model loaded
    │
    ├─ No → Load model from ANN_Project/best_model.pth
    └─ Yes → Use cached model
    │
    ▼
Load NBA player data
    │
    ▼
Preprocess features
    │
    ▼
Run team selection algorithm
    │
    ▼
Return team composition
    │
    ▼
Frontend displays results
```

### Example: CNN Image Classification

```
User uploads image
    │
    ▼
Frontend converts to FormData
    │
    ▼
POST to /api/cnn/predict
    │
    ▼
CNN router receives image file
    │
    ├─ Model not loaded → Load from CNN_Project/models/
    └─ Model loaded → Use cached
    │
    ▼
Preprocess image (resize, normalize)
    │
    ▼
Run inference
    │
    ▼
Return predictions + confidence
    │
    ▼
Frontend displays results
```

## Deployment Flow

### Frontend (Vercel)

```
Git Push → GitHub
    │
    ▼
Vercel webhook triggered
    │
    ▼
Clone repository
    │
    ▼
cd launcher/frontend
    │
    ▼
npm install
    │
    ▼
npm run build
    │
    ▼
Deploy to CDN
    │
    ▼
Available at https://your-app.vercel.app
```

### Backend (Railway)

```
Git Push → GitHub
    │
    ▼
Railway webhook triggered
    │
    ▼
Clone repository
    │
    ▼
Detect railway.toml
    │
    ▼
Install Python 3.11
    │
    ▼
pip install -r backend/requirements.txt
    │
    ▼
Start: uvicorn main:app --port $PORT
    │
    ▼
Health check /health
    │
    ▼
Available at https://your-app.railway.app
```

## Performance Optimizations

### Frontend
- **Code Splitting**: Each page is lazy-loaded
- **Tree Shaking**: Vite removes unused code
- **CSS Purging**: Tailwind removes unused styles
- **Asset Optimization**: Vite optimizes images/fonts
- **Caching**: Browser caches static assets

### Backend
- **Lazy Loading**: Models load on demand
- **CPU-Optimized PyTorch**: Uses CPU-specific builds
- **Connection Pooling**: Reuses connections
- **Response Compression**: Gzip compression
- **Request Timeout**: 30s timeout prevents hanging

## Security

### Frontend
- **HTTPS Only**: Enforced by Vercel
- **CORS**: Backend validates origins
- **XSS Protection**: React escapes content
- **CSP Headers**: Content Security Policy

### Backend
- **Input Validation**: Pydantic models
- **Rate Limiting**: Can add middleware
- **CORS Whitelist**: Specific origins only
- **File Upload Limits**: Max file size enforced

## Monitoring

### Vercel
- Build logs
- Function logs
- Analytics dashboard
- Error tracking

### Railway
- Application logs
- Resource usage (CPU, memory)
- Request metrics
- Health check monitoring

## Scalability

### Current Limits
- **Railway Free Tier**: 500 hours/month, $5 credit
- **Vercel Free Tier**: 100GB bandwidth/month
- **Memory**: Models use ~500MB-1GB RAM total

### Scale-Up Path
1. **Upgrade Railway Plan**: More RAM, no sleep
2. **Add Caching**: Redis for predictions
3. **Load Balancing**: Multiple Railway instances
4. **CDN**: CloudFlare for assets
5. **Model Optimization**: Quantization, pruning

## Failure Modes and Recovery

### Model Loading Fails
- Health check returns `not_loaded`
- Frontend shows user-friendly error
- Logs contain detailed error message

### API Timeout
- 30s timeout on requests
- Frontend shows timeout error
- User can retry

### Railway Sleep (Free Tier)
- First request takes 30-60s (cold start)
- Subsequent requests are fast
- Upgrade to prevent sleep

### Out of Memory
- Railway restarts service
- Models reload on next request
- Consider model optimization

## Future Enhancements

### Potential Improvements
- [ ] Model caching with Redis
- [ ] Batch prediction endpoints
- [ ] WebSocket for real-time updates
- [ ] User authentication
- [ ] Result history storage
- [ ] A/B testing framework
- [ ] Model versioning
- [ ] Prometheus metrics
- [ ] Docker containerization
- [ ] CI/CD pipeline with tests
