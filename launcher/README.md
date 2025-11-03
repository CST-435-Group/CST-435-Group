# CST-435 ML Projects Launcher

Unified project launcher for accessing multiple machine learning projects: ANN (NBA Team Selection), CNN (Fruit Classification), and NLP (Sentiment Analysis).

## Architecture

### Frontend (Vercel)
- **Framework**: React 18 + Vite + Tailwind CSS
- **Routing**: React Router with lazy loading
- **Features**:
  - Home page with project selection
  - Individual pages for each ML project
  - Responsive design
  - Optimized for production

### Backend (Railway)
- **Framework**: FastAPI
- **Architecture**: API Gateway pattern
- **Routes**:
  - `/api/ann/*` - NBA Team Selection (ANN)
  - `/api/cnn/*` - Fruit Classification (CNN)
  - `/api/nlp/*` - Sentiment Analysis (NLP)
- **Features**:
  - Lazy loading of ML models (memory efficient)
  - Unified health checks
  - CORS configuration
  - CPU-optimized PyTorch

## Project Structure

```
launcher/
├── frontend/               # React frontend
│   ├── src/
│   │   ├── pages/         # Individual project pages
│   │   ├── components/    # Shared components
│   │   └── services/      # API clients
│   ├── package.json
│   └── vite.config.js
├── backend/               # FastAPI backend
│   ├── routers/          # Project-specific routers
│   │   ├── ann.py
│   │   ├── cnn.py
│   │   └── nlp.py
│   ├── main.py           # API Gateway
│   └── requirements.txt
├── vercel.json           # Vercel deployment config
└── railway.toml          # Railway deployment config
```

## Local Development

### Prerequisites
- Node.js 18+
- Python 3.11+
- Git

### Frontend Setup

```bash
cd launcher/frontend
npm install
npm run dev
```

Frontend will run on http://localhost:3000

### Backend Setup

```bash
cd launcher/backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

Backend will run on http://localhost:8000

### Environment Variables

Create `.env` files in the root:

**Frontend (.env)**
```
VITE_API_URL=http://localhost:8000
```

**Backend (.env)**
```
PORT=8000
```

## Deployment

### Vercel Deployment (Frontend)

1. **Install Vercel CLI**
```bash
npm i -g vercel
```

2. **Login to Vercel**
```bash
vercel login
```

3. **Deploy**
```bash
cd launcher
vercel
```

4. **Set Environment Variables**
Go to Vercel Dashboard → Project Settings → Environment Variables:
- `VITE_API_URL` = Your Railway backend URL (e.g., `https://your-app.railway.app`)

5. **Production Deployment**
```bash
vercel --prod
```

### Railway Deployment (Backend)

1. **Create Railway Account**
   - Go to https://railway.app
   - Sign up with GitHub

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `CST-435-Group` repository

3. **Configure Build**
   - Railway will automatically detect `railway.toml` and `nixpacks.toml`
   - **Important**: Railway looks at the `launcher` directory
   - No need to set root directory manually
   - Railway will use the start command from `railway.toml`

4. **If Build Fails with "Unable to generate build plan"**

   **Quick Fix Option 1** - Set Root Directory:
   - Go to Settings → "Root Directory"
   - Set to `backend`
   - Update "Start Command" to: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Redeploy

   **Quick Fix Option 2** - Use Dockerfile:
   - Go to Settings → "Builder"
   - Select "Dockerfile"
   - Set "Dockerfile Path" to `launcher/Dockerfile`
   - Redeploy

   See `RAILWAY_FIX.md` for detailed troubleshooting.

5. **Environment Variables**
   - `PORT` - Automatically set by Railway
   - `PYTHON_VERSION` - Already configured in `railway.toml`

6. **Model Files**
   - Ensure model files exist in their respective project directories:
     - `ANN_Project/best_model.pth`
     - `CNN_Project/models/best_model.pth`
     - `CNN_Project/models/model_metadata.json`

7. **First Deployment**
   - First deployment will take 10-15 minutes (downloading PyTorch + models)
   - Subsequent deployments will be faster
   - Watch logs for progress

## API Endpoints

### General
- `GET /` - API information
- `GET /health` - Health check for all services

### ANN (NBA Team Selection)
- `GET /api/ann/` - Project info
- `GET /api/ann/health` - Health check
- `GET /api/ann/data-info` - Dataset statistics
- `POST /api/ann/select-team` - Select optimal team
- `GET /api/ann/players` - Get top players

### CNN (Fruit Classification)
- `GET /api/cnn/` - Project info
- `GET /api/cnn/health` - Health check
- `GET /api/cnn/info` - Model information
- `GET /api/cnn/fruit-list` - List of recognizable fruits
- `POST /api/cnn/predict` - Classify image (multipart/form-data)
- `POST /api/cnn/predict-base64` - Classify base64 image

### NLP (Sentiment Analysis)
- `GET /api/nlp/` - Project info
- `GET /api/nlp/health` - Health check
- `POST /api/nlp/analyze` - Analyze single text
- `POST /api/nlp/analyze/batch` - Analyze multiple texts
- `GET /api/nlp/examples` - Example reviews
- `GET /api/nlp/sentiment-scale` - Sentiment scale info

## Features

### Home Page
- Project cards with descriptions
- Quick navigation to individual projects
- Responsive grid layout

### ANN Project Page
- View dataset statistics
- Select team using 3 different algorithms:
  - Greedy (top 5 players)
  - Balanced (position-aware)
  - Exhaustive search
- View top players
- Team composition analysis

### CNN Project Page
- Upload fruit images
- Real-time classification
- Confidence scores
- Top 3 predictions
- Support for multiple fruit types

### NLP Project Page
- Analyze sentiment of text
- 7-point sentiment scale (-3 to +3)
- Example reviews for testing
- Confidence scores
- Probability distribution

## Performance Optimizations

### Frontend
- Lazy loading of pages
- Code splitting
- Optimized images
- Minimized bundle size

### Backend
- Lazy loading of ML models
- CPU-optimized PyTorch
- CORS configuration
- Request timeout handling
- Health checks

## Troubleshooting

### Frontend Issues

**Problem**: API calls fail with CORS error
**Solution**: Ensure backend CORS is configured to allow your Vercel domain

**Problem**: 404 on page refresh
**Solution**: Vercel automatically handles this with `rewrites` in `vercel.json`

### Backend Issues

**Problem**: Model files not found
**Solution**: Ensure model files are in correct locations:
- `ANN_Project/best_model.pth`
- `CNN_Project/models/best_model.pth`
- `CNN_Project/models/model_metadata.json`

**Problem**: Out of memory on Railway
**Solution**: Models are lazy-loaded. Only one project loaded at a time. If still issues, upgrade Railway plan.

**Problem**: Slow first request
**Solution**: This is normal - models load on first request. Subsequent requests are fast.

### Deployment Issues

**Problem**: Vercel build fails
**Solution**:
- Check Node.js version (need 18+)
- Ensure `package.json` is in `frontend/` directory
- Check build logs in Vercel dashboard

**Problem**: Railway deployment fails
**Solution**:
- Check Python version (need 3.11)
- Ensure `requirements.txt` includes all dependencies
- Check Railway logs for specific errors

## Links

- **Vercel Docs**: https://vercel.com/docs
- **Railway Docs**: https://docs.railway.app
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **React Router Docs**: https://reactrouter.com

## Contributing

This project was developed for CST-435 Neural Networks course.

## License

Educational use only - CST-435 Course Project

## Generating per-project cost JSON files

This repo includes a helper script that writes `docs/cost_report.json` for each project using the `cost_analysis` utilities.

From the repo root run (PowerShell):

```powershell
# generate for all known projects
python .\tools\generate_cost_reports.py --all

# generate only ANN and CNN
python .\tools\generate_cost_reports.py --projects ann cnn

# generate using the phased breakdown (Phase 1 / Phase 2)
python .\tools\generate_cost_reports.py --all --use-phased
```

The script will create `docs/` directories when needed and write `cost_report.json` with the computed values. The backend docs router prefers these files if present.
