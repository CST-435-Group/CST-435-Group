# Quick Start Guide

## Initial Setup (One Time)

### 1. Install Frontend Dependencies
```bash
cd launcher/frontend
npm install
```

### 2. Install Backend Dependencies
```bash
cd launcher/backend
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## Running Locally

### Option 1: Run Both Services Manually

**Terminal 1 - Backend:**
```bash
cd launcher/backend
# Activate venv if not already active
python main.py
```
Backend will be at: http://localhost:8000

**Terminal 2 - Frontend:**
```bash
cd launcher/frontend
npm run dev
```
Frontend will be at: http://localhost:3000

### Option 2: Use the Root Scripts

From the `launcher` directory:

```bash
# Install all dependencies
npm run install:all

# Terminal 1 - Backend
npm run dev:backend

# Terminal 2 - Frontend
npm run dev:frontend
```

## Testing the Setup

1. **Backend API**:
   - Open http://localhost:8000/docs
   - You should see the FastAPI interactive documentation

2. **Frontend**:
   - Open http://localhost:3000
   - You should see the project selection home page

3. **Health Check**:
   - Visit http://localhost:8000/health
   - Should return status for all services

## Project URLs

### Local Development
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Individual Projects
- **Home**: http://localhost:3000/
- **ANN Project**: http://localhost:3000/ann
- **CNN Project**: http://localhost:3000/cnn
- **NLP Project**: http://localhost:3000/nlp

## Troubleshooting

### Backend won't start

**Error: "Module not found"**
```bash
# Make sure you're in the backend directory
cd launcher/backend

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Error: "Model files not found"**
- Ensure the following files exist:
  - `ANN_Project/best_model.pth`
  - `CNN_Project/models/best_model.pth`
  - `CNN_Project/models/model_metadata.json`
  - `nlp-react/backend/` (NLP model)

### Frontend won't start

**Error: "Cannot find module"**
```bash
# Make sure you're in the frontend directory
cd launcher/frontend

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

**Error: "API calls fail"**
- Make sure backend is running on port 8000
- Check the console for CORS errors
- Verify `vite.config.js` proxy configuration

### Models not loading

**First request is slow**
- This is normal! Models load lazily on first request
- Subsequent requests will be much faster

**Out of memory**
- Models are lazy-loaded, only one at a time
- Close other applications to free up RAM
- If still issues, comment out the models you're not testing

## Next Steps

1. **Test each project**:
   - Navigate to each project page
   - Try the features
   - Check API responses in browser dev tools

2. **Deploy to Vercel/Railway**:
   - See `README.md` for deployment instructions
   - Set up environment variables
   - Deploy frontend to Vercel
   - Deploy backend to Railway

3. **Configure production**:
   - Update CORS origins in `backend/main.py`
   - Set `VITE_API_URL` in Vercel environment variables
   - Test production deployment

## Common Commands

```bash
# Frontend
cd launcher/frontend
npm run dev          # Start dev server
npm run build        # Build for production
npm run preview      # Preview production build

# Backend
cd launcher/backend
python main.py       # Start FastAPI server
uvicorn main:app --reload  # Start with auto-reload

# From root
npm run install:all      # Install all dependencies
npm run dev:frontend     # Run frontend
npm run dev:backend      # Run backend
```

## Need Help?

- Check `README.md` for detailed documentation
- View API docs at http://localhost:8000/docs
- Check browser console for frontend errors
- Check terminal for backend errors
