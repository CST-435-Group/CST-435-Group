# Railway Deployment Fix

## Problem

Railway's Nixpacks couldn't detect the project type because the `launcher` directory has both `frontend/` and `backend/` subdirectories without a clear entry point at the root.

**Error:**
```
Nixpacks was unable to generate a build plan for this app.
```

## Solution

I've added several files to help Railway understand this is a Python backend project:

### Files Added

1. **`nixpacks.toml`** - Tells Nixpacks how to build the project
2. **`requirements.txt`** (root) - References `backend/requirements.txt`
3. **`main.py`** (root) - Entry point that imports from `backend/`
4. **Updated `railway.toml`** - Points to nixpacks config

## Deployment Options

### Option 1: Use railway.toml (Recommended)

The `railway.toml` now includes:

```toml
[build]
builder = "NIXPACKS"
nixpacksConfigPath = "nixpacks.toml"

[deploy]
startCommand = "cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT"
```

**Steps:**
1. Push changes to GitHub
2. Railway will auto-detect `railway.toml`
3. It will use `nixpacks.toml` for build configuration
4. Should deploy successfully now

### Option 2: Use Root-Level main.py

If Option 1 doesn't work, Railway can now detect the root `main.py`:

**Steps:**
1. Go to Railway dashboard
2. Go to your project settings
3. Change start command to: `python main.py`
4. Redeploy

### Option 3: Set Root Directory to backend/

**Steps:**
1. Go to Railway dashboard
2. Project Settings → Service Settings
3. Set "Root Directory" to `backend`
4. Railway will now build from the backend directory
5. Remove the "cd backend &&" from start command

### Option 4: Use Dockerfile (Most Control)

If Nixpacks still has issues, create a Dockerfile:

```dockerfile
# launcher/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy backend code
COPY backend/ /app/backend/

# Copy parent directories for imports
COPY ANN_Project/ /app/ANN_Project/
COPY CNN_Project/ /app/CNN_Project/
COPY nlp-react/ /app/nlp-react/

# Install dependencies
WORKDIR /app/backend
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Then in Railway:
- Settings → Deploy → Dockerfile Path: `launcher/Dockerfile`

## Recommended Approach

**Try in this order:**

1. ✅ **Push the new files to GitHub** (nixpacks.toml, requirements.txt, updated railway.toml)
2. ✅ **Redeploy on Railway** - It should work now
3. ❌ If still fails → Try Option 3 (set root directory to `backend`)
4. ❌ If still fails → Try Option 4 (Dockerfile)

## Verification

After deployment succeeds, verify:

```bash
# Check health endpoint
curl https://your-app.railway.app/health

# Should return:
{
  "status": "healthy",
  "services": {
    "ann": {"status": "not_loaded", "model_loaded": false},
    "cnn": {"status": "not_loaded", "model_loaded": false},
    "nlp": {"status": "not_loaded", "model_loaded": false}
  }
}

# Check API docs
curl https://your-app.railway.app/docs
# Should return HTML page
```

## Current File Structure

```
launcher/
├── main.py               # NEW: Root entry point
├── requirements.txt      # NEW: References backend/requirements.txt
├── nixpacks.toml         # NEW: Build configuration
├── railway.toml          # UPDATED: Points to nixpacks.toml
├── backend/
│   ├── main.py          # Actual FastAPI app
│   └── requirements.txt # Python dependencies
└── frontend/
    └── ...
```

## Troubleshooting

### Build still fails with same error

**Solution:** Railway may be caching. Try:
1. Settings → Deployments → Click "..." on latest → Redeploy
2. Or make a small change and push again

### Build fails with "module not found"

**Solution:** The imports from parent projects (ANN, CNN, NLP) may not be available.

**Fix:** Add them to the build:
1. In Railway dashboard
2. Settings → Variables
3. Add: `PYTHONPATH=/app:/app/backend`

### Models not found after deployment

**Solution:** Model files need to be in the repository or uploaded separately.

Ensure these exist:
- `../ANN_Project/best_model.pth`
- `../CNN_Project/models/best_model.pth`
- `../CNN_Project/models/model_metadata.json`

### Out of memory

**Solution:**
- Models are lazy-loaded, should be fine
- If needed, comment out model loaders you're not using
- Or upgrade Railway plan

## Success Checklist

After Railway deployment succeeds:

- [ ] Health check returns 200: `curl https://your-app.railway.app/health`
- [ ] API docs accessible: `https://your-app.railway.app/docs`
- [ ] Can access ANN health: `/api/ann/health`
- [ ] Can access CNN health: `/api/cnn/health`
- [ ] Can access NLP health: `/api/nlp/health`
- [ ] First model request works (may be slow)
- [ ] Subsequent requests are fast
- [ ] No errors in Railway logs

## Next Steps

Once Railway deployment succeeds:

1. Copy your Railway URL (e.g., `https://cst-435-launcher-production.up.railway.app`)
2. Go to Vercel dashboard
3. Add environment variable:
   - Name: `VITE_API_URL`
   - Value: Your Railway URL
4. Redeploy Vercel frontend
5. Test the full stack!

## Need Help?

If you're still having issues:

1. Check Railway logs: Dashboard → Deployments → View logs
2. Look for specific error messages
3. Try the Dockerfile approach (Option 4)
4. Share the full error message for more help
