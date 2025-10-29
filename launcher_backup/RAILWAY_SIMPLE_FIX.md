# Railway Simple Fix - USE THIS!

## The Issue

The Dockerfile is trying to copy files from parent directories (`../ANN_Project`, etc.) but Docker can only access files within the `launcher` directory.

## The EASIEST Solution

**Don't use the Dockerfile at all. Just tell Railway to build from the `backend` directory directly.**

## Step-by-Step Fix

### 1. Go to Railway Dashboard
- Navigate to your project
- Click on your service

### 2. Settings → Service Settings

**Update these settings:**

1. **Root Directory**
   - Click "Configure"
   - Set to: `launcher/backend`
   - Click "Save"

2. **Start Command**
   - Click "Configure"
   - Set to: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Click "Save"

3. **Builder** (optional, but recommended)
   - Click "Configure"
   - Select "Nixpacks" (not Dockerfile)
   - Click "Save"

### 3. Redeploy

Click the "Deploy" button or push a new commit.

## Why This Works

- Railway will now build **only** from the `backend` directory
- It will detect `requirements.txt` in that directory
- It will install Python dependencies
- The parent projects (`ANN_Project`, `CNN_Project`, `nlp-react`) will still be accessible because they're in the repository
- Python can import from them using relative paths

## What About the Imports?

The backend routers import from parent directories like this:

```python
# In backend/routers/ann.py
ann_project_path = Path(__file__).parent.parent.parent.parent / "ANN_Project"
sys.path.insert(0, str(ann_project_path))
```

This will work because:
- `Path(__file__)` = `/app/launcher/backend/routers/ann.py`
- `.parent.parent.parent.parent` = `/app/` (repository root)
- `/app/ANN_Project` exists in the Railway deployment

## Verify It Works

After deployment:

```bash
# Test health check
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
```

## If Models Don't Load

The models might not load if the files aren't in the repository. Check Railway logs:

```
Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Then make a request:
```bash
curl https://your-app.railway.app/api/nlp/examples
```

If you get an error about missing model files, you'll need to ensure:
- `ANN_Project/best_model.pth` is in the repository
- `CNN_Project/models/best_model.pth` is in the repository
- `nlp-react/backend/` code is in the repository

## Alternative: Deploy Backend Separately

If you want even more control, deploy just the backend as a separate Railway project:

1. Create a new Railway project
2. Set root directory to `launcher/backend`
3. Everything else automatic

## Summary

**The fix is simple: Tell Railway to build from `launcher/backend` instead of `launcher`.**

Settings you need:
- ✅ Root Directory: `launcher/backend`
- ✅ Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- ✅ Builder: Nixpacks (not Dockerfile)

That's it! Redeploy and it should work.
