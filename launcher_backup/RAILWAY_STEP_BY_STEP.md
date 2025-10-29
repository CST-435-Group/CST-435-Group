# Railway Deployment - Step by Step

## The Simplest Way to Deploy

Follow these exact steps to get your backend running on Railway.

---

## Step 1: Go to Railway Dashboard

1. Open https://railway.app
2. Log in with your GitHub account
3. Click on your project (or create a new one if you haven't)

---

## Step 2: Configure Settings

Click on **Settings** in the left sidebar (or top menu).

### Setting A: Root Directory

1. Scroll to **"Root Directory"**
2. Click **"Configure"** or **"Edit"**
3. Enter: `launcher/backend`
4. Click **"Save"** or press Enter

**Why?** This tells Railway to build only from the backend directory, avoiding Docker context issues.

---

### Setting B: Start Command

1. Scroll to **"Start Command"** (also called "Custom Start Command")
2. Click **"Configure"** or **"Edit"**
3. Enter: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Click **"Save"** or press Enter

**Why?** This is the command Railway runs to start your FastAPI server.

---

### Setting C: Builder (Optional but Recommended)

1. Scroll to **"Builder"**
2. Click **"Configure"** or **"Edit"**
3. Select **"Nixpacks"** (NOT "Dockerfile")
4. Click **"Save"**

**Why?** Nixpacks automatically detects Python and installs dependencies.

---

## Step 3: Deploy

1. Click the **"Deploy"** button at the top
2. Or push a commit to trigger auto-deploy

Railway will:
- ✅ Clone your repository
- ✅ Navigate to `launcher/backend`
- ✅ Detect `requirements.txt`
- ✅ Install Python 3.11
- ✅ Install dependencies (`pip install -r requirements.txt`)
- ✅ Start the server with your command

**Build Time:** 5-10 minutes (first time, includes PyTorch)

---

## Step 4: Watch the Logs

While deploying, click **"View Logs"** or **"Deployments"** to see progress.

You should see:
```
Installing Python 3.11...
Installing dependencies...
Collecting fastapi==0.109.0
Collecting uvicorn[standard]==0.27.0
...
Building wheel for uvicorn...
Successfully installed fastapi-0.109.0 uvicorn-0.27.0 ...
Starting server...
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Step 5: Get Your URL

After deployment succeeds:

1. Railway will show your public URL
2. Copy it (looks like: `https://your-project.up.railway.app`)
3. Test it:

```bash
curl https://your-project.up.railway.app/health
```

Should return:
```json
{
  "status": "healthy",
  "services": {
    "ann": {"status": "not_loaded", "model_loaded": false},
    "cnn": {"status": "not_loaded", "model_loaded": false},
    "nlp": {"status": "not_loaded", "model_loaded": false}
  }
}
```

---

## Step 6: Test API Endpoints

Visit your API documentation:
```
https://your-project.up.railway.app/docs
```

You should see the FastAPI Swagger UI with all your endpoints listed.

Try some endpoints:
```bash
# Get NLP examples
curl https://your-project.up.railway.app/api/nlp/examples

# Analyze sentiment (will load NLP model on first request - slow)
curl -X POST https://your-project.up.railway.app/api/nlp/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'
```

---

## Common Issues

### Issue: Build fails with "No such file or directory"

**Solution:** Double-check your Root Directory setting:
- Should be: `launcher/backend` (with the slash)
- NOT: `backend`
- NOT: `launcher`

### Issue: Build fails with "No module named 'fastapi'"

**Solution:** Ensure `requirements.txt` exists in `launcher/backend/` directory.

### Issue: Server starts but API returns 404

**Solution:**
- Check that Start Command is exactly: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Ensure `main.py` exists in `launcher/backend/`

### Issue: Models don't load / Import errors

**Solution:**
- Check Railway logs for specific error
- Model files must be in the repository:
  - `ANN_Project/best_model.pth`
  - `CNN_Project/models/best_model.pth`
  - `nlp-react/backend/model.py`

### Issue: First request is very slow

**This is normal!**
- Models load on first request (lazy loading)
- First request: 10-30 seconds
- Subsequent requests: <1 second

### Issue: Railway sleeps after inactivity

**This is expected on free tier:**
- After 5-10 minutes of no requests, Railway sleeps the service
- First request after sleep: 30-60 seconds (cold start)
- Subsequent requests: fast
- Upgrade to Pro plan to prevent sleep

---

## Verification Checklist

After deployment, verify:

- [ ] Railway build completed successfully
- [ ] No errors in deployment logs
- [ ] Public URL is accessible
- [ ] Health check returns 200: `/health`
- [ ] API docs accessible: `/docs`
- [ ] Can access each project's health check:
  - [ ] `/api/ann/health`
  - [ ] `/api/cnn/health`
  - [ ] `/api/nlp/health`

---

## Next: Deploy Frontend to Vercel

Once your Railway backend is working:

1. Copy your Railway URL
2. Go to Vercel dashboard
3. Deploy your frontend from `launcher/frontend`
4. Set environment variable:
   - Name: `VITE_API_URL`
   - Value: Your Railway URL (e.g., `https://your-app.up.railway.app`)
5. Redeploy Vercel

Then your full stack will be live!

---

## Need Help?

- **Detailed troubleshooting**: See `RAILWAY_FIX.md`
- **Simple explanation**: See `RAILWAY_SIMPLE_FIX.md`
- **Full documentation**: See `README.md`

## Summary

**The three critical settings:**

1. ✅ Root Directory: `launcher/backend`
2. ✅ Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. ✅ Builder: Nixpacks

That's all you need! 🚀
