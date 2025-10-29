# Deploy Now - Quick Guide

## You Just Fixed the Railway Error!

I've added the necessary files to fix the Nixpacks detection issue. Here's what to do now:

## Step 1: Commit and Push Changes

```bash
cd "C:\Users\Soren\OneDrive\Documents\school\College Senior\CST-405\Compiler\CST-435-Group"

git add launcher/
git commit -m "Add Railway deployment configuration files"
git push
```

## Step 2: Deploy to Railway

### Option A: Automatic (if Railway is already connected)

Railway will automatically detect your push and start building. Watch the logs in Railway dashboard.

### Option B: Manual Trigger

1. Go to Railway dashboard: https://railway.app
2. Select your project
3. Click "Deploy" → "Redeploy"

## Step 3: Monitor the Build

Railway will now:
1. ✅ Detect Python project (from root `requirements.txt`)
2. ✅ Use `nixpacks.toml` for build configuration
3. ✅ Install dependencies from `backend/requirements.txt`
4. ✅ Start server with command from `railway.toml`

**Build time:** 5-10 minutes (first time, includes PyTorch)

## Step 4: Verify Deployment

Once build completes, test your API:

```bash
# Replace with your Railway URL
export RAILWAY_URL="https://your-app.up.railway.app"

# Test health check
curl $RAILWAY_URL/health

# Should return:
# {
#   "status": "healthy",
#   "services": {
#     "ann": {"status": "not_loaded", ...},
#     "cnn": {"status": "not_loaded", ...},
#     "nlp": {"status": "not_loaded", ...}
#   }
# }

# Test API docs (should return HTML)
curl $RAILWAY_URL/docs

# Test an endpoint (will load model on first request - slow)
curl $RAILWAY_URL/api/nlp/examples
```

## Step 5: Deploy Frontend to Vercel

```bash
cd launcher
vercel login
vercel
```

When prompted:
- **Project name**: `cst-435-launcher`
- **Link to existing project**: No
- **Which directory**: `./` (current)

After deployment, set environment variable:

```bash
# Set your Railway URL
vercel env add VITE_API_URL production
# Paste your Railway URL when prompted

# Redeploy to apply env var
vercel --prod
```

## Step 6: Test Full Stack

Visit your Vercel URL and:
- ✅ Home page loads with project cards
- ✅ Click ANN project → loads ANN page
- ✅ Click CNN project → loads CNN page
- ✅ Click NLP project → loads NLP page
- ✅ Try features (may be slow first time - models loading)

## What If It Still Fails?

### Railway Build Fails

If you still get the same error, try:

**Quick Fix - Set Root Directory:**
1. Railway Dashboard → Settings
2. "Root Directory" → Set to `backend`
3. "Start Command" → Set to `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Redeploy

**Alternative - Use Dockerfile:**
1. Railway Dashboard → Settings
2. "Builder" → Select "Dockerfile"
3. "Dockerfile Path" → Set to `launcher/Dockerfile`
4. Redeploy

### Railway Builds But Models Don't Load

Check Railway logs for specific errors:
- Missing model files? Ensure they're in the repository
- Import errors? May need to adjust Python paths
- Out of memory? Models are lazy-loaded, should be fine

### Frontend Can't Connect to Backend

1. Check CORS in `backend/main.py`
2. Add your Vercel domain to `allow_origins`
3. Redeploy Railway backend

## Files I Added to Fix Railway

```
launcher/
├── nixpacks.toml        # NEW: Tells Nixpacks how to build
├── requirements.txt     # NEW: Root-level to help detection
├── main.py              # NEW: Alternative entry point
├── railway.toml         # UPDATED: Points to nixpacks config
├── Dockerfile           # NEW: Backup option if Nixpacks fails
└── RAILWAY_FIX.md       # NEW: Detailed troubleshooting
```

## Success Checklist

- [ ] Code pushed to GitHub
- [ ] Railway build completes successfully
- [ ] Railway health check returns 200
- [ ] Railway API docs accessible
- [ ] Vercel frontend deploys
- [ ] VITE_API_URL env var set in Vercel
- [ ] Can access all 3 projects from Vercel URL
- [ ] API calls work (check browser console)

## Quick Commands Reference

```bash
# Check Railway deployment status
railway status

# View Railway logs
railway logs

# Check Vercel deployment
vercel ls

# Redeploy Vercel
vercel --prod

# Test API locally before deploying
cd launcher/backend
python main.py
# Visit http://localhost:8000/docs
```

## Need More Help?

- **Railway Issues**: Check `RAILWAY_FIX.md` for detailed troubleshooting
- **General Docs**: Check `README.md`
- **Architecture**: Check `ARCHITECTURE.md`
- **Deployment**: Check `DEPLOYMENT_CHECKLIST.md`

## Pro Tips

1. **First deployment takes longest** (10-15 min for PyTorch)
2. **Railway free tier sleeps** after inactivity (30-60s wake time)
3. **Models load on first request** per project (slow first time)
4. **Check Railway logs** if something isn't working
5. **Test locally first** to catch issues early

---

You're almost there! Just commit, push, and watch Railway deploy. Good luck! 🚀
