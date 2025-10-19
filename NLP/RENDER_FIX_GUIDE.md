# 🔧 Render Deployment Fix Guide

## Problem: Build Failed with Rust Compilation Error

Your deployment failed with this error:
```
error: failed to create directory `/usr/local/cargo/registry/cache/`
Read-only file system (os error 30)
```

## Root Cause
The specific version `pydantic-core==2.14.6` (dependency of `pydantic==2.5.3`) requires Rust compilation, which has filesystem permission issues on Render.

---

## ✅ Solution 1: Update Pydantic (RECOMMENDED)

### Step 1: Update requirements.txt

**I've already updated your file to:**
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.6.0  # ← Updated from 2.5.3
torch>=2.0.0
transformers>=4.30.0
python-multipart==0.0.6
```

### Step 2: Commit and Push

```bash
cd "C:\Users\Soren\OneDrive\Documents\school\College Senior\CST-405\Compiler\CST-435-Group\NLP"

git add nlp-react/backend/requirements.txt
git commit -m "Fix: Update pydantic to avoid Rust compilation issues on Render"
git push origin main
```

### Step 3: Redeploy on Render

1. Go to your Render dashboard
2. Find your service
3. Click **"Manual Deploy"** → **"Deploy latest commit"**
4. Wait 10-15 minutes (first time downloads AI model)

---

## ✅ Solution 2: Alternative Requirements (If Solution 1 Fails)

If the above doesn't work, try this more explicit version:

### Create a new file `requirements-render.txt`:

```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.6.0
pydantic-core==2.16.1
torch>=2.0.0,<2.5.0
transformers>=4.30.0,<5.0.0
python-multipart==0.0.6
numpy<2.0.0
```

### Update Render Build Command:

In Render dashboard → Settings → Build Command:
```
pip install --upgrade pip && pip install -r requirements-render.txt
```

---

## ✅ Solution 3: Use Precompiled Wheels

If both above fail, use CPU-only PyTorch (smaller, faster install):

### requirements.txt:
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic>=2.6.0
torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu
transformers>=4.30.0
python-multipart==0.0.6
```

### Render Build Command:
```
pip install --upgrade pip && pip install torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
```

---

## ✅ Solution 4: Specify Python Version

Sometimes the Python version matters. Add this to Render:

### In Render Dashboard → Environment:

Add environment variable:
```
PYTHON_VERSION = 3.11
```

(Default is 3.13, which might have compatibility issues)

---

## 🧪 Testing After Fix

Once deployed, test these endpoints:

### 1. Health Check
```
https://your-backend-url.onrender.com/health
```
Should return:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. API Docs
```
https://your-backend-url.onrender.com/docs
```
Should show interactive API documentation

### 3. Test Analysis
Go to `/docs` and try the `/analyze` endpoint with:
```json
{
  "text": "This movie was amazing!"
}
```

---

## 📋 Quick Checklist

- [ ] Updated `requirements.txt` with `pydantic==2.6.0`
- [ ] Committed and pushed changes to GitHub
- [ ] Triggered manual redeploy on Render
- [ ] Waited 10-15 minutes for build
- [ ] Checked build logs for success
- [ ] Tested `/health` endpoint
- [ ] Tested `/docs` page
- [ ] Tried sample analysis

---

## 🔍 How to Check Build Logs

1. Go to your Render dashboard
2. Click on your service
3. Click **"Logs"** tab
4. Select **"Build"** instead of "Deploy"
5. Watch for:
   - ✅ `Successfully installed fastapi...`
   - ✅ `Downloading transformers...`
   - ✅ `Downloading DistilBERT model...`
   - ✅ `Build succeeded ✓`

---

## 🚨 If Still Failing

### Check for These Common Issues:

1. **Wrong Root Directory**
   - Must be: `nlp-react/backend`
   - NOT: `backend` or `nlp-react` or empty

2. **Wrong Start Command**
   - Must be: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Check for typos!

3. **Memory Issues (512MB limit on free tier)**
   - PyTorch + Transformers model ≈ 400-450MB
   - Should fit, but barely
   - Solution: Use CPU-only torch or upgrade to paid tier

4. **Python Version Issues**
   - Try setting `PYTHON_VERSION=3.11` environment variable

---

## 💡 Pro Tips

### Speed Up Future Deploys
After first successful deploy, subsequent deploys are MUCH faster (2-3 minutes) because:
- Dependencies are cached
- Model is cached
- Only code changes rebuild

### Keep Backend Awake (Optional)
Free tier sleeps after 15 minutes. To keep it awake:

1. Sign up at [UptimeRobot.com](https://uptimerobot.com) (free)
2. Add HTTP monitor
3. URL: `https://your-backend-url.onrender.com/health`
4. Interval: 5 minutes
5. Backend stays awake 24/7!

---

## 📞 Still Need Help?

If you're still stuck after trying all solutions:

1. **Copy the full build log** from Render
2. **Check if the error changed** (different error = progress!)
3. **Share the new error** with instructor or classmates
4. **Try Railway.app instead** (alternative platform, sometimes more forgiving)

---

## 🎯 Expected Timeline

- **Fix requirements.txt**: 2 minutes (done! ✓)
- **Commit and push**: 1 minute
- **Render rebuild**: 10-15 minutes (first time)
- **Testing**: 2 minutes
- **Total**: ~15-20 minutes

---

Good luck! The updated `pydantic==2.6.0` should fix your issue. 🚀
