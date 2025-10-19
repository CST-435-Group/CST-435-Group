# 🚀 Quick Render Deployment Reference

## Your Project Configuration

### ✅ Backend Deployment (Render Web Service)

| Setting | Value |
|---------|-------|
| **Service Name** | `sentiment-analyzer-api` |
| **Language** | Python |
| **Region** | Oregon (US West) or Ohio (US East) |
| **Branch** | `main` |
| **Root Directory** | `nlp-react/backend` ⚠️ CRITICAL |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Plan** | Free |

### Environment Variables (Optional)
```
PYTHON_VERSION = 3.11
MODEL_NAME = distilbert-base-uncased
```

---

## ✅ Frontend Deployment (Vercel)

| Setting | Value |
|---------|-------|
| **Framework** | Vite |
| **Root Directory** | `nlp-react/frontend` ⚠️ CRITICAL |
| **Build Command** | `npm run build` |
| **Output Directory** | `dist` |
| **Install Command** | `npm install` |

### Environment Variables (REQUIRED)
```
VITE_API_URL = https://your-backend-url.onrender.com
```
⚠️ **NO TRAILING SLASH!**

---

## 📝 Git Commands to Deploy Fix

```bash
# Navigate to project
cd "C:\Users\Soren\OneDrive\Documents\school\College Senior\CST-405\Compiler\CST-435-Group\NLP"

# Check status
git status

# Add changes
git add .

# Commit with message
git commit -m "Fix: Update pydantic version for Render compatibility"

# Push to GitHub
git push origin main
```

---

## 🧪 Testing Endpoints After Deployment

### Backend Health Check
```
GET https://your-backend-url.onrender.com/health
```
Expected Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Backend API Docs
```
https://your-backend-url.onrender.com/docs
```

### Test Analysis
```
POST https://your-backend-url.onrender.com/analyze
Content-Type: application/json

{
  "text": "This movie was absolutely amazing!"
}
```

---

## ⏱️ Expected Deployment Times

| Event | Time |
|-------|------|
| First backend build | 10-15 minutes |
| Subsequent builds | 2-3 minutes |
| Frontend build | 2-3 minutes |
| Backend wake-up (after sleep) | 30-60 seconds |

---

## 🎯 Deployment Checklist

### Before Deploying
- [x] Fixed `requirements.txt` (updated pydantic)
- [ ] Committed changes to Git
- [ ] Pushed to GitHub
- [ ] Created Render account
- [ ] Created Vercel account

### Backend Deployment
- [ ] Created Web Service on Render
- [ ] Set Root Directory to `nlp-react/backend`
- [ ] Verified Build Command
- [ ] Verified Start Command
- [ ] Deployed (waited 10-15 min)
- [ ] Copied backend URL
- [ ] Tested `/health` endpoint

### Frontend Deployment
- [ ] Created project on Vercel
- [ ] Set Root Directory to `nlp-react/frontend`
- [ ] Added `VITE_API_URL` environment variable
- [ ] Deployed (waited 2-3 min)
- [ ] Tested full app
- [ ] Verified sentiment analysis works

---

## 🐛 Common Issues & Quick Fixes

| Problem | Solution |
|---------|----------|
| Build fails with Rust error | Use updated `requirements.txt` with `pydantic==2.6.0` |
| "Model not loaded" | Wait 30-60 seconds, refresh page |
| "API Disconnected" | Check `VITE_API_URL` has no trailing slash |
| 404 on backend | Verify Root Directory is `nlp-react/backend` |
| Frontend build fails | Verify Root Directory is `nlp-react/frontend` |
| CORS errors | Already fixed in `main.py` |

---

## 📞 URLs After Deployment

Fill these in after deploying:

**Backend:**
```
Health: https://________________.onrender.com/health
API Docs: https://________________.onrender.com/docs
```

**Frontend:**
```
Live App: https://________________.vercel.app
```

---

## 🔄 Redeployment Process

### If you make code changes:

1. **Edit files locally**
2. **Test locally** (optional but recommended)
3. **Commit changes:**
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin main
   ```
4. **Render/Vercel auto-deploys** (or click "Manual Deploy")

---

## 💡 Keep Backend Awake

### Option 1: UptimeRobot (Free)
1. Go to [uptimerobot.com](https://uptimerobot.com)
2. Sign up (free)
3. Add HTTP Monitor
4. URL: `https://your-backend.onrender.com/health`
5. Interval: 5 minutes
6. Done! Backend stays awake

### Option 2: Upgrade
- Render: $7/month = no sleep
- Railway: Pay per use (similar cost)

---

## ✅ Success Criteria

Your deployment is successful when:

1. ✅ Backend `/health` returns `"model_loaded": true`
2. ✅ Frontend loads without errors
3. ✅ Green "AI Model Ready" indicator shows
4. ✅ Can analyze text and get results
5. ✅ Probability chart displays
6. ✅ Example buttons work
7. ✅ Works on mobile device

---

**Last Updated:** October 19, 2025
**Your Project:** CST-405 NLP Sentiment Analyzer
