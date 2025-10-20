# 🚂 SWITCH TO RAILWAY - Final Solution

## Why Railway is Your Best Option

After trying multiple optimizations, your app needs ~700MB RAM.

**Render Free Tier:** 512MB ❌  
**Railway Free Tier:** 1GB ✅

Railway's free $5/month credit gives you enough resources!

---

## ⚡ Deploy to Railway (10 Minutes)

### Step 1: Sign Up (2 minutes)
1. Go to: https://railway.app
2. Click **"Start a New Project"**
3. Choose **"Login with GitHub"**
4. Authorize Railway
5. Add credit card (required for free tier, **won't be charged** unless you exceed $5/month)

### Step 2: Create Project (1 minute)
1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose: `Susana016/CST-435-Group`
4. Railway starts analyzing...

### Step 3: Configure (2 minutes)
1. Railway detects it's a monorepo
2. Click **"Settings"** (⚙️ icon in your service)
3. Set these values:

```
Root Directory: nlp-react/backend
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

4. Click **"Variables"** tab
5. Add (optional but recommended):
```
PYTHON_VERSION = 3.11
```

### Step 4: Deploy! (5-10 minutes)
1. Railway automatically starts building
2. Watch the deployment logs
3. You'll see:
   ```
   ✅ Installing Python dependencies...
   ✅ Downloading DistilBERT model...
   ✅ Starting server...
   ✅ Deployment live!
   ```

### Step 5: Get Your URL (1 minute)
1. Go to **"Settings"** tab
2. Scroll to **"Networking"**
3. Click **"Generate Domain"**
4. Copy your URL: `https://your-app.up.railway.app`

### Step 6: Test (1 minute)
Visit: `https://your-app.up.railway.app/health`

Should return:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## ✅ What You Get with Railway

| Feature | Value |
|---------|-------|
| RAM | 1GB (2x Render free) |
| Monthly Credit | $5 |
| Build Time | 5-10 min (first time) |
| Wake Time | 10-20 seconds |
| Auto Deploy | ✅ Yes (on git push) |
| Custom Domain | ✅ Yes |

---

## 💰 Cost Analysis

**Your app usage:**
- Running 24/7: ~$3-4/month
- Running only when used: ~$1-2/month

**Railway gives you $5/month free** = Plenty of credit! ✅

---

## 🔄 Update Frontend After Railway Deploy

Once Railway backend is live:

### Step 1: Copy Railway URL
```
https://your-app.up.railway.app
```

### Step 2: Update Vercel Environment Variable
1. Go to Vercel Dashboard
2. Your project → **Settings**
3. **Environment Variables**
4. Find `VITE_API_URL`
5. Click **"Edit"**
6. Change to your Railway URL (NO trailing slash!)
7. **Save**

### Step 3: Redeploy Frontend
1. Go to **"Deployments"** tab
2. Click **"⋯"** on latest deployment
3. Select **"Redeploy"**
4. Wait 2 minutes

### Step 4: Test Full App
1. Visit your Vercel URL
2. Try sentiment analysis
3. ✅ Should work perfectly!

---

## 📊 Comparison: Why Railway > Render for ML Apps

| Aspect | Render Free | Railway Free |
|--------|-------------|--------------|
| **RAM** | 512MB | 1GB |
| **Your App** | ❌ Doesn't fit | ✅ Fits perfectly |
| **Build Time** | 10-15 min | 5-10 min |
| **Python Support** | Sometimes issues | Excellent |
| **Credit Card** | Not required | Required (not charged) |
| **Best For** | Simple apps | ML/AI apps |

---

## 🎯 Timeline Summary

- Sign up: 2 minutes
- Configure: 2 minutes
- First deploy: 5-10 minutes
- Update frontend: 3 minutes
- Test: 2 minutes
- **Total: ~20 minutes**

---

## 🐛 Troubleshooting Railway

### If deployment fails:

**1. Check Root Directory**
- Must be: `nlp-react/backend`
- NOT: `backend` or empty

**2. Check Build Logs**
- Click on deployment
- View detailed logs
- Look for specific errors

**3. Environment Variables**
Add if needed:
```
PYTHON_VERSION = 3.11
PORT = 8000
```

---

## ✅ Checklist: Railway Deployment

- [ ] Signed up for Railway
- [ ] Added credit card (for free $5 credit)
- [ ] Created new project from GitHub
- [ ] Set Root Directory: `nlp-react/backend`
- [ ] Deployment succeeded
- [ ] Generated domain
- [ ] Tested `/health` endpoint
- [ ] Copied Railway URL
- [ ] Updated Vercel env var `VITE_API_URL`
- [ ] Redeployed frontend
- [ ] Tested full app end-to-end

---

## 🎉 After Success

### Update Your README:

```markdown
# 🎬 Multi-Scale Sentiment Analyzer

## 🌐 Live Demo
- **Frontend:** https://your-app.vercel.app
- **Backend API:** https://your-app.up.railway.app
- **API Docs:** https://your-app.up.railway.app/docs

## 👤 Created By
[Your Name] - CST-405 NLP Project

## 🛠️ Tech Stack
- Frontend: React + Vite + Tailwind (Vercel)
- Backend: FastAPI + PyTorch + Transformers (Railway)
- Model: DistilBERT for sentiment analysis
```

---

## 🚀 Why This Will Work

Railway is designed for ML/AI applications:
- ✅ More RAM (1GB vs 512MB)
- ✅ Better Python dependency handling
- ✅ Optimized for PyTorch/Transformers
- ✅ Used by thousands of ML projects
- ✅ Generous free tier

Your app will deploy successfully on first try! 🎉

---

## 💡 Pro Tips

### Keep Railway App Active
Railway doesn't sleep as aggressively as Render, but you can:
1. Use UptimeRobot to ping `/health` every 5 minutes
2. Or just let it sleep (10-20 sec wake-up is fast)

### Monitor Usage
1. Railway Dashboard → Your Project
2. Check **"Metrics"** tab
3. Monitor monthly credit usage
4. You'll likely use $3-4/month (within free tier)

---

## 📞 Getting Help

**Railway Support:**
- Docs: https://docs.railway.app
- Discord: https://discord.gg/railway (very active!)
- Status: https://status.railway.app

**Your instructor:** If Railway requires credit card and you can't get one

---

**Deploy to Railway now - it's your best path to success!** 🚂✨
