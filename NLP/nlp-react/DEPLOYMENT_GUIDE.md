# 🚀 Deployment Guide - Complete Walkthrough

Deploy your Sentiment Analyzer to the cloud using **free tiers**!

---

## 📋 Deployment Strategy

**Important:** Vercel does NOT support Python backends. We'll deploy in two parts:

1. **Backend (Python/FastAPI)** → Choose ONE:
   - **Option A: Render.com** (Recommended - easier)
   - **Option B: Railway.app** (Faster, but requires credit card)

2. **Frontend (React)** → **Vercel** (Best for React apps)

**Total time:** 30-45 minutes

---

## 🤔 Which Backend Platform Should I Choose?

### Platform Comparison

| Feature | Render.com | Railway.app |
|---------|-----------|-------------|
| **Free Tier** | ✅ 750 hrs/month | ✅ $5 credit/month |
| **Credit Card Required** | ❌ No | ⚠️ Yes |
| **Sleep After Inactivity** | ✅ Yes (15 min) | ✅ Yes (varies) |
| **Wake-up Time** | 30-60 seconds | 10-20 seconds |
| **Setup Difficulty** | ⭐⭐ Easy | ⭐⭐⭐ Medium |
| **Build Time (First)** | 10-15 min | 5-10 min |
| **Best For** | Beginners, no credit card | Faster performance |

### 💡 Recommendation

**Choose Render** if:
- ✅ You don't have a credit card
- ✅ This is your first deployment
- ✅ You prefer easier setup

**Choose Railway** if:
- ✅ You have a credit card
- ✅ You want faster wake-up times
- ✅ You're comfortable with slightly more complex setup

---

## 📝 Before You Start

### Checklist

- [ ] Code working locally (both frontend and backend)
- [ ] All changes committed to Git
- [ ] GitHub repository created
- [ ] All changes pushed to GitHub
- [ ] GitHub account created
- [ ] Vercel account created (use GitHub login)
- [ ] Render OR Railway account created

### Commit Your Changes

```bash
# Check what files changed
git status

# Add all changes
git add .

# Commit
git commit -m "Add modifications and prepare for deployment"

# Push to GitHub
git push origin main
```

---

## Part 1: Deploy Backend

Choose **ONE** of these options:

---

## OPTION A: Deploy Backend to Render.com

### Why Render?
- ✅ **No credit card required**
- ✅ Simple setup process
- ✅ Good free tier (750 hours/month)
- ✅ Great for learning

### Step 1: Create Render Account

1. Go to **[render.com](https://render.com)**
2. Click **"Get Started"**
3. Choose **"Sign in with GitHub"**
4. Authorize Render to access your repositories

### Step 2: Create Web Service

1. From Render dashboard, click **"New +"**
2. Select **"Web Service"**

### Step 3: Connect Repository

1. Click **"Connect a repository"**
2. If not connected yet, click **"Configure GitHub"**
3. Choose **"All repositories"** OR select specific repo
4. Click **"Install"**
5. Back in Render, click **"Connect"** next to your `nlp-react` repository

### Step 4: Configure Service Settings

**Fill in these EXACT values:**

```
Name: sentiment-analyzer-api
(or any name you prefer - no spaces)

Language: Python

Region: Oregon (US West) or Ohio (US East)
(Choose closest to your location)

Branch: main

Root Directory: backend
⚠️ CRITICAL: Click "Edit" and type: backend

Build Command: pip install -r requirements.txt

Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Step 5: Environment Variables (Optional)

You can add these if needed:
```
MODEL_NAME: distilbert-base-uncased
```

Click **"Add Environment Variable"** if you want to add any.

### Step 6: Select Plan

1. Scroll down to **"Instance Type"**
2. Select **"Free"** ($0/month)
3. Read the note about sleep after 15 minutes of inactivity

### Step 7: Deploy!

1. Click **"Create Web Service"** at the bottom
2. Wait for deployment (THIS WILL TAKE 10-15 MINUTES!)
3. Watch the logs - you'll see:
   ```
   Installing dependencies...
   Collecting torch...
   Downloading DistilBERT model (250MB)...
   Build successful!
   Starting server...
   Your service is live!
   ```

### Step 8: Get Your Backend URL

Once deployed (green checkmark), you'll see:
```
Your service is live at:
https://sentiment-analyzer-api.onrender.com
```

**Copy this URL!** You'll need it for the frontend.

### Step 9: Test Your Backend

Open in browser:
```
https://your-backend-url.onrender.com/health
```

You should see:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

If it shows "service unavailable" wait 30-60 seconds (it's waking up).

### Step 10: Check API Documentation

Visit:
```
https://your-backend-url.onrender.com/docs
```

You'll see the automatic FastAPI documentation!

---

## OPTION B: Deploy Backend to Railway.app

### Why Railway?
- ✅ Faster deployment and wake-up
- ✅ Nicer interface
- ✅ Better developer experience
- ⚠️ Requires credit card (for free $5/month credit)

### Step 1: Create Railway Account

1. Go to **[railway.app](https://railway.app)**
2. Click **"Start a New Project"** or **"Login"**
3. Choose **"Login with GitHub"**
4. Authorize Railway

### Step 2: Create New Project

1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Click **"Configure GitHub App"**
4. Install Railway on your repositories
5. Select your `nlp-react` repository

### Step 3: Configure Service

1. Railway will detect it's a monorepo
2. Click **"Add variables"** (top right)
3. Add these:

```
Root Directory:
backend

Install Command:
pip install -r requirements.txt

Build Command:
(leave empty)

Start Command:
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Step 4: Add Environment Variables

In the **Variables** tab, add:

```
PORT: 8000
PYTHON_VERSION: 3.11
```

### Step 5: Deploy!

1. Railway will automatically start deploying
2. Watch the build logs (click on deployment)
3. First deployment: 5-10 minutes
4. You'll see:
   ```
   Installing dependencies...
   Downloading model...
   Starting server...
   Deployment successful!
   ```

### Step 6: Generate Domain

1. Go to **Settings** tab
2. Scroll to **"Networking"**
3. Click **"Generate Domain"**
4. You'll get a URL like: `https://sentiment-api.up.railway.app`

**Copy this URL!**

### Step 7: Test Your Backend

Open in browser:
```
https://your-railway-url.up.railway.app/health
```

Should see:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## Part 2: Deploy Frontend to Vercel

**This is the same regardless of which backend option you chose!**

### Why Vercel for Frontend?
- ✅ **Perfect for React/Vite apps**
- ✅ Super fast (global CDN)
- ✅ Automatic HTTPS
- ✅ Free tier is generous
- ✅ No credit card required
- ✅ Auto-deploys from GitHub

### Step 1: Create Vercel Account

1. Go to **[vercel.com](https://vercel.com)**
2. Click **"Sign Up"**
3. Choose **"Continue with GitHub"**
4. Authorize Vercel

### Step 2: Import Project

1. From Vercel dashboard, click **"Add New..."**
2. Select **"Project"**
3. Find your `nlp-react` repository
4. Click **"Import"**

### Step 3: Configure Build Settings

**CRITICAL SETTINGS:**

```
Framework Preset: Vite
(Should auto-detect)

Root Directory: frontend
⚠️ MUST EDIT: Click "Edit" and select "frontend"

Build Command: npm run build
(Auto-detected)

Output Directory: dist
(Auto-detected)

Install Command: npm install
(Auto-detected)
```

### Step 4: Add Environment Variables

**THIS IS THE MOST IMPORTANT STEP!**

Before deploying, add your backend URL:

1. Expand **"Environment Variables"** section
2. Add this variable:

```
Key:   VITE_API_URL
Value: https://your-backend-url.onrender.com
       (or your Railway URL)
⚠️ NO TRAILING SLASH!
```

**Examples:**
```
✅ Correct: https://sentiment-api.onrender.com
❌ Wrong:   https://sentiment-api.onrender.com/
❌ Wrong:   http://localhost:8000
```

3. Click **"Add"**

### Step 5: Deploy!

1. Click **"Deploy"**
2. Wait 2-3 minutes
3. Watch the build logs
4. Success = confetti animation! 🎉

### Step 6: Get Your Frontend URL

You'll see:
```
🎉 Congratulations!
Visit: https://your-app.vercel.app
```

**Copy this URL!**

---

## Part 3: Testing Your Deployed App

### Step 1: Visit Your App

1. Open your Vercel URL in browser
2. **Wait 30-60 seconds on first visit** (backend waking up)
3. You'll see the app load

### Step 2: Check Connection Status

Look for the status indicator at the top:
- 🟢 **Green = "AI Model Ready"** - Good!
- 🔴 **Red = "API Disconnected"** - Wait or troubleshoot

### Step 3: Test Functionality

1. **Enter a review:**
   ```
   This movie was absolutely phenomenal! Best film I've seen this year!
   ```

2. **Click "Analyze Sentiment"**

3. **Wait for results** (first request may take 30-60 seconds)

4. **Verify results appear:**
   - Sentiment score displayed
   - Confidence percentage shown
   - Probability chart rendered
   - Emoji and label correct

### Step 4: Test Multiple Times

Try these examples:
- Very positive review
- Very negative review
- Neutral review
- Click example buttons

### Step 5: Test from Different Devices

- Different browser
- Mobile phone
- Incognito/private mode

---

## Part 4: Update Your Repository

### Add Deployment URLs to README

1. **Edit your README.md:**

Add this at the very top:

```markdown
# 🎬 Multi-Scale Sentiment Analyzer

## 🌐 Live Demo

**Try it now:** [https://your-app.vercel.app](https://your-app.vercel.app)

### API Endpoints
- **Frontend:** https://your-app.vercel.app
- **Backend API:** https://your-backend-url.onrender.com/docs

## 👤 Student Project
Created by: **[Your Name]**
Course: AIT-204 - Natural Language Processing
Institution: Grand Canyon University

---

[Rest of README continues...]
```

2. **Commit and push:**

```bash
git add README.md
git commit -m "Add deployment URLs to README"
git push origin main
```

Vercel will **automatically redeploy** when you push!

---

## 🔄 Updating Your Deployment

### When You Make Code Changes

**Frontend changes:**
```bash
# Make changes to files in frontend/
git add .
git commit -m "Update: describe your changes"
git push origin main
```
→ Vercel **auto-deploys** (takes 2-3 minutes)

**Backend changes:**
```bash
# Make changes to files in backend/
git add .
git commit -m "Update: describe your changes"
git push origin main
```
→ Render/Railway **auto-deploys** (takes 5-10 minutes)

### Manual Redeploy

**Vercel:**
1. Go to project dashboard
2. **Deployments** tab
3. Click **"⋯"** → **"Redeploy"**

**Render:**
1. Go to service dashboard
2. Click **"Manual Deploy"** button
3. Select **"Deploy latest commit"**

**Railway:**
1. Go to service dashboard
2. Click **"Deploy"** button
3. Or push to GitHub (auto-deploys)

---

## 💰 Free Tier Limitations

### Render.com Free Tier
- ✅ **750 hours/month** (enough for 24/7 for one service)
- ⚠️ **Sleeps after 15 min** of inactivity
- ⚠️ **30-60 second wake-up** time
- ⚠️ **512MB RAM**
- ✅ **Automatic HTTPS**
- ✅ **No credit card required**

### Railway.app Free Tier
- ✅ **$5 credit/month** (≈500 hours)
- ⚠️ **Credit card required** (not charged until you exceed free tier)
- ⚠️ **Sleeps after inactivity**
- ⚠️ **10-20 second wake-up** time
- ✅ **Better performance**
- ✅ **Automatic HTTPS**

### Vercel Free Tier
- ✅ **100GB bandwidth/month**
- ✅ **Unlimited projects**
- ✅ **No sleep time** (always instant)
- ✅ **Global CDN**
- ✅ **Automatic HTTPS**
- ✅ **No credit card required**

### Keeping Backend Awake (Optional)

If you hate the 30-60 second wait:

**Option 1: UptimeRobot (Free)**
1. Sign up at [uptimerobot.com](https://uptimerobot.com)
2. Add new monitor:
   - Type: HTTP(s)
   - URL: Your backend health endpoint
   - Interval: 5 minutes
3. This pings your backend every 5 minutes, keeping it awake

**Option 2: Cron-job.org (Free)**
1. Sign up at [cron-job.org](https://cron-job.org)
2. Create new cron job
3. Set to ping your backend every 10 minutes

**Option 3: Upgrade to Paid**
- Render: $7/month (no sleep)
- Railway: Pay for what you use beyond $5

---

## 🐛 Troubleshooting

### Backend Issues

**❌ "Build failed" on Render/Railway**

**Problem:** Dependencies won't install

**Solutions:**
```bash
# Check requirements.txt has all dependencies
# Make sure Root Directory is set to "backend"
# Verify Python version compatibility (3.8+)
```

**Check the build logs** for specific error messages.

---

**❌ "Application failed to respond"**

**Problem:** Server won't start

**Solutions:**
```bash
# Verify Start Command is correct:
uvicorn main:app --host 0.0.0.0 --port $PORT

# Check main.py is in backend/ directory
# Look for error in deployment logs
```

---

**❌ "Model download timeout"**

**Problem:** First deployment times out

**Solution:**
- This is normal! The model is 250MB
- Render: Takes 10-15 minutes first time
- Railway: Takes 5-10 minutes first time
- Be patient, watch the logs
- Subsequent deploys are MUCH faster (2-3 min)

---

**❌ Backend URL shows 404**

**Problem:** Wrong URL or service not running

**Solutions:**
```bash
# Check the URL is correct (no typos)
# Make sure service shows "Active" in dashboard
# Try the /health endpoint first
# Wait 30-60 seconds (might be waking up)
```

---

### Frontend Issues

**❌ "Vercel build failed"**

**Problem:** Build process errors

**Solutions:**
```bash
# Check Root Directory is set to "frontend"
# Verify Build Command: npm run build
# Check Output Directory: dist
# Look at build logs for specific error
```

---

**❌ "API Disconnected" error**

**Problem:** Frontend can't reach backend

**Solutions:**
```bash
# Check VITE_API_URL is set correctly
# Verify URL has NO trailing slash
# Make sure backend is deployed and running
# Test backend /health endpoint directly
# Redeploy frontend after adding env var
```

---

**❌ Environment variable not working**

**Problem:** VITE_API_URL not being used

**Solutions:**
```bash
# Variables must start with VITE_
# Must add BEFORE first deployment, or redeploy after
# Check spelling: VITE_API_URL (all caps)
# Verify value has no quotes
# No trailing slash in URL
```

---

### Connection Issues

**❌ CORS errors in browser console**

**Problem:** Backend blocking frontend requests

**Solution:**
```python
# Check backend/main.py CORS config:
allow_origins=[
    "http://localhost:3000",
    "https://*.vercel.app",  # This should allow all Vercel
    "*"  # Or allow all (less secure)
]
```

Already configured correctly in the starter code!

---

**❌ "Backend sleeping" / slow first request**

**Problem:** Free tier sleeps after inactivity

**Solution:**
- This is NORMAL on free tier
- First request takes 30-60 seconds
- Subsequent requests are fast
- Use UptimeRobot to keep awake (see above)
- Or upgrade to paid plan

---

**❌ App works locally but not deployed**

**Problem:** Environment differences

**Solutions:**
```bash
# Check environment variables are set
# Verify backend URL is production URL (not localhost)
# Test backend independently: /health endpoint
# Check browser console for errors (F12)
# Try in incognito/private mode
```

---

### Testing Individual Parts

**Test Backend Alone:**
```
# Health check
https://your-backend-url.com/health

# API docs
https://your-backend-url.com/docs

# Try POST /analyze from docs page
```

**Test Frontend Alone:**
```
# Run locally but use production backend
# Edit frontend/.env:
VITE_API_URL=https://your-production-backend.com

# Then: npm run dev
```

---

## 📋 Deployment Checklist

Before submitting, verify:

### Backend ✓
- [ ] Deployed to Render or Railway
- [ ] Service shows "Active"
- [ ] /health endpoint returns 200
- [ ] /docs page loads
- [ ] Can test /analyze from docs

### Frontend ✓
- [ ] Deployed to Vercel
- [ ] VITE_API_URL environment variable set
- [ ] App loads without errors
- [ ] Connection indicator is green
- [ ] Can analyze text successfully

### Integration ✓
- [ ] Frontend can reach backend
- [ ] Results display correctly
- [ ] Example buttons work
- [ ] Charts render properly
- [ ] No CORS errors

### Documentation ✓
- [ ] README has deployment URLs
- [ ] URLs are clickable links
- [ ] Modification summary added
- [ ] Code is committed and pushed

---

## 🎯 Final Test Script

Use this to verify everything works:

1. ✅ Visit your Vercel URL
2. ✅ Wait for green "AI Model Ready" indicator
3. ✅ Enter: "This movie was amazing!"
4. ✅ Click "Analyze Sentiment"
5. ✅ Verify: Score shows +2 or +3
6. ✅ Check: Confidence percentage displayed
7. ✅ Check: Chart renders correctly
8. ✅ Click an example button
9. ✅ Verify: New analysis appears
10. ✅ Test on mobile device
11. ✅ Share URL with a friend - does it work for them?

**If all 11 pass, you're done!** 🎉

---

## 🚀 Performance Tips

### Speed Up Backend Wake-Up

**Use UptimeRobot:**
```
1. Sign up (free)
2. Add HTTP monitor
3. URL: https://your-backend/health
4. Interval: 5 minutes
5. Backend stays awake!
```

### Optimize Frontend Loading

Already optimized:
- ✅ Vite code splitting
- ✅ Lazy loading ready
- ✅ Production build minified
- ✅ Vercel CDN (global)

---

## 💡 Common Questions

**Q: Why two different platforms?**
A: Vercel is great for React but doesn't support Python. Render/Railway are great for Python but not as good for React.

**Q: Can I use just one platform?**
A: Technically yes (Render can do both), but it's not ideal. Vercel is much better for React frontends.

**Q: Which is better, Render or Railway?**
A: Render = easier, no credit card. Railway = faster, requires credit card. Both work great!

**Q: Why does my backend sleep?**
A: Free tiers save resources by sleeping after inactivity. This is normal! Paid plans don't sleep.

**Q: Can I use this in my portfolio?**
A: YES! That's the point. Share your live URL proudly.

**Q: What if I hit free tier limits?**
A: Very unlikely for a class project. If you do, upgrade is ~$7-12/month total.

---

## 📞 Getting Help

### Platform-Specific Help

**Render:**
- Docs: [render.com/docs](https://render.com/docs)
- Community: [community.render.com](https://community.render.com)
- Status: [status.render.com](https://status.render.com)

**Railway:**
- Docs: [docs.railway.app](https://docs.railway.app)
- Discord: [discord.gg/railway](https://discord.gg/railway)
- Status: [status.railway.app](https://status.railway.app)

**Vercel:**
- Docs: [vercel.com/docs](https://vercel.com/docs)
- Discord: [vercel.com/discord](https://vercel.com/discord)
- Status: [vercel-status.com](https://vercel-status.com)

### Course Support

- Check this guide thoroughly
- Review troubleshooting section
- Ask your instructor
- Help classmates (you'll learn more!)
- Search the error message on Google

---

## 🎉 Success!

Once everything is working:

1. **Test thoroughly** - Make sure all features work
2. **Document your URLs** - Add to README
3. **Take screenshots** - For your portfolio
4. **Share your work** - Post on LinkedIn, Twitter
5. **Help others** - Share what you learned
6. **Celebrate!** - You deployed a full-stack AI app! 🎊

**Your live URLs:**
- 🌐 Frontend: `https://your-app.vercel.app`
- 🔧 Backend: `https://your-api.render.com` or `.railway.app`
- 📚 API Docs: `https://your-api.url/docs`

---

**Congratulations on deploying your first full-stack application!** 🚀

Now you have a portfolio-worthy project to show employers!
