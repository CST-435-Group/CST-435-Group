# 🚨 RAILWAY ROOT DIRECTORY FIX

## The Problem

Railway is STILL using Python 3.13.9 even after you:
- ✅ Created `.python-version` file
- ✅ Updated `requirements.txt`
- ✅ Pushed to GitHub

**Why?** Railway can't find your `.python-version` file because the **Root Directory** setting is wrong!

---

## ✅ SOLUTION: Fix Railway Root Directory

### Step 1: Go to Railway Settings

1. Open Railway Dashboard
2. Click on your service
3. Click **"Settings"** (⚙️ icon or left sidebar)

### Step 2: Check Root Directory

Scroll down to find **"Root Directory"** or **"Source"** section.

**Current setting is probably:**
- Empty (blank)
- OR `backend`
- OR `CST-435-Group/NLP/nlp-react/backend`

**Must be EXACTLY:**
```
nlp-react/backend
```

### Step 3: Update Root Directory

1. Click **"Edit"** next to Root Directory
2. Type: `nlp-react/backend`
3. Click **"Save"** or **"Update"**

### Step 4: Redeploy

1. After saving, Railway should auto-redeploy
2. OR click **"Deploy"** button manually
3. Watch the build logs

---

## 🔍 How to Verify It Worked

In the Railway logs, you should NOW see:

### ✅ CORRECT:
```
python  │  3.11.9  │  from .python-version file
```

NOT:
```
python  │  3.13.9  │  railpack default (3.13)
```

---

## 🎯 Complete Railway Configuration

Here's what ALL your settings should be:

### Service Settings:
```
Name: sentiment-analyzer-backend (or whatever you chose)
```

### Source:
```
Repository: Susana016/CST-435-Group
Branch: main
Root Directory: nlp-react/backend  ← CRITICAL!
```

### Start Command:
```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

OR leave it blank (Railway auto-detects from main.py)

### Environment Variables:
```
(Optional, but recommended:)
PYTHON_VERSION = 3.11.9
```

---

## 📸 Visual Guide

When you look at Railway Settings, you should see:

```
┌─────────────────────────────────────┐
│ Source                              │
├─────────────────────────────────────┤
│ Repository: Susana016/CST-435-Group│
│ Branch: main                        │
│ Root Directory: nlp-react/backend  │← HERE!
└─────────────────────────────────────┘
```

---

## 🚨 If Root Directory Field Doesn't Exist

Some Railway interfaces don't show "Root Directory" directly. Try:

### Method 1: railway.toml File

Create a file in your repo root:

**File:** `railway.toml` (in the root of your repository)

```toml
[build]
buildCommand = "pip install -r requirements.txt"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT"
workingDirectory = "nlp-react/backend"
```

Then:
```bash
git add railway.toml
git commit -m "Add Railway config with correct working directory"
git push origin main
```

### Method 2: Use Railway CLI

If you have Railway CLI installed:

```bash
railway link
railway up --service backend --rootDir nlp-react/backend
```

### Method 3: Recreate Service

1. Delete current Railway service
2. Create new service
3. **During setup**, specify Root Directory: `nlp-react/backend`

---

## ⏰ Timeline After Fix

- Update Root Directory: 30 seconds
- Railway redeploy: 5-10 minutes
- **Total: ~10 minutes**

---

## 🎉 What Will Happen After Fix

1. ✅ Railway finds `.python-version` in `nlp-react/backend/`
2. ✅ Railway installs Python 3.11.9 (not 3.13.9)
3. ✅ Railway finds `requirements.txt` with `pydantic>=2.6.0`
4. ✅ pydantic 2.6.0 installs successfully
5. ✅ No Rust compilation errors
6. ✅ Build succeeds!
7. ✅ Deployment works!
8. ✅ App is live! 🎊

---

## 📋 Troubleshooting

### If you can't find Root Directory setting:

**Look for these similar names:**
- "Working Directory"
- "Project Directory"
- "Source Path"
- "Build Path"

**Or check in these sections:**
- Settings → Source
- Settings → Build
- Settings → Deploy
- Project Settings

### If Railway keeps using wrong directory:

Try the `railway.toml` method above - it overrides dashboard settings.

---

## 💡 Why This Matters

```
Without Root Directory:
Railway looks at: CST-435-Group/NLP/
Can't find: .python-version ❌
Uses: Python 3.13.9 (default) ❌
Fails: pydantic incompatible ❌

With Root Directory = nlp-react/backend:
Railway looks at: CST-435-Group/NLP/nlp-react/backend/
Finds: .python-version ✅
Uses: Python 3.11.9 ✅
Works: pydantic compatible ✅
```

---

## 🎯 Quick Action Checklist

- [ ] Go to Railway Dashboard
- [ ] Click your service
- [ ] Click Settings
- [ ] Find Root Directory field
- [ ] Set to: `nlp-react/backend`
- [ ] Save
- [ ] Wait for redeploy
- [ ] Check logs show Python 3.11.9
- [ ] Build succeeds!

---

**Fix the Root Directory setting RIGHT NOW!** This is the missing piece! 🚀
