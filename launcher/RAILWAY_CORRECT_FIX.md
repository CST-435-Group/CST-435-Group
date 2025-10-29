# Railway Correct Fix - THE REAL SOLUTION

## The Problem

You set Root Directory to `launcher/backend`, which means Railway can only see files in that directory. But the backend needs to import from:
- `ANN_Project/` (parent directory)
- `CNN_Project/` (parent directory)
- `nlp-react/` (parent directory)

These are **outside** `launcher/backend`, so they're not accessible.

## The Solution ✅

**Change Railway settings:**

### Setting 1: Root Directory
**Change from:** `launcher/backend`
**Change to:** `launcher`

This lets Railway see the parent project directories.

### Setting 2: Start Command
**Set to:** `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`

This changes into the backend directory before starting the server.

---

## Exact Steps

1. **Go to Railway Dashboard** → Your Project → **Settings**

2. **Find "Root Directory"**
   - Click Edit
   - Change to: `launcher`
   - Save

3. **Find "Start Command"** (or "Custom Start Command")
   - Click Edit
   - Set to: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Save

4. **Redeploy**
   - Go to Deployments tab
   - Click "Redeploy"

---

## Why This Works

**Directory Structure Railway Will See:**

```
/app/                        ← Railway root
├── backend/                 ← Where we start the server
│   ├── main.py
│   ├── routers/
│   └── requirements.txt
├── frontend/
└── ...other files
```

But the repository ROOT also has:
```
/                            ← Repository root (one level up from /app)
├── launcher/                ← /app points here
├── ANN_Project/             ← Accessible!
├── CNN_Project/             ← Accessible!
└── nlp-react/               ← Accessible!
```

The backend routers look for parent projects like this:
```python
Path(__file__).parent.parent.parent.parent / "ANN_Project"
```

From `/app/backend/routers/ann.py`:
- `.parent` → `/app/backend/routers`
- `.parent` → `/app/backend`
- `.parent` → `/app` (launcher)
- `.parent` → `/` (repository root)
- `/ANN_Project` → Found! ✅

---

## After Changing Settings

Railway will:
1. ✅ Clone entire repository
2. ✅ Set working directory to `launcher`
3. ✅ See `backend/requirements.txt`
4. ✅ Install dependencies
5. ✅ Run: `cd backend && uvicorn main:app ...`
6. ✅ Backend can import from parent projects

---

## Verification

After redeploying, check the logs. You should see:
```
✅ Found ANN_Project at: /app/../ANN_Project
✅ Found CNN_Project at: /app/../CNN_Project
✅ Found NLP project at: /app/../nlp-react
```

Instead of:
```
❌ ANN_Project not found in any of these locations
```

Then test:
```bash
curl https://your-app.railway.app/health
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

## Summary

**Change these 2 settings in Railway:**

| Setting | Value |
|---------|-------|
| Root Directory | `launcher` (not `launcher/backend`!) |
| Start Command | `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT` |

That's it! Redeploy and it should work.
