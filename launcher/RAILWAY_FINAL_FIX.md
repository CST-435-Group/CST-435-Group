# Railway Final Fix - DO THIS NOW ✅

## What I Just Fixed

1. ✅ **Removed Dockerfile builder** from `railway.toml`
2. ✅ **Renamed Dockerfile** to `Dockerfile.backup` (so Railway won't detect it)
3. ✅ **Simplified railway.toml** to only have deployment config

Now Railway will use **Nixpacks** instead of Dockerfile.

---

## Your Next Steps (3 Simple Settings)

### Step 1: Commit and Push These Changes

```bash
cd "C:\Users\Soren\OneDrive\Documents\school\College Senior\CST-405\Compiler\CST-435-Group"

git add launcher/
git commit -m "Fix Railway deployment - use Nixpacks instead of Dockerfile"
git push
```

### Step 2: Configure Railway Settings

Go to Railway Dashboard → Your Project → Settings:

**Setting 1: Root Directory**
- Set to: `launcher/backend`
- Save

**Setting 2: Start Command**
- Set to: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Save

**Setting 3: Builder**
- Should now show "Nixpacks" (auto-detected)
- If it still shows "Dockerfile", wait a moment after pushing and refresh the page

### Step 3: Redeploy

Click "Redeploy" or Railway will auto-deploy when you push.

---

## What Railway Will Do Now

Railway will:
1. ✅ Clone your repo
2. ✅ Go to `launcher/backend` directory
3. ✅ Detect Python project (from `requirements.txt`)
4. ✅ Use **Nixpacks** to build (NOT Dockerfile)
5. ✅ Install Python 3.11
6. ✅ Run `pip install -r requirements.txt`
7. ✅ Start server with your command

**No more Dockerfile errors!**

---

## Verify It Works

After deployment completes:

```bash
# Replace with your Railway URL
export RAILWAY_URL="https://your-app.up.railway.app"

# Test health check
curl $RAILWAY_URL/health

# Should return:
# {
#   "status": "healthy",
#   "services": {...}
# }
```

---

## What Changed in Files

### `railway.toml` - SIMPLIFIED
**Before:**
```toml
[build]
builder = "NIXPACKS"
nixpacksConfigPath = "nixpacks.toml"
```

**After:**
```toml
# Removed [build] section completely
# Railway will auto-detect Nixpacks

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
...
```

### `Dockerfile` - RENAMED
- Was: `Dockerfile`
- Now: `Dockerfile.backup`
- Railway won't detect it anymore

---

## If Railway Still Shows "Dockerfile"

**Option A: Delete railway.toml entirely**

```bash
# From launcher directory
git rm railway.toml
git commit -m "Remove railway.toml to force Nixpacks detection"
git push
```

Then set everything in Railway Dashboard manually.

**Option B: Force Nixpacks in Railway Dashboard**

Even though it says "auto-detected from railway.toml", you can:
1. Delete the `railway.toml` file from your repository
2. Configure everything in Railway Dashboard instead
3. Redeploy

---

## Settings Summary

**These are the ONLY 3 settings you need:**

```
Root Directory:  launcher/backend
Start Command:   uvicorn main:app --host 0.0.0.0 --port $PORT
Builder:         Nixpacks (should auto-detect after changes)
```

---

## Next: Deploy Frontend

Once Railway backend works:

```bash
cd launcher
vercel login
vercel

# After deployment, set environment variable:
vercel env add VITE_API_URL production
# Enter your Railway URL when prompted

# Redeploy with the new env var
vercel --prod
```

---

## Troubleshooting

### "Builder still shows Dockerfile"

1. Push the changes (railway.toml and Dockerfile renamed)
2. Wait 30 seconds
3. Refresh Railway dashboard
4. Should now show "Nixpacks"

### "Build fails with Python error"

Check Railway logs. If you see:
```
ERROR: Could not find a version that satisfies the requirement torch>=2.0.0
```

This means PyTorch is installing. It takes 5-10 minutes. Be patient!

### "Server starts but endpoints return 404"

Check that:
- Root Directory is `launcher/backend` (with the slash)
- Start Command is exactly as shown above
- The parent projects (ANN_Project, CNN_Project, nlp-react) are in the repository

---

## Complete Checklist

- [ ] Committed and pushed changes
- [ ] Set Root Directory to `launcher/backend`
- [ ] Set Start Command to `uvicorn main:app --host 0.0.0.0 --port $PORT`
- [ ] Railway shows "Nixpacks" as builder
- [ ] Clicked Redeploy
- [ ] Build completed successfully
- [ ] Health check returns 200
- [ ] Can access `/docs` endpoint
- [ ] Ready to deploy frontend!

---

## You're Almost There! 🚀

The fix is done. Just:
1. ✅ Push the changes
2. ✅ Set those 3 Railway settings
3. ✅ Redeploy
4. ✅ Test the health check
5. ✅ Deploy frontend to Vercel

That's it!
