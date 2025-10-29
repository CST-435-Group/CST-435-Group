# Vercel Build Error Fix

## Current Error

```
sh: line 1: vite: command not found
Error: Command "vite build" exited with 127
```

## The Problem

Vercel is trying to run `vite build` directly, but it should run `npm run build` instead. The build command must use npm to properly access the installed dependencies.

## The Solution

In Vercel Dashboard, you need to configure these settings correctly:

### Go to: Project Settings → General

**1. Root Directory**
- Set to: `launcher/frontend` ✅ (This is correct!)

**2. Build Command**
- **MUST BE**: `npm run build`
- ❌ DO NOT use: `vite build` (vite won't be in PATH)
- ❌ DO NOT use: `cd frontend && npm run build` (you're already in frontend!)

**3. Output Directory**
- Set to: `dist`

**4. Install Command**
- **LEAVE EMPTY** or set to: `npm install`
- DO NOT use `cd frontend && npm install`

---

## Step-by-Step Fix

1. Go to Vercel Dashboard: https://vercel.com
2. Click on your project
3. Go to **Settings** → **General**
4. Scroll down to **Build & Development Settings**
5. Click **Override** toggle if it's not already on
6. **Build Command**:
   - Change from `vite build` to: `npm run build`
   - Click **Save**
7. **Install Command**:
   - Should be: `npm install` (or leave empty for auto-detect)
   - Click **Save**
8. **Output Directory**:
   - Should be: `dist`
   - Click **Save**
9. Go to **Deployments** tab
10. Click **Redeploy** (or wait for auto-redeploy)

---

## Why This Works

When you set Root Directory to `launcher/frontend`, Vercel:
1. ✅ Changes to that directory first
2. ✅ Runs `npm install` from there
3. ✅ Runs `npm run build` from there
4. ✅ Looks for `dist/` output from there

You DON'T need to `cd frontend` in the commands because Vercel is already in the `frontend` directory!

---

## Correct Settings Summary

| Setting | Value |
|---------|-------|
| **Root Directory** | `launcher/frontend` |
| **Framework Preset** | Vite |
| **Build Command** | `npm run build` (or leave empty) |
| **Output Directory** | `dist` |
| **Install Command** | `npm install` (or leave empty) |

---

## Alternative: Let Vercel Auto-Detect Everything

Actually, the BEST approach is:

1. **Root Directory**: Set to `launcher/frontend`
2. **Everything else**: LEAVE EMPTY / Use defaults

Vercel will automatically detect:
- ✅ Framework: Vite
- ✅ Install command: `npm install`
- ✅ Build command: `npm run build`
- ✅ Output directory: `dist`

---

## After Making Changes

1. Go to Deployments
2. Click **Redeploy**
3. Watch the build logs
4. Should see:
   ```
   Installing dependencies...
   Running "npm install"
   Building...
   Running "npm run build"
   ✓ Built in X seconds
   ```

That's it! The build should succeed now.
