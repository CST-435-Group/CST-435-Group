# Vercel Deployment Setup Guide

## Prerequisites

1. Your Railway backend is deployed and running (✅ Done!)
2. You have a Vercel account (sign up at https://vercel.com)
3. Your GitHub repository is accessible

---

## Step 1: Get Your Railway Backend URL

First, you need your Railway backend URL to connect the frontend to it.

1. Go to Railway Dashboard
2. Click on your deployed service
3. Go to **Settings** tab
4. Find **Domains** section
5. Copy your Railway URL (e.g., `https://your-app.railway.app`)

---

## Step 2: Deploy to Vercel

### Option A: Deploy via Vercel Dashboard (Recommended)

1. Go to https://vercel.com/new
2. Click **Import Git Repository**
3. Select your GitHub repository: `CST-435-Group`
4. Configure the project:

   **Framework Preset**: Vite

   **Root Directory**: `launcher/frontend`

   **Build Command**: `npm install && npm run build`

   **Output Directory**: `dist`

   **Install Command**: `npm install`

5. Add Environment Variables (IMPORTANT):
   - Click **Environment Variables**
   - Add this variable:
     - **Name**: `VITE_API_URL`
     - **Value**: `https://your-railway-url.railway.app/api` (replace with your actual Railway URL)

6. Click **Deploy**

### Option B: Deploy via Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Navigate to launcher directory
cd launcher

# Login to Vercel
vercel login

# Deploy (will prompt for configuration)
vercel

# When prompted:
# - Set up and deploy? Yes
# - Which scope? (select your account)
# - Link to existing project? No
# - Project name? cst-435-launcher (or your preferred name)
# - In which directory is your code located? frontend
# - Want to modify settings? Yes
# - Build Command: npm run build
# - Output Directory: dist
# - Development Command: npm run dev
```

---

## Step 3: Configure Environment Variables

After deployment, you need to set the backend API URL:

1. Go to your project in Vercel Dashboard
2. Click **Settings** → **Environment Variables**
3. Add variable:
   - **Name**: `VITE_API_URL`
   - **Value**: `https://your-railway-url.railway.app/api`
   - **Environments**: Check all (Production, Preview, Development)
4. Click **Save**

**Important**: After adding environment variables, you must **redeploy** for them to take effect!

---

## Step 4: Verify Deployment

After deployment completes:

1. **Get your Vercel URL** (e.g., `https://your-app.vercel.app`)
2. **Open it in browser**
3. **You should see**: Main launcher page with three project cards (ANN, CNN, NLP)
4. **Test each project**:
   - Click on ANN → Should load NBA team selection interface
   - Click on CNN → Should load fruit classification interface
   - Click on NLP → Should load sentiment analysis interface

---

## Step 5: Update CORS in Railway Backend

After you get your Vercel URL, you may need to add it to the CORS configuration:

The backend already has `"https://*.vercel.app"` in CORS allowed origins, so it should work automatically. If you have issues:

1. Go to `launcher/backend/main.py`
2. Find the CORS middleware
3. Add your specific Vercel URL to `allow_origins` if needed

---

## Troubleshooting

### Issue: "Cannot connect to backend" or API errors

**Solution 1**: Check environment variable
```bash
# In Vercel Dashboard → Settings → Environment Variables
# Verify VITE_API_URL is set correctly
VITE_API_URL=https://your-railway-url.railway.app/api
```

After changing, redeploy!

**Solution 2**: Check CORS configuration in Railway backend
- Make sure `allow_origins` includes `"https://*.vercel.app"`

### Issue: Build fails with "npm ERR!"

**Solution**: Check the build command and root directory
- Root Directory: `launcher/frontend`
- Build Command: `npm install && npm run build`
- Output Directory: `dist`

### Issue: 404 on page refresh

This is already handled! The `vercel.json` has rewrite rules:
```json
{
  "rewrites": [
    { "source": "/(.*)", "destination": "/" }
  ]
}
```

---

## Quick Reference

### Vercel Configuration Summary

| Setting | Value |
|---------|-------|
| **Framework** | Vite |
| **Root Directory** | `launcher/frontend` |
| **Build Command** | `npm install && npm run build` |
| **Output Directory** | `dist` |
| **Install Command** | `npm install` |
| **Node Version** | 18.x (default) |

### Environment Variables

| Variable | Value |
|----------|-------|
| `VITE_API_URL` | `https://your-railway-url.railway.app/api` |

---

## Testing the Full Stack

Once both are deployed:

1. **Frontend URL**: `https://your-app.vercel.app`
2. **Backend URL**: `https://your-railway-url.railway.app`

**Test flow**:
1. Visit frontend URL
2. Click on **ANN Project** card
3. Click "Select Optimal Team"
4. Should make API call to Railway backend
5. Should display team selection results

If this works, your full stack is deployed successfully! 🎉

---

## Next Steps After Deployment

1. **Custom Domain** (Optional)
   - Add custom domain in Vercel Dashboard → Settings → Domains

2. **Analytics** (Optional)
   - Enable Vercel Analytics in project settings

3. **Git Integration**
   - Vercel automatically redeploys on push to main branch
   - Each PR gets a preview deployment

4. **Monitor Logs**
   - Vercel Dashboard → Deployments → View Function Logs
   - Railway Dashboard → Deployments → View Logs

---

## Summary: What You Need

1. ✅ Railway backend URL
2. ✅ Vercel account
3. ✅ GitHub repository access
4. 📝 Deploy via Vercel Dashboard
5. 📝 Set `VITE_API_URL` environment variable
6. 📝 Test the deployment

That's it! Your unified launcher will be live on Vercel. 🚀
