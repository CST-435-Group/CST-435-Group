# Vercel Deployment - Quick Start

## Step 1: Get Railway Backend URL

1. Go to Railway Dashboard: https://railway.app
2. Click your project
3. Go to **Settings** → **Networking** → **Public Networking**
4. Copy your domain (e.g., `cst-435-production.up.railway.app`)
5. **Full API URL will be**: `https://YOUR-DOMAIN.railway.app/api`

---

## Step 2: Deploy to Vercel

1. **Go to**: https://vercel.com/new

2. **Import your repository**: `CST-435-Group`

3. **Configure Project Settings**:

   | Setting | Value |
   |---------|-------|
   | **Framework Preset** | Vite |
   | **Root Directory** | `launcher/frontend` ← Click "Edit" and set this! |
   | **Build Command** | `npm run build` (auto-detected) |
   | **Output Directory** | `dist` (auto-detected) |
   | **Install Command** | `npm install` (auto-detected) |

4. **Add Environment Variable**:
   - Click **Environment Variables**
   - **Name**: `VITE_API_URL`
   - **Value**: `https://YOUR-RAILWAY-DOMAIN.railway.app/api` (use your actual Railway domain!)
   - **Environments**: Check all three (Production, Preview, Development)
   - Click **Add**

5. **Click Deploy** 🚀

---

## Step 3: Wait for Build

The build will take 2-3 minutes. You'll see:
- ✅ Installing dependencies
- ✅ Building application
- ✅ Deploying to Vercel Edge Network

---

## Step 4: Test Your Deployment

Once deployed, Vercel will give you a URL like: `https://cst-435-launcher.vercel.app`

**Test the app**:
1. Open the Vercel URL in your browser
2. You should see three project cards: ANN, CNN, NLP
3. Click **ANN Project** → Try "Select Optimal Team"
4. Click **CNN Project** → Try uploading an image
5. Click **NLP Project** → Try analyzing sentiment

If all three work, you're done! 🎉

---

## Troubleshooting

### Issue: "Failed to fetch" or network errors

**Check**: Did you set `VITE_API_URL` correctly?
- Go to Vercel Dashboard → Settings → Environment Variables
- Make sure it's: `https://YOUR-RAILWAY-DOMAIN.railway.app/api` (with `/api` at the end!)
- After changing, go to Deployments → click ⋯ → Redeploy

### Issue: Build fails

**Check**: Did you set Root Directory to `launcher/frontend`?
- Go to Vercel Dashboard → Settings → General → Root Directory
- Should be: `launcher/frontend`
- After changing, redeploy

### Issue: 404 on page refresh

This should already be fixed by `vercel.json` file. If not:
- Make sure `launcher/frontend/vercel.json` exists
- Should contain rewrites configuration

---

## Summary Checklist

- [ ] Get Railway domain from Railway Dashboard
- [ ] Go to https://vercel.com/new
- [ ] Import CST-435-Group repository
- [ ] Set Root Directory to `launcher/frontend`
- [ ] Add environment variable `VITE_API_URL` = `https://YOUR-DOMAIN.railway.app/api`
- [ ] Click Deploy
- [ ] Wait for build to complete
- [ ] Test all three projects (ANN, CNN, NLP)

---

## What's Next?

After successful deployment:

1. **Share your app**: Send the Vercel URL to others
2. **Custom domain** (optional): Add in Vercel Settings → Domains
3. **Auto-deploy**: Vercel automatically deploys when you push to GitHub
4. **Monitor**: Check logs in Vercel Dashboard → Deployments

Your unified launcher is now live! 🚀
