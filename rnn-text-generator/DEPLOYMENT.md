# Deployment Guide

This guide covers deploying the RNN Text Generator to Railway (backend) and Vercel (frontend).

## Table of Contents
- [Prerequisites](#prerequisites)
- [Backend Deployment (Railway)](#backend-deployment-railway)
- [Frontend Deployment (Vercel)](#frontend-deployment-vercel)
- [Post-Deployment Configuration](#post-deployment-configuration)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Railway account: https://railway.app
- Vercel account: https://vercel.com
- Trained model files in `backend/saved_models/`
- Git repository (GitHub, GitLab, or Bitbucket)

## Backend Deployment (Railway)

### Step 1: Prepare Your Repository

1. Ensure your model files are ready:
   ```
   backend/saved_models/
   ├── model.pt                 # Model weights
   ├── tokenizer.pkl            # Tokenizer with vocabulary
   └── model_config.json        # Model configuration
   ```

2. **IMPORTANT**: Model files can be large. Railway has a 500MB deployment limit.
   - If your models are too large, consider:
     - Training with limited vocabulary (recommended: use `train_optimal.py`)
     - Uploading models to cloud storage (S3, Google Cloud Storage) and downloading during startup
     - Using Railway volumes for persistent storage

### Step 2: Deploy to Railway

1. **Log in to Railway**: https://railway.app

2. **Create a New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Authorize Railway to access your repository
   - Select your repository

3. **Configure the Service**:
   - Railway will auto-detect Python and use `railway.toml` for configuration
   - Set the root directory to `backend` if prompted

4. **Set Environment Variables**:
   Go to your project settings and add:
   ```
   ALLOWED_ORIGINS=https://your-app.vercel.app
   PYTHONUNBUFFERED=1
   ```
   - Replace `https://your-app.vercel.app` with your actual Vercel URL (you'll get this after deploying frontend)
   - You can add multiple origins separated by commas: `https://your-app.vercel.app,https://www.your-domain.com`

5. **Deploy**:
   - Railway will automatically build and deploy
   - Wait for deployment to complete (check logs for any errors)
   - Note your Railway URL (e.g., `https://your-app.railway.app`)

### Step 3: Verify Backend Deployment

Test your backend API:
```bash
curl https://your-app.railway.app/
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Frontend Deployment (Vercel)

### Step 1: Deploy to Vercel

1. **Log in to Vercel**: https://vercel.com

2. **Import Your Repository**:
   - Click "New Project"
   - Import your Git repository
   - Select the repository

3. **Configure the Project**:
   - Framework Preset: `Create React App` (should auto-detect)
   - Root Directory: `frontend`
   - Build Command: `npm run build` (default)
   - Output Directory: `build` (default)

4. **Set Environment Variables**:
   Add the following environment variable:
   ```
   REACT_APP_API_URL=https://your-app.railway.app
   ```
   - Replace `https://your-app.railway.app` with your actual Railway backend URL
   - Make sure there's NO trailing slash

5. **Deploy**:
   - Click "Deploy"
   - Wait for build and deployment to complete
   - Note your Vercel URL (e.g., `https://your-app.vercel.app`)

### Step 2: Verify Frontend Deployment

1. Visit your Vercel URL
2. Check that the "Model Ready" indicator shows green
3. Try generating text to ensure frontend-backend communication works

## Post-Deployment Configuration

### Update Backend CORS Settings

Now that you have your Vercel URL, update the Railway backend:

1. Go to your Railway project
2. Update the `ALLOWED_ORIGINS` environment variable:
   ```
   ALLOWED_ORIGINS=https://your-app.vercel.app
   ```
3. Redeploy the backend (Railway will auto-redeploy when you change env vars)

### Custom Domains (Optional)

**Vercel**:
1. Go to Project Settings → Domains
2. Add your custom domain
3. Follow DNS configuration instructions

**Railway**:
1. Go to Project Settings → Domains
2. Add your custom domain
3. Configure DNS as instructed

Don't forget to update `ALLOWED_ORIGINS` if you add custom domains!

## Troubleshooting

### Backend Issues

**Model not loading:**
- Check Railway logs: Are model files present?
- Verify model files are under 500MB combined
- Check `saved_models/` directory structure

**CORS errors:**
- Verify `ALLOWED_ORIGINS` in Railway includes your Vercel URL
- Make sure there are no trailing slashes
- Check that both http/https protocols match

**Port binding errors:**
- Railway provides `$PORT` automatically
- Ensure `railway.toml` uses `--port $PORT` in the start command

### Frontend Issues

**Can't connect to backend:**
- Verify `REACT_APP_API_URL` is set correctly in Vercel
- Check that Railway backend is running (visit the health endpoint)
- Ensure there's no trailing slash in `REACT_APP_API_URL`

**Environment variables not working:**
- Environment variables must start with `REACT_APP_` in Create React App
- Redeploy after changing environment variables
- Clear browser cache

**Build failures:**
- Check build logs in Vercel
- Ensure all dependencies are in `package.json`
- Verify Node version compatibility

### Common Issues

**Large Model Files:**
If your model files exceed Railway's limits:

1. **Option A**: Use volume storage
   ```bash
   # Add to railway.toml
   [volumes]
   models = "/app/saved_models"
   ```

2. **Option B**: Download from cloud storage on startup
   - Upload models to S3/GCS
   - Add download script to `railway.toml` build command

3. **Option C**: Use smaller models
   - Train with limited vocabulary: `python app/train_optimal.py`
   - This typically produces 10-20MB model files

**Logs and Debugging:**
- Railway: Check deployment logs in project dashboard
- Vercel: Check function logs and build logs
- Use health check endpoints to verify services

## Environment Variables Reference

### Backend (Railway)
| Variable | Required | Example | Description |
|----------|----------|---------|-------------|
| `ALLOWED_ORIGINS` | Yes | `https://your-app.vercel.app` | Comma-separated CORS origins |
| `PYTHONUNBUFFERED` | No | `1` | Disable Python output buffering |
| `PORT` | Auto | `8000` | Automatically provided by Railway |

### Frontend (Vercel)
| Variable | Required | Example | Description |
|----------|----------|---------|-------------|
| `REACT_APP_API_URL` | Yes | `https://your-app.railway.app` | Backend API base URL (no trailing slash) |

## Continuous Deployment

Both Railway and Vercel support automatic deployments:

- **Railway**: Automatically redeploys when you push to your repository's main branch
- **Vercel**: Automatically redeploys when you push to your repository's main branch

To disable auto-deployment, configure in each platform's settings.

## Cost Considerations

**Railway**:
- Free tier: $5 credit/month
- Pay-as-you-go after free credits
- Costs depend on CPU/memory usage and uptime

**Vercel**:
- Hobby plan: Free
- Includes 100GB bandwidth/month
- Serverless functions included

## Security Best Practices

1. **Use HTTPS only** - Both platforms provide SSL automatically
2. **Restrict CORS origins** - Don't use `*` in production
3. **Environment variables** - Never commit `.env` files
4. **API rate limiting** - Consider adding rate limiting to backend
5. **Monitor logs** - Regularly check deployment logs for errors

## Support

- Railway Docs: https://docs.railway.app
- Vercel Docs: https://vercel.com/docs
- Project Issues: Check your repository's issue tracker
