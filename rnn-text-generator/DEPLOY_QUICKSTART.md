# Deployment Quick Start

Quick reference for deploying to Railway and Vercel.

## 🚀 Quick Deploy Steps

### 1. Deploy Backend to Railway

```bash
# Ensure you have trained models in backend/saved_models/
cd backend
ls saved_models/  # Should see model.pt, tokenizer.pkl, model_config.json
```

1. Go to https://railway.app
2. New Project → Deploy from GitHub
3. Select your repo
4. Set environment variables:
   - `ALLOWED_ORIGINS` = `https://your-app.vercel.app` (update after deploying frontend)
   - `PYTHONUNBUFFERED` = `1`
5. Deploy and copy your Railway URL

### 2. Deploy Frontend to Vercel

1. Go to https://vercel.com
2. New Project → Import your repo
3. Configure:
   - Root Directory: `frontend`
   - Framework: Create React App (auto-detected)
4. Set environment variable:
   - `REACT_APP_API_URL` = `https://your-app.railway.app` (your Railway URL)
5. Deploy and copy your Vercel URL

### 3. Update Backend CORS

1. Go back to Railway project
2. Update environment variable:
   - `ALLOWED_ORIGINS` = `https://your-app.vercel.app` (your Vercel URL)
3. Backend will auto-redeploy

### 4. Test Your Deployment

Visit your Vercel URL and verify:
- ✅ "Model Ready" indicator is green
- ✅ Can generate text successfully
- ✅ Analytics dashboard loads

## 📋 Environment Variables Checklist

### Railway (Backend)
- [ ] `ALLOWED_ORIGINS` = Your Vercel URL (no trailing slash)
- [ ] `PYTHONUNBUFFERED` = `1`

### Vercel (Frontend)
- [ ] `REACT_APP_API_URL` = Your Railway URL (no trailing slash)

## ⚠️ Important Notes

- **Model Size**: Railway has 500MB limit. Use `train_optimal.py` for smaller models.
- **No Trailing Slashes**: Make sure URLs don't end with `/`
- **CORS**: Frontend won't work until you set correct `ALLOWED_ORIGINS`
- **Env Vars**: Must start with `REACT_APP_` for frontend

## 🔗 Your URLs

Fill these in after deployment:

```
Railway Backend:  https://__________________.railway.app
Vercel Frontend:  https://__________________.vercel.app
```

## 📚 Full Documentation

See [DEPLOYMENT.md](./DEPLOYMENT.md) for complete documentation including:
- Troubleshooting
- Custom domains
- Large model handling
- Security best practices

## 🆘 Quick Troubleshooting

**CORS Error?**
→ Check `ALLOWED_ORIGINS` in Railway matches your Vercel URL exactly

**Model Not Loading?**
→ Check Railway logs. Verify model files exist and are under 500MB

**Can't Connect to Backend?**
→ Verify `REACT_APP_API_URL` in Vercel is correct

**Build Failed?**
→ Check build logs in Vercel/Railway dashboard
