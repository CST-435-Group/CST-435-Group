# Deployment Checklist

## Pre-Deployment Checks

### Model Files
- [ ] `ANN_Project/best_model.pth` exists
- [ ] `ANN_Project/data/nba_players.csv` exists
- [ ] `CNN_Project/models/best_model.pth` exists
- [ ] `CNN_Project/models/model_metadata.json` exists
- [ ] `nlp-react/backend/model.py` exists
- [ ] `nlp-react/backend/main.py` exists

### Local Testing
- [ ] Backend starts without errors: `cd launcher/backend && python main.py`
- [ ] Frontend builds successfully: `cd launcher/frontend && npm run build`
- [ ] API health check works: http://localhost:8000/health
- [ ] Frontend loads: http://localhost:3000
- [ ] Can navigate between pages
- [ ] API calls work from frontend

### Code Quality
- [ ] All imports resolve correctly
- [ ] No syntax errors
- [ ] Environment variables documented
- [ ] README is comprehensive
- [ ] QUICKSTART guide is clear

## Vercel Deployment (Frontend)

### Setup
- [ ] Vercel account created
- [ ] Vercel CLI installed: `npm i -g vercel`
- [ ] Logged in to Vercel: `vercel login`

### Deployment Steps
1. [ ] Navigate to launcher directory: `cd launcher`
2. [ ] Run: `vercel`
3. [ ] Follow prompts:
   - Project name: `cst-435-launcher`
   - Directory: `./` (current directory)
   - Vercel will detect settings from `vercel.json`
4. [ ] Set environment variables in Vercel dashboard:
   - `VITE_API_URL` = Your Railway backend URL
5. [ ] Deploy to production: `vercel --prod`

### Verification
- [ ] Frontend is accessible at Vercel URL
- [ ] Home page loads correctly
- [ ] All navigation links work
- [ ] Check browser console for errors
- [ ] Test on mobile device

## Railway Deployment (Backend)

### Setup
- [ ] Railway account created (https://railway.app)
- [ ] GitHub repository connected

### Deployment Steps
1. [ ] Create new project in Railway
2. [ ] Select "Deploy from GitHub repo"
3. [ ] Choose `CST-435-Group` repository
4. [ ] Railway auto-detects `railway.toml`
5. [ ] Set root directory to `launcher` (if needed)
6. [ ] Configure environment variables:
   - `PORT` (auto-set by Railway)
   - `PYTHON_VERSION` = `3.11`

### Verification
- [ ] Deployment succeeds (check logs)
- [ ] API is accessible at Railway URL
- [ ] Health check endpoint works: `https://your-app.railway.app/health`
- [ ] API docs accessible: `https://your-app.railway.app/docs`
- [ ] Test each project endpoint:
  - [ ] `/api/ann/health`
  - [ ] `/api/cnn/health`
  - [ ] `/api/nlp/health`

## Post-Deployment

### Frontend Configuration
- [ ] Update CORS origins in `backend/main.py` to include Vercel domain
- [ ] Redeploy backend if CORS was updated
- [ ] Test API calls from Vercel-hosted frontend

### Testing All Features

**ANN Project**
- [ ] Data info loads
- [ ] Team selection works with all methods
- [ ] Player list displays
- [ ] No errors in console

**CNN Project**
- [ ] Model info displays
- [ ] Image upload works
- [ ] Predictions are accurate
- [ ] Confidence scores display

**NLP Project**
- [ ] Sentiment scale displays
- [ ] Text analysis works
- [ ] Example buttons work
- [ ] Results display correctly

### Performance
- [ ] First request completes (may be slow - models loading)
- [ ] Subsequent requests are fast
- [ ] No memory errors
- [ ] Pages load within 3 seconds

### Documentation
- [ ] Update README with deployment URLs
- [ ] Add screenshot to repository
- [ ] Document any deployment issues
- [ ] Update QUICKSTART if needed

## Monitoring

### Railway
- [ ] Check Railway logs for errors
- [ ] Monitor memory usage
- [ ] Set up alerts (if needed)
- [ ] Note cold start times

### Vercel
- [ ] Check Vercel analytics
- [ ] Monitor build times
- [ ] Check for failed builds
- [ ] Review function logs

## Common Issues and Solutions

### Issue: Models not loading on Railway
**Solution**:
- Check if model files are in correct locations
- Review Railway logs for specific error
- Ensure Python version is correct (3.11)

### Issue: CORS errors from Vercel to Railway
**Solution**:
- Add Vercel domain to CORS allowed origins in `backend/main.py`
- Redeploy backend

### Issue: High memory usage on Railway
**Solution**:
- Models are lazy-loaded, should be fine
- If still issues, comment out unused model loaders
- Consider Railway Pro plan

### Issue: Slow first request
**Solution**:
- This is expected (models load on first request)
- Subsequent requests are fast
- Could implement warm-up function (optional)

### Issue: Build fails on Vercel
**Solution**:
- Check Node.js version (18+ required)
- Verify `vercel.json` configuration
- Check build logs in Vercel dashboard

### Issue: Build fails on Railway
**Solution**:
- Check Python version (3.11 required)
- Verify `railway.toml` configuration
- Check if all dependencies are in `requirements.txt`

## Rollback Plan

### Vercel
- Previous deployments available in dashboard
- Click "Promote to Production" on previous deployment

### Railway
- Previous deployments in Railway dashboard
- Redeploy from specific commit
- Or revert Git commit and redeploy

## Success Criteria

- [ ] All three projects accessible
- [ ] No critical errors in logs
- [ ] All features working as expected
- [ ] Performance acceptable
- [ ] Mobile responsive
- [ ] Documentation complete

## Final Notes

- First deployment takes longest (10-15 minutes for models)
- Keep Railway free tier limits in mind (500 hours/month)
- Monitor usage to avoid unexpected costs
- Consider adding error tracking (Sentry, etc.) for production
