# Git Commands to Deploy the Fix

## Step 1: Check Git Status
```bash
cd "C:\Users\Soren\OneDrive\Documents\school\College Senior\CST-405\Compiler\CST-435-Group\NLP"
git status
```

You should see:
```
modified:   nlp-react/backend/requirements.txt
```

## Step 2: Add the Changes
```bash
git add nlp-react/backend/requirements.txt
```

Or add everything:
```bash
git add .
```

## Step 3: Commit the Changes
```bash
git commit -m "Fix: Update pydantic to 2.6.0 to resolve Render build error"
```

## Step 4: Push to GitHub
```bash
git push origin main
```

If your default branch is `master` instead of `main`:
```bash
git push origin master
```

## Step 5: Verify on GitHub
1. Go to your GitHub repository
2. Navigate to: `nlp-react/backend/requirements.txt`
3. Verify it shows `pydantic==2.6.0`

## Step 6: Trigger Redeploy on Render

### Option A: Automatic (if connected to GitHub)
Render should automatically detect the push and redeploy

### Option B: Manual Deploy
1. Go to Render dashboard
2. Click on your service
3. Click **"Manual Deploy"**
4. Select **"Clear build cache & deploy"** (important!)
5. Wait 10-15 minutes

---

## Alternative: Use Windows Git Command
If you prefer, open Command Prompt or Git Bash and run:

```cmd
cd "C:\Users\Soren\OneDrive\Documents\school\College Senior\CST-405\Compiler\CST-435-Group\NLP"
git add .
git commit -m "Fix: Update pydantic to 2.6.0 for Render compatibility"
git push origin main
```

---

## Troubleshooting

### If git says "nothing to commit"
The file might not have been saved. Verify by opening:
`nlp-react/backend/requirements.txt`

Should show:
```
pydantic==2.6.0  ✓
NOT pydantic==2.5.3  ✗
```

### If push fails with authentication error
You may need to use a Personal Access Token:
1. Go to GitHub Settings → Developer Settings → Personal Access Tokens
2. Generate new token with `repo` permissions
3. Use token as password when pushing

### If Render still shows old version
1. Click **"Clear build cache & deploy"**
2. Or delete the service and recreate (last resort)

---

After pushing, Render should automatically redeploy with the fixed requirements.txt!
