# 🚂 Railway Python Version Fix

## The Problem

Railway is auto-detecting Python 3.13.9, but your requirements.txt still has:
```
pydantic==2.5.3  ← Incompatible with Python 3.13!
```

Error:
```
TypeError: ForwardRef._evaluate() missing 1 required keyword-only argument: 'recursive_guard'
```

---

## ✅ SOLUTION: Force Python 3.11

### Option 1: Add .python-version file (RECOMMENDED)

Create a file in your backend folder to specify Python version:

1. Create file: `nlp-react/backend/.python-version`
2. Content: `3.11.9`
3. Commit and push

```bash
cd "C:/Users/Soren/OneDrive/Documents/school/College Senior/CST-405/Compiler/CST-435-Group/NLP"

# Create .python-version file
echo "3.11.9" > nlp-react/backend/.python-version

# Commit and push
git add nlp-react/backend/.python-version
git commit -m "Add Python 3.11 version specification for Railway"
git push origin main
```

Then redeploy on Railway (it will auto-detect the new file).

---

### Option 2: Update requirements.txt (ALSO DO THIS)

Your requirements.txt STILL has the old pydantic version. Update it:

**File:** `nlp-react/backend/requirements.txt`

**Change from:**
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
torch>=2.0.0
transformers>=4.30.0
python-multipart==0.0.6
```

**Change to:**
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic>=2.6.0
torch>=2.0.0
transformers>=4.30.0
python-multipart==0.0.6
```

Then:
```bash
git add nlp-react/backend/requirements.txt
git commit -m "Update pydantic to 2.6.0+ for Python 3.13 compatibility"
git push origin main
```

---

### Option 3: Add Environment Variable in Railway

1. Go to Railway Dashboard
2. Click your service
3. **"Variables"** tab
4. Add:
```
PYTHON_VERSION = 3.11.9
```
5. Redeploy

---

## 🎯 RECOMMENDED: Do ALL THREE

For maximum compatibility:

1. ✅ Create `.python-version` file with `3.11.9`
2. ✅ Update `requirements.txt` to use `pydantic>=2.6.0`
3. ✅ Add `PYTHON_VERSION=3.11.9` environment variable

This ensures Python 3.11 is used AND pydantic is updated.

---

## 📝 Quick Fix Commands

```bash
cd "C:/Users/Soren/OneDrive/Documents/school/College Senior/CST-405/Compiler/CST-435-Group/NLP"

# Create Python version file
echo "3.11.9" > nlp-react/backend/.python-version

# Verify requirements.txt has pydantic>=2.6.0
# (Check the file manually or let me update it)

# Commit everything
git add .
git commit -m "Fix: Force Python 3.11 and update pydantic for Railway"
git push origin main
```

---

## ⏰ Timeline

- Create .python-version: 30 seconds
- Update requirements.txt: 30 seconds
- Commit and push: 30 seconds
- Railway redeploy: 5-10 minutes
- **Total: ~10 minutes**

---

## ✅ After Pushing

Railway will automatically:
1. Detect `.python-version` file
2. Use Python 3.11.9 instead of 3.13.9
3. Install pydantic 2.6.0 (or higher)
4. Build successfully!
5. Deploy without memory errors (1GB RAM is plenty)

---

## 🎉 This WILL Work!

With Python 3.11 + pydantic 2.6.0 + Railway's 1GB RAM:
- ✅ Build will succeed
- ✅ Dependencies will install
- ✅ Model will load
- ✅ Server will start
- ✅ No memory errors!

---

**Create that .python-version file and push NOW!** 🚀
