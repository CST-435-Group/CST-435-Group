# 🚀 Quick Start Guide - Anime Image Classifier

Get your CNN trained and app running in 3 steps!

---

## ⚡ Super Fast Start (3 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model (20-60 min)
python train_model.py

# 3. Launch app
streamlit run streamlit_app.py
```

**Done!** Your app opens at http://localhost:8501 🎉

---

## 📋 Detailed Setup

### Prerequisites
- Python 3.8+
- Kaggle account
- 5GB free disk space
- Optional: NVIDIA GPU (3-4x faster)

### Step 1: Kaggle API Setup (2 minutes)

1. **Get your API key:**
   - Go to https://www.kaggle.com/settings
   - Scroll to "API" section
   - Click "Create New API Token"
   - Downloads `kaggle.json`

2. **Place the file:**
   ```
   Windows: C:\Users\YourUsername\.kaggle\kaggle.json
   ```

3. **Verify:**
   ```bash
   kaggle datasets list
   ```

### Step 2: Install Dependencies (5 minutes)

**Option A: Automatic (Windows)**
```bash
setup.bat
```

**Option B: Manual**
```bash
# For GPU (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# All other packages
pip install kagglehub pandas numpy pillow opencv-python tqdm matplotlib seaborn streamlit requests scikit-learn
```

**Verify GPU:**
```bash
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

### Step 3: Train Model (20-60 minutes)

```bash
python train_model.py
```

**What happens:**
1. Downloads Safebooru dataset (3M rows)
2. Selects 5 tags with 500+ images each
3. Downloads ~2,500-5,000 images
4. Converts to grayscale 128×128
5. Trains CNN for 50 epochs
6. Saves best model

**Expected output:**
```
[STEP 1] Downloading Safebooru dataset...
✅ Dataset downloaded

[STEP 2] Loading metadata CSV...
✅ Loaded 2,736,037 rows

[STEP 3] Analyzing tags...
Selected tags:
  1. cat: 45,234 images
  2. sword: 28,567 images
  3. school_uniform: 67,890 images
  4. wings: 34,123 images
  5. glasses: 89,456 images

[STEP 4] Downloading images...
✅ Downloaded 2,500 images

[STEP 5-9] Training...
Epoch 50/50: Val Acc: 72.34%
✅ Training complete!
```

**Training Time:**
- GPU: 20-40 minutes
- CPU: 1-2 hours

### Step 4: Launch Streamlit App (instant)

```bash
streamlit run streamlit_app.py
```

Opens at: http://localhost:8501

**Features:**
- View 10 images at a time
- See predictions vs. true labels
- Color-coded correct/incorrect
- Confidence scores
- Top-3 predictions per image
- Batch accuracy metrics

---

## 🎯 What You'll See

### In the App:

**Sidebar:**
```
📊 Model Information
Test Accuracy: 72.34%
Number of Classes: 5

🏷️ Classification Tags
1. cat
2. sword
3. school_uniform
4. wings
5. glasses

📈 Dataset Info
Total Images: 2,500
Training: 1,750
Validation: 375
Test: 375
```

**Main Display:**
- Grid of 10 images (2 rows × 5 columns)
- Each image shows:
  - The actual image (128×128 grayscale)
  - Predicted tag (green ✅ if correct, red ❌ if wrong)
  - Confidence percentage
  - Top 3 predictions (expandable)

**Metrics:**
- Batch Accuracy: % correct in current 10
- Correct/Incorrect counts
- Results table with all predictions

---

## 🎨 Customization Tips

### Change the Tags

Edit `train_model.py` line 100:
```python
SELECTED_TAGS = [
    'cat',           # Replace with your tags
    'dog',           # Must have 500+ images each
    'car',
    'tree',
    'house'
]
```

**Good tag ideas:**
- Visual objects: cat, dog, car, tree, house
- Characters: 1girl, 1boy, solo, multiple_girls
- Settings: outdoor, indoor, classroom, forest
- Features: red_eyes, blue_hair, long_hair, wings
- Objects: sword, gun, book, phone, food

### Faster Training (Testing)

Edit `train_model.py`:
```python
num_epochs = 10      # Line 477 (instead of 50)
MAX_IMAGES = 500     # Line 148 (instead of 1000)
```

### Better Accuracy

Edit `train_model.py`:
```python
num_epochs = 100     # Train longer
MAX_IMAGES = 2000    # More data
learning_rate = 0.0001  # Fine-tune (Line 472)
```

---

## 🐛 Common Issues & Fixes

### "Kaggle API credentials not found"
```bash
# Fix: Place kaggle.json in correct location
# Windows: C:\Users\YourUsername\.kaggle\kaggle.json

# Verify with:
kaggle datasets list
```

### "Not enough images for tag"
```python
# Fix: Choose more common tags in train_model.py:
SELECTED_TAGS = ['1girl', 'solo', 'long_hair', 'short_hair', 'smile']
```

### "CUDA out of memory"
```python
# Fix: Reduce batch size in train_model.py line 369:
batch_size = 16  # Instead of 32
```

### "Streamlit app shows error"
```bash
# Fix: Make sure training completed
# Check that these files exist:
dir models
# Should see: best_model.pth, model_metadata.json
```

### Training is too slow
```python
# Option 1: Use GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Option 2: Reduce images
MAX_IMAGES = 500  # In train_model.py line 148

# Option 3: Reduce epochs
num_epochs = 20  # In train_model.py line 477
```

---

## 📊 Understanding the Output

### During Training:
```
Epoch 1/50
████████████ 100% | loss: 1.234, acc: 45.2%
Train Loss: 1.2340 | Train Acc: 45.23%
Val Loss: 1.4567 | Val Acc: 38.91%
💾 Saved best model
```

- **Loss**: Lower is better (shows how wrong predictions are)
- **Accuracy**: Higher is better (% correct predictions)
- **Val Acc**: Most important (performance on unseen data)
- 💾 means model improved and was saved

### In Streamlit App:
```
✅ school_uniform
Confidence: 87.3%
```
- ✅ Green = Correct prediction
- 87.3% = Model confidence

```
❌ sword
True: cat
Confidence: 65.2%
```
- ❌ Red = Wrong prediction
- True label was "cat"
- Model was 65.2% confident (but wrong!)

---

## 🎯 Expected Results

### Model Performance:
- **Training Accuracy:** 75-90%
- **Test Accuracy:** 65-75%
- **Common issues:**
  - Similar tags (1girl vs. solo) harder to distinguish
  - More data = better accuracy
  - Distinct tags = easier classification

### Best Tag Combinations (Recommended):
```python
# Option 1: Objects (Easiest)
['cat', 'dog', 'car', 'tree', 'house']

# Option 2: Fantasy Elements
['wings', 'horns', 'tail', 'elf', 'dragon']

# Option 3: Accessories
['glasses', 'hat', 'bow', 'ribbon', 'flower']

# Option 4: Settings
['beach', 'forest', 'city', 'school', 'indoor']
```

---

## 📁 Generated Files

After training:
```
preprocessed_images/     # 2,500-5,000 images
├── cat/                # ~500-1000 each
├── sword/
├── school_uniform/
├── wings/
└── glasses/

models/
├── best_model.pth           # ~50MB trained model
├── model_metadata.json      # Accuracy stats
└── training_history.json    # Training curves

data/
└── dataset_metadata.json    # Image paths
```

---

## 🎉 Success Checklist

Before launching app:
- [ ] Kaggle credentials configured
- [ ] Dependencies installed
- [ ] Training completed without errors
- [ ] `models/best_model.pth` exists
- [ ] Test accuracy printed at end
- [ ] No error messages

App working correctly:
- [ ] Opens at http://localhost:8501
- [ ] Sidebar shows metrics
- [ ] Images display in grid
- [ ] Predictions show green/red
- [ ] "Load 10 Random Images" button works
- [ ] Confidence percentages shown

---

## 💡 Pro Tips

### 1. Start Small
```python
# First run: Test with small dataset
MAX_IMAGES = 500
num_epochs = 10
# Verify everything works (~10 min)

# Then: Full training
MAX_IMAGES = 1000
num_epochs = 50
```

### 2. Monitor Training
- Watch validation accuracy
- If val_acc stops improving → training done
- If train_acc >> val_acc → overfitting

### 3. Pick Good Tags
- ✅ Visually distinct objects
- ✅ 1000+ images available
- ❌ Avoid similar tags (1girl vs. solo)
- ❌ Avoid rare tags (<500 images)

### 4. Use GPU
```bash
# Check if GPU detected:
python -c "import torch; print(torch.cuda.get_device_name(0))"

# If not, install CUDA PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📞 Quick Commands Reference

```bash
# Setup
pip install -r requirements.txt

# Train
python train_model.py

# Launch app
streamlit run streamlit_app.py

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Verify Kaggle
kaggle datasets list
```

---

## 🆘 Need Help?

1. **Read the error message** - usually tells you what's wrong
2. **Check README.md** - full documentation
3. **Verify files exist** - models/, data/, preprocessed_images/
4. **Try small test first** - 500 images, 10 epochs

---

**Ready to start?** 🚀

```bash
python train_model.py
```

Then:

```bash
streamlit run streamlit_app.py
```

**That's it!** Your anime classifier is live! 🎨

---

*Estimated total time: 30-90 minutes (depending on GPU)*
