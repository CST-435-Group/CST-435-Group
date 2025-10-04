# 🍎 Fruit Image Classification - CNN Project
## CST-435 Neural Networks Assignment

Train a CNN to classify fruit images, then deploy with an interactive Streamlit web app!

---

## 📋 Project Overview

This project:
1. **Downloads** Fruits dataset from Kaggle (100+ fruit types)
2. **Selects** 5 fruit categories with 500+ images each
3. **Preprocesses** images to grayscale 128×128 pixels
4. **Trains** a CNN model using PyTorch with GPU acceleration
5. **Deploys** the best model in a Streamlit web app
6. **Displays** 10 images at a time with predictions and accuracy metrics

---

## 🚀 Quick Start (2 Commands!)

```bash
# 1. Train model (10-30 min)
python train_model.py

# 2. Launch app
streamlit run streamlit_app.py
```

**That's it!** Your app opens at http://localhost:8501 🎉

---

## 📦 Prerequisites

### 1. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Kaggle API:
Place your `kaggle.json` file in:
- Windows: `C:\Users\YourUsername\.kaggle\kaggle.json`

Get your API key from https://www.kaggle.com/settings

---

## 🎯 How It Works

### Training Phase (`train_model.py`):
1. ✅ Downloads Fruits dataset from Kaggle
2. ✅ Explores available fruit categories
3. ✅ **Automatically selects 5 fruits** with most images
4. ✅ Preprocesses all images to **grayscale 128×128**
5. ✅ Trains CNN for 50 epochs
6. ✅ Saves best model

**Example fruits selected:**
- Apple
- Banana  
- Orange
- Strawberry
- Grape

(Automatically chosen based on image count)

### Deployment Phase (`streamlit_app.py`):
1. ✅ Loads trained model
2. ✅ Shows model accuracy in sidebar
3. ✅ Displays 10 random images in grid
4. ✅ Predicts fruit for each image
5. ✅ Shows confidence scores
6. ✅ Color-codes correct (✅) vs incorrect (❌)
7. ✅ Calculates batch accuracy

---

## 🏗️ CNN Architecture

```
Input: 1×128×128 Grayscale
    ↓
Conv2D(32) + BatchNorm + ReLU + MaxPool
    ↓
Conv2D(64) + BatchNorm + ReLU + MaxPool
    ↓
Conv2D(128) + BatchNorm + ReLU + MaxPool
    ↓
Flatten → 32,768 features
    ↓
Dense(256) + ReLU + Dropout(0.5)
    ↓
Dense(128) + ReLU + Dropout(0.5)
    ↓
Dense(5) → Predictions
```

**Features:**
- 3 Convolutional blocks
- Batch Normalization for stability
- Max Pooling for spatial reduction
- Dropout for regularization
- ~2M trainable parameters

---

## 📊 Expected Results

### Model Performance:
- **Training Accuracy:** 85-95%
- **Test Accuracy:** 80-90%

(Fruits are easier to distinguish than anime tags!)

### Training Time:
| Hardware | Time (50 epochs) |
|----------|-----------------|
| RTX 3060 GPU | 10-15 min |
| GTX 1660 GPU | 15-20 min |
| CPU Only | 30-45 min |

### Dataset:
- **Total Images:** 2,500-5,000
- **Per Class:** 500-1,000 images
- **Image Size:** 128×128 grayscale
- **Classes:** 5 fruits (auto-selected)

---

## 🖼️ Streamlit App Features

### Main Display:
```
🎲 Load 10 Random Images
─────────────────────────────

[Img 1]    [Img 2]    [Img 3]    [Img 4]    [Img 5]
✅ Apple   ❌ Banana  ✅ Orange  ✅ Apple   ✅ Grape
92.3%     True:Apple  88.1%      95.7%      87.2%
          71.2%

[Img 6]    [Img 7]    [Img 8]    [Img 9]    [Img 10]
...

Batch Accuracy: 80%
Correct: 8 | Incorrect: 2 | Total: 10
```

### Interactive Features:
- 🎲 Load new random batches
- 📊 See confidence scores
- 🔍 Expand to view top-3 predictions
- 📋 Detailed results table
- 🎯 Per-class performance metrics

---

## 🎓 Assignment Requirements - ALL MET!

✅ **Dataset:** Fruits from Kaggle
✅ **Preprocessing:** Grayscale 128×128 conversion
✅ **Classes:** 5 distinct fruits (auto-selected)
✅ **Minimum Images:** 500+ per class
✅ **Training:** Complete CNN training pipeline
✅ **Model:** Saved best performing model
✅ **Deployment:** Interactive Streamlit web app
✅ **Visualization:** 10 images at a time
✅ **Metrics:** Accuracy, confidence, predictions shown

**Bonus Features:**
- Automatic fruit selection
- Batch Normalization
- Dropout regularization
- Learning rate scheduling
- GPU acceleration
- Professional web interface

---

## 📁 Files Generated

### During Training:
```
preprocessed_images/
├── Apple/              # ~500-1000 images
├── Banana/             # ~500-1000 images
├── Orange/             # ~500-1000 images
├── Strawberry/         # ~500-1000 images
└── Grape/              # ~500-1000 images

models/
├── best_model.pth          # Trained model (~50MB)
├── model_metadata.json     # Accuracy and info
└── training_history.json   # Training curves

data/
└── dataset_metadata.json   # Image paths and labels
```

---

## 🔧 Customization

### Want Different Fruits?
The script automatically selects the top 5 fruits with most images!

Or manually edit `train_model.py` after line 102:
```python
# After seeing available fruits, manually select:
SELECTED_FRUITS = ['Apple', 'Banana', 'Mango', 'Pineapple', 'Watermelon']
```

### Change Training Parameters:
```python
# In train_model.py:
num_epochs = 50         # Line 477
batch_size = 32         # Line 369
MAX_IMAGES_PER_FRUIT = 1000  # Line 132
```

---

## 🐛 Troubleshooting

### "Kaggle API not found"
```bash
# Place kaggle.json in:
C:\Users\YourUsername\.kaggle\kaggle.json
```

### "CUDA out of memory"
```python
# Reduce batch size in train_model.py line 369:
batch_size = 16  # Instead of 32
```

### "Not enough images"
The script automatically picks fruits with enough images!

### Training is slow
- Use GPU if available
- Reduce `MAX_IMAGES_PER_FRUIT = 500`
- Reduce `num_epochs = 25`

---

## 📊 What You'll See During Training

```
[STEP 1] Downloading Fruits dataset from Kaggle...
✅ Dataset downloaded

[STEP 2] Exploring dataset structure...
Found 120 directories

📊 Available fruits and image counts:
   1. Apple             : 3,000 images
   2. Banana            : 2,800 images
   3. Orange            : 2,500 images
   ...

[STEP 3] Selecting 5 fruit categories...
🎯 Selected fruits:
  1. Apple: 3000 images
  2. Banana: 2800 images
  3. Orange: 2500 images
  4. Strawberry: 2200 images
  5. Grape: 2000 images

[STEP 4] Collecting and preprocessing images...
Processing 'Apple'...
  ✅ Successfully processed 1000 images

✅ Total images processed: 5000

[STEP 5-8] Training...
Epoch 50/50: Val Acc: 87.32%
✅ Training complete!

[STEP 9] Evaluating...
Test Accuracy: 86.45%
```

---

## 🎉 Success Indicators

### During Training:
- ✅ Dataset downloads successfully
- ✅ 5 fruits selected automatically
- ✅ ~5,000 images processed
- ✅ Training accuracy increases each epoch
- ✅ Model saved: `models/best_model.pth`

### In Streamlit App:
- ✅ App opens at http://localhost:8501
- ✅ Sidebar shows ~80-90% accuracy
- ✅ 10 fruit images display in grid
- ✅ Predictions show with confidence
- ✅ Green/red color coding works
- ✅ "Load 10 Random Images" button works

---

## 💡 Pro Tips

### First Run:
1. Let it auto-select the fruits (top 5 by count)
2. Use default settings for initial training
3. Verify everything works end-to-end

### Second Run:
1. Try manually selecting specific fruits
2. Increase training epochs to 100
3. Experiment with more images per fruit

### For Best Accuracy:
```python
MAX_IMAGES_PER_FRUIT = 2000  # More data
num_epochs = 100             # Train longer
```

---

## 📞 Quick Commands Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py

# Launch app
streamlit run streamlit_app.py

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🌟 Why This Project Rocks

- 🎯 **Auto fruit selection** - No manual configuration needed
- 📦 **Real dataset** - Actual fruit images from Kaggle
- 🚀 **Fast training** - Fruits are easier to classify than anime
- 💻 **Professional UI** - Beautiful Streamlit interface
- 🎮 **GPU support** - 3x faster with CUDA
- ✅ **High accuracy** - Expect 80-90% accuracy!
- 🔧 **Easy to use** - Just 2 commands to run

---

## 🎓 Perfect for Assignment

This project demonstrates:
- ✅ Dataset selection and preprocessing
- ✅ Grayscale image conversion
- ✅ CNN architecture design
- ✅ Model training and optimization
- ✅ Model evaluation
- ✅ Deployment with web interface
- ✅ Interactive visualization

---

**Ready to classify fruits?** 🍎🍌🍊

```bash
python train_model.py
```

Then:

```bash
streamlit run streamlit_app.py
```

**Your fruit classifier will be live in ~20 minutes!** 🎉

---

*CST-435 Neural Networks Assignment - PyTorch + Streamlit Edition*
