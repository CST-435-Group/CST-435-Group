# 🚀 GAN Quick Start Guide

Get your GAN running in 5 minutes!

---

## Step 1: Install Dependencies (2 min)

```bash
cd GAN
pip install -r requirements.txt
```

**Having issues?** See [INSTALL.md](INSTALL.md) for detailed instructions.

---

## Step 2: Verify Setup (30 sec)

```bash
python test_setup.py
```

**Expected output:**
```
[TEST 1] Checking PyTorch installation...
  ✅ PyTorch 2.x.x installed
  ✅ CUDA available: 11.8
  ✅ GPU: NVIDIA GeForce RTX 3060

[TEST 2] Checking dependencies...
  ✅ torchvision installed
  ✅ Pillow (PIL) installed
  ...

All tests passed! Ready to train GAN.
```

**If you see errors:**
- Missing torchvision: `pip install torchvision`
- No preprocessed images: Run CNN_Project first (see Step 3)

---

## Step 3: Ensure CNN_Project Data Exists

**If you haven't run CNN_Project yet:**
```bash
cd ../CNN_Project
python train_model.py
cd ../GAN
```

**If you already have CNN_Project data:**
Skip this step - you're ready!

---

## Step 4: Train GAN (30-60 min)

```bash
python train_gan.py
```

**What you'll see:**
```
[STEP 1] Initializing GAN models...
Generator: 8,500,000 parameters
Discriminator: 2,800,000 parameters

[STEP 2] Loading real images from CNN_Project...
Found 5000 images

[STEP 3] Starting training...

Epoch 1/100 (1.0% complete)
  D_loss: 1.2345 | G_loss: 2.3456
  D(real): 0.856 | D(fake): 0.123
  Epoch Time: 32.5s | Avg: 32.5s/epoch
  Elapsed: 0:00:32 | ETA: 0:53:28
```

**Training will save models every epoch automatically!**
- Safe to Ctrl+C anytime
- Models saved to `GAN/models/`
- Progress images saved to `GAN/training_progress/`

---

## Step 5: Check Models (After Training Starts)

Open a new terminal and check available models:
```bash
python check_models.py
```

**Output:**
```
AVAILABLE MODELS
================================================================================

✅ best_model.pth
   ⭐ Best quality (RECOMMENDED)
   Size: 85.3 MB
   Epoch: 45
   D(fake): 0.523

✅ latest_model.pth
   🔄 Most recent checkpoint
   Size: 85.3 MB
   Epoch: 50
```

---

## Step 6: Generate Images (After Any Training)

```bash
# Use best model (recommended)
python generate_images.py --num 50 --classify
```

**Output:**
```
Generated 50 images in GAN/generated_images/
✅ Grid saved to GAN/generated_images/grid.png

CNN Classifier loaded (5 classes)
Classification Results:
  Image   1: Apple           (confidence: 85.2%)
  Image   2: Banana          (confidence: 78.9%)
  ...
```

---

## Quick Commands Cheat Sheet

```bash
# Install
pip install -r requirements.txt

# Verify setup
python test_setup.py

# Train (can Ctrl+C anytime)
python train_gan.py

# Check models
python check_models.py

# Generate images
python generate_images.py --num 50

# Generate and classify
python generate_images.py --num 50 --classify

# Use specific model
python generate_images.py --model GAN/models/latest_model.pth --num 50
```

---

## Common Scenarios

### Scenario 1: First Time Setup
```bash
cd GAN
pip install -r requirements.txt
python test_setup.py
python train_gan.py
# Wait for training or Ctrl+C when you want
python generate_images.py --num 50 --classify
```

### Scenario 2: I Stopped Training Early
```bash
# Check what models are available
python check_models.py

# Use best model found so far
python generate_images.py --model GAN/models/best_model.pth --num 50
```

### Scenario 3: Continue Viewing Training Progress
```bash
# Training is running, want to check progress
# Open new terminal:
cd GAN

# View latest progress images
start training_progress/  # Windows
open training_progress/   # Mac
xdg-open training_progress/  # Linux

# Or view live loss plot
start training_history_live.png  # Windows
```

### Scenario 4: Training on Different Computer
```bash
# Copy entire GAN/models/ folder to new computer
# Then:
cd GAN
pip install -r requirements.txt
python generate_images.py --model models/best_model.pth --num 50
```

---

## Expected Training Times

| Hardware | Time per Epoch | 100 Epochs |
|----------|---------------|------------|
| RTX 3060 | ~30s | ~50 min |
| GTX 1660 | ~45s | ~75 min |
| CPU Only | ~2-3 min | ~3-5 hours |

---

## Troubleshooting Quick Fixes

### "No module named 'torchvision'"
```bash
pip install torchvision
```

### "No images found in preprocessed_images"
```bash
cd ../CNN_Project
python train_model.py
cd ../GAN
```

### "CUDA out of memory"
Edit `train_gan.py` line 252:
```python
batch_size = 32  # Changed from 64
```

### Training seems stuck
- Check GPU usage: `nvidia-smi`
- Check disk space
- Look for errors in terminal

---

## What Success Looks Like

### After 10 epochs:
- Blob-like shapes appearing
- D(fake) moving from 0.1 → 0.3

### After 50 epochs:
- Recognizable fruit-like shapes
- D(fake) around 0.4-0.5
- CNN classifies some as correct fruits

### After 100 epochs:
- Clear fruit images
- D(fake) stable around 0.5
- CNN confidently classifies most images

---

## Next Steps After Training

1. **View generated images:**
   - Open `GAN/generated_images/`
   - View grid: `GAN/generated_images/grid_64.png`

2. **Check training progress:**
   - View `GAN/training_history.png`
   - Browse `GAN/training_progress/epoch_*.png`

3. **Generate more images:**
   ```bash
   python generate_images.py --num 100 --classify
   ```

4. **Experiment:**
   - Try different models (latest vs best vs checkpoints)
   - Generate different quantities
   - Test with CNN classifier

---

## Files You'll Have After Training

```
GAN/
├── models/
│   ├── best_model.pth         ⭐ Use this!
│   ├── latest_model.pth
│   ├── checkpoint_*.pth
│   └── *.json (metadata)
├── generated_images/
│   ├── sample_*.png (100 images)
│   └── grid_64.png
├── training_progress/
│   └── epoch_*.png (every 5 epochs)
└── training_history.png
```

---

## Getting Help

- **Setup issues:** See [INSTALL.md](INSTALL.md)
- **Usage questions:** See [README.md](README.md)
- **Architecture details:** See `GAN_ARCHITECTURE_20x20.md`

---

**🎉 You're ready to generate some fruit images!**

```bash
python train_gan.py
```
