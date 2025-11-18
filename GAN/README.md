# 🎨 GAN for 128×128 Grayscale Fruit Images
## CST-435 Neural Networks Assignment

Generate synthetic fruit images using a Generative Adversarial Network (GAN) trained on real fruit images from the CNN_Project.

---

## 📋 Project Overview

This GAN project:
1. **Trains** a GAN on preprocessed fruit images from CNN_Project
2. **Generates** realistic 128×128 grayscale fruit images
3. **Integrates** with CNN classifier to evaluate generated images
4. **Uses** PyTorch with GPU acceleration for fast training

---

## 🚀 Quick Start (2 Commands!)

```bash
# 1. Train GAN (30-60 min with GPU)
python GAN/train_gan.py

# 2. Generate new images
python GAN/generate_images.py --num 50 --classify
```

**That's it!** Your generated images are in `GAN/generated_images/` 🎉

---

## 📦 Prerequisites

### 1. Install Dependencies:
```bash
cd GAN
pip install -r requirements.txt
```

**For detailed installation instructions (GPU setup, troubleshooting, etc.):**
See [INSTALL.md](INSTALL.md)

GPU recommended but not required.

### 2. Verify Installation:
```bash
python test_setup.py
```

### 3. Complete CNN_Project First:
```bash
cd ../CNN_Project
python train_model.py
cd ../GAN
```

This creates the preprocessed fruit images that the GAN will learn from.

---

## 🏗️ GAN Architecture

### Generator (G)
**Purpose:** Transform random noise into 128×128 grayscale images

```
INPUT: Random Noise ξ ~ N(0,1)
Shape: [Batch, 100]
    ↓
Dense → [Batch, 8×8×512]
    ↓
BatchNorm + LeakyReLU(0.2)
    ↓
Reshape → [Batch, 512, 8, 8]
    ↓
Conv2DTranspose(256, 5×5, stride=2) → [Batch, 256, 16, 16]
    ↓
BatchNorm + LeakyReLU(0.2)
    ↓
Conv2DTranspose(128, 5×5, stride=2) → [Batch, 128, 32, 32]
    ↓
BatchNorm + LeakyReLU(0.2)
    ↓
Conv2DTranspose(64, 5×5, stride=2) → [Batch, 64, 64, 64]
    ↓
BatchNorm + LeakyReLU(0.2)
    ↓
Conv2DTranspose(1, 5×5, stride=2) → [Batch, 1, 128, 128]
    ↓
Tanh → OUTPUT: [-1, 1]
```

**Parameters:** ~8.5M

### Discriminator (D)
**Purpose:** Classify images as real or fake

```
INPUT: Image [Batch, 1, 128, 128]
    ↓
Conv2D(64, 5×5, stride=2) → [Batch, 64, 64, 64]
    ↓
LeakyReLU(0.2) + Dropout(0.3)
    ↓
Conv2D(128, 5×5, stride=2) → [Batch, 128, 32, 32]
    ↓
LeakyReLU(0.2) + Dropout(0.3)
    ↓
Conv2D(256, 5×5, stride=2) → [Batch, 256, 16, 16]
    ↓
LeakyReLU(0.2) + Dropout(0.3)
    ↓
Conv2D(512, 5×5, stride=2) → [Batch, 512, 8, 8]
    ↓
LeakyReLU(0.2) + Dropout(0.3)
    ↓
Flatten → Dense(1) → OUTPUT: Logit
```

**Parameters:** ~2.8M

---

## 🎯 Training Process

### Two-Step Alternating Training

```
FOR each epoch:
    FOR each batch of real images:

        ┌─────────────────────────────────┐
        │ STEP A: Train Discriminator     │
        └─────────────────────────────────┘
        1. Get real images from CNN_Project
        2. Generate fake images with G(ξ)
        3. D predicts real vs fake
        4. Compute loss: L_D = CE(D(real), 1) + CE(D(fake), 0)
        5. Update D weights

        ┌─────────────────────────────────┐
        │ STEP B: Train Generator          │
        └─────────────────────────────────┘
        1. Generate fake images with G(ξ)
        2. D predicts on fake images
        3. Compute loss: L_G = CE(D(G(ξ)), 1)
        4. Update G weights
```

### Hyperparameters
```
Image Size:           128 × 128 × 1
Noise Dimension:      100
Batch Size:           64
Epochs:               100
Learning Rate:        0.0002
Optimizer:            Adam (β1=0.5, β2=0.999)
Loss Function:        BCEWithLogitsLoss
Image Normalization:  [-1, 1]
```

---

## 📊 Expected Results

### Training Time:
| Hardware | Time (100 epochs) |
|----------|------------------|
| RTX 3060 GPU | 30-45 min |
| GTX 1660 GPU | 45-60 min |
| CPU Only | 2-3 hours |

### Training Metrics:
- **Discriminator Loss:** Should stabilize around 0.5-1.0
- **Generator Loss:** Should stabilize around 0.5-2.0
- **D(real):** Should stay near 1.0 (correctly identifies real images)
- **D(fake):** Should approach 0.5 (generator fooling discriminator)

### Signs of Success:
✅ Losses stabilize (don't oscillate wildly)
✅ D(fake) approaches 0.5
✅ Generated images look like fruit shapes
✅ Diverse outputs (no mode collapse)

### Signs of Failure:
❌ Mode collapse: all images look identical
❌ Discriminator loss → 0 (too strong)
❌ Generator loss explodes
❌ Generated images are just noise

---

## 📁 Files Generated

```
GAN/
├── models/
│   ├── best_model.pth             # ⭐ BEST quality model (RECOMMENDED)
│   ├── latest_model.pth           # 🔄 Updated EVERY epoch (safe to Ctrl+C)
│   ├── final_gan.pth              # Final model after all epochs
│   ├── checkpoint_epoch_*.pth     # Every 10 epochs (for resume)
│   ├── models_index.json          # Summary of all models
│   ├── best_metadata.json         # Best model info
│   ├── latest_metadata.json       # Latest model info
│   └── final_metadata.json        # Final stats
├── generated_images/
│   ├── sample_000.png - 099.png   # Individual images
│   └── grid_64.png                # 8×8 grid view
├── training_progress/
│   └── epoch_*.png                # Progress images every 5 epochs
├── training_history.png           # Final loss curves
└── training_history_live.png      # Live updating loss curves
```

### ⚠️ Important: Interrupt-Safe Training!

**You can kill training at ANY TIME (Ctrl+C) and still use your models!**

- `best_model.pth` - Automatically saved when D(fake) is closest to 0.5
- `latest_model.pth` - Updated EVERY epoch, always usable
- `checkpoint_epoch_*.pth` - Saved every 10 epochs

All models are immediately usable with `generate_images.py` - no post-processing needed!

---

## 🖼️ Usage Examples

### Basic Training:
```bash
python GAN/train_gan.py
```

### Check Available Models:
```bash
# See what models are available and their status
python GAN/check_models.py
```

### Generate Images:
```bash
# Use best model (default, recommended)
python GAN/generate_images.py --num 50

# Generate and classify with CNN
python GAN/generate_images.py --num 100 --classify

# Use latest model (most recent epoch)
python GAN/generate_images.py --model GAN/models/latest_model.pth --num 50

# Use specific checkpoint
python GAN/generate_images.py --model GAN/models/checkpoint_epoch_050.pth --num 20
```

### If You Need to Stop Training Early:
```bash
# Just hit Ctrl+C - your models are safe!
# Then check what's available:
python GAN/check_models.py

# Use the best model found so far:
python GAN/generate_images.py --model GAN/models/best_model.pth --num 50
```

---

## 🔧 Advanced Options

### Custom Training Settings
Edit `train_gan.py` to modify:
```python
num_epochs = 100        # Line 268
batch_size = 64         # Line 241
lr = 0.0002            # Line 249
noise_dim = 100        # Line 223
```

### Resume Training
```python
# Load checkpoint and continue training
checkpoint = torch.load('GAN/models/checkpoint_epoch_050.pth')
generator.load_state_dict(checkpoint['generator_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
# Continue training...
```

---

## 🧪 Integration with CNN Classifier

The GAN automatically tests generated images with the CNN classifier:

```python
# After training completes
print("\n[STEP 8] Testing generated images with CNN classifier...")

# Load CNN model from CNN_Project
cnn_model = FruitCNN(num_classes=5)
cnn_model.load_state_dict(torch.load('CNN_Project/models/best_model.pth'))

# Generate images
fake_images = generator(noise)

# Classify them
predictions = cnn_model(fake_images)
# See what fruit classes the CNN thinks they are!
```

This helps evaluate if the GAN is generating realistic fruit images.

---

## 📊 Monitoring Training

### During Training:
```
Epoch 1/100
  D_loss: 1.2345 | G_loss: 2.3456
  D(real): 0.856 | D(fake): 0.234
  [SAVED] Generated images to GAN/training_progress/epoch_005.png
```

### What to Look For:
- **D(real)** near 1.0 → Discriminator correctly identifies real images
- **D(fake)** approaching 0.5 → Generator is fooling discriminator
- **Losses stabilizing** → Training is converging
- **Progress images improving** → Generator learning realistic features

---

## 🎓 Key Concepts

### Why GAN for Image Generation?
- **Unsupervised learning:** No need for labels
- **Learns distribution:** Captures the "essence" of fruit images
- **High quality:** Can generate realistic, diverse images
- **Data augmentation:** Can expand training datasets

### Generator vs Discriminator:
```
Generator (Artist):
  - Tries to create fake images that look real
  - Wants to fool the discriminator
  - Loss goes down when discriminator is fooled

Discriminator (Critic):
  - Tries to tell real from fake
  - Wants to correctly classify images
  - Loss goes down when it's accurate
```

### Why 128×128?
- Matches CNN_Project input requirements
- Large enough for recognizable features
- Small enough for fast training
- Good balance of quality and performance

---

## 💾 Model Saving Strategy

The GAN trainer saves models in **3 different ways** to ensure you never lose progress:

### 1. **Latest Model** (Every Epoch)
- **File:** `latest_model.pth`
- **Updated:** After EVERY single epoch
- **Use case:** If you stop training early, this is always available
- **Safe to:** Kill training anytime (Ctrl+C)

### 2. **Best Model** (Quality-Based)
- **File:** `best_model.pth`
- **Updated:** When D(fake) gets closest to 0.5
- **Use case:** Generates highest quality images
- **Metric:** D(fake) = 0.5 means generator perfectly fools discriminator

### 3. **Checkpoints** (Every 10 Epochs)
- **Files:** `checkpoint_epoch_010.pth`, `checkpoint_epoch_020.pth`, etc.
- **Updated:** Every 10 epochs
- **Use case:** Resume training or compare different epochs
- **Includes:** Full training state (optimizer, history, etc.)

### Metadata Files (Human-Readable)
Each model has a corresponding `.json` metadata file:
- `best_metadata.json` - Best model stats
- `latest_metadata.json` - Latest model stats
- `models_index.json` - Summary of ALL models

**Example metadata:**
```json
{
  "epoch": 45,
  "noise_dim": 100,
  "avg_D_loss": 0.6234,
  "avg_G_loss": 1.2345,
  "avg_D_fake": 0.523,
  "timestamp": "2025-01-18 14:30:45"
}
```

---

## 🐛 Troubleshooting

### "No images found in preprocessed_images"
```bash
# Run CNN training first
cd CNN_Project
python train_model.py
cd ..
```

### "CUDA out of memory"
```python
# Reduce batch size in train_gan.py
batch_size = 32  # Instead of 64
```

### Training is unstable
- Lower learning rate: `lr = 0.0001`
- Add gradient clipping
- Increase dropout in discriminator

### Mode collapse (all images look the same)
- Continue training longer
- Try different random seed
- Increase batch size
- Add noise to discriminator inputs

---

## 📞 Quick Commands Reference

```bash
# Train GAN
python GAN/train_gan.py

# Check available models (after training starts)
python GAN/check_models.py

# Generate images (uses best_model.pth by default)
python GAN/generate_images.py --num 50

# Generate and classify
python GAN/generate_images.py --num 100 --classify

# Use latest model (most recent checkpoint)
python GAN/generate_images.py --model GAN/models/latest_model.pth --num 50

# Use specific checkpoint
python GAN/generate_images.py --model GAN/models/checkpoint_epoch_050.pth --num 20

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# View model metadata (Windows)
type GAN\models\best_metadata.json

# View model metadata (Mac/Linux)
cat GAN/models/best_metadata.json
```

---

## 🌟 Why This Project Rocks

- 🎨 **Generates new images** - Creates fruit images that don't exist
- 🔗 **Integrated pipeline** - Works seamlessly with CNN_Project
- 🚀 **GPU accelerated** - Fast training with CUDA
- 📊 **Automatic evaluation** - Tests with CNN classifier
- 💾 **Regular checkpoints** - Save progress every 10 epochs
- 📈 **Visual feedback** - See progress every 5 epochs
- 🎯 **Production ready** - 128×128 format matches CNN input

---

## 🎓 Perfect for Assignment

This project demonstrates:
- ✅ GAN architecture design
- ✅ Generator and Discriminator implementation
- ✅ Adversarial training process
- ✅ Integration with existing models
- ✅ GPU optimization with PyTorch
- ✅ Image generation pipeline
- ✅ Model evaluation

---

## 📚 Architecture Details

See `GAN_ARCHITECTURE_20x20.md` for detailed architecture documentation.

The 128×128 version scales up from the 20×20 design:
- 4 upsampling stages instead of 3
- 4 downsampling stages instead of 2
- Higher channel counts (up to 512)
- More parameters (~11M total vs ~2.5M)

---

**Ready to generate fruit images?** 🍎🍌🍊

```bash
python GAN/train_gan.py
```

Then:

```bash
python GAN/generate_images.py --num 50 --classify
```

**Your synthetic fruit images will be ready in ~45 minutes!** 🎉

---

*CST-435 Neural Networks Assignment - PyTorch GAN Edition*
