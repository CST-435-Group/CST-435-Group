# Single Fruit GAN Training Guide

## 🎯 What This Does

Instead of training on ALL fruits mixed together (which confuses the GAN), this trains on **ONE specific fruit type at a time**.

**Much better results!** The GAN learns what an Apple looks like, not a mix of Apple/Banana/Peach/etc.

---

## 🍎 Available Fruits

Based on your CNN_Project training, you can generate:

1. **Pear**
2. **Apple**
3. **Tomato**
4. **Peach**
5. **Cucumber**

---

## 🚀 Quick Start

### Train Apple GAN:
```bash
python train_gan_single_fruit.py --fruit Apple --epochs 50
```

### Train Pear GAN:
```bash
python train_gan_single_fruit.py --fruit Pear --epochs 50
```

### All Options:
```bash
python train_gan_single_fruit.py --fruit <FRUIT> --epochs <NUM> --batch-size <SIZE> --lr <RATE>
```

**Parameters:**
- `--fruit`: **REQUIRED** - Choose: Pear, Apple, Tomato, Peach, or Cucumber
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.0002)

---

## 📂 File Organization

Each fruit gets its own directories:

```
GAN/
├── models/
│   ├── Apple/
│   │   ├── best_model.pth
│   │   ├── latest_model.pth
│   │   └── checkpoint_*.pth
│   ├── Pear/
│   │   ├── best_model.pth
│   │   └── ...
│   └── ... (one folder per fruit)
├── generated_images/
│   ├── Apple/
│   │   ├── sample_000.png - 099.png
│   │   └── grid_64.png
│   ├── Pear/
│   └── ...
└── training_progress/
    ├── Apple/
    │   └── epoch_*.png
    └── ...
```

---

## 🎓 How Evaluation Works

### During Training:
1. GAN trains ONLY on the fruit you specified (e.g., only Apple images)
2. Models are saved every epoch
3. Best model tracked (when D(fake) ≈ 0.5)

### After Training (CNN Evaluation):
1. Generates 50 test images
2. Feeds them to CNN classifier
3. Counts how many CNN thinks are the target fruit
4. Reports accuracy

**Example:**
```
Apple Generation Accuracy:
  42/50 images classified as Apple (84.0%)

Sample predictions:
  Image 1: Apple ✅
  Image 2: Apple ✅
  Image 3: Pear ❌  # GAN generated something that looks more like pear
  Image 4: Apple ✅
  ...
```

---

## 📊 Training Each Fruit

### 1. Train Apple GAN
```bash
python train_gan_single_fruit.py --fruit Apple --epochs 50
```

**Expected:**
- Training time: ~40-50 min (GPU) or ~4-5 hours (CPU)
- Uses ~1,000-2,000 Apple images
- Generates realistic apple shapes
- CNN accuracy: 70-90%

### 2. Train Pear GAN
```bash
python train_gan_single_fruit.py --fruit Pear --epochs 50
```

### 3. Train Tomato GAN
```bash
python train_gan_single_fruit.py --fruit Tomato --epochs 50
```

### 4. Train Peach GAN
```bash
python train_gan_single_fruit.py --fruit Peach --epochs 50
```

### 5. Train Cucumber GAN
```bash
python train_gan_single_fruit.py --fruit Cucumber --epochs 50
```

---

## 💡 Tips for Best Results

### Recommended Epochs by Fruit:
- **Apple**: 50-75 epochs (round shape, easy)
- **Pear**: 75-100 epochs (distinct shape, medium)
- **Tomato**: 50-75 epochs (round like apple)
- **Peach**: 75-100 epochs (fuzzy texture, harder)
- **Cucumber**: 100+ epochs (elongated, unique shape)

### Faster Training (Testing):
```bash
# Quick test with 25 epochs
python train_gan_single_fruit.py --fruit Apple --epochs 25
```

### Higher Quality (More Time):
```bash
# Train longer for better results
python train_gan_single_fruit.py --fruit Pear --epochs 150
```

### Smaller Batch for Limited VRAM:
```bash
# If you get CUDA out of memory errors
python train_gan_single_fruit.py --fruit Apple --epochs 50 --batch-size 32
```

---

## 🔬 Comparing Results

### Train All 5 Fruits:
```bash
# Terminal 1
python train_gan_single_fruit.py --fruit Apple --epochs 50

# Terminal 2
python train_gan_single_fruit.py --fruit Pear --epochs 50

# Terminal 3
python train_gan_single_fruit.py --fruit Tomato --epochs 50

# Terminal 4
python train_gan_single_fruit.py --fruit Peach --epochs 50

# Terminal 5
python train_gan_single_fruit.py --fruit Cucumber --epochs 50
```

**Or run sequentially:**
```bash
python train_gan_single_fruit.py --fruit Apple --epochs 50
python train_gan_single_fruit.py --fruit Pear --epochs 50
python train_gan_single_fruit.py --fruit Tomato --epochs 50
python train_gan_single_fruit.py --fruit Peach --epochs 50
python train_gan_single_fruit.py --fruit Cucumber --epochs 50
```

---

## 📈 Monitoring Training

### What You'll See:
```
Epoch 25/50 (50.0% complete)
  D_loss: 0.8234 | G_loss: 1.2345
  D(real): 0.756 | D(fake): 0.423
  Epoch Time: 42.3s | Avg: 43.1s/epoch
  Elapsed: 0:17:57 | ETA: 0:17:58
  [BEST] New best model! D(fake)=0.423 (distance: 0.077)
  [SAVED] Progress images to GAN/training_progress/Apple/epoch_025.png
```

### Good Signs:
- ✅ D(fake) moving toward 0.5
- ✅ D(real) staying around 0.7-0.9
- ✅ Losses stabilizing (not wildly oscillating)
- ✅ Progress images showing fruit-like shapes

### Warning Signs:
- ❌ D(fake) stuck near 0 or 1
- ❌ D_loss going to 0 (discriminator too strong)
- ❌ G_loss exploding
- ❌ Progress images still look like noise after 25+ epochs

---

## 🎯 CNN Evaluation Scores

### What's Good:
- **80-100%**: Excellent! GAN generates very realistic fruits
- **60-80%**: Good - recognizable as target fruit
- **40-60%**: Okay - some fruit-like features
- **<40%**: Needs more training or parameter tuning

### Example Results:
```
Apple Generation Accuracy:
  45/50 images classified as Apple (90.0%)  ← Excellent!

Pear Generation Accuracy:
  32/50 images classified as Pear (64.0%)   ← Good

Cucumber Generation Accuracy:
  18/50 images classified as Cucumber (36.0%)  ← Needs more epochs
```

---

## 🔧 Troubleshooting

### "No [Fruit] images found!"
**Solution:** Check that CNN_Project was trained with that fruit. View available fruits:
```bash
dir "C:\Users\Soren\OneDrive\Documents\school\College Senior\CST-405\Compiler\CST-435-Group\CNN_Project\preprocessed_images"
```

### D(fake) stuck near 0
**Solution:** Discriminator is too strong. Try:
- Lower discriminator learning rate: `--lr 0.0001`
- Train more epochs to let generator catch up

### D(fake) stuck near 1
**Solution:** Generator is too strong. Try:
- Increase discriminator learning rate slightly
- Check if you have enough training images

### CUDA out of memory
**Solution:** Reduce batch size:
```bash
python train_gan_single_fruit.py --fruit Apple --epochs 50 --batch-size 32
```

---

## 📊 Expected Training Times

| Hardware | Time per Epoch | 50 Epochs | 100 Epochs |
|----------|---------------|-----------|------------|
| RTX 3050 Ti | ~40-45s | ~33-38 min | ~67-75 min |
| RTX 3060 | ~35-40s | ~29-33 min | ~58-67 min |
| CPU Only | ~3-5 min | ~2.5-4 hours | ~5-8 hours |

**Note:** Times vary based on number of fruit images available.

---

## 📁 Files Generated

After training Apple GAN for 50 epochs:

```
GAN/models/Apple/
├── best_model.pth              # Best quality model
├── best_metadata.json          # Best model stats
├── latest_model.pth            # Most recent epoch
├── latest_metadata.json        # Latest stats
├── checkpoint_epoch_010.pth    # Every 10 epochs
├── checkpoint_epoch_020.pth
├── ...
├── checkpoint_epoch_050.pth
└── training_history_live.png   # Loss curves

GAN/generated_images/Apple/
├── sample_000.png - 099.png    # 100 generated apples
└── grid_64.png                 # 8×8 grid view

GAN/training_progress/Apple/
├── epoch_005.png               # Progress every 5 epochs
├── epoch_010.png
├── ...
└── epoch_050.png
```

---

## 🎨 Generating More Images

After training, generate new images:

```bash
# Using the generate script
python generate_images.py --model GAN/models/Apple/best_model.pth --num 50 --classify
```

The `--classify` flag will use the CNN to evaluate if they look like the target fruit!

---

## 🏆 Best Practices

1. **Start with one fruit** - Test with Apple (easiest)
2. **Train 50 epochs minimum** - Less may not converge
3. **Check progress images** - View every 5 epochs
4. **Monitor D(fake)** - Should approach 0.5
5. **Use best_model.pth** - It's automatically selected for quality
6. **Evaluate with CNN** - See if generated images are convincing

---

## 🎯 Assignment Use

Perfect for demonstrating:
- ✅ Focused GAN training (one task at a time)
- ✅ Clear evaluation metrics (CNN classifier accuracy)
- ✅ Comparable results across different fruits
- ✅ Quality tracking (best model selection)
- ✅ Visual progress (generated image grids)

---

**Ready to generate some fruit!** 🍎🍐🍅🍑🥒

```bash
python train_gan_single_fruit.py --fruit Apple --epochs 50
```
