# GAN Training Stability Guide

## Problem: Gray Images / Mode Collapse

When your GAN starts generating gray/uniform images and the generator loss explodes, it means the **discriminator has become too strong**.

### Symptoms:
- D_loss approaches 0.0000
- G_loss explodes (20, 50, 100+)
- D(real) = 1.000, D(fake) = 0.000
- Generated images become gray or uniform

### What's Happening:
The discriminator becomes perfect at distinguishing real from fake. When this happens:
1. Generator receives no useful gradient signal
2. Generator can't learn to improve
3. Generator "gives up" and outputs gray mush

## Solutions Implemented

### 1. Lower Discriminator Learning Rate ✅
```python
LR_G = 0.0002  # Generator learning rate
LR_D = 0.0001  # Discriminator learning rate (HALF of generator)
```
**Why:** Slows down discriminator training so generator can keep up

### 2. Label Smoothing ✅
```python
LABEL_SMOOTHING = 0.1  # Real labels = 0.9 instead of 1.0
```
**Why:** Prevents discriminator from becoming overconfident

### 3. Add Noise to Inputs ✅
```python
NOISE_STD = 0.05  # Add small noise to real and fake images
```
**Why:** Prevents discriminator from memorizing training data

### 4. Train Discriminator Less Frequently (Optional)
```python
D_TRAIN_RATIO = 1  # Train discriminator every N batches (1 = every batch)
```
**Why:** Gives generator more chances to catch up. Try D_TRAIN_RATIO = 2 if still having issues.

## Monitoring Training

Watch these metrics each epoch:

### Healthy Training:
```
D_loss: 0.5-1.5 | G_loss: 2-15
D(real): 0.7-0.9 | D(fake): 0.1-0.3
```
Both models are learning. Generator is making progress.

### Discriminator Too Strong:
```
D_loss: 0.0-0.1 | G_loss: 20-100
D(real): 0.95-1.0 | D(fake): 0.0-0.05
⚠️ WARNING: Discriminator too strong!
```
Discriminator is winning. You'll see warnings and gray images.

### Generator Too Strong (Rare):
```
D_loss: 2.0+ | G_loss: 0.5-2.0
D(real): 0.4-0.6 | D(fake): 0.4-0.6
```
Generator is fooling discriminator too easily. Increase LR_D or D_TRAIN_RATIO.

## Tuning Guide

If you still get gray images after epoch 20-30:

### Step 1: Lower Discriminator Learning Rate More
```python
LR_D = 0.00005  # Even slower
```

### Step 2: Increase Label Smoothing
```python
LABEL_SMOOTHING = 0.2  # More smoothing
```

### Step 3: Train Discriminator Less Often
```python
D_TRAIN_RATIO = 2  # Train D every 2 batches instead of every batch
```

### Step 4: Add More Noise
```python
NOISE_STD = 0.1  # More noise to inputs
```

## When to Stop Training

1. **Early Stop if Collapsed:** If images are still gray after 50 epochs with fixes, restart with lower LR_D
2. **Good Results:** Stop when generated images look realistic (usually 100-150 epochs)
3. **Oscillation:** If quality goes up and down, that's normal - pick the best checkpoint

## Restarting from Checkpoint

If your training collapsed at epoch 90, you can:

1. **Lower LR_D** in the code
2. **Load from epoch 40** (before it collapsed):
   ```python
   # In train_gan_dual_conditional.py, add at start:
   generator.load_state_dict(torch.load('GAN/models_dual_conditional/generator_epoch_040.pth'))
   discriminator.load_state_dict(torch.load('GAN/models_dual_conditional/discriminator_epoch_040.pth'))
   ```
3. Continue training with new settings

## Architecture Changes (Advanced)

If tuning doesn't help:

1. **Add Dropout to Discriminator** (makes it weaker)
2. **Add Spectral Normalization** (limits discriminator power)
3. **Use Wasserstein Loss** (more stable than BCE)
4. **Try Progressive Training** (start with smaller images)

These require modifying `models_dual_conditional.py`.

## Summary

**Most important fix:** Lower LR_D to 0.0001 or 0.00005

**Second most important:** Label smoothing to 0.1-0.2

**Third:** Add noise to discriminator inputs

The goal is to keep the discriminator and generator **balanced** - neither too strong nor too weak.
