# GAN Architecture for 20x20 Black & White Images

## Overview
This document describes the complete architecture of a Generative Adversarial Network (GAN) designed to generate 20x20 pixel black and white images.

---

## Complete GAN Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           GAN TRAINING ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────────┘

INPUT: Random Noise Vector ξ ~ N(0,1)                    INPUT: Real Images X_real
           │                                                        │
           │ [Batch × 100]                                         │ [Batch × 20 × 20 × 1]
           ▼                                                        ▼
    ┌──────────────┐                                        ┌──────────────┐
    │  GENERATOR   │                                        │ REAL DATA    │
    │      G       │                                        │ SAMPLES      │
    └──────────────┘                                        └──────────────┘
           │                                                        │
           │ [Batch × 20 × 20 × 1]                                 │
           │ X_fake                                                │
           ▼                                                        ▼
    ┌──────────────┐                                        ┌──────────────┐
    │ DISCRIMINATOR│◄───────────────────────────────────────│ DISCRIMINATOR│
    │      D       │         (Same Weights θ_D)             │      D       │
    │              │                                        │              │
    └──────────────┘                                        └──────────────┘
           │                                                        │
           │ Ŷ_fake                                                │ Ŷ_real
           │ [Batch × 1]                                           │ [Batch × 1]
           ▼                                                        ▼
    ┌──────────────┐                                        ┌──────────────┐
    │   STEP A:    │                                        │   STEP B:    │
    │  Update θ_G  │                                        │  Update θ_D  │
    └──────────────┘                                        └──────────────┘
           │                                                        │
           │ L_G = CE(Ŷ_fake, 1)                                   │ L_D = CE(Ŷ_real, 1)
           │                                                        │     + CE(Ŷ_fake, 0)
           ▼                                                        ▼
    Backprop through G                                      Backprop through D
    (Discriminator frozen)                                  (Generator frozen)
```

---

## Detailed Generator Architecture (G)

### Purpose
Transform random noise into realistic 20x20 black and white images

### Layer-by-Layer Breakdown

```
INPUT: Noise Vector ξ
Shape: [Batch, 100]
Distribution: Normal(0, 1)
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 1: Dense (Fully Connected)               │
│ ─────────────────────────────────────────────  │
│ Units: 6400 (5 × 5 × 256)                     │
│ Activation: None                               │
│ Use Bias: False                                │
│ Output Shape: [Batch, 6400]                    │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 2: Batch Normalization                   │
│ ─────────────────────────────────────────────  │
│ Normalizes activations                         │
│ Output Shape: [Batch, 6400]                    │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 3: LeakyReLU                             │
│ ─────────────────────────────────────────────  │
│ Alpha: 0.2                                     │
│ Output Shape: [Batch, 6400]                    │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 4: Reshape                               │
│ ─────────────────────────────────────────────  │
│ Target Shape: (5, 5, 256)                      │
│ Output Shape: [Batch, 5, 5, 256]               │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 5: Conv2DTranspose (Upsampling)          │
│ ─────────────────────────────────────────────  │
│ Filters: 128                                   │
│ Kernel Size: (5, 5)                            │
│ Strides: (2, 2)                                │
│ Padding: 'same'                                │
│ Use Bias: False                                │
│ Output Shape: [Batch, 10, 10, 128]             │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 6: Batch Normalization                   │
│ ─────────────────────────────────────────────  │
│ Output Shape: [Batch, 10, 10, 128]             │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 7: LeakyReLU                             │
│ ─────────────────────────────────────────────  │
│ Alpha: 0.2                                     │
│ Output Shape: [Batch, 10, 10, 128]             │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 8: Conv2DTranspose (Upsampling)          │
│ ─────────────────────────────────────────────  │
│ Filters: 64                                    │
│ Kernel Size: (5, 5)                            │
│ Strides: (2, 2)                                │
│ Padding: 'same'                                │
│ Use Bias: False                                │
│ Output Shape: [Batch, 20, 20, 64]              │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 9: Batch Normalization                   │
│ ─────────────────────────────────────────────  │
│ Output Shape: [Batch, 20, 20, 64]              │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 10: LeakyReLU                            │
│ ─────────────────────────────────────────────  │
│ Alpha: 0.2                                     │
│ Output Shape: [Batch, 20, 20, 64]              │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 11: Conv2DTranspose (Output Layer)       │
│ ─────────────────────────────────────────────  │
│ Filters: 1 (grayscale)                         │
│ Kernel Size: (5, 5)                            │
│ Strides: (1, 1)                                │
│ Padding: 'same'                                │
│ Use Bias: False                                │
│ Activation: tanh                               │
│ Output Shape: [Batch, 20, 20, 1]               │
└────────────────────────────────────────────────┘
                │
                ▼
         OUTPUT: X_fake
         Shape: [Batch, 20, 20, 1]
         Value Range: [-1, 1]
```

### Generator Parameters
- Total Layers: 11
- Trainable Parameters: ~2.1M
- Key Features:
  - Progressive upsampling: 5×5 → 10×10 → 20×20
  - Batch normalization for training stability
  - LeakyReLU for non-linearity
  - Tanh output for normalized pixel values

---

## Detailed Discriminator Architecture (D)

### Purpose
Classify images as real (from dataset) or fake (from generator)

### Layer-by-Layer Breakdown

```
INPUT: Image X
Shape: [Batch, 20, 20, 1]
Source: Real data OR Generator output
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 1: Conv2D (Downsampling)                 │
│ ─────────────────────────────────────────────  │
│ Filters: 64                                    │
│ Kernel Size: (5, 5)                            │
│ Strides: (2, 2)                                │
│ Padding: 'same'                                │
│ Input Shape: [Batch, 20, 20, 1]                │
│ Output Shape: [Batch, 10, 10, 64]              │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 2: LeakyReLU                             │
│ ─────────────────────────────────────────────  │
│ Alpha: 0.2                                     │
│ Output Shape: [Batch, 10, 10, 64]              │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 3: Dropout                               │
│ ─────────────────────────────────────────────  │
│ Rate: 0.3 (30% dropout)                        │
│ Output Shape: [Batch, 10, 10, 64]              │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 4: Conv2D (Downsampling)                 │
│ ─────────────────────────────────────────────  │
│ Filters: 128                                   │
│ Kernel Size: (5, 5)                            │
│ Strides: (2, 2)                                │
│ Padding: 'same'                                │
│ Output Shape: [Batch, 5, 5, 128]               │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 5: LeakyReLU                             │
│ ─────────────────────────────────────────────  │
│ Alpha: 0.2                                     │
│ Output Shape: [Batch, 5, 5, 128]               │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 6: Dropout                               │
│ ─────────────────────────────────────────────  │
│ Rate: 0.3 (30% dropout)                        │
│ Output Shape: [Batch, 5, 5, 128]               │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 7: Flatten                               │
│ ─────────────────────────────────────────────  │
│ Output Shape: [Batch, 3200]                    │
│ (5 × 5 × 128 = 3200)                           │
└────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────┐
│ Layer 8: Dense (Output Layer)                  │
│ ─────────────────────────────────────────────  │
│ Units: 1                                       │
│ Activation: None (logits)                      │
│ Output Shape: [Batch, 1]                       │
└────────────────────────────────────────────────┘
                │
                ▼
         OUTPUT: Ŷ (logits)
         Shape: [Batch, 1]
         Interpretation:
           > 0 → classified as REAL
           < 0 → classified as FAKE
```

### Discriminator Parameters
- Total Layers: 8
- Trainable Parameters: ~410K
- Key Features:
  - Progressive downsampling: 20×20 → 10×10 → 5×5
  - Dropout for regularization
  - LeakyReLU for non-linearity
  - Single output (logit) for binary classification

---

## Loss Functions

### Generator Loss (L_G)

```
Objective: Fool the discriminator into classifying fake images as real

L_G = BinaryCrossEntropy(Ŷ_fake, 1)

Where:
- Ŷ_fake: Discriminator's output for generated images
- Target: 1 (we want discriminator to think they're real)

Mathematical Form:
L_G = -log(sigmoid(Ŷ_fake))
    = -log(D(G(ξ)))
```

### Discriminator Loss (L_D)

```
Objective: Correctly classify real images as real and fake images as fake

L_D = BinaryCrossEntropy(Ŷ_real, 1) + BinaryCrossEntropy(Ŷ_fake, 0)

Where:
- Ŷ_real: Discriminator's output for real images
- Ŷ_fake: Discriminator's output for generated images
- Targets: 1 for real, 0 for fake

Mathematical Form:
L_D = -log(sigmoid(Ŷ_real)) - log(1 - sigmoid(Ŷ_fake))
    = -log(D(X_real)) - log(1 - D(G(ξ)))
```

---

## Training Process

### Two-Step Alternating Training

```
FOR each epoch:
    FOR each batch of real images:

        ┌─────────────────────────────────────────────┐
        │ STEP A: Train Generator (Update θ_G)       │
        └─────────────────────────────────────────────┘

        1. Generate noise: ξ ~ N(0, 1)
        2. Generate fake images: X_fake = G(ξ)
        3. Get discriminator predictions: Ŷ_fake = D(X_fake)
        4. Compute generator loss: L_G = CE(Ŷ_fake, 1)
        5. Compute gradients: ∇θ_G L_G
        6. Update generator weights: θ_G ← θ_G - α · ∇θ_G L_G

        Note: Discriminator weights (θ_D) are FROZEN

        ┌─────────────────────────────────────────────┐
        │ STEP B: Train Discriminator (Update θ_D)   │
        └─────────────────────────────────────────────┘

        1. Use same X_fake from Step A
        2. Get discriminator predictions on fake: Ŷ_fake = D(X_fake)
        3. Get discriminator predictions on real: Ŷ_real = D(X_real)
        4. Compute discriminator loss: L_D = CE(Ŷ_real, 1) + CE(Ŷ_fake, 0)
        5. Compute gradients: ∇θ_D L_D
        6. Update discriminator weights: θ_D ← θ_D - α · ∇θ_D L_D

        Note: Generator weights (θ_G) are FROZEN
```

---

## Hyperparameters

```
┌─────────────────────────────────────────────────────┐
│ Training Configuration                              │
├─────────────────────────────────────────────────────┤
│ Image Size:           20 × 20 × 1                   │
│ Noise Dimension:      100                           │
│ Batch Size:           256                           │
│ Epochs:               50-200                        │
│ Learning Rate (G):    1e-4 (0.0001)                 │
│ Learning Rate (D):    1e-4 (0.0001)                 │
│ Optimizer:            Adam                          │
│ Beta1 (Adam):         0.5                           │
│ Beta2 (Adam):         0.999                         │
│ Image Normalization:  [-1, 1]                       │
│                       (x - 127.5) / 127.5           │
└─────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

### Forward Pass

```
Training Step A (Generator):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Noise ξ [100]
    ↓
Generator G [θ_G]
    ↓
Fake Images X_fake [20×20×1]
    ↓
Discriminator D [θ_D frozen]
    ↓
Predictions Ŷ_fake [1]
    ↓
Loss L_G = CE(Ŷ_fake, 1)
    ↓
Gradients ∇θ_G
    ↓
Update θ_G


Training Step B (Discriminator):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Branch 1 (Fake):              Branch 2 (Real):
X_fake [20×20×1]              X_real [20×20×1]
    ↓                             ↓
Discriminator D [θ_D]         Discriminator D [θ_D]
    ↓                             ↓
Ŷ_fake [1]                    Ŷ_real [1]
    ↓                             ↓
CE(Ŷ_fake, 0) ────────┬──────── CE(Ŷ_real, 1)
                      ↓
              Loss L_D = Sum
                      ↓
              Gradients ∇θ_D
                      ↓
              Update θ_D

Note: Generator weights (θ_G) are FROZEN
```

---

## Architecture Comparison: 28×28 vs 20×20

### Generator Differences

| Layer          | 28×28 MNIST      | 20×20 Custom    |
|----------------|------------------|-----------------|
| Initial Dense  | 7×7×256 = 12,544 | 5×5×256 = 6,400 |
| Initial Shape  | (7, 7, 256)      | (5, 5, 256)     |
| Upsample 1     | 7×7 → 7×7        | 5×5 → 10×10     |
| Upsample 2     | 7×7 → 14×14      | 10×10 → 20×20   |
| Upsample 3     | 14×14 → 28×28    | (Not needed)    |
| Final Output   | (28, 28, 1)      | (20, 20, 1)     |

### Discriminator Differences

| Layer          | 28×28 MNIST      | 20×20 Custom    |
|----------------|------------------|-----------------|
| Input          | (28, 28, 1)      | (20, 20, 1)     |
| Downsample 1   | 28×28 → 14×14    | 20×20 → 10×10   |
| Downsample 2   | 14×14 → 7×7      | 10×10 → 5×5     |
| Flatten Size   | 7×7×128 = 6,272  | 5×5×128 = 3,200 |
| Output         | (1,)             | (1,)            |

---

## Key Design Decisions

### 1. Why Conv2DTranspose in Generator?
- Learns upsampling patterns specific to target domain
- Better than simple interpolation
- Creates checkerboard patterns that can be learned away

### 2. Why LeakyReLU instead of ReLU?
- Prevents "dying ReLU" problem
- Allows small negative gradients to flow
- Typical alpha = 0.2

### 3. Why Batch Normalization in Generator?
- Stabilizes training
- Helps gradients flow
- Prevents mode collapse

### 4. Why Dropout in Discriminator?
- Prevents overfitting
- Makes discriminator slightly "worse" so generator can catch up
- Typical rate = 0.3

### 5. Why Tanh activation on Generator output?
- Outputs in range [-1, 1]
- Matches normalized image data
- Centers data around 0

### 6. Why no activation on Discriminator output?
- We use from_logits=True in loss function
- More numerically stable
- Prevents saturation

---

## Expected Training Behavior

### Early Epochs (1-10)
- Generator produces random noise-like images
- Discriminator easily distinguishes real from fake
- High discriminator accuracy (>90%)
- High generator loss

### Middle Epochs (10-30)
- Generator starts producing blob-like shapes
- Discriminator accuracy drops to 60-70%
- Generator and discriminator losses stabilize
- Mode collapse may occur (generator produces similar images)

### Late Epochs (30-50+)
- Generator produces recognizable patterns
- Discriminator struggles (accuracy ~50-60%)
- Losses oscillate but remain stable
- Generated images should look realistic

### Signs of Success
- Discriminator accuracy approaches 50% (can't tell real from fake)
- Generated images are diverse (no mode collapse)
- Images contain coherent structures
- Losses don't diverge

### Signs of Failure
- Mode collapse: all generated images look identical
- Discriminator loss goes to 0 (too strong)
- Generator loss explodes (can't fool discriminator)
- Oscillating losses without convergence

---

## Summary

This GAN architecture is specifically designed for 20×20 black and white images with:

- **Generator**: 11 layers, progressive upsampling from 5×5 to 20×20
- **Discriminator**: 8 layers, progressive downsampling from 20×20 to 5×5
- **Training**: Alternating optimization of G and D
- **Loss**: Binary cross-entropy with logits
- **Optimizer**: Adam with learning rate 1e-4

The architecture follows the same principles as the standard DCGAN (Deep Convolutional GAN) but is scaled appropriately for the smaller 20×20 image size.
