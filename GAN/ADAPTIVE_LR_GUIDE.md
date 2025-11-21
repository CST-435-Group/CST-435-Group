# Adaptive Learning Rate Guide

## How It Works

The adaptive learning rate system automatically adjusts the discriminator's learning rate to keep it balanced with the generator.

### Target Metrics

```python
TARGET_D_REAL = 0.85  # Discriminator should classify ~85% of real images correctly
TARGET_D_FAKE = 0.15  # Discriminator should mistakenly classify ~15% of fake images as real
```

**Why these targets?**
- D(real) = 0.85 means discriminator isn't perfect (not overfitting)
- D(fake) = 0.15 means generator is learning to fool discriminator
- This keeps both networks learning and improving

### Adjustment Logic

After each epoch, the system checks:

#### Discriminator Too Strong
```
If D(real) > 0.90 AND D(fake) < 0.10:
  → Lower LR_D by 5% (multiply by 0.95)
  → Slows down discriminator training
  → Gives generator a chance to catch up
```

**Example:**
```
Epoch 33: D(real)=0.896, D(fake)=0.004
📉 Lowered LR_D to 0.000095 (discriminator too strong)
```

#### Discriminator Too Weak
```
If D(real) < 0.75 AND D(fake) > 0.25:
  → Raise LR_D by 5% (divide by 0.95)
  → Speeds up discriminator training
  → Prevents generator from dominating
```

**Example:**
```
Epoch 50: D(real)=0.72, D(fake)=0.30
📈 Raised LR_D to 0.000112 (discriminator too weak)
```

#### Balanced
```
If metrics are within acceptable range:
  → Keep LR_D unchanged
  → Both networks learning well
```

**Example:**
```
Epoch 25: D(real)=0.87, D(fake)=0.13
✅ LR_D stable at 0.000100 (balanced)
```

## What You'll See During Training

### Healthy Training (Adaptive LR Working)
```
Epoch 20/200
  D_loss: 0.8432 | G_loss: 5.2341
  D(real): 0.862 | D(fake): 0.142
  Time: 285.3s
  ✅ LR_D stable at 0.000100 (balanced)
```
Both networks are balanced. No adjustment needed.

### Discriminator Getting Strong (System Responds)
```
Epoch 33/200
  D_loss: 0.3381 | G_loss: 4.8045
  D(real): 0.896 | D(fake): 0.004
  Time: 275.5s
  📉 Lowered LR_D to 0.000095 (discriminator too strong)
```
System detects discriminator dominance and slows it down.

### After Multiple Adjustments
```
Epoch 45/200
  D_loss: 0.7821 | G_loss: 6.1234
  D(real): 0.841 | D(fake): 0.168
  Time: 290.2s
  ✅ LR_D stable at 0.000073 (balanced)
```
LR_D has been reduced over time to maintain balance.

## Configuration

### Default Settings (Recommended)
```python
ADAPTIVE_LR = True
TARGET_D_REAL = 0.85
TARGET_D_FAKE = 0.15
LR_ADJUSTMENT_RATE = 0.95  # 5% change per adjustment
MIN_LR_D = 0.00001  # Won't go below this
MAX_LR_D = 0.0003   # Won't go above this
```

### If Discriminator Still Too Strong

Make targets more lenient:
```python
TARGET_D_REAL = 0.80  # Lower target (discriminator can be less accurate)
TARGET_D_FAKE = 0.20  # Higher target (more false positives OK)
```

### If Adjustments Too Aggressive

Slow down adjustment rate:
```python
LR_ADJUSTMENT_RATE = 0.98  # 2% change instead of 5%
```

### If Adjustments Too Slow

Speed up adjustment rate:
```python
LR_ADJUSTMENT_RATE = 0.90  # 10% change per adjustment
```

## Monitoring

### Good Signs ✅
- Frequent "✅ LR_D stable" messages
- D(real) between 0.80-0.90
- D(fake) between 0.10-0.20
- G_loss stays below 15
- Generated images improve over time

### Warning Signs ⚠️
- Constant "📉 Lowered LR_D" every epoch → Lower MIN_LR_D or adjust targets
- LR_D hits MIN_LR_D and stays there → Discriminator architecture too strong
- G_loss still exploding → May need architectural changes

## Advantages Over Fixed LR

### Fixed LR Problems:
- Discriminator eventually dominates
- Gray images appear
- Must manually tune and restart training

### Adaptive LR Benefits:
- ✅ Automatically prevents discriminator dominance
- ✅ Self-correcting throughout training
- ✅ No manual intervention needed
- ✅ Better image quality maintained
- ✅ Training can run full 200 epochs

## Advanced: Learning Rate History

To track LR changes over training, the current LR is printed each epoch:
```
Epoch 1: LR_D = 0.000100
Epoch 20: LR_D = 0.000095
Epoch 40: LR_D = 0.000081
Epoch 60: LR_D = 0.000073
...
```

You can see how the system adapts to keep models balanced!

## Turning It Off

If you want to test without adaptive LR:
```python
ADAPTIVE_LR = False  # Use fixed LR throughout training
```

But the adaptive system is recommended for best results!
