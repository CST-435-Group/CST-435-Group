# 3-Tank Conditional GAN Workflow
## Leopard 2, M1 Abrams, T-90M

Perfect for your curated War Thunder footage!

---

## Your Setup

**Three tank classes:**
1. **Leopard 2** (German main battle tank)
2. **M1 Abrams** (American main battle tank)
3. **T-90M** (Russian main battle tank)

**Current status:** Frames extracted and curated in `GAN/datasets/military_vehicles_raw/`

---

## Complete Workflow

### Step 1: Organize Frames by Tank Type

You have two options:

#### Option A: Batch Organization (FASTER - Recommended)
Use this if you have consecutive frames of the same tank (e.g., frames 0-200 are all Leopard 2)

```bash
python GAN/organize_by_tank_type.py
# Choose option 2 (Batch)
```

**Interactive prompts:**
```
Start frame number: 0
End frame number: 200
Which tank? [1] Leopard 2
✓ Copied 201 frames to leopard2/

Start frame number: 201
End frame number: 450
Which tank? [2] M1 Abrams
✓ Copied 250 frames to m1abrams/

Start frame number: 451
End frame number: 700
Which tank? [3] T-90M
✓ Copied 250 frames to t90m/
```

#### Option B: One-by-One (More Accurate)
Use this if frames are mixed

```bash
python GAN/organize_by_tank_type.py
# Choose option 1 (One-by-one)
```

Each frame shows, you press:
- `1` = Leopard 2
- `2` = M1 Abrams
- `3` = T-90M
- `s` = Skip bad frame

**Result:** Frames organized into:
```
GAN/datasets/military_vehicles_organized/
  leopard2/
    leopard2_00000.png
    leopard2_00001.png
    ...
  m1abrams/
    m1abrams_00000.png
    m1abrams_00001.png
    ...
  t90m/
    t90m_00000.png
    t90m_00001.png
    ...
```

---

### Step 2: Preprocess Organized Frames

```bash
python GAN/preprocess_dataset.py
```

**When prompted:**
```
Available datasets:
  1. military_vehicles_organized (XXX images)

Select dataset: 1
Proceed? (y/n): y
```

**What happens:**
- Smart resize/crop to 200x200
- Filters bad images
- Auto-detects 3 classes from folders

**Output:**
```
Dataset Statistics:
  Total images: 650
  Number of classes: 3

Class distribution:
  leopard2: 220 images (33.8%)
  m1abrams: 215 images (33.1%)
  t90m: 215 images (33.1%)

Ready for training: GAN/datasets/military_vehicles_organized_processed/
```

**Balanced dataset = better results!**

---

### Step 3: Train Conditional GAN

```bash
python GAN/train_gan_conditional.py
```

**Update the dataset path in the script first:**

Open `train_gan_conditional.py` and change line 40:
```python
DATASET_PATH = 'GAN/datasets/military_vehicles_organized_processed'
```

**Training output:**
```
CONDITIONAL GAN TRAINING
Device: cuda

[STEP 1] Loading dataset...
Found 650 images

Dataset Statistics:
  Total images: 650
  Number of classes: 3

Class distribution:
  leopard2: 220 images (33.8%)
  m1abrams: 215 images (33.1%)
  t90m: 215 images (33.1%)

CLASS LABELS: leopard2, m1abrams, t90m

[STEP 2] Initializing Conditional GAN...
Generator: 8,936,643 parameters
Discriminator: 4,515,905 parameters

[STEP 3] Training for 200 epochs...
Epoch 1/200: [████████] 21/21 [00:35<00:00]
...
```

**Training time:** ~35-40 seconds per epoch × 200 = ~2-2.5 hours

**You can stop early:** Even at epoch 50-100, results will be decent!

---

### Step 4: Generate Specific Tanks

#### Generate 10 Leopard 2 tanks:
```bash
python GAN/generate_images_conditional.py --class leopard2 --num 10
```

**Output:**
```
Generated 10 images of class: leopard2
📁 Location: GAN/generated_images_conditional/
🖼️ Grid: GAN/generated_images_conditional/leopard2_grid.png
```

#### Generate 10 M1 Abrams tanks:
```bash
python GAN/generate_images_conditional.py --class m1abrams --num 10
```

#### Generate 10 T-90M tanks:
```bash
python GAN/generate_images_conditional.py --class t90m --num 10
```

#### Or generate all three at once:
```bash
python GAN/generate_images_conditional.py --all-classes --num 5
```

**Output:** Grid with 5 Leopard 2, 5 M1 Abrams, 5 T-90M (15 total)

---

## Quick Reference Commands

```bash
# 1. Organize frames (if not already done)
python GAN/organize_by_tank_type.py

# 2. Preprocess
python GAN/preprocess_dataset.py

# 3. Train (update DATASET_PATH first!)
python GAN/train_gan_conditional.py

# 4. Generate
python GAN/generate_images_conditional.py --class leopard2 --num 10
python GAN/generate_images_conditional.py --class m1abrams --num 10
python GAN/generate_images_conditional.py --class t90m --num 10
python GAN/generate_images_conditional.py --all-classes --num 5
```

---

## Tips for Best Results

### During Organization:
- **Be consistent** - Make sure frames are labeled correctly
- **Skip bad frames** - Blurry, too dark, odd angles
- **Balance classes** - Try to get similar numbers for each tank (~200-250 each)

### During Training:
- **Monitor progress** - Check `GAN/training_progress_conditional/epoch_XXX.png`
- **Early epochs (1-50)** - Random colored blobs
- **Mid epochs (50-100)** - Tank shapes emerge
- **Late epochs (100-200)** - Detailed tanks
- **Can stop early** - If quality looks good at epoch 100, you're done!

### During Generation:
- **Use best model** - `best_model.pth` (best D(fake) score)
- **Or use latest** - `latest_model.pth` (most recent)
- **Try different amounts** - Generate 5, 10, 20 to compare

---

## Expected Results

### Dataset:
- **Total frames:** ~650 (after curation)
- **Per tank:** ~220 each
- **Resolution:** 200×200 RGB

### Training:
- **Architecture:** Conditional GAN
- **Classes:** 3 (perfectly balanced)
- **Epochs:** 200 (or stop at 100 if good)
- **Time:** ~2-3 hours

### Generation:
- **Control:** Full - choose which tank to generate
- **Quality:** High - photorealistic War Thunder graphics
- **Variety:** Good - different angles/lighting from footage

---

## Demo Script

Perfect for showing off your project:

```bash
# Generate 5 of each tank type
echo "Generating Leopard 2 tanks..."
python GAN/generate_images_conditional.py --class leopard2 --num 5

echo "Generating M1 Abrams tanks..."
python GAN/generate_images_conditional.py --class m1abrams --num 5

echo "Generating T-90M tanks..."
python GAN/generate_images_conditional.py --class t90m --num 5

echo "All three classes in one grid..."
python GAN/generate_images_conditional.py --all-classes --num 3
```

**Result:** Perfect demonstration of conditional generation!

---

## Troubleshooting

### "Dataset path not found"
Update line 40 in `train_gan_conditional.py`:
```python
DATASET_PATH = 'GAN/datasets/military_vehicles_organized_processed'
```

### "Only found 1 class" or "Only found 2 classes"
Make sure organization step created 3 folders:
- `leopard2/`
- `m1abrams/`
- `t90m/`

### "Unbalanced classes"
Try to get similar numbers:
- Leopard 2: ~200-250
- M1 Abrams: ~200-250
- T-90M: ~200-250

### "Generated images look bad"
- Train for more epochs (try 200-300)
- Check training progress images
- Make sure dataset has good variety

---

## Why 3 Tanks Works Great

**Perfect for conditional GAN:**
- ✅ Clear visual differences (German vs American vs Russian design)
- ✅ Balanced classes (similar amounts each)
- ✅ High-quality source (War Thunder graphics)
- ✅ Easy to demonstrate (generate each type on demand)
- ✅ Not too many classes (easier to learn than 10+)

**Impressive for presentations:**
- "I can generate any of these three tanks on command!"
- Shows understanding of conditional GANs
- Real-world application (game asset generation)
- High visual quality

---

## Summary

1. ✅ Extracted War Thunder frames
2. ✅ Curated/removed bad frames
3. ⏭️ **Next:** Organize by tank type
4. ⏭️ **Then:** Preprocess
5. ⏭️ **Finally:** Train conditional GAN
6. 🎉 **Result:** Generate specific tanks on demand!

**Your project will stand out!** 🎮🤖🚀
