# Video-Based View Classification Workflow

Much easier than manually classifying thousands of images! Record videos of specific tank views and extract frames with automatic labeling.

## Workflow

### Step 1: Record Videos by View

Record separate videos for each tank type + view combination:

**Examples:**
- `M1A1_front.mp4` - Video of M1A1 Abrams from front view only
- `M1A1_side.mp4` - Video of M1A1 Abrams from side view only
- `M1A1_back.mp4` - Video of M1A1 Abrams from back view only
- `M1A1_top.mp4` - (OPTIONAL) Video of M1A1 Abrams from top view only
- `Leopard2_front.mp4` - Video of Leopard2 from front view only
- etc.

**Tips:**
- Keep each video focused on ONE view angle
- Record 30-60 seconds per view (gives 100-200 frames)
- Use War Thunder replay viewer or test drive
- Higher quality = better results
- **View angles are auto-discovered** - use any combination you want (front, side, back, top, etc.). Top view is optional!

### Step 2: Extract Frames with Labels

Run the extraction script for each video:

```bash
python extract_video_frames_with_labels.py
```

**Example interaction:**
```
Enter path to video file: C:\Videos\M1A1_top.mp4
Enter tank type: M1A1_Abrams
Enter view angle: top
Frame skip (default 5): 10
```

This creates:
```
GAN/datasets/military_vehicles_raw/
  M1A1_Abrams_top/
    M1A1_Abrams_top_000000.png
    M1A1_Abrams_top_000001.png
    ...
```

**Repeat for each video** - the script won't overwrite existing frames, so you can run it multiple times.

### Step 3: Preprocess for Training

Once you've extracted all videos, preprocess the frames:

```bash
python preprocess_with_folders.py
```

This will:
- Crop images to square (center crop)
- Resize to 200x200
- Move to `GAN/datasets/military_vehicles_with_views/`
- Preserve tank type + view labels

**Output:**
```
GAN/datasets/military_vehicles_with_views/
  M1A1_Abrams_front/
    *.png
  M1A1_Abrams_side/
    *.png
  M1A1_Abrams_top/
    *.png
  M1A1_Abrams_back/
    *.png
  Leopard2_front/
    *.png
  ...
```

### Step 4: Train Dual Conditional GAN

Now train the GAN with automatic tank+view conditioning:

```bash
python train_gan_dual_conditional.py
```

The training script:
- Automatically parses tank type and view from folder names
- Trains generator with two embeddings (tank + view)
- Generates sample grids showing all combinations

## Benefits Over Manual Classification

✅ **Much faster** - Record 5-10 videos instead of classifying 1000+ images
✅ **More accurate** - Each video is a consistent view angle
✅ **Easy to add more data** - Just record another video
✅ **No manual labeling** - View label comes from your video recording
✅ **Consistent quality** - All frames from one video are similar

## Recommended Recording Strategy

For each tank type, record 4 videos:

1. **Top view** - Use free camera looking straight down
2. **Front view** - Camera facing front of tank
3. **Side view** - Camera facing left or right side
4. **Back view** - Camera facing rear of tank

**Pro tip:** Use War Thunder's replay viewer to:
- Pause time
- Rotate camera smoothly around tank
- Get consistent lighting
- Capture clean backgrounds

## Example Full Workflow

```bash
# Record videos:
# - M1A1_top.mp4
# - M1A1_front.mp4
# - M1A1_side.mp4
# - M1A1_back.mp4

# Extract frames from each video
python extract_video_frames_with_labels.py
# Tank: M1A1_Abrams, View: top

python extract_video_frames_with_labels.py
# Tank: M1A1_Abrams, View: front

python extract_video_frames_with_labels.py
# Tank: M1A1_Abrams, View: side

python extract_video_frames_with_labels.py
# Tank: M1A1_Abrams, View: back

# Preprocess all frames
python preprocess_with_folders.py

# Train GAN
python train_gan_dual_conditional.py
```

## Adding More Tanks

Just record 4 more videos (front/side/top/back) for the new tank and run the extraction script:

```bash
python extract_video_frames_with_labels.py
# Tank: T90, View: top
# etc.
```

Then re-run preprocessing and continue training from your last checkpoint.
