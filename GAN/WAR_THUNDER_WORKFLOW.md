# War Thunder Video Frame Extraction Workflow

Perfect way to get high-quality, photorealistic tank images with consistent lighting and angles!

---

## Why This is Brilliant 💡

**Advantages of War Thunder footage:**
- ✅ Photorealistic 3D models
- ✅ Consistent quality
- ✅ Perfect 360° views
- ✅ Clean backgrounds
- ✅ No people/infantry
- ✅ Multiple angles/lighting
- ✅ Specific tank models
- ✅ You control everything!

**Better than web scraping because:**
- No random photos
- No watermarks
- No people in shots
- Consistent style
- As many angles as you want

---

## Step-by-Step Guide

### 1. Record War Thunder Footage

**In-game settings:**
- Go to test drive or hangar view
- Select specific tank (e.g., M1 Abrams, Leopard 2, T-90)
- Use free camera mode
- Go 360° around the tank 2-3 times
- Rotate at different heights (low, eye level, high)
- Record at 1080p or higher

**Recording tips:**
- Use OBS, Nvidia ShadowPlay, or Windows Game Bar
- 30-60 seconds per tank is plenty
- Slow rotation = better frames
- Multiple tanks = multiple videos
- Clear lighting conditions

**Recommended tanks to record:**
- M1 Abrams (USA)
- Leopard 2A5/2A6 (Germany)
- T-90A/T-72 (Russia)
- Challenger 2 (UK)
- Type 90/Type 10 (Japan)
- Leclerc (France)

---

### 2. Extract Frames from Video

```bash
python GAN/extract_video_frames.py
```

**Interactive prompts:**

```
[STEP 1] Select video file
Enter path to video file: C:\Videos\warthunder_m1abrams.mp4

[STEP 2] Select output directory
Default: GAN/datasets/military_vehicles_raw
Enter output directory (or press Enter for default): [Press Enter]

[STEP 3] Frame extraction settings
Frame skip (default 5): 10
# Recommend 10 for 60fps, 5 for 30fps

[STEP 4] Maximum frames (optional)
Max frames (or press Enter for no limit): 200
# 200 frames = plenty of angles

Proceed? (y/n): y
```

**Output:**
```
VIDEO INFO
File: warthunder_m1abrams.mp4
Total frames: 1800
FPS: 60.00
Duration: 30.0 seconds
Frame skip: Saving every 10 frame(s)
Estimated frames to save: ~180

[INFO] Starting numbering from: 0
Output directory: GAN/datasets/military_vehicles_raw

EXTRACTING FRAMES
Processing video: 100%|████████████| 1800/1800

EXTRACTION COMPLETE
Total frames processed: 1800
Frames saved: 180
Saved to: GAN/datasets/military_vehicles_raw
File naming: frame_000000.png to frame_000179.png
```

---

### 3. Extract More Videos (Won't Overwrite!)

**Video 2: Leopard 2**
```bash
python GAN/extract_video_frames.py
```

```
Enter path to video file: C:\Videos\warthunder_leopard2.mp4
[Press Enter for default directory]
Frame skip: 10
Max frames: 200
Proceed? (y/n): y
```

**Output:**
```
[INFO] Starting numbering from: 180  ← Continues from previous!
Frames saved: 180
File naming: frame_000180.png to frame_000359.png
```

**No overwriting! Automatically continues numbering!**

---

### 4. Extract Multiple Tanks

Repeat for each tank type:
- M1 Abrams → frame_000000 to frame_000179
- Leopard 2 → frame_000180 to frame_000359
- T-90 → frame_000360 to frame_000539
- Challenger 2 → frame_000540 to frame_000719
- etc.

**After 5 videos:** ~900 frames of high-quality tank images!

---

### 5. Preprocess Frames

```bash
python GAN/preprocess_dataset.py
```

This will:
- Smart resize/crop to 200x200
- Filter out any bad frames
- Organize for training

**Output:**
```
Found 900 images
Processing...
Successful: 850
Ready for training!
```

---

### 6. Train Conditional GAN

```bash
python GAN/train_gan_conditional.py
```

**Auto-detects classes from filenames!**

If you name your videos:
- `tank_m1abrams.mp4` → Class: "tank"
- `tank_leopard2.mp4` → Class: "tank"
- `tank_t90.mp4` → Class: "tank"

Or for different vehicle types:
- `tank_m1abrams.mp4` → Class: "tank"
- `helicopter_apache.mp4` → Class: "helicopter"
- `jet_f16.mp4` → Class: "jet"

---

## Frame Skip Recommendations

**For 60 FPS video:**
- `frame_skip = 10` → Save 6 frames/second (recommended)
- `frame_skip = 5` → Save 12 frames/second (lots of images)
- `frame_skip = 30` → Save 2 frames/second (fewer images)

**For 30 FPS video:**
- `frame_skip = 5` → Save 6 frames/second (recommended)
- `frame_skip = 3` → Save 10 frames/second (lots of images)
- `frame_skip = 15` → Save 2 frames/second (fewer images)

**360° rotation in 20 seconds:**
- 6 frames/sec × 20 sec = **120 frames** (good coverage)
- Captures every ~3° of rotation
- Perfect for training!

---

## Example Session

### Recording Session (1 hour):
```
Record 10 different tanks, 30 seconds each
Total footage: 5 minutes
```

### Extraction Session (10 minutes):
```bash
# Extract video 1
python GAN/extract_video_frames.py
# Video: tank1.mp4, Skip: 10, Max: 200

# Extract video 2
python GAN/extract_video_frames.py
# Video: tank2.mp4, Skip: 10, Max: 200

# ... repeat for all 10 videos
```

**Result:** ~2000 high-quality tank frames!

### Preprocessing (2 minutes):
```bash
python GAN/preprocess_dataset.py
```

**Result:** ~1800 clean 200x200 images ready for training

### Training (2-3 hours):
```bash
python GAN/train_gan_conditional.py
```

**Result:** GAN that generates photorealistic tanks! 🎮→🤖

---

## Tips for Best Results

### In-Game Recording:
1. **Use hangar/test drive** - Clean background
2. **Free camera mode** - Smooth rotation
3. **Multiple passes** - Different heights/angles
4. **Slow rotation** - Better frame coverage
5. **Good lighting** - Consistent quality
6. **Zoom appropriately** - Fill frame with tank

### Frame Extraction:
1. **Frame skip 5-10** - Good balance
2. **Max 150-200 per video** - Prevents redundancy
3. **Multiple videos** - Different tanks/angles
4. **Check output** - Make sure frames look good

### Dataset Quality:
- **Target: 500-1000 frames** minimum
- **Variety:** Different tanks, angles, lighting
- **Consistency:** Same game, settings, quality
- **Clean:** No HUD, crosshairs, or UI

---

## Advanced: Multiple Vehicle Types

### Record different categories:

**Tanks:**
- M1 Abrams, Leopard 2, T-90, Challenger 2
- Save to: `GAN/datasets/military_vehicles_raw/tanks/`

**Aircraft:**
- F-16, MiG-29, F-22
- Save to: `GAN/datasets/military_vehicles_raw/jets/`

**Helicopters:**
- Apache, Mi-24, Black Hawk
- Save to: `GAN/datasets/military_vehicles_raw/helicopters/`

**Conditional GAN will learn all three types!**

---

## Comparison: War Thunder vs Web Scraping

| Feature | War Thunder | Web Scraping |
|---------|-------------|--------------|
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Consistency** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Control** | ⭐⭐⭐⭐⭐ | ⭐ |
| **No people** | ✅ | ❌ |
| **Speed** | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Variety** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Effort** | Medium | Low |
| **Results** | Excellent | Good |

**Best approach: Combine both!**
- War Thunder for main tanks
- Web scraping for variety/rare vehicles

---

## Quick Commands Summary

```bash
# 1. Extract frames from video
python GAN/extract_video_frames.py

# 2. Preprocess frames
python GAN/preprocess_dataset.py

# 3. Train conditional GAN
python GAN/train_gan_conditional.py

# 4. Generate specific tank types
python GAN/generate_images_conditional.py --class tank --num 10
```

---

## Troubleshooting

### "Could not open video"
- Check video codec (MP4/H.264 works best)
- Try converting with VLC: Media → Convert

### "OpenCV not installed"
- Script auto-installs, just run again

### "Too many similar frames"
- Increase frame_skip (try 15 or 20)
- Rotate faster in-game

### "Frames look dark/washed out"
- Adjust in-game brightness
- Re-record with better lighting

---

## Perfect for:
- 🎮 War Thunder (tanks, planes, ships)
- 🚁 DCS World (aircraft)
- ⚔️ Arma 3 (military vehicles)
- 🎯 Any game with vehicle test drives

**High-quality 3D models → Perfect training data!** 🚀
