# Dual Conditional GAN - Tank Type + View Angle

This improves the GAN by conditioning on **two** inputs instead of one:
1. **Tank type** (M1A1_Abrams, M1A2_Abrams, etc.)
2. **View angle** (front, side, back, top - or any custom views you want)

This helps the GAN learn each perspective separately, since front/side/back views of tanks look very different.

**Note:** View angles are automatically discovered from your folder names. You can use any combination (front/side/back/top/etc.). Top view is optional!

## Workflow

### Step 1: Classify Training Images by View

Run the classification tool to label each image with its view angle:

```bash
python classify_views.py
```

**Instructions:**
- The tool shows 100 images at a time in a grid
- Press a number key to classify ALL unclassified images in the current view:
  - `1` = Front view
  - `2` = Side view (left or right)
  - `3` = Top view
  - `4` = Back view
  - `5` = Skip/Unclear
- Green titles = already classified
- Black titles = needs classification
- Press `Enter` to save and move to next batch
- Press `Q` to quit

**Output:**
- `view_classifications.json` - saves your classifications
- `curated_data_with_views/` - new dataset with view labels in class names
  - Example: `M1A1_Abrams_front/`, `M1A1_Abrams_side/`, etc.

### Step 2: Train the Dual Conditional GAN

Once you've classified the images, train the new GAN:

```bash
python train_gan_dual_conditional.py
```

**What's different:**
- Generator takes **two** inputs: tank type embedding + view angle embedding
- Discriminator also sees both conditions
- Training automatically extracts tank type and view from class names
- Sample images show a grid of all tank types × all view angles

**Output:**
- `models_dual_conditional/` - trained models
- `training_progress_dual_conditional/` - sample grids showing all combinations
- `label_mappings.json` - mapping of tank types and views to indices

### Step 3: Generate Specific Views

To generate specific tank+view combinations, you can use the trained model:

```python
import torch
from models_dual_conditional import DualConditionalGenerator
import json

# Load model and mappings
generator = DualConditionalGenerator(num_tanks=10, num_views=4)
# ... load weights ...

with open('models_dual_conditional/label_mappings.json') as f:
    mappings = json.load(f)

# Generate M1A1 Abrams front view
tank_idx = mappings['tank_to_idx']['M1A1_Abrams']
view_idx = mappings['view_to_idx']['front']

noise = torch.randn(1, 100)
tank_label = torch.tensor([tank_idx])
view_label = torch.tensor([view_idx])

fake_img = generator(noise, tank_label, view_label)
```

## Benefits

1. **Better quality** - GAN doesn't confuse different perspectives
2. **Controlled generation** - Can specify exactly what view to generate
3. **Easier training** - Each condition has more consistent visual features
4. **More flexible** - Can generate any tank type from any angle

## Architecture Details

**Generator:**
- Input: noise (100D) + tank embedding (50D) + view embedding (50D) = 200D total
- Two separate embedding layers for tank type and view angle
- Embeddings are learned during training

**Discriminator:**
- Input: image (3×128×128) + tank embedding (50D) + view embedding (50D)
- Must determine if image is real AND matches the specified tank type AND view angle
- Encourages generator to produce correct combinations
