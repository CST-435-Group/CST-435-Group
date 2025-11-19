# Model Chunking System

## Overview

This GAN project includes an automatic model chunking system that prevents model files from exceeding GitHub's 100MB file size limit. The system automatically splits large models into smaller chunks during training and reassembles them transparently when loading.

## How It Works

### Automatic Chunking During Training

When you train the GAN, models are automatically saved with chunking:

```python
from model_utils import save_model_chunked

# Instead of torch.save(model_data, 'model.pth')
save_model_chunked(model_data, 'model.pth')
```

**What happens:**
- If model < 90MB: Saved as a single .pth file (normal)
- If model > 90MB: Automatically split into multiple chunks

### File Structure for Chunked Models

For a large model, you'll see these files:

```
models/
├── best_model.chunk_000.pth    (29.6 MB - generator state dict)
├── best_model.chunk_001.pth    (16.5 MB - discriminator state dict)
├── best_model.manifest.json    (metadata about chunks)
└── best_model.pth               (small placeholder)
```

### Transparent Loading

Loading works the same way regardless of chunking:

```python
from model_utils import load_model_chunked

# Works with both chunked and non-chunked models
model_data = load_model_chunked('model.pth', device='cpu')
```

**What happens:**
- If chunked: Reads manifest, loads all chunks, assembles in RAM
- If not chunked: Loads normally

## For Users

### Training

Just run training scripts normally:

```bash
python GAN/train_gan.py
python GAN/train_gan_single_fruit.py --fruit Apple --epochs 50
```

Models are automatically chunked if needed. You'll see messages like:

```
[CHUNKING] Model size 137.8MB exceeds 90MB, splitting...
[CHUNKED] Saved 2 chunks (max 29.6MB)
```

### Generating Images

Use the generate script normally - it handles chunked models automatically:

```bash
python GAN/generate_images.py --model GAN/models/best_model.pth --num 50
```

### Checking Models

```bash
python GAN/check_models.py
```

This will show which models are chunked:

```
✅ best_model.pth
   Size: 137.8 MB (chunked into 2 files)
   📦 Chunked model (GitHub-friendly)
```

## For Developers

### Manual Conversion

To convert existing large models:

```bash
python GAN/convert_existing_models.py
```

### Testing

Test the chunking system:

```bash
python GAN/test_chunking.py
```

### API Reference

```python
from model_utils import save_model_chunked, load_model_chunked, clean_chunks

# Save with automatic chunking
save_model_chunked(model_data, 'model.pth')

# Load (handles both chunked and regular models)
model_data = load_model_chunked('model.pth', device='cpu')

# Clean up all chunks and manifests
clean_chunks('model.pth')
```

## GitHub Compatibility

All chunk files are guaranteed to be under 90MB, making them safe for GitHub:

- ✅ Each chunk < 90MB
- ✅ All chunks must be committed
- ✅ Manifest files must be committed
- ✅ Models load correctly from cloned repos

## Benefits

1. **No Manual Splitting**: Happens automatically during training
2. **Transparent**: Existing code works without changes
3. **GitHub-Friendly**: All files under 100MB limit
4. **No Data Loss**: Complete models preserved across chunks
5. **Performance**: Models reassembled efficiently in RAM

## Technical Details

### Chunk Strategy

- Threshold: 90MB (with safety margin below 100MB)
- Splitting: State dicts split by parameter groups
- Assembly: Chunks loaded and merged in RAM
- Format: Standard PyTorch .pth files

### Manifest Structure

```json
{
  "version": "1.0",
  "is_chunked": true,
  "total_size_mb": 137.8,
  "total_chunks": 2,
  "state_dicts": {
    "generator_state_dict": {
      "chunked": false,
      "chunk_id": 0
    },
    "discriminator_state_dict": {
      "chunked": false,
      "chunk_id": 1
    }
  },
  "chunk_files": [
    {
      "chunk_id": 0,
      "filename": "model.chunk_000.pth",
      "size_mb": 29.6,
      "state_dict": "generator_state_dict"
    },
    ...
  ]
}
```

## Troubleshooting

### "Model not loaded" error

Make sure all chunk files and manifest are present:

```bash
ls GAN/models/best_model*
# Should show: .pth, .chunk_000.pth, .chunk_001.pth, .manifest.json
```

### Large files in git status

If you see files > 100MB in `git status`, run:

```bash
python GAN/convert_existing_models.py
```

### Chunks not loading

Verify manifest exists and all chunks listed are present.

## Files

- `model_utils.py` - Core chunking utilities
- `train_gan.py` - Updated to use chunking
- `train_gan_single_fruit.py` - Updated to use chunking
- `generate_images.py` - Updated to load chunked models
- `check_models.py` - Updated to show chunk info
- `test_chunking.py` - Test suite
- `convert_existing_models.py` - Conversion utility
