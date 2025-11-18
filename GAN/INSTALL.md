# GAN Project Installation Guide

## Quick Install (3 Steps)

### Step 1: Install Dependencies
```bash
cd GAN
pip install -r requirements.txt
```

**Note:** If you have GPU and want CUDA support (highly recommended for faster training):
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only (slower but works)
pip install torch torchvision
```

### Step 2: Verify Installation
```bash
python test_setup.py
```

### Step 3: Start Training
```bash
python train_gan.py
```

---

## Detailed Installation

### Requirements
- Python 3.8 or higher
- CUDA-capable GPU (recommended but not required)
- 4GB+ RAM
- 2GB+ disk space for models

### Package Versions
- **torch**: >=2.0.0 (PyTorch for deep learning)
- **torchvision**: >=0.15.0 (Image utilities for PyTorch)
- **Pillow**: >=9.0.0 (Image processing)
- **numpy**: >=1.21.0 (Numerical computing)
- **tqdm**: >=4.65.0 (Progress bars)
- **matplotlib**: >=3.5.0 (Plotting)
- **scikit-learn**: >=1.0.0 (Machine learning utilities)

### Installation Options

#### Option 1: Standard Installation (CPU)
```bash
pip install -r requirements.txt
```

#### Option 2: GPU Installation (NVIDIA CUDA)
**Check your CUDA version first:**
```bash
nvidia-smi
```

**Then install matching PyTorch:**
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

#### Option 3: Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv gan_env

# Activate (Windows)
gan_env\Scripts\activate

# Activate (Mac/Linux)
source gan_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Verify GPU Support

After installation, verify CUDA is working:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output (if GPU available):
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
```

Expected output (if CPU only):
```
CUDA available: False
GPU: None
```

---

## Common Issues

### Issue: "No module named 'torch'"
**Solution:**
```bash
pip install torch torchvision
```

### Issue: "No module named 'torchvision'"
**Solution:**
```bash
pip install torchvision
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in `train_gan.py`:
```python
batch_size = 32  # Change from 64 to 32
```

### Issue: "Could not find a version that satisfies the requirement torch"
**Solution:** Upgrade pip:
```bash
python -m pip install --upgrade pip
pip install torch torchvision
```

### Issue: PyTorch not detecting GPU
**Solution:**
1. Check CUDA version: `nvidia-smi`
2. Install matching PyTorch version
3. Restart Python/terminal

---

## Platform-Specific Notes

### Windows
- Use `pip` or `pip3`
- Make sure Python is in PATH
- Admin rights may be needed for some installations

### Mac (Apple Silicon M1/M2)
```bash
# PyTorch has Metal Performance Shaders (MPS) support
pip install torch torchvision
# GPU acceleration will use MPS instead of CUDA
```

### Linux
```bash
# Usually straightforward
pip install -r requirements.txt
```

---

## Alternative: Conda Installation

If you use Anaconda/Miniconda:

```bash
# Create environment
conda create -n gan_env python=3.10

# Activate
conda activate gan_env

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install tqdm scikit-learn matplotlib Pillow
```

---

## Verify Full Setup

Run the comprehensive setup test:
```bash
python test_setup.py
```

This will check:
- ✅ PyTorch installation
- ✅ CUDA availability
- ✅ All dependencies
- ✅ CNN_Project data
- ✅ GAN model architecture
- ✅ Data loading

---

## Next Steps

After successful installation:

1. **Check prerequisites:**
   ```bash
   python test_setup.py
   ```

2. **Train the GAN:**
   ```bash
   python train_gan.py
   ```

3. **Monitor progress:**
   - Watch terminal output
   - Check `GAN/training_history_live.png`
   - View progress images in `GAN/training_progress/`

4. **Generate images:**
   ```bash
   python generate_images.py --num 50 --classify
   ```

---

## Getting Help

- Check test_setup.py output for specific errors
- Review README.md for usage examples
- Ensure CNN_Project was trained first
- Verify GPU with `nvidia-smi` (if using CUDA)

---

**Installation Time:**
- With pip: ~2-5 minutes
- With conda: ~5-10 minutes
- Depends on internet speed and system
