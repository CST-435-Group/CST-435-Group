# GPU Setup Guide for RTX 3050

**Your System:**
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- CUDA: 13.0
- Python: 3.14.0
- Current PyTorch: CPU-only (2.9.1)

---

## ⚠️ Current Issue: Python 3.14 Compatibility

**PyTorch doesn't support Python 3.14 yet.** PyTorch currently supports Python 3.8-3.12.

You have **3 options** to get GPU acceleration:

---

## Option 1: Use Python 3.12 (RECOMMENDED) ⭐

This is the easiest and most reliable option.

### Step 1: Install Python 3.12
Download from: https://www.python.org/downloads/release/python-3127/

**Important:** During installation:
- ✅ Check "Add Python 3.12 to PATH"
- ✅ Install for all users (optional)

### Step 2: Create Virtual Environment
```bash
# Navigate to GAN folder
cd "C:\Users\Soren\OneDrive\Documents\school\College Senior\CST-405\Compiler\CST-435-Group\GAN"

# Create virtual environment with Python 3.12
py -3.12 -m venv gan_env

# Activate it
gan_env\Scripts\activate

# You should see (gan_env) in your prompt
```

### Step 3: Install CUDA PyTorch
```bash
# Install PyTorch with CUDA 12.1 (works with CUDA 13.0)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### Step 4: Verify GPU Works
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA: True
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

### Step 5: Run Training
```bash
python train_gan.py
```

---

## Option 2: Wait for PyTorch 3.14 Support

PyTorch typically adds support for new Python versions within a few months.

**Check for updates:**
```bash
pip install --upgrade torch torchvision
```

**Monitor PyTorch releases:**
https://pytorch.org/get-started/locally/

**In the meantime:** Train with CPU (slower but works)

---

## Option 3: Use Conda with Python 3.12

If you prefer Conda over virtual environments:

### Step 1: Install Anaconda/Miniconda
Download from: https://www.anaconda.com/products/distribution

### Step 2: Create Environment
```bash
# Create new environment with Python 3.12
conda create -n gan_env python=3.12

# Activate it
conda activate gan_env
```

### Step 3: Install PyTorch with CUDA
```bash
# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install tqdm scikit-learn matplotlib Pillow seaborn
```

### Step 4: Verify and Run
```bash
python -c "import torch; print(torch.cuda.is_available())"
python train_gan.py
```

---

## Speed Comparison

**Your RTX 3050 vs CPU:**

| Hardware | Time per Epoch | 100 Epochs |
|----------|---------------|------------|
| **RTX 3050 (GPU)** | ~40-50s | ~65-85 min |
| **CPU Only** | ~3-5 min | **5-8 hours** |

**GPU is 5-10x faster!** Worth setting up Python 3.12.

---

## Current Workaround: Train on CPU

While you set up Python 3.12, you can still train on CPU:

```bash
python train_gan.py
```

**Tips for CPU training:**
1. Let it run overnight (5-8 hours for 100 epochs)
2. Reduce batch size for faster epochs (edit train_gan.py line 260):
   ```python
   batch_size = 32  # Instead of 64
   ```
3. Train fewer epochs initially to test:
   ```python
   num_epochs = 25  # Instead of 100
   ```

---

## Verifying Your Current Setup

Run this to see your current status:

```bash
python -c "import sys; import torch; print(f'Python: {sys.version}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**Current output:**
```
Python: 3.14.0
PyTorch: 2.9.1+cpu
CUDA Available: False
```

**After Python 3.12 setup:**
```
Python: 3.12.7
PyTorch: 2.x.x+cu121
CUDA Available: True
```

---

## Quick Setup Guide (Python 3.12)

```bash
# 1. Install Python 3.12 from python.org
# 2. Open new PowerShell in GAN folder
cd "C:\Users\Soren\OneDrive\Documents\school\College Senior\CST-405\Compiler\CST-435-Group\GAN"

# 3. Create virtual environment
py -3.12 -m venv gan_env

# 4. Activate
gan_env\Scripts\activate

# 5. Install CUDA PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 6. Install other dependencies
pip install -r requirements.txt

# 7. Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# 8. Run training
python train_gan.py
```

---

## Files Modified for Windows Compatibility

✅ **train_gan.py** - Fixed multiprocessing (num_workers=0 on Windows)
✅ **test_setup.py** - Fixed multiprocessing, removed full training import

You can now run:
```bash
python test_setup.py
```

Should complete without multiprocessing errors.

---

## Next Steps

**Recommended path:**
1. Install Python 3.12 alongside 3.14 (they can coexist)
2. Create virtual environment with Python 3.12
3. Install CUDA PyTorch
4. Train with GPU (5-10x faster)

**Alternative:**
1. Train on CPU for now (works, just slower)
2. Wait for PyTorch Python 3.14 support
3. Upgrade when available

---

## Questions?

- **Can I keep Python 3.14?** Yes! Multiple Python versions can coexist. Use virtual environments.
- **Will this affect my other projects?** No, virtual environments are isolated.
- **How long to set up?** ~15-20 minutes for Option 1.
- **Can I switch between CPU and GPU?** Yes, just activate the appropriate environment.

---

## Verify After Setup

```bash
# After setup, verify everything:
python test_setup.py

# Should show:
# ✅ CUDA available: 12.1
# ✅ GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

Then start training!
