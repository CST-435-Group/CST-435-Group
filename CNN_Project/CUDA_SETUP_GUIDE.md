# 🎮 PyTorch + CUDA Setup Guide for Windows

## Complete Guide to GPU-Accelerated Deep Learning on Windows

---

## 🎯 Quick Decision Tree

**Do you have an NVIDIA GPU?**
- ✅ **YES** → Follow "GPU Setup" below (recommended - 3-4x faster!)
- ❌ **NO** → Follow "CPU-Only Setup" below (still works, just slower)

---

## 🎮 GPU Setup (Recommended)

### Step 1: Check Your GPU

**Open Command Prompt and run:**
```bash
nvidia-smi
```

**If this works,** you'll see:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 545.84       Driver Version: 545.84       CUDA Version: 12.3    |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0  On |                  N/A |
```

**Note your CUDA Version** (e.g., 12.3 or 11.8) - you'll need this!

**If `nvidia-smi` doesn't work:**
- You need to install NVIDIA drivers first
- Go to: https://www.nvidia.com/Download/index.aspx
- Select your GPU and download the latest driver
- Install and restart your computer
- Try `nvidia-smi` again

---

### Step 2: Install PyTorch with CUDA

**Visit:** https://pytorch.org/get-started/locally/

**Or use these quick commands:**

#### For CUDA 11.8 (Most Compatible):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### For CUDA 12.1 (Newer):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### For CUDA 11.7:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

**Note:** You don't need to install CUDA separately! PyTorch comes with everything needed.

---

### Step 3: Install Other Dependencies

```bash
pip install numpy matplotlib seaborn pandas scikit-learn tqdm jupyter
```

---

### Step 4: Verify CUDA is Working

**Run the test script:**
```bash
python test_installation.py
```

**Expected output:**
```
✅ PyTorch         - OK
✅ TorchVision     - OK

🎮 GPU/CUDA Information:
   ✅ CUDA is available!
   CUDA Version: 11.8
   GPU 0: NVIDIA GeForce RTX 3060
   GPU Memory: 12.00 GB

✅ You're ready to run the CNN project!
```

**Or test directly in Python:**
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 💻 CPU-Only Setup

**If you don't have an NVIDIA GPU:**

### Step 1: Install PyTorch (CPU Version)
```bash
pip3 install torch torchvision torchaudio
```

### Step 2: Install Other Dependencies
```bash
pip install numpy matplotlib seaborn pandas scikit-learn tqdm jupyter
```

### Step 3: Verify Installation
```bash
python test_installation.py
```

**Expected output:**
```
✅ PyTorch         - OK
✅ TorchVision     - OK

ℹ️  CUDA not available (CPU will be used)
⚠️  Training will be slower on CPU

✅ You're ready to run the CNN project!
```

**Note:** Training will take 15-20 minutes instead of 5-10 minutes, but it will still work!

---

## 🆘 Troubleshooting

### Issue 1: "CUDA not available" despite having GPU

**Solutions:**
1. **Reinstall PyTorch with CUDA:**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Check GPU drivers:**
   ```bash
   nvidia-smi
   ```
   If this doesn't work, update your drivers from NVIDIA website

3. **Verify PyTorch installation:**
   ```python
   import torch
   print(torch.__version__)  # Should show something like: 2.1.0+cu118
   ```

---

### Issue 2: "RuntimeError: CUDA out of memory"

**Solutions:**
1. **Reduce batch size in notebook:**
   ```python
   batch_size = 32  # Instead of 64
   # or even
   batch_size = 16
   ```

2. **Close other GPU-intensive applications:**
   - Games
   - Chrome (can use GPU)
   - Video editing software
   - Other deep learning programs

3. **Clear GPU cache (in notebook):**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

### Issue 3: ImportError or ModuleNotFoundError

**Solutions:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Reinstall all packages
pip install torch torchvision numpy matplotlib seaborn pandas scikit-learn tqdm jupyter --upgrade
```

---

### Issue 4: Training is slow even with GPU

**Check:**
1. **Verify GPU is being used:**
   ```python
   # In notebook, check output
   print(device)  # Should show: cuda
   ```

2. **Monitor GPU usage:**
   ```bash
   nvidia-smi -l 1  # Updates every 1 second
   ```
   GPU utilization should be 80-100% during training

3. **Ensure data is on GPU:**
   ```python
   # This should happen automatically, but verify
   images = images.to(device)
   ```

---

### Issue 5: "No module named 'torch'"

**Solutions:**
```bash
# Make sure you're using the right Python
python --version  # Should be 3.8+

# Try with python3 explicitly
python3 -m pip install torch torchvision torchaudio

# Or use pip3
pip3 install torch torchvision torchaudio
```

---

## 🎯 GPU Requirements

### Minimum GPU Requirements:
- **NVIDIA GPU:** Any CUDA-capable GPU (GTX 10-series or newer)
- **GPU Memory:** 2 GB minimum (4 GB recommended)
- **CUDA Support:** Compute Capability 3.5 or higher

### Tested GPUs:
- ✅ RTX 40-series (4090, 4080, 4070, etc.) - Excellent
- ✅ RTX 30-series (3090, 3080, 3070, 3060, etc.) - Excellent
- ✅ RTX 20-series (2080, 2070, 2060, etc.) - Great
- ✅ GTX 16-series (1660, 1650, etc.) - Good
- ✅ GTX 10-series (1080, 1070, 1060, etc.) - Good
- ⚠️ GTX 900-series (980, 970, etc.) - Works but older

### Performance Expectations:
| GPU | Approx. Time (50 epochs) |
|-----|-------------------------|
| RTX 4090 | ~3-4 minutes |
| RTX 3080/3090 | ~5-6 minutes |
| RTX 3060/3070 | ~6-8 minutes |
| GTX 1660 Ti | ~8-10 minutes |
| GTX 1060 (6GB) | ~10-12 minutes |
| CPU (no GPU) | ~15-20 minutes |

---

## 📊 Verify Your Setup

### Full Verification Script:

```python
import torch
import torchvision
import sys

print("=" * 70)
print("PYTORCH + CUDA VERIFICATION")
print("=" * 70)

# Python version
print(f"\nPython Version: {sys.version}")

# PyTorch version
print(f"PyTorch Version: {torch.__version__}")
print(f"TorchVision Version: {torchvision.__version__}")

# CUDA availability
print(f"\nCUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        
        props = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"  Multiprocessors: {props.multi_processor_count}")
    
    # Test GPU
    print("\n🧪 Testing GPU...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("✅ GPU test successful!")
    
    print("\n🚀 Your GPU is ready for deep learning!")
else:
    print("\n⚠️  CUDA not available")
    print("   Training will use CPU (slower but still works)")
    print("\n   To enable GPU:")
    print("   1. Ensure you have NVIDIA GPU")
    print("   2. Install GPU drivers from nvidia.com")
    print("   3. Reinstall PyTorch with CUDA:")
    print("      pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "=" * 70)
```

**Save as `verify_cuda.py` and run:**
```bash
python verify_cuda.py
```

---

## 🚀 Next Steps

Once everything is verified:

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook CNN_Image_Classification.ipynb
   ```

2. **Or run standalone script:**
   ```bash
   python cnn_standalone.py
   ```

3. **Monitor GPU during training:**
   ```bash
   # Open another terminal
   nvidia-smi -l 1
   ```

4. **Enjoy the speed! 🎮⚡**

---

## 💡 Pro Tips

### Maximize GPU Performance:
1. **Close unnecessary applications** before training
2. **Update GPU drivers** regularly
3. **Increase batch size** if you have more GPU memory
4. **Use `pin_memory=True`** in DataLoader (already set in our code)
5. **Monitor temperature** - keep GPU under 85°C

### Common GPU Memory Issues:
```python
# Check available memory
print(torch.cuda.memory_allocated() / 1e9, "GB")
print(torch.cuda.memory_reserved() / 1e9, "GB")

# Clear cache if needed
torch.cuda.empty_cache()
```

### Multiple GPUs:
```python
# Use first GPU (default)
device = torch.device('cuda:0')

# Use second GPU
device = torch.device('cuda:1')
```

---

## 📚 Additional Resources

### Official Documentation:
- **PyTorch Installation:** https://pytorch.org/get-started/locally/
- **CUDA Toolkit:** https://developer.nvidia.com/cuda-downloads
- **NVIDIA Drivers:** https://www.nvidia.com/Download/index.aspx

### Learning Resources:
- **PyTorch Tutorials:** https://pytorch.org/tutorials/
- **CUDA Programming:** https://docs.nvidia.com/cuda/
- **PyTorch Forums:** https://discuss.pytorch.org/

---

## ✅ Installation Checklist

- [ ] NVIDIA GPU present (`nvidia-smi` works)
- [ ] Latest GPU drivers installed
- [ ] Python 3.8+ installed
- [ ] PyTorch with CUDA installed
- [ ] Other dependencies installed (numpy, matplotlib, etc.)
- [ ] `test_installation.py` shows CUDA available
- [ ] GPU detected in Python: `torch.cuda.is_available()` returns `True`
- [ ] Ready to train with GPU acceleration! 🚀

---

**Congratulations! You're now ready to train CNNs with GPU acceleration! 🎉🎮**

*Training time: 5-10 minutes with GPU vs 15-20 minutes with CPU*

---

*Last Updated: October 2025*
*For CST-435 Neural Networks Assignment*
