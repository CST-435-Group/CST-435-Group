"""
Test script to verify all dependencies are installed correctly for PyTorch
Run this before starting the main CNN project
"""

import sys

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✅ {package_name:20s} - OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name:20s} - MISSING ({str(e)})")
        return False

def main():
    print("=" * 70)
    print("CNN PROJECT - DEPENDENCY CHECK (PyTorch Version)")
    print("=" * 70)
    print("\nChecking required packages...\n")
    
    all_ok = True
    
    # Test core packages
    all_ok &= test_import('torch', 'PyTorch')
    all_ok &= test_import('torchvision', 'TorchVision')
    all_ok &= test_import('numpy', 'NumPy')
    all_ok &= test_import('pandas', 'Pandas')
    all_ok &= test_import('matplotlib', 'Matplotlib')
    all_ok &= test_import('seaborn', 'Seaborn')
    all_ok &= test_import('sklearn', 'Scikit-learn')
    all_ok &= test_import('PIL', 'Pillow')
    all_ok &= test_import('tqdm', 'TQDM')
    
    print("\n" + "=" * 70)
    
    if all_ok:
        print("✅ ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📋 Version Information:")
        
        # Display versions
        import torch
        import torchvision
        import numpy as np
        import pandas as pd
        import matplotlib
        import sklearn
        
        print(f"   PyTorch:     {torch.__version__}")
        print(f"   TorchVision: {torchvision.__version__}")
        print(f"   NumPy:       {np.__version__}")
        print(f"   Pandas:      {pd.__version__}")
        print(f"   Matplotlib:  {matplotlib.__version__}")
        print(f"   Scikit-learn: {sklearn.__version__}")
        
        # Check CUDA availability
        print("\n🎮 GPU/CUDA Information:")
        if torch.cuda.is_available():
            print(f"   ✅ CUDA is available!")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                
            # Check GPU memory
            if torch.cuda.device_count() > 0:
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"   GPU Memory: {total_memory:.2f} GB")
                
            print("\n   🚀 Your CNN will train MUCH faster with GPU acceleration!")
        else:
            print("   ℹ️  CUDA not available (CPU will be used)")
            print("   ⚠️  Training will be slower on CPU")
            print("\n   To enable CUDA:")
            print("      1. Install NVIDIA GPU drivers")
            print("      2. Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
            print("      3. Reinstall PyTorch with CUDA:")
            print("         pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        
        print("\n✅ You're ready to run the CNN project!")
        print("   Next step: Run 'jupyter notebook CNN_Image_Classification.ipynb'")
        
    else:
        print("❌ SOME DEPENDENCIES ARE MISSING!")
        print("=" * 70)
        print("\n📦 To install missing packages, run:")
        print("\n   For PyTorch with CUDA (recommended for GPU):")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n   For PyTorch CPU-only:")
        print("   pip3 install torch torchvision torchaudio")
        print("\n   For other dependencies:")
        print("   pip install numpy pandas matplotlib seaborn scikit-learn pillow tqdm jupyter")
        print("\n   Or use requirements.txt:")
        print("   pip install -r requirements.txt")
        
    print("\n" + "=" * 70)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
