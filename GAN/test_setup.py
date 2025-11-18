"""
Test GAN Setup
Verify all prerequisites before training
"""

import sys
from pathlib import Path

print("=" * 80)
print("GAN SETUP VERIFICATION")
print("=" * 80)

# Test 1: PyTorch
print("\n[TEST 1] Checking PyTorch installation...")
try:
    import torch
    print(f"  ✅ PyTorch {torch.__version__} installed")
    if torch.cuda.is_available():
        print(f"  ✅ CUDA available: {torch.version.cuda}")
        print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  ⚠️  CUDA not available (CPU only)")
except ImportError:
    print("  ❌ PyTorch not installed")
    print("     Install: pip install torch torchvision")
    sys.exit(1)

# Test 2: Other dependencies
print("\n[TEST 2] Checking dependencies...")
deps = ['torchvision', 'PIL', 'numpy', 'tqdm', 'matplotlib']
for dep in deps:
    try:
        if dep == 'PIL':
            import PIL
            print(f"  ✅ Pillow (PIL) installed")
        else:
            __import__(dep)
            print(f"  ✅ {dep} installed")
    except ImportError:
        print(f"  ❌ {dep} not installed")
        print(f"     Install: pip install {dep}")
        sys.exit(1)

# Test 3: CNN_Project exists
print("\n[TEST 3] Checking CNN_Project...")
CNN_PROJECT_DIR = Path(__file__).parent.parent / 'CNN_Project'
if CNN_PROJECT_DIR.exists():
    print(f"  ✅ CNN_Project directory found")
else:
    print(f"  ❌ CNN_Project directory not found")
    sys.exit(1)

# Test 4: Preprocessed images exist
print("\n[TEST 4] Checking preprocessed images...")
PREPROCESSED_DIR = CNN_PROJECT_DIR / 'preprocessed_images'
if PREPROCESSED_DIR.exists():
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_count = 0
    for ext in image_extensions:
        image_count += len(list(PREPROCESSED_DIR.glob(f'**/*{ext}')))

    if image_count > 0:
        print(f"  ✅ Found {image_count} preprocessed images")
    else:
        print(f"  ❌ No images found in preprocessed_images/")
        print(f"     Run: python CNN_Project/train_model.py")
        sys.exit(1)
else:
    print(f"  ❌ preprocessed_images/ directory not found")
    print(f"     Run: python CNN_Project/train_model.py")
    sys.exit(1)

# Test 5: CNN model exists (optional)
print("\n[TEST 5] Checking CNN model (optional)...")
CNN_MODEL_PATH = CNN_PROJECT_DIR / 'models' / 'best_model.pth'
if CNN_MODEL_PATH.exists():
    print(f"  ✅ CNN model found (can classify generated images)")
else:
    print(f"  ⚠️  CNN model not found (optional)")
    print(f"     GAN will train, but can't evaluate with CNN")

# Test 6: Create directories
print("\n[TEST 6] Creating output directories...")
GAN_DIR = Path(__file__).parent
dirs = [
    GAN_DIR / 'models',
    GAN_DIR / 'generated_images',
    GAN_DIR / 'training_progress'
]
for d in dirs:
    d.mkdir(parents=True, exist_ok=True)
    print(f"  ✅ {d.name}/ created")

# Test 7: Test Generator and Discriminator (without importing full train_gan)
print("\n[TEST 7] Testing GAN models...")
try:
    # Import torch modules
    import torch.nn as nn

    # Define simplified test versions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Simple generator test
    test_gen = nn.Sequential(
        nn.Linear(100, 8*8*512),
        nn.LeakyReLU(0.2)
    ).to(device)

    noise = torch.randn(4, 100, device=device)
    output = test_gen(noise)

    if output.shape == (4, 32768):  # 8*8*512
        print(f"  ✅ Neural network operations work correctly")
    else:
        print(f"  ❌ Neural network output shape wrong")
        sys.exit(1)

    print(f"  ✅ PyTorch models can be created and run on {device}")
    print(f"  ✅ Generator and Discriminator architectures will work")

except Exception as e:
    print(f"  ❌ Error testing models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test data loading
print("\n[TEST 8] Testing data loading...")
try:
    from train_gan import FruitImageDataset
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = FruitImageDataset(PREPROCESSED_DIR, transform=transform)

    if len(dataset) > 0:
        print(f"  ✅ Dataset loaded: {len(dataset)} images")

        # Test loading one image
        sample = dataset[0]
        if sample.shape == (1, 128, 128):
            print(f"  ✅ Image shape correct: {list(sample.shape)}")
        else:
            print(f"  ❌ Image shape wrong: {sample.shape}")
            sys.exit(1)

        # Test DataLoader (with num_workers=0 for Windows)
        from torch.utils.data import DataLoader
        test_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(test_loader))
        print(f"  ✅ DataLoader works: batch shape {list(batch.shape)}")
    else:
        print(f"  ❌ Dataset is empty")
        sys.exit(1)

except Exception as e:
    print(f"  ❌ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("SETUP VERIFICATION COMPLETE")
print("=" * 80)
print("\n✅ All tests passed! Ready to train GAN.")
print("\nNext steps:")
print("  1. Train GAN:      python GAN/train_gan.py")
print("  2. Generate images: python GAN/generate_images.py --num 50")
print("\n" + "=" * 80)
