"""
Check Available GAN Models
Shows what models are available and their status
"""

import os
import json
from pathlib import Path

print("=" * 80)
print("GAN MODELS STATUS")
print("=" * 80)

models_dir = Path('GAN/models')

if not models_dir.exists():
    print("\n❌ GAN/models directory not found!")
    print("Run: python GAN/train_gan.py")
    exit(1)

# Check for models index
index_path = models_dir / 'models_index.json'
if index_path.exists():
    with open(index_path, 'r') as f:
        index = json.load(f)

    print(f"\n{'='*80}")
    print("TRAINING STATUS")
    print(f"{'='*80}")
    if index.get('training_completed'):
        print("✅ Training completed")
        print(f"Total epochs: {index.get('total_epochs', 'N/A')}")
        print(f"Total time: {index.get('total_training_time', 'N/A')}")
        print(f"Best epoch: {index.get('best_epoch', 'N/A')}")
    else:
        print("⚠️  Training in progress or interrupted")

# Check key models
print(f"\n{'='*80}")
print("AVAILABLE MODELS")
print(f"{'='*80}")

key_models = {
    'best_model.pth': '⭐ Best quality (RECOMMENDED)',
    'latest_model.pth': '🔄 Most recent checkpoint',
    'final_gan.pth': '✅ Final model after all epochs'
}

for model_name, description in key_models.items():
    model_path = models_dir / model_name
    metadata_path = models_dir / model_name.replace('.pth', '_metadata.json')

    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ {model_name}")
        print(f"   {description}")
        print(f"   Size: {size_mb:.1f} MB")

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
            print(f"   Epoch: {meta.get('epoch', 'N/A')}")
            print(f"   D_loss: {meta.get('avg_D_loss', 0):.4f}")
            print(f"   G_loss: {meta.get('avg_G_loss', 0):.4f}")
            print(f"   D(fake): {meta.get('avg_D_fake', 0):.3f}")

            if 'best_d_fake_score' in meta:
                print(f"   Quality score: {meta['best_d_fake_score']:.4f} (lower is better)")
    else:
        print(f"\n❌ {model_name}")
        print(f"   Not found (training may not have reached this point)")

# Check checkpoints
print(f"\n{'='*80}")
print("CHECKPOINTS (every 10 epochs)")
print(f"{'='*80}")

checkpoints = sorted(models_dir.glob('checkpoint_epoch_*.pth'))
if checkpoints:
    print(f"\nFound {len(checkpoints)} checkpoint(s):")
    for cp in checkpoints:
        size_mb = cp.stat().st_size / (1024 * 1024)
        print(f"  ✅ {cp.name} ({size_mb:.1f} MB)")
else:
    print("\n❌ No checkpoints found")
    print("   (Checkpoints are saved every 10 epochs)")

# Usage instructions
print(f"\n{'='*80}")
print("USAGE")
print(f"{'='*80}")

if (models_dir / 'best_model.pth').exists():
    print("\n✅ You can generate images now!")
    print("\nRecommended command:")
    print("  python GAN/generate_images.py --num 50 --classify")
    print("\nOr specify a model:")
    print("  python GAN/generate_images.py --model GAN/models/best_model.pth --num 50")
    print("  python GAN/generate_images.py --model GAN/models/latest_model.pth --num 50")
else:
    print("\n⚠️  No usable models found yet")
    print("Continue training or wait for first epoch to complete")

# Model info
print(f"\n{'='*80}")
print("MODEL INFO")
print(f"{'='*80}")

metadata_files = list(models_dir.glob('*_metadata.json'))
if metadata_files:
    # Get most recent metadata
    latest_meta_path = models_dir / 'latest_metadata.json'
    if latest_meta_path.exists():
        with open(latest_meta_path, 'r') as f:
            meta = json.load(f)

        print(f"\nNoise dimension: {meta.get('noise_dim', 100)}")
        print(f"Image size: 128x128 grayscale")
        print(f"Output range: [-1, 1]")
        print(f"Compatible with: CNN classifier from CNN_Project")
        print(f"Last updated: {meta.get('timestamp', 'N/A')}")

print("\n" + "=" * 80)
