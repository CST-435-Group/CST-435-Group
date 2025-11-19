"""
Convert existing large model files to chunked format
Run this once to convert models that exceed GitHub's 100MB limit
"""

import torch
import os
from pathlib import Path
from model_utils import save_model_chunked, clean_chunks

print("=" * 80)
print("CONVERTING EXISTING MODELS TO CHUNKED FORMAT")
print("=" * 80)

# Check both GAN/models and GAN/GAN/models
search_paths = [Path('GAN/models'), Path('GAN/GAN/models')]
model_files = []

for search_path in search_paths:
    if search_path.exists():
        model_files.extend(list(search_path.glob('**/*.pth')))

if not model_files:
    print("\nNo model files found in GAN/models/")
    exit(0)

print(f"\nFound {len(model_files)} model file(s)")

converted_count = 0
skipped_count = 0

for model_path in model_files:
    # Skip if already chunked (check for manifest)
    manifest_path = model_path.parent / f"{model_path.stem}.manifest.json"
    if manifest_path.exists():
        print(f"\n[SKIP] {model_path}")
        print(f"  Already chunked")
        skipped_count += 1
        continue

    # Check file size
    size_mb = model_path.stat().st_size / (1024 * 1024)

    print(f"\n[PROCESSING] {model_path}")
    print(f"  Current size: {size_mb:.1f} MB")

    if size_mb < 90:
        print(f"  [SKIP] Under 90MB threshold - no chunking needed")
        skipped_count += 1
        continue

    try:
        # Load the model
        print(f"  Loading model...")
        model_data = torch.load(model_path, map_location='cpu')

        # Check if it's just a placeholder for chunked model
        if isinstance(model_data, dict) and model_data.get('is_chunked'):
            print(f"  [SKIP] Already a placeholder for chunked model")
            skipped_count += 1
            continue

        # Delete the original file first
        print(f"  Removing original file...")
        os.remove(model_path)

        # Save with chunking
        print(f"  Saving with chunking...")
        save_model_chunked(model_data, str(model_path))

        print(f"  [SUCCESS] Converted to chunked format")
        converted_count += 1

    except Exception as e:
        print(f"  [ERROR] Failed to convert: {e}")
        # Try to restore if possible
        continue

print("\n" + "=" * 80)
print("CONVERSION COMPLETE")
print("=" * 80)
print(f"\nConverted: {converted_count} model(s)")
print(f"Skipped: {skipped_count} model(s)")

if converted_count > 0:
    print(f"\n[SUCCESS] {converted_count} large model(s) have been chunked for GitHub")
    print("You can now commit these files to Git:")
    print("  - Each chunk is under 90MB")
    print("  - Models will load transparently when needed")

print("\n" + "=" * 80)
