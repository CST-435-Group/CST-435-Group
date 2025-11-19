"""
Preprocess Downloaded Images for GAN Training
- Smart resize: shortest side to 200px, then center crop to 200x200
- Convert to RGB
- Remove corrupted images, GIFs, and bad aspect ratios
- Filter out images with people (if possible)
- Organize into training directory
"""

import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil
import numpy as np

TARGET_SIZE = (200, 200)
MIN_ASPECT_RATIO = 0.5  # Reject images narrower than 1:2
MAX_ASPECT_RATIO = 2.0  # Reject images wider than 2:1

def smart_crop_resize(img, target_size=(200, 200)):
    """
    Smart resize and crop to target size maintaining aspect ratio

    Process:
    1. Resize so shortest side = 200px
    2. Center crop to 200x200

    Args:
        img: PIL Image
        target_size: Target dimensions (width, height)

    Returns:
        PIL Image resized and cropped to target_size
    """
    target_width, target_height = target_size

    # Get current dimensions
    width, height = img.size

    # Calculate scaling factor (shortest side to target size)
    if width < height:
        # Width is shorter
        scale = target_width / width
    else:
        # Height is shorter
        scale = target_height / height

    # Resize maintaining aspect ratio
    new_width = int(width * scale)
    new_height = int(height * scale)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Center crop to target size
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    img = img.crop((left, top, right, bottom))

    return img

def has_people_features(img):
    """
    Simple heuristic to detect if image might contain people
    Checks for skin-tone colors in upper portion of image

    Returns:
        True if likely contains people, False otherwise
    """
    try:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Sample upper 40% of image (where faces usually are)
        width, height = img.size
        upper_region = img.crop((0, 0, width, int(height * 0.4)))

        # Convert to numpy array
        img_array = np.array(upper_region)

        # Define skin tone ranges (R > G > B, with specific ranges)
        # This is a simple heuristic
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]

        # Skin tone detection (rough heuristic)
        skin_mask = (
            (r > 95) & (g > 40) & (b > 20) &
            (r > g) & (r > b) &
            (abs(r - g) > 15)
        )

        skin_percentage = np.sum(skin_mask) / skin_mask.size

        # If more than 10% skin tones in upper region, likely has people
        return skin_percentage > 0.10
    except:
        return False

def preprocess_images(input_dir, output_dir, target_size=(200, 200)):
    """
    Preprocess images for GAN training with smart cropping and filtering

    Args:
        input_dir: Directory with raw downloaded images
        output_dir: Directory for processed images
        target_size: Target image size (width, height)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PREPROCESSING IMAGES FOR GAN TRAINING")
    print("=" * 80)
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print(f"Method: Smart resize (shortest side) + center crop")

    # Find all images recursively
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    all_files = []

    print("\n[STEP 1] Scanning for images...")
    for ext in image_extensions:
        all_files.extend(input_path.rglob(f'*{ext}'))
        all_files.extend(input_path.rglob(f'*{ext.upper()}'))

    print(f"Found {len(all_files)} image files")

    if len(all_files) == 0:
        print("[ERROR] No images found!")
        return 0

    # Process images
    print("\n[STEP 2] Processing images...")
    print("  - Converting to RGB")
    print("  - Smart resizing and cropping")
    print("  - Filtering GIFs and bad images")
    print("  - Detecting and filtering people (heuristic)")

    successful = 0
    failed = 0
    skipped_gif = 0
    skipped_aspect = 0
    skipped_people = 0
    skipped_small = 0

    for i, img_path in enumerate(tqdm(all_files, desc="Processing")):
        try:
            # Skip GIF files
            if img_path.suffix.lower() == '.gif':
                skipped_gif += 1
                continue

            # Open image
            img = Image.open(img_path)

            # Get original dimensions
            width, height = img.size

            # Skip very small images (less than 100px on any side)
            if width < 100 or height < 100:
                skipped_small += 1
                continue

            # Check aspect ratio (reject extreme ratios)
            aspect_ratio = width / height
            if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                skipped_aspect += 1
                continue

            # Convert to RGB (handles grayscale, RGBA, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Check for people (simple heuristic)
            if has_people_features(img):
                skipped_people += 1
                continue

            # Smart resize and crop
            img = smart_crop_resize(img, target_size)

            # Save with new name
            output_file = output_path / f"image_{successful:05d}.png"
            img.save(output_file, 'PNG', quality=95)

            successful += 1

        except Exception as e:
            failed += 1
            continue

    print(f"\n[STEP 3] Cleanup and summary")
    print(f"  Successful: {successful}")
    print(f"  Failed/Corrupted: {failed}")
    print(f"  Skipped (GIF): {skipped_gif}")
    print(f"  Skipped (bad aspect ratio): {skipped_aspect}")
    print(f"  Skipped (likely people): {skipped_people}")
    print(f"  Skipped (too small): {skipped_small}")
    print(f"  Total processed: {len(all_files)}")
    print(f"  Success rate: {(successful/len(all_files)*100):.1f}%")

    # Calculate dataset size
    if successful > 0:
        sample_size = os.path.getsize(output_path / "image_00000.png")
        total_size_mb = (sample_size * successful) / (1024 * 1024)
        print(f"\n[INFO] Dataset size: ~{total_size_mb:.1f} MB")
        print(f"[INFO] Images per batch (64): {successful // 64} batches")

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nProcessed images: {output_dir}")
    print(f"Ready for training: {successful} images")

    if successful > 500:
        print("\n✅ Dataset size is good for GAN training (>500 images)")
    elif successful > 200:
        print("\n⚠️  Dataset is small but workable (200-500 images)")
    else:
        print("\n❌ Dataset is too small (<200 images) - download more")

    return successful

def find_raw_datasets():
    """Find available raw datasets"""
    datasets_dir = Path('GAN/datasets')

    if not datasets_dir.exists():
        return []

    raw_dirs = [d for d in datasets_dir.iterdir()
                if d.is_dir() and ('raw' in d.name or 'downloaded' in d.name)]

    return raw_dirs

def main():
    print("=" * 80)
    print("IMAGE PREPROCESSING FOR GAN")
    print("=" * 80)

    # Find available datasets
    raw_datasets = find_raw_datasets()

    if not raw_datasets:
        print("\n[ERROR] No raw datasets found!")
        print("\nRun 'python GAN/download_dataset.py' first to download images")
        return

    print("\nAvailable datasets:")
    for i, dataset in enumerate(raw_datasets, 1):
        # Count images
        image_count = len(list(dataset.rglob('*.jpg'))) + \
                     len(list(dataset.rglob('*.png'))) + \
                     len(list(dataset.rglob('*.jpeg')))
        print(f"{i}. {dataset.name} ({image_count} images)")

    # Get user choice
    if len(raw_datasets) == 1:
        choice = 1
        print(f"\nAuto-selecting: {raw_datasets[0].name}")
    else:
        choice = input(f"\nSelect dataset (1-{len(raw_datasets)}): ").strip()
        try:
            choice = int(choice)
        except:
            print("[ERROR] Invalid choice")
            return

    if choice < 1 or choice > len(raw_datasets):
        print("[ERROR] Invalid choice")
        return

    input_dir = str(raw_datasets[choice - 1])

    # Output directory
    dataset_name = raw_datasets[choice - 1].name.replace('_raw', '').replace('_downloaded', '')
    output_dir = f'GAN/datasets/{dataset_name}_processed'

    print(f"\n[PROCESSING] {dataset_name}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    confirm = input("\nProceed? (y/n): ").strip().lower()

    if confirm != 'y':
        print("Cancelled")
        return

    # Process images
    count = preprocess_images(input_dir, output_dir, TARGET_SIZE)

    if count > 0:
        print("\n[NEXT STEP] Update your training script:")
        print(f"  DATASET_PATH = '{output_dir}'")
        print(f"  IMAGE_SIZE = 200")
        print(f"  CHANNELS = 3  # RGB color images")

if __name__ == "__main__":
    main()
