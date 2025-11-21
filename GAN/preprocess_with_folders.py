"""
Preprocess Images While Preserving Folder Structure
Perfect for when you already have organized folders (e.g., Leopard 2 tank/, M1 Abrams/, etc.)
"""

import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

TARGET_SIZE = (200, 200)
MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 2.0

def smart_crop_resize(img, target_size=(200, 200)):
    """Smart resize and crop to target size maintaining aspect ratio"""
    target_width, target_height = target_size
    width, height = img.size

    # Calculate scaling factor (shortest side to target size)
    if width < height:
        scale = target_width / width
    else:
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
    """Simple heuristic to detect if image might contain people"""
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')

        width, height = img.size
        upper_region = img.crop((0, 0, width, int(height * 0.4)))
        img_array = np.array(upper_region)

        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]

        skin_mask = (
            (r > 95) & (g > 40) & (b > 20) &
            (r > g) & (r > b) &
            (abs(r - g) > 15)
        )

        skin_percentage = np.sum(skin_mask) / skin_mask.size
        return skin_percentage > 0.10
    except:
        return False

def preprocess_with_folders(input_dir, output_dir, target_size=(200, 200)):
    """
    Preprocess images while maintaining folder structure

    Input structure:
      input_dir/
        M1A1_Abrams_front/
          *.png
        M1A1_Abrams_side/
          *.png
        Leopard2_top/
          *.png

    Output structure:
      output_dir/
        M1A1_Abrams_front/
          *.png
        M1A1_Abrams_side/
          *.png
        Leopard2_top/
          *.png
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    print("=" * 80)
    print("PREPROCESSING WITH FOLDER STRUCTURE PRESERVATION")
    print("=" * 80)
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")

    # Find all subdirectories (tank types + views)
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]

    if len(subdirs) == 0:
        print("\n[ERROR] No subdirectories found!")
        print("Expected structure: input_dir/TankType_view/images...")
        return

    print(f"\n[STEP 1] Found {len(subdirs)} classes:")
    for subdir in subdirs:
        print(f"  - {subdir.name}")

    # No name mapping - preserve folder names as-is
    # Folders should already be named like: M1A1_Abrams_front, Leopard2_side, etc.
    class_mapping = {}
    for subdir in subdirs:
        class_mapping[subdir.name] = subdir.name

    print(f"\nPreserving original class names (no mapping)")

    # Process each class folder
    print(f"\n[STEP 2] Processing images...")

    total_stats = {
        'successful': 0,
        'failed': 0,
        'skipped_gif': 0,
        'skipped_aspect': 0,
        'skipped_people': 0,
        'skipped_small': 0
    }

    class_counts = {}

    for subdir in subdirs:
        class_name = class_mapping[subdir.name]
        class_output_dir = output_path / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Processing: {subdir.name} -> {class_name}/")

        # Find all images in this subfolder
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        class_files = []

        for ext in image_extensions:
            class_files.extend(list(subdir.glob(f'*{ext}')))
            class_files.extend(list(subdir.glob(f'*{ext.upper()}')))

        print(f"    Found {len(class_files)} images")

        if len(class_files) == 0:
            continue

        # Process images for this class
        successful = 0
        stats = {
            'failed': 0,
            'skipped_gif': 0,
            'skipped_aspect': 0,
            'skipped_people': 0,
            'skipped_small': 0
        }

        for img_path in tqdm(class_files, desc=f"    {class_name}", leave=False):
            try:
                # Skip GIF files
                if img_path.suffix.lower() == '.gif':
                    stats['skipped_gif'] += 1
                    continue

                # Open image
                img = Image.open(img_path)
                width, height = img.size

                # Skip very small images
                if width < 100 or height < 100:
                    stats['skipped_small'] += 1
                    continue

                # Check aspect ratio
                aspect_ratio = width / height
                if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                    stats['skipped_aspect'] += 1
                    continue

                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Check for people (disabled for tanks - War Thunder shouldn't have people)
                # if has_people_features(img):
                #     stats['skipped_people'] += 1
                #     continue

                # Smart resize and crop
                img = smart_crop_resize(img, target_size)

                # Save with class-specific naming
                output_file = class_output_dir / f"{class_name}_{successful:05d}.png"
                img.save(output_file, 'PNG', quality=95)

                successful += 1

            except Exception as e:
                stats['failed'] += 1
                continue

        print(f"    → Saved: {successful} images")

        # Update totals
        total_stats['successful'] += successful
        for key in stats:
            total_stats[key] += stats[key]

        class_counts[class_name] = successful

    # Summary
    print(f"\n{'='*80}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"\nTotal Results:")
    print(f"  Successful: {total_stats['successful']}")
    print(f"  Failed/Corrupted: {total_stats['failed']}")
    print(f"  Skipped (GIF): {total_stats['skipped_gif']}")
    print(f"  Skipped (bad aspect ratio): {total_stats['skipped_aspect']}")
    print(f"  Skipped (too small): {total_stats['skipped_small']}")

    print(f"\nPer-class breakdown:")
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / total_stats['successful'] * 100) if total_stats['successful'] > 0 else 0
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")

    print(f"\nOutput structure:")
    for class_name in sorted(class_counts.keys()):
        print(f"  {output_dir}/{class_name}/")
        print(f"    {class_name}_00000.png")
        print(f"    {class_name}_00001.png")
        print(f"    ...")

    if total_stats['successful'] > 0:
        print(f"\n✅ Ready for training!")
        print(f"\nNext step: Update train_gan_conditional.py:")
        print(f"  DATASET_PATH = '{output_dir}'")

    return total_stats['successful']

def main():
    print("=" * 80)
    print("PREPROCESS IMAGES WITH FOLDER STRUCTURE")
    print("=" * 80)

    input_dir = 'GAN/datasets/military_vehicles_raw'
    output_dir = 'GAN/datasets/military_vehicles_with_views'

    print(f"\nThis script will:")
    print(f"  1. Find all subdirectories in: {input_dir}")
    print(f"  2. Process each as a separate class (preserving tank+view names)")
    print(f"  3. Maintain folder structure in: {output_dir}")
    print(f"  4. Resize/crop all images to 200x200")
    print(f"  5. Filter out bad images")

    confirm = input("\nProceed? (y/n): ").strip().lower()

    if confirm != 'y':
        print("Cancelled")
        return

    count = preprocess_with_folders(input_dir, output_dir, TARGET_SIZE)

    if count > 0:
        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print(f"\n✅ Processed {count} images")
        print(f"📁 Output: {output_dir}")
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\nTrain dual conditional GAN:")
        print("   python train_gan_dual_conditional.py")
        print("\nGenerate specific tank+view combinations:")
        print("   The GAN will automatically parse tank type and view angle from folder names")

if __name__ == "__main__":
    main()
