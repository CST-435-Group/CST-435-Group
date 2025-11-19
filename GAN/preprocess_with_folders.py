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
        Leopard 2 tank/
          frame_001.png
          frame_002.png
        M1 Abrams/
          frame_003.png
          frame_004.png

    Output structure:
      output_dir/
        leopard2/
          leopard2_00000.png
          leopard2_00001.png
        m1abrams/
          m1abrams_00000.png
          m1abrams_00001.png
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    print("=" * 80)
    print("PREPROCESSING WITH FOLDER STRUCTURE PRESERVATION")
    print("=" * 80)
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")

    # Find all subdirectories (tank types)
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]

    if len(subdirs) == 0:
        print("\n[ERROR] No subdirectories found!")
        print("Expected structure: input_dir/TankType/images...")
        return

    print(f"\n[STEP 1] Found {len(subdirs)} classes:")
    for subdir in subdirs:
        print(f"  - {subdir.name}")

    # Create mapping of folder names to clean class names
    class_mapping = {}
    for subdir in subdirs:
        # Clean folder name for class name
        # "Leopard 2 tank" -> "leopard2"
        # "M1 Abrams tank" -> "m1abrams"
        # "T-90 battle tank" -> "t90"
        original_name = subdir.name.lower()

        # Remove common words first
        clean_name = original_name.replace(' tank', '').replace('tank', '')
        clean_name = clean_name.replace(' battle', '').replace('battle', '')
        clean_name = clean_name.replace(' main', '').replace('main', '')

        # Remove spaces, hyphens, underscores
        clean_name = clean_name.replace(' ', '').replace('-', '').replace('_', '').strip()

        # Special handling for specific tank types
        if 'leopard' in clean_name and '2' in clean_name:
            clean_name = 'leopard2'
        elif 'leopard' in clean_name:
            clean_name = 'leopard'
        elif 'm1' in clean_name or 'abrams' in clean_name:
            clean_name = 'm1abrams'
        elif 't90m' in clean_name or 't-90m' in original_name:
            clean_name = 't90m'
        elif 't90' in clean_name or 't-90' in original_name:
            clean_name = 't90'

        # If clean_name is empty after all replacements, use a sanitized version of original
        if not clean_name:
            clean_name = ''.join(c for c in original_name if c.isalnum()).lower()

        class_mapping[subdir.name] = clean_name

    print(f"\nClass name mapping:")
    for original, clean in class_mapping.items():
        print(f"  '{original}' -> '{clean}'")

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
    output_dir = 'GAN/datasets/military_vehicles_processed'

    print(f"\nThis script will:")
    print(f"  1. Find all subdirectories in: {input_dir}")
    print(f"  2. Process each as a separate class")
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
        print("\n1. Update training script:")
        print("   Open: GAN/train_gan_conditional.py")
        print(f"   Change line 40: DATASET_PATH = '{output_dir}'")
        print("\n2. Train:")
        print("   python GAN/train_gan_conditional.py")
        print("\n3. Generate:")
        print("   python GAN/generate_images_conditional.py --class leopard2 --num 10")
        print("   python GAN/generate_images_conditional.py --class m1abrams --num 10")
        print("   python GAN/generate_images_conditional.py --class t90 --num 10")

if __name__ == "__main__":
    main()
