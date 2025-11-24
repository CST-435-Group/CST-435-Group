"""
Organize classified images into folder structure for dual conditional GAN training

Reads view_classifications.json and organizes images into:
  datasets/military_vehicles_with_views/
    {tank_type}_front/
    {tank_type}_side/
    {tank_type}_back/
    {tank_type}_top/
    ...
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def organize_views_dataset(
    classifications_file='view_classifications.json',
    output_dir='datasets/military_vehicles_with_views',
    copy_files=True
):
    """
    Organize classified images into folder structure

    Args:
        classifications_file: Path to view_classifications.json
        output_dir: Output directory for organized dataset
        copy_files: If True, copy files; if False, create symlinks
    """

    # Load classifications
    print(f"Loading classifications from {classifications_file}...")
    with open(classifications_file, 'r') as f:
        classifications = json.load(f)

    # Filter out skipped images
    valid_classifications = {
        path: view for path, view in classifications.items()
        if view not in ['skip', None, '']
    }

    print(f"Total classified images: {len(classifications)}")
    print(f"Valid images (not skipped): {len(valid_classifications)}")

    # Count by view
    view_counts = {}
    for view in valid_classifications.values():
        view_counts[view] = view_counts.get(view, 0) + 1

    print(f"\nView distribution:")
    for view, count in sorted(view_counts.items()):
        print(f"  {view}: {count} images")

    # Count by tank type
    tank_counts = {}
    for path in valid_classifications.keys():
        # Extract tank type from path (assumes path like: .../tank_type/image.png)
        parts = Path(path).parts
        if len(parts) >= 2:
            tank_type = parts[-2]  # Second to last part is the tank type folder
            tank_counts[tank_type] = tank_counts.get(tank_type, 0) + 1

    print(f"\nTank type distribution:")
    for tank, count in sorted(tank_counts.items()):
        print(f"  {tank}: {count} images")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nCreating organized dataset in: {output_path}")

    # Organize images
    print(f"\nOrganizing images...")
    stats = {}
    errors = []

    for img_path, view in tqdm(valid_classifications.items(), desc="Processing"):
        src_path = Path(img_path)

        # Check if source exists
        if not src_path.exists():
            errors.append(f"Source not found: {src_path}")
            continue

        # Extract tank type from path
        parts = src_path.parts
        if len(parts) < 2:
            errors.append(f"Invalid path structure: {src_path}")
            continue

        tank_type = parts[-2]

        # Create destination folder: {tank_type}_{view}
        class_name = f"{tank_type}_{view}"
        dest_folder = output_path / class_name
        dest_folder.mkdir(parents=True, exist_ok=True)

        # Destination file
        dest_file = dest_folder / src_path.name

        # Copy or link file
        try:
            if copy_files:
                shutil.copy2(src_path, dest_file)
            else:
                # Create relative symlink if possible
                if dest_file.exists():
                    dest_file.unlink()
                os.symlink(src_path.resolve(), dest_file)

            # Track stats
            if class_name not in stats:
                stats[class_name] = 0
            stats[class_name] += 1

        except Exception as e:
            errors.append(f"Error processing {src_path}: {e}")

    # Print summary
    print(f"\n{'='*60}")
    print("ORGANIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nCreated {len(stats)} class folders:")
    for class_name, count in sorted(stats.items()):
        print(f"  {class_name}: {count} images")

    if errors:
        print(f"\n⚠️  Encountered {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    print(f"\n✅ Dataset ready for training at: {output_path}")
    print(f"   Run: python train_gan_dual_conditional.py")


if __name__ == '__main__':
    import sys

    # Parse args
    copy_files = '--symlink' not in sys.argv

    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        print("\nOptions:")
        print("  --symlink   Create symlinks instead of copying files (saves disk space)")
        sys.exit(0)

    organize_views_dataset(copy_files=copy_files)
