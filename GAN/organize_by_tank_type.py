"""
Organize War Thunder Frames by Tank Type
Helps you label frames as Leopard 2, M1 Abrams, or T-90M

Usage:
  python GAN/organize_by_tank_type.py
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import cv2

def show_image(image_path):
    """Display image using OpenCV for classification"""
    img = cv2.imread(str(image_path))
    if img is not None:
        # Resize for display if too large
        height, width = img.shape[:2]
        max_display = 800
        if width > max_display or height > max_display:
            scale = max_display / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))

        cv2.imshow('Classify This Tank', img)
        cv2.waitKey(1)  # Update window

def organize_frames(input_dir, output_dir):
    """
    Organize frames into tank-specific folders

    Args:
        input_dir: Directory with raw frames
        output_dir: Directory for organized frames
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create class folders
    classes = ['leopard2', 'm1abrams', 't90m']
    class_folders = {}

    for class_name in classes:
        folder = output_path / class_name
        folder.mkdir(parents=True, exist_ok=True)
        class_folders[class_name] = folder

    print("=" * 80)
    print("ORGANIZE FRAMES BY TANK TYPE")
    print("=" * 80)
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")
    print("\nClasses:")
    print("  [1] Leopard 2")
    print("  [2] M1 Abrams")
    print("  [3] T-90M")
    print("  [s] Skip this frame")
    print("  [q] Quit")

    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    all_files = []

    for ext in image_extensions:
        all_files.extend(list(input_path.glob(f'*{ext}')))
        all_files.extend(list(input_path.glob(f'*{ext.upper()}')))

    all_files.sort()  # Sort by filename

    if len(all_files) == 0:
        print(f"\n[ERROR] No images found in {input_dir}")
        return

    print(f"\nFound {len(all_files)} frames to organize")

    # Count existing files in each class folder
    counts = {
        'leopard2': len(list(class_folders['leopard2'].glob('*.png'))),
        'm1abrams': len(list(class_folders['m1abrams'].glob('*.png'))),
        't90m': len(list(class_folders['t90m'].glob('*.png')))
    }

    print(f"\nExisting images in output:")
    print(f"  Leopard 2: {counts['leopard2']}")
    print(f"  M1 Abrams: {counts['m1abrams']}")
    print(f"  T-90M: {counts['t90m']}")

    # Process frames
    print("\n" + "=" * 80)
    print("CLASSIFICATION")
    print("=" * 80)
    print("\nFor each frame, press:")
    print("  1 = Leopard 2")
    print("  2 = M1 Abrams")
    print("  3 = T-90M")
    print("  s = Skip")
    print("  q = Quit")
    print("\n" + "=" * 80 + "\n")

    organized = {'leopard2': 0, 'm1abrams': 0, 't90m': 0}
    skipped = 0

    try:
        for i, img_path in enumerate(all_files):
            print(f"\n[{i+1}/{len(all_files)}] {img_path.name}")

            # Show image
            show_image(img_path)

            # Get user input
            while True:
                choice = input("Classify (1/2/3/s/q): ").strip().lower()

                if choice == 'q':
                    print("\nQuitting...")
                    cv2.destroyAllWindows()
                    raise KeyboardInterrupt

                if choice == 's':
                    skipped += 1
                    break

                if choice in ['1', '2', '3']:
                    # Map choice to class
                    class_map = {'1': 'leopard2', '2': 'm1abrams', '3': 't90m'}
                    selected_class = class_map[choice]

                    # Copy file to class folder
                    output_file = class_folders[selected_class] / f"{selected_class}_{counts[selected_class]:05d}.png"

                    # Load and save as PNG (ensures consistent format)
                    img = Image.open(img_path)
                    img.save(output_file, 'PNG')

                    counts[selected_class] += 1
                    organized[selected_class] += 1
                    print(f"  → Saved as {selected_class}")
                    break
                else:
                    print("  Invalid choice. Use 1, 2, 3, s, or q")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        cv2.destroyAllWindows()

    # Summary
    print("\n" + "=" * 80)
    print("ORGANIZATION COMPLETE")
    print("=" * 80)
    print(f"\nProcessed: {i+1}/{len(all_files)} frames")
    print(f"\nOrganized this session:")
    print(f"  Leopard 2: {organized['leopard2']}")
    print(f"  M1 Abrams: {organized['m1abrams']}")
    print(f"  T-90M: {organized['t90m']}")
    print(f"  Skipped: {skipped}")
    print(f"\nTotal in output folders:")
    print(f"  Leopard 2: {counts['leopard2']}")
    print(f"  M1 Abrams: {counts['m1abrams']}")
    print(f"  T-90M: {counts['t90m']}")
    print(f"\nOutput location: {output_dir}")

def batch_organize_frames(input_dir, output_dir):
    """
    Organize frames by asking for batch classification
    Useful when you have consecutive frames of the same tank
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create class folders
    classes = ['leopard2', 'm1abrams', 't90m']
    class_folders = {}

    for class_name in classes:
        folder = output_path / class_name
        folder.mkdir(parents=True, exist_ok=True)
        class_folders[class_name] = folder

    print("=" * 80)
    print("BATCH ORGANIZE FRAMES BY TANK TYPE")
    print("=" * 80)
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")

    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    all_files = []

    for ext in image_extensions:
        all_files.extend(list(input_path.glob(f'*{ext}')))
        all_files.extend(list(input_path.glob(f'*{ext.upper()}')))

    all_files.sort()

    if len(all_files) == 0:
        print(f"\n[ERROR] No images found in {input_dir}")
        return

    print(f"\nFound {len(all_files)} frames")

    # Count existing files
    counts = {
        'leopard2': len(list(class_folders['leopard2'].glob('*.png'))),
        'm1abrams': len(list(class_folders['m1abrams'].glob('*.png'))),
        't90m': len(list(class_folders['t90m'].glob('*.png')))
    }

    print("\n" + "=" * 80)
    print("BATCH CLASSIFICATION")
    print("=" * 80)
    print("\nYou'll classify ranges of frames (e.g., frame_0000 to frame_0150)")
    print("Useful when consecutive frames show the same tank")
    print("\n" + "=" * 80 + "\n")

    print(f"First frame: {all_files[0].name}")
    print(f"Last frame: {all_files[-1].name}")

    while True:
        print("\n" + "=" * 80)
        print("Enter range to classify:")

        start_input = input("Start frame number (or 'q' to quit): ").strip()
        if start_input.lower() == 'q':
            break

        end_input = input("End frame number: ").strip()

        try:
            start_idx = int(start_input)
            end_idx = int(end_input)
        except:
            print("[ERROR] Invalid numbers")
            continue

        # Show sample from range
        sample_frames = [f for f in all_files if start_idx <= int(''.join(filter(str.isdigit, f.stem))) <= end_idx]

        if len(sample_frames) == 0:
            print(f"[ERROR] No frames found in range {start_idx}-{end_idx}")
            continue

        print(f"\nFound {len(sample_frames)} frames in range")
        print(f"First: {sample_frames[0].name}")
        print(f"Last: {sample_frames[-1].name}")

        # Show first frame
        if len(sample_frames) > 0:
            show_image(sample_frames[0])

        print("\nWhich tank type are these frames?")
        print("  [1] Leopard 2")
        print("  [2] M1 Abrams")
        print("  [3] T-90M")
        print("  [s] Skip this batch")

        choice = input("Choose (1/2/3/s): ").strip().lower()

        if choice == 's':
            cv2.destroyAllWindows()
            continue

        if choice not in ['1', '2', '3']:
            print("[ERROR] Invalid choice")
            cv2.destroyAllWindows()
            continue

        class_map = {'1': 'leopard2', '2': 'm1abrams', '3': 't90m'}
        selected_class = class_map[choice]

        # Copy all frames in range
        print(f"\nCopying {len(sample_frames)} frames to {selected_class}...")

        for frame_path in sample_frames:
            output_file = class_folders[selected_class] / f"{selected_class}_{counts[selected_class]:05d}.png"
            img = Image.open(frame_path)
            img.save(output_file, 'PNG')
            counts[selected_class] += 1

        print(f"✓ Copied {len(sample_frames)} frames")
        cv2.destroyAllWindows()

    cv2.destroyAllWindows()

    # Summary
    print("\n" + "=" * 80)
    print("BATCH ORGANIZATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal in output folders:")
    print(f"  Leopard 2: {counts['leopard2']}")
    print(f"  M1 Abrams: {counts['m1abrams']}")
    print(f"  T-90M: {counts['t90m']}")
    print(f"\nOutput location: {output_dir}")

def main():
    print("=" * 80)
    print("ORGANIZE WAR THUNDER FRAMES BY TANK TYPE")
    print("=" * 80)

    input_dir = 'GAN/datasets/military_vehicles_raw'
    output_dir = 'GAN/datasets/military_vehicles_organized'

    print("\nOrganization methods:")
    print("  [1] One-by-one - Classify each frame individually")
    print("  [2] Batch - Classify ranges of frames (faster if consecutive frames are same tank)")

    method = input("\nChoose method (1/2): ").strip()

    if method == '1':
        organize_frames(input_dir, output_dir)
    elif method == '2':
        batch_organize_frames(input_dir, output_dir)
    else:
        print("Invalid choice")
        return

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Preprocess organized frames:")
    print("   python GAN/preprocess_dataset.py")
    print("   Select: military_vehicles_organized")
    print("\n2. Train conditional GAN:")
    print("   python GAN/train_gan_conditional.py")
    print("\n3. Generate specific tanks:")
    print("   python GAN/generate_images_conditional.py --class leopard2 --num 10")
    print("   python GAN/generate_images_conditional.py --class m1abrams --num 10")
    print("   python GAN/generate_images_conditional.py --class t90m --num 10")

if __name__ == "__main__":
    main()
