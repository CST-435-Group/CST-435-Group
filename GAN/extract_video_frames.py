"""
Extract Frames from Video (e.g., War Thunder gameplay)
Perfect for getting high-quality tank/vehicle images from gameplay recordings

Usage:
  python GAN/extract_video_frames.py
"""

import os
import sys
import cv2
from pathlib import Path
from tqdm import tqdm

def get_next_image_number(output_dir):
    """
    Find the highest numbered image in the directory and return next number

    Args:
        output_dir: Directory to check

    Returns:
        Next available image number
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        return 0

    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    all_files = []

    for ext in image_extensions:
        all_files.extend(list(output_path.rglob(f'*{ext}')))
        all_files.extend(list(output_path.rglob(f'*{ext.upper()}')))

    if len(all_files) == 0:
        return 0

    # Extract numbers from filenames
    max_number = -1

    for file_path in all_files:
        filename = file_path.stem  # Filename without extension

        # Try to extract number from filename
        # Handles formats like: "image_00123.png", "frame_456.jpg", "tank_789.png"
        import re
        numbers = re.findall(r'\d+', filename)

        if numbers:
            # Get the last number in the filename
            try:
                num = int(numbers[-1])
                if num > max_number:
                    max_number = num
            except:
                pass

    return max_number + 1

def extract_frames(video_path, output_dir, frame_skip=5, max_frames=None):
    """
    Extract frames from video file

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_skip: Save every Nth frame (default: 5 = save every 5th frame)
        max_frames: Maximum frames to extract (None = extract all)

    Returns:
        Number of frames extracted
    """
    video_path = Path(video_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return 0

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_seconds = total_frames / fps if fps > 0 else 0

    print(f"\n{'='*80}")
    print(f"VIDEO INFO")
    print(f"{'='*80}")
    print(f"File: {video_path.name}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration_seconds:.1f} seconds")
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"\nFrame skip: Saving every {frame_skip} frame(s)")

    estimated_output = total_frames // frame_skip
    if max_frames and estimated_output > max_frames:
        estimated_output = max_frames

    print(f"Estimated frames to save: ~{estimated_output}")

    # Get starting number (continue from existing files)
    start_number = get_next_image_number(output_dir)
    print(f"\n[INFO] Starting numbering from: {start_number}")
    print(f"Output directory: {output_dir}")

    # Extract frames
    print(f"\n{'='*80}")
    print("EXTRACTING FRAMES")
    print(f"{'='*80}\n")

    frame_count = 0
    saved_count = 0
    current_number = start_number

    pbar = tqdm(total=total_frames, desc="Processing video", unit="frames")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save every Nth frame
        if frame_count % frame_skip == 0:
            # Save frame
            output_file = output_path / f"frame_{current_number:06d}.png"
            cv2.imwrite(str(output_file), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            saved_count += 1
            current_number += 1

            # Check max frames limit
            if max_frames and saved_count >= max_frames:
                print(f"\n[INFO] Reached max frames limit: {max_frames}")
                break

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nTotal frames processed: {frame_count}")
    print(f"Frames saved: {saved_count}")
    print(f"Saved to: {output_dir}")
    print(f"File naming: frame_{start_number:06d}.png to frame_{current_number-1:06d}.png")

    return saved_count

def main():
    print("=" * 80)
    print("VIDEO FRAME EXTRACTOR")
    print("Extract frames from War Thunder (or any video) for GAN training")
    print("=" * 80)

    # Check if opencv is installed
    try:
        import cv2
    except ImportError:
        print("\n[ERROR] OpenCV not installed!")
        print("\nInstalling opencv-python...")
        os.system(f'"{sys.executable}" -m pip install opencv-python')
        print("\nPlease run the script again.")
        return

    # Get video file
    print("\n[STEP 1] Select video file")
    video_path = input("Enter path to video file: ").strip().strip('"')

    if not video_path:
        print("[ERROR] No video path provided")
        return

    video_path = Path(video_path)

    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        return

    # Get output directory
    print("\n[STEP 2] Select output directory")
    print("Default: GAN/datasets/military_vehicles_raw")

    output_dir = input("Enter output directory (or press Enter for default): ").strip().strip('"')

    if not output_dir:
        output_dir = 'GAN/datasets/military_vehicles_raw'

    # Get frame skip
    print("\n[STEP 3] Frame extraction settings")
    print("Frame skip: Save every Nth frame")
    print("  1 = Save every frame (LOTS of images)")
    print("  5 = Save every 5th frame (recommended for 60fps video)")
    print("  10 = Save every 10th frame (recommended for 30fps video)")
    print("  30 = Save every 30th frame (1 per second at 30fps)")

    frame_skip_input = input("Frame skip (default 5): ").strip()

    try:
        frame_skip = int(frame_skip_input) if frame_skip_input else 5
    except:
        frame_skip = 5

    # Get max frames
    print("\n[STEP 4] Maximum frames (optional)")
    print("Limit total frames extracted (leave empty for no limit)")

    max_frames_input = input("Max frames (or press Enter for no limit): ").strip()

    try:
        max_frames = int(max_frames_input) if max_frames_input else None
    except:
        max_frames = None

    # Confirm
    print("\n" + "=" * 80)
    print("READY TO EXTRACT")
    print("=" * 80)
    print(f"Video: {video_path.name}")
    print(f"Output: {output_dir}")
    print(f"Frame skip: Every {frame_skip} frame(s)")
    print(f"Max frames: {max_frames if max_frames else 'No limit'}")

    confirm = input("\nProceed? (y/n): ").strip().lower()

    if confirm != 'y':
        print("Cancelled")
        return

    # Extract frames
    saved_count = extract_frames(video_path, output_dir, frame_skip, max_frames)

    if saved_count > 0:
        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print(f"\n✅ Extracted {saved_count} frames")
        print(f"📁 Location: {output_dir}")
        print("\nNext steps:")
        print("  1. Extract more videos if needed (script won't overwrite)")
        print("  2. Run preprocessing: python GAN/preprocess_dataset.py")
        print("  3. Train GAN: python GAN/train_gan_conditional.py")

if __name__ == "__main__":
    main()
