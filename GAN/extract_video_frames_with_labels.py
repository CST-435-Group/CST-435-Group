"""
Extract Frames from Video with Tank Type and View Labels
Records videos of specific tank views (e.g., M1A1 top view, M1A1 side view)
and extracts frames with proper labeling for dual conditional GAN

Usage:
  python extract_video_frames_with_labels.py
"""

import os
import sys
import cv2
from pathlib import Path
from tqdm import tqdm

def extract_frames_with_labels(video_path, tank_type, view_angle, output_dir, frame_skip=5, max_frames=None):
    """
    Extract frames from video and save with tank type + view labels

    Args:
        video_path: Path to video file
        tank_type: Tank type (e.g., "M1A1_Abrams", "Leopard2", "T90")
        view_angle: View angle ("front", "side", "top", "back")
        output_dir: Base output directory
        frame_skip: Save every Nth frame (default: 5)
        max_frames: Maximum frames to extract (None = extract all)

    Returns:
        Number of frames extracted
    """
    video_path = Path(video_path)

    # Create output directory: output_dir/tank_type_view/
    class_name = f"{tank_type}_{view_angle}"
    output_path = Path(output_dir) / class_name
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'='*80}")
    print(f"VIDEO INFO")
    print(f"{'='*80}")
    print(f"File: {video_path.name}")
    print(f"Tank: {tank_type}")
    print(f"View: {view_angle}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration_seconds:.1f} seconds")
    print(f"Resolution: {width}x{height}")
    print(f"\nFrame skip: Saving every {frame_skip} frame(s)")

    estimated_output = total_frames // frame_skip
    if max_frames and estimated_output > max_frames:
        estimated_output = max_frames

    print(f"Estimated frames to save: ~{estimated_output}")
    print(f"Output directory: {output_path}")

    # Find next available number
    existing_files = list(output_path.glob("*.png")) + list(output_path.glob("*.jpg"))
    if existing_files:
        # Extract numbers from existing files
        import re
        numbers = []
        for f in existing_files:
            matches = re.findall(r'\d+', f.stem)
            if matches:
                numbers.append(int(matches[-1]))
        start_number = max(numbers) + 1 if numbers else 0
    else:
        start_number = 0

    print(f"\n[INFO] Starting numbering from: {start_number}")

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
            # Save frame with class name prefix
            output_file = output_path / f"{class_name}_{current_number:06d}.png"
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
    print(f"Saved to: {output_path}")
    print(f"Class: {class_name}")

    return saved_count

def main():
    print("=" * 80)
    print("VIDEO FRAME EXTRACTOR WITH LABELS")
    print("Extract frames with tank type + view angle labels for dual conditional GAN")
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

    # Get tank type
    print("\n[STEP 2] Tank type")
    print("Examples: M1A1_Abrams, M1A2_Abrams, Leopard2, T90")
    print("Use underscores for spaces (e.g., M1A1_Abrams, not M1A1 Abrams)")

    tank_type = input("Enter tank type: ").strip().replace(' ', '_')

    if not tank_type:
        print("[ERROR] Tank type required")
        return

    # Get view angle
    print("\n[STEP 3] View angle")
    print("Common options: front, side, back")
    print("Optional: top (or any custom view name)")
    print("(View angles are auto-discovered during training)")

    view_angle = input("Enter view angle: ").strip().lower()

    if not view_angle:
        print(f"[ERROR] View angle required")
        return

    # Get output directory
    print("\n[STEP 4] Output directory")
    print("Default: GAN/datasets/military_vehicles_raw")
    print("(Preprocessing will move to military_vehicles_with_views)")

    output_dir = input("Enter output directory (or press Enter for default): ").strip().strip('"')

    if not output_dir:
        output_dir = 'GAN/datasets/military_vehicles_raw'

    # Get frame skip
    print("\n[STEP 5] Frame extraction settings")
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
    print("\n[STEP 6] Maximum frames (optional)")
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
    print(f"Tank: {tank_type}")
    print(f"View: {view_angle}")
    print(f"Class name: {tank_type}_{view_angle}")
    print(f"Output: {output_dir}/{tank_type}_{view_angle}/")
    print(f"Frame skip: Every {frame_skip} frame(s)")
    print(f"Max frames: {max_frames if max_frames else 'No limit'}")

    confirm = input("\nProceed? (y/n): ").strip().lower()

    if confirm != 'y':
        print("Cancelled")
        return

    # Extract frames
    saved_count = extract_frames_with_labels(
        video_path,
        tank_type,
        view_angle,
        output_dir,
        frame_skip,
        max_frames
    )

    if saved_count > 0:
        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print(f"\n✅ Extracted {saved_count} frames")
        print(f"📁 Location: {output_dir}/{tank_type}_{view_angle}/")
        print(f"🏷️  Class: {tank_type}_{view_angle}")
        print("\nNext steps:")
        print("  1. Extract more videos if needed (different tanks/views)")
        print("  2. Run preprocessing: python preprocess_with_folders.py")
        print("  3. Train dual conditional GAN: python train_gan_dual_conditional.py")
        print("\nWorkflow:")
        print("  - Record separate videos for each tank+view combination")
        print("  - Run this script for each video with appropriate labels")
        print("  - Preprocessing will crop/resize to 200x200 and organize for training")

if __name__ == "__main__":
    main()
