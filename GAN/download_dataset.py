"""
Download Dataset for GAN Training
Supports multiple dataset options for 200x200 color images

Choose from:
1. Military Vehicles (tanks, aircraft, ships)
2. Pokemon
3. Anime Faces
4. Cars
5. Custom Bing Image Search
"""

import os
import sys
from pathlib import Path

def print_menu():
    print("=" * 80)
    print("GAN DATASET DOWNLOADER - 200x200 COLOR IMAGES")
    print("=" * 80)
    print("\nChoose a dataset:")
    print("1. Military Vehicles (Tanks, Jets, Ships) - Custom download")
    print("2. Pokemon Images - Kaggle")
    print("3. Anime Faces - Kaggle")
    print("4. Stanford Cars - Requires manual download")
    print("5. Custom Bing Image Search - Any topic!")
    print("\nRecommended: Option 1 (Military Vehicles) or 5 (Custom)")
    print("=" * 80)

def install_requirements():
    """Install required packages for downloading"""
    print("\n[SETUP] Installing required packages...")

    packages = [
        'bing-image-downloader',
        'pillow',
        'requests',
        'tqdm',
        'numpy'
    ]

    for package in packages:
        os.system(f'"{sys.executable}" -m pip install {package} -q')

    print("[SUCCESS] Required packages installed")

def download_bing_images(query, output_dir, limit=1000):
    """Download images using Bing Image Downloader"""
    try:
        from bing_image_downloader import downloader

        print(f"\n[DOWNLOAD] Searching for '{query}'...")
        print(f"Target: {limit} images")
        print(f"Output: {output_dir}")

        downloader.download(
            query,
            limit=limit,
            output_dir=output_dir,
            adult_filter_off=True,
            force_replace=False,
            timeout=15,
            verbose=True
        )

        # Count downloaded images
        image_dir = Path(output_dir) / query
        if image_dir.exists():
            images = list(image_dir.glob('*.*'))
            print(f"\n[SUCCESS] Downloaded {len(images)} images")
            return len(images)
        return 0

    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return 0

def download_military_vehicles():
    """Download military vehicles dataset with better filtering"""
    print("\n" + "=" * 80)
    print("DOWNLOADING MILITARY VEHICLES DATASET")
    print("=" * 80)

    install_requirements()

    output_dir = 'GAN/datasets/military_vehicles_raw'
    os.makedirs(output_dir, exist_ok=True)

    # IMPROVED: More specific search terms focused on VEHICLES only
    # Added negative keywords to exclude people/infantry
    categories = [
        'M1 Abrams tank',
        'Leopard 2 tank',
        'T-90 battle tank',
        'armored tank side view',
        'F-16 fighter jet',
        'F-22 Raptor aircraft',
        'MiG-29 fighter jet',
        'Su-27 fighter aircraft',
        'aircraft carrier ship',
        'destroyer warship',
        'battleship USS',
        'frigate naval vessel',
        'Apache attack helicopter',
        'Black Hawk helicopter',
        'Chinook helicopter military',
        'submarine underwater vessel'
    ]

    print(f"\n[INFO] Downloading {len(categories)} categories...")
    print("Each category: ~60-80 images")
    print("Total target: ~1000 images")
    print("[IMPROVED] Better search terms to exclude infantry/people")

    total_downloaded = 0

    for i, category in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}] Category: {category}")
        count = download_bing_images(category, output_dir, limit=80)
        total_downloaded += count

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\nTotal images downloaded: {total_downloaded}")
    print(f"Location: {output_dir}")
    print("\nNext step: Run preprocessing script to:")
    print("  - Resize to 200x200 (smart crop)")
    print("  - Remove corrupted/invalid images")
    print("  - Filter out GIFs and bad images")
    print("  - Organize for training")

def download_pokemon():
    """Download Pokemon images"""
    print("\n[INFO] Pokemon dataset requires Kaggle API")
    print("\nSteps:")
    print("1. Install kaggle: pip install kaggle")
    print("2. Get API credentials from kaggle.com/account")
    print("3. Download: kaggle datasets download -d kvpratama/pokemon-images-dataset")
    print("4. Extract to GAN/datasets/pokemon/")

def download_anime_faces():
    """Download Anime faces dataset"""
    print("\n[INFO] Anime faces dataset requires Kaggle API")
    print("\nSteps:")
    print("1. Install kaggle: pip install kaggle")
    print("2. Download: kaggle datasets download -d splcher/animefacedataset")
    print("3. Extract to GAN/datasets/anime_faces/")

def custom_search():
    """Custom Bing search for any topic"""
    print("\n" + "=" * 80)
    print("CUSTOM IMAGE SEARCH")
    print("=" * 80)

    install_requirements()

    query = input("\nEnter search query (e.g., 'cyberpunk city', 'dragons', 'spaceships'): ").strip()

    if not query:
        print("[ERROR] Empty query")
        return

    limit = input("Number of images to download (default 1000): ").strip()
    limit = int(limit) if limit else 1000

    # Clean query for directory name
    safe_name = "".join(c if c.isalnum() else "_" for c in query).lower()
    output_dir = f'GAN/datasets/{safe_name}_raw'

    download_bing_images(query, output_dir, limit)

    print(f"\n[NEXT] Run preprocessing to prepare images for training")

def main():
    print_menu()

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == '1':
        download_military_vehicles()
    elif choice == '2':
        download_pokemon()
    elif choice == '3':
        download_anime_faces()
    elif choice == '4':
        print("\n[INFO] Stanford Cars requires manual download")
        print("Visit: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset")
    elif choice == '5':
        custom_search()
    else:
        print("[ERROR] Invalid choice")

if __name__ == "__main__":
    main()
