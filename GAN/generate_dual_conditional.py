"""
Generate Images using Dual Conditional GAN (Tank Type + View Angle)

Usage:
  python generate_dual_conditional.py                          # Interactive menu
  python generate_dual_conditional.py --tank M1A1_Abrams --view front --num 10
  python generate_dual_conditional.py --all --num 5            # Generate all combinations
  python generate_dual_conditional.py --grid                   # Generate a grid of all combos
"""

import os
import json
import torch
import argparse
from pathlib import Path
from torchvision.utils import save_image, make_grid
from models_dual_conditional import DualConditionalGenerator

# Paths
MODEL_DIR = 'models_dual_conditional'
OUTPUT_DIR = 'generated_images_dual_conditional'
LATENT_DIM = 100
EMBED_DIM = 50


def load_generator(model_path, num_tanks, num_views, device):
    """Load trained generator"""
    generator = DualConditionalGenerator(
        latent_dim=LATENT_DIM,
        num_tanks=num_tanks,
        num_views=num_views,
        embed_dim=EMBED_DIM
    ).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    generator.load_state_dict(state_dict)
    generator.eval()

    return generator


def generate_images(generator, tank_idx, view_idx, num_images, device):
    """Generate images for a specific tank+view combination"""
    with torch.no_grad():
        noise = torch.randn(num_images, LATENT_DIM, device=device)
        tank_labels = torch.full((num_images,), tank_idx, dtype=torch.long, device=device)
        view_labels = torch.full((num_images,), view_idx, dtype=torch.long, device=device)

        fake_images = generator(noise, tank_labels, view_labels)
        # Denormalize from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2

    return fake_images


def generate_grid(generator, num_tanks, num_views, device, samples_per_combo=1):
    """Generate a grid showing all tank+view combinations"""
    all_images = []

    with torch.no_grad():
        for view_idx in range(num_views):
            for tank_idx in range(num_tanks):
                noise = torch.randn(samples_per_combo, LATENT_DIM, device=device)
                tank_labels = torch.full((samples_per_combo,), tank_idx, dtype=torch.long, device=device)
                view_labels = torch.full((samples_per_combo,), view_idx, dtype=torch.long, device=device)

                fake_img = generator(noise, tank_labels, view_labels)
                all_images.append(fake_img)

    # Concatenate and denormalize
    all_images = torch.cat(all_images, dim=0)
    all_images = (all_images + 1) / 2

    return all_images


def main():
    parser = argparse.ArgumentParser(description='Generate images using Dual Conditional GAN')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to generator model (default: latest)')
    parser.add_argument('--tank', type=str, default=None,
                        help='Tank type to generate (e.g., M1A1_Abrams, Leopard2, T90)')
    parser.add_argument('--view', type=str, default=None,
                        help='View angle to generate (e.g., front, side, back)')
    parser.add_argument('--num', type=int, default=10,
                        help='Number of images to generate')
    parser.add_argument('--all', action='store_true',
                        help='Generate images for all tank+view combinations')
    parser.add_argument('--grid', action='store_true',
                        help='Generate a grid showing all combinations')
    parser.add_argument('--grid-size', type=int, default=30,
                        help='Size of the grid (default: 30x30)')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("DUAL CONDITIONAL GAN - IMAGE GENERATION")
    print("=" * 60)
    print(f"Device: {device}")

    # Load label mappings
    mappings_path = Path(MODEL_DIR) / 'label_mappings.json'
    if not mappings_path.exists():
        print(f"[ERROR] Label mappings not found: {mappings_path}")
        return

    with open(mappings_path) as f:
        mappings = json.load(f)

    tank_to_idx = mappings['tank_to_idx']
    idx_to_tank = {int(k): v for k, v in mappings['idx_to_tank'].items()}
    view_to_idx = mappings['view_to_idx']
    idx_to_view = {int(k): v for k, v in mappings['idx_to_view'].items()}

    num_tanks = len(tank_to_idx)
    num_views = len(view_to_idx)

    print(f"\nAvailable tanks ({num_tanks}):")
    for name, idx in sorted(tank_to_idx.items(), key=lambda x: x[1]):
        print(f"  [{idx}] {name}")

    print(f"\nAvailable views ({num_views}):")
    for name, idx in sorted(view_to_idx.items(), key=lambda x: x[1]):
        print(f"  [{idx}] {name}")

    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = Path(MODEL_DIR) / 'latest_generator.pth'
        if not model_path.exists():
            model_path = Path(MODEL_DIR) / 'generator_epoch_200.pth'

    if not model_path.exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        return

    print(f"\nLoading model: {model_path}")
    generator = load_generator(model_path, num_tanks, num_views, device)
    print("Model loaded successfully!")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate based on mode
    if args.grid:
        grid_size = args.grid_size
        total_images = grid_size * grid_size
        print(f"\nGenerating {grid_size}x{grid_size} grid ({total_images} images)...")

        all_images = []
        with torch.no_grad():
            # Generate in batches to avoid memory issues
            batch_size = min(64, total_images)
            remaining = total_images

            while remaining > 0:
                current_batch = min(batch_size, remaining)

                noise = torch.randn(current_batch, LATENT_DIM, device=device)
                # Randomly select tank and view for variety
                tank_labels = torch.randint(0, num_tanks, (current_batch,), device=device)
                view_labels = torch.randint(0, num_views, (current_batch,), device=device)

                fake_images = generator(noise, tank_labels, view_labels)
                fake_images = (fake_images + 1) / 2  # Denormalize
                all_images.append(fake_images.cpu())

                remaining -= current_batch
                print(f"  Generated {total_images - remaining}/{total_images}...")

        all_images = torch.cat(all_images, dim=0)
        grid = make_grid(all_images, nrow=grid_size, padding=2)
        save_path = output_path / f'grid_{grid_size}x{grid_size}.png'
        save_image(grid, save_path)

        print(f"\nSaved {grid_size}x{grid_size} grid to: {save_path}")

    elif args.all:
        print(f"\nGenerating {args.num} images for each combination...")

        for tank_name, tank_idx in tank_to_idx.items():
            for view_name, view_idx in view_to_idx.items():
                combo_name = f"{tank_name}_{view_name}"
                print(f"  Generating {combo_name}...")

                images = generate_images(generator, tank_idx, view_idx, args.num, device)

                # Save individual images
                combo_dir = output_path / combo_name
                combo_dir.mkdir(parents=True, exist_ok=True)

                for i in range(args.num):
                    save_image(images[i], combo_dir / f"{combo_name}_{i:04d}.png")

                # Save grid for this combo
                if args.num > 1:
                    grid = make_grid(images, nrow=min(5, args.num), padding=2)
                    save_image(grid, combo_dir / f"{combo_name}_grid.png")

        print(f"\nSaved {num_tanks * num_views * args.num} images to: {output_path}")

    else:
        # Interactive or specific tank+view
        if args.tank and args.view:
            tank_name = args.tank
            view_name = args.view
        else:
            # Interactive selection
            print("\n" + "-" * 40)
            print("SELECT TANK TYPE:")
            for name, idx in sorted(tank_to_idx.items(), key=lambda x: x[1]):
                print(f"  [{idx}] {name}")

            try:
                tank_choice = input("\nEnter tank number or name: ").strip()
                if tank_choice.isdigit():
                    tank_idx = int(tank_choice)
                    tank_name = idx_to_tank[tank_idx]
                else:
                    tank_name = tank_choice
                    tank_idx = tank_to_idx.get(tank_name)
                    if tank_idx is None:
                        # Try partial match
                        for name in tank_to_idx:
                            if tank_choice.lower() in name.lower():
                                tank_name = name
                                tank_idx = tank_to_idx[name]
                                break
            except (ValueError, KeyError):
                print("[ERROR] Invalid tank selection")
                return

            print("\n" + "-" * 40)
            print("SELECT VIEW ANGLE:")
            for name, idx in sorted(view_to_idx.items(), key=lambda x: x[1]):
                print(f"  [{idx}] {name}")

            try:
                view_choice = input("\nEnter view number or name: ").strip()
                if view_choice.isdigit():
                    view_idx = int(view_choice)
                    view_name = idx_to_view[view_idx]
                else:
                    view_name = view_choice
                    view_idx = view_to_idx.get(view_name)
                    if view_idx is None:
                        for name in view_to_idx:
                            if view_choice.lower() in name.lower():
                                view_name = name
                                view_idx = view_to_idx[name]
                                break
            except (ValueError, KeyError):
                print("[ERROR] Invalid view selection")
                return

        # Validate selections
        if tank_name not in tank_to_idx:
            print(f"[ERROR] Unknown tank: {tank_name}")
            print(f"Available: {list(tank_to_idx.keys())}")
            return

        if view_name not in view_to_idx:
            print(f"[ERROR] Unknown view: {view_name}")
            print(f"Available: {list(view_to_idx.keys())}")
            return

        tank_idx = tank_to_idx[tank_name]
        view_idx = view_to_idx[view_name]

        print(f"\nGenerating {args.num} images of {tank_name} ({view_name} view)...")

        images = generate_images(generator, tank_idx, view_idx, args.num, device)

        # Save
        combo_name = f"{tank_name}_{view_name}"
        combo_dir = output_path / combo_name
        combo_dir.mkdir(parents=True, exist_ok=True)

        for i in range(args.num):
            save_path = combo_dir / f"{combo_name}_{i:04d}.png"
            save_image(images[i], save_path)

        # Save grid
        if args.num > 1:
            grid = make_grid(images, nrow=min(5, args.num), padding=2)
            grid_path = combo_dir / f"{combo_name}_grid.png"
            save_image(grid, grid_path)
            print(f"Grid saved: {grid_path}")

        print(f"\nSaved {args.num} images to: {combo_dir}")

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
