import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import os
import time
import json
from tqdm import tqdm
from models_dual_conditional import DualConditionalGenerator, DualConditionalDiscriminator, weights_init

# Hyperparameters
BATCH_SIZE = 16
LATENT_DIM = 100
EMBED_DIM = 50
NUM_EPOCHS = 200
LR_G = 0.0002
LR_D = 0.0001  # Lower discriminator learning rate to prevent it from getting too strong
BETA1 = 0.5
IMAGE_SIZE = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training stability settings
LABEL_SMOOTHING = 0.1  # Smooth real labels to 0.9 instead of 1.0
NOISE_STD = 0.05  # Add small noise to discriminator inputs
D_TRAIN_RATIO = 1  # Train discriminator every N batches (1 = every batch)

# Dynamic learning rate adjustment
ADAPTIVE_LR = True  # Enable adaptive learning rate
TARGET_D_REAL = 0.85  # Target for D(real) - aim for 85% accuracy on real images
TARGET_D_FAKE = 0.15  # Target for D(fake) - aim for 15% false positives
LR_ADJUSTMENT_RATE = 0.95  # Multiply/divide LR by this amount when adjusting
MIN_LR_D = 0.00001  # Minimum discriminator learning rate
MAX_LR_D = 0.0003  # Maximum discriminator learning rate

# Paths
DATA_DIR = 'GAN/datasets/military_vehicles_with_views'
CHECKPOINT_DIR = 'GAN/models_dual_conditional'
PROGRESS_DIR = 'GAN/training_progress_dual_conditional'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PROGRESS_DIR, exist_ok=True)


class DualLabelTankDataset(Dataset):
    """Dataset that extracts both tank type and view from class names"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.tank_labels = []
        self.view_labels = []

        # Tank type and view mappings (auto-discovered)
        self.tank_to_idx = {}
        self.idx_to_tank = {}
        self.view_to_idx = {}
        self.idx_to_view = {}

        # First pass: collect all tank types and views
        tank_types = set()
        view_types = set()
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                # Parse: "M1A1_Abrams_front" -> tank="M1A1_Abrams", view="front"
                parts = class_name.rsplit('_', 1)
                if len(parts) == 2:
                    tank_type, view = parts
                    tank_types.add(tank_type)
                    view_types.add(view)

        # Create tank type mapping
        for idx, tank_type in enumerate(sorted(tank_types)):
            self.tank_to_idx[tank_type] = idx
            self.idx_to_tank[idx] = tank_type

        # Create view mapping (auto-discovered, so you can use front/side/back without needing top)
        for idx, view in enumerate(sorted(view_types)):
            self.view_to_idx[view] = idx
            self.idx_to_view[idx] = view

        # Second pass: load all images
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            # Parse class name
            parts = class_name.rsplit('_', 1)
            if len(parts) != 2:
                print(f"Warning: Skipping malformed class name: {class_name}")
                continue

            tank_type, view = parts

            if view not in self.view_to_idx:
                print(f"Warning: Unknown view '{view}' in class {class_name}")
                continue

            if tank_type not in self.tank_to_idx:
                print(f"Warning: Unknown tank type '{tank_type}' in class {class_name}")
                continue

            tank_idx = self.tank_to_idx[tank_type]
            view_idx = self.view_to_idx[view]

            # Load images
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    self.image_paths.append(img_path)
                    self.tank_labels.append(tank_idx)
                    self.view_labels.append(view_idx)

        print(f"\nDataset loaded:")
        print(f"  Total images: {len(self.image_paths)}")
        print(f"  Tank types: {len(self.tank_to_idx)} - {list(self.tank_to_idx.keys())}")
        print(f"  Views: {len(self.view_to_idx)} - {list(self.view_to_idx.keys())}")

        # Print distribution
        print("\nView distribution:")
        for view, idx in self.view_to_idx.items():
            count = self.view_labels.count(idx)
            print(f"  {view}: {count} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        tank_label = self.tank_labels[idx]
        view_label = self.view_labels[idx]

        return image, tank_label, view_label


def save_model_chunked(model, path, chunk_size_mb=90):
    """Save model in chunks if it exceeds size limit"""
    state_dict = model.state_dict()

    # Save to temporary buffer to check size
    temp_path = path + '.tmp'
    torch.save(state_dict, temp_path)

    file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)

    if file_size_mb <= chunk_size_mb:
        # Small enough, just rename
        os.replace(temp_path, path)
        return path
    else:
        # Need to chunk
        print(f"  [CHUNKING] Model size {file_size_mb:.1f}MB exceeds {chunk_size_mb}MB, splitting...")
        os.remove(temp_path)

        # Split state dict into chunks
        items = list(state_dict.items())
        chunk_paths = []
        current_chunk = {}
        current_size = 0
        chunk_idx = 0
        max_chunk_size = 0

        for key, tensor in items:
            tensor_size = tensor.element_size() * tensor.nelement()

            # Start new chunk if adding this would exceed limit
            if current_size + tensor_size > chunk_size_mb * 1024 * 1024 and current_chunk:
                chunk_path = f"{path}.chunk{chunk_idx}"
                torch.save(current_chunk, chunk_path)
                chunk_size = os.path.getsize(chunk_path) / (1024 * 1024)
                max_chunk_size = max(max_chunk_size, chunk_size)
                chunk_paths.append(chunk_path)

                current_chunk = {}
                current_size = 0
                chunk_idx += 1

            current_chunk[key] = tensor
            current_size += tensor_size

        # Save last chunk
        if current_chunk:
            chunk_path = f"{path}.chunk{chunk_idx}"
            torch.save(current_chunk, chunk_path)
            chunk_size = os.path.getsize(chunk_path) / (1024 * 1024)
            max_chunk_size = max(max_chunk_size, chunk_size)
            chunk_paths.append(chunk_path)

        # Create manifest
        manifest = {
            'chunks': [os.path.basename(p) for p in chunk_paths],
            'num_chunks': len(chunk_paths)
        }
        manifest_path = path + '.manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)

        print(f"  [CHUNKED] Saved {len(chunk_paths)} chunks (max {max_chunk_size:.1f}MB)")
        print(f"  [MANIFEST] {os.path.basename(manifest_path)}")
        return manifest_path


def generate_sample_images(generator, dataset, epoch, device, num_samples=100):
    """Generate sample images showing each tank type + view combination"""
    generator.eval()

    with torch.no_grad():
        # Create grid showing all tank types x all views
        num_tanks = len(dataset.tank_to_idx)
        num_views = len(dataset.view_to_idx)

        all_images = []

        for view_idx in range(num_views):
            for tank_idx in range(num_tanks):
                # Generate one sample for this tank+view combo
                noise = torch.randn(1, LATENT_DIM, device=device)
                tank_label = torch.tensor([tank_idx], device=device)
                view_label = torch.tensor([view_idx], device=device)

                fake_img = generator(noise, tank_label, view_label)
                all_images.append(fake_img)

        # Create grid: rows=views, cols=tanks
        grid = torch.cat(all_images, dim=0)
        grid = make_grid(grid, nrow=num_tanks, normalize=True, value_range=(-1, 1), padding=2)

        # Save
        save_path = os.path.join(PROGRESS_DIR, f'epoch_{epoch:03d}.png')
        save_image(grid, save_path)
        print(f"  [SAVED] {num_tanks}x{num_views} grid → {save_path}")

    generator.train()


def train():
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load dataset
    dataset = DualLabelTankDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    num_tanks = len(dataset.tank_to_idx)
    num_views = len(dataset.view_to_idx)

    # Save label mappings
    label_info = {
        'tank_to_idx': dataset.tank_to_idx,
        'idx_to_tank': dataset.idx_to_tank,
        'view_to_idx': dataset.view_to_idx,
        'idx_to_view': dataset.idx_to_view
    }
    with open(os.path.join(CHECKPOINT_DIR, 'label_mappings.json'), 'w') as f:
        json.dump(label_info, f, indent=2)

    # Initialize models
    generator = DualConditionalGenerator(
        latent_dim=LATENT_DIM,
        num_tanks=num_tanks,
        num_views=num_views,
        embed_dim=EMBED_DIM
    ).to(DEVICE)

    discriminator = DualConditionalDiscriminator(
        num_tanks=num_tanks,
        num_views=num_views,
        embed_dim=EMBED_DIM
    ).to(DEVICE)

    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LR_G, betas=(BETA1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(BETA1, 0.999))

    # Track current learning rates
    current_lr_d = LR_D
    current_lr_g = LR_G

    # Loss
    criterion = nn.BCELoss()

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting Training on {DEVICE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Tank types: {num_tanks}")
    print(f"  View angles: {num_views}")
    print(f"  Latent dim: {LATENT_DIM}")
    print(f"  Embed dim: {EMBED_DIM}")
    print(f"  LR_G: {LR_G} | LR_D: {LR_D} (starting)")
    print(f"  Label smoothing: {LABEL_SMOOTHING}")
    print(f"  Noise: {NOISE_STD}")
    print(f"  D train ratio: {D_TRAIN_RATIO}")
    if ADAPTIVE_LR:
        print(f"  🔄 Adaptive LR: ON")
        print(f"     Target D(real): {TARGET_D_REAL} | Target D(fake): {TARGET_D_FAKE}")
        print(f"     LR_D range: {MIN_LR_D} to {MAX_LR_D}")
    else:
        print(f"  🔄 Adaptive LR: OFF")
    print(f"{'='*60}\n")

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        d_losses = []
        g_losses = []
        d_real_scores = []
        d_fake_scores = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        batch_idx = 0
        for real_imgs, tank_labels, view_labels in pbar:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(DEVICE)
            tank_labels = tank_labels.to(DEVICE)
            view_labels = view_labels.to(DEVICE)

            # Labels with smoothing (real = 0.9 instead of 1.0 to prevent discriminator overconfidence)
            real_labels = torch.ones(batch_size, 1, device=DEVICE) * (1.0 - LABEL_SMOOTHING)
            fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

            # ---------------------
            # Train Discriminator (but not every batch if D_TRAIN_RATIO > 1)
            # ---------------------
            if batch_idx % D_TRAIN_RATIO == 0:
                optimizer_D.zero_grad()

                # Add small noise to real images to prevent discriminator from memorizing
                noisy_real_imgs = real_imgs + torch.randn_like(real_imgs) * NOISE_STD

                # Real images
                real_output = discriminator(noisy_real_imgs, tank_labels, view_labels)
                d_real_loss = criterion(real_output, real_labels)

                # Fake images
                noise = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
                fake_imgs = generator(noise, tank_labels, view_labels)

                # Add small noise to fake images too
                noisy_fake_imgs = fake_imgs.detach() + torch.randn_like(fake_imgs) * NOISE_STD
                fake_output = discriminator(noisy_fake_imgs, tank_labels, view_labels)
                d_fake_loss = criterion(fake_output, fake_labels)

                # Total discriminator loss
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                optimizer_D.step()
            else:
                # Still need to generate fakes for generator training
                noise = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
                fake_imgs = generator(noise, tank_labels, view_labels)
                d_loss = torch.tensor(0.0)
                real_output = torch.tensor(0.0)
                fake_output = torch.tensor(0.0)

            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate fake images and try to fool discriminator
            fake_output_for_g = discriminator(fake_imgs, tank_labels, view_labels)
            g_loss = criterion(fake_output_for_g, torch.ones(batch_size, 1, device=DEVICE))  # Generator wants discriminator to output 1

            g_loss.backward()
            optimizer_G.step()

            # Stats (only record discriminator stats when it was actually trained)
            if batch_idx % D_TRAIN_RATIO == 0:
                d_losses.append(d_loss.item())
                d_real_scores.append(real_output.mean().item())
                d_fake_scores.append(fake_output.mean().item())

            g_losses.append(g_loss.item())

            # Update progress bar
            pbar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}' if batch_idx % D_TRAIN_RATIO == 0 else 'skip',
                'G_loss': f'{g_loss.item():.4f}'
            })

            batch_idx += 1

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_d_loss = sum(d_losses)/len(d_losses) if d_losses else 0
        avg_g_loss = sum(g_losses)/len(g_losses)
        avg_d_real = sum(d_real_scores)/len(d_real_scores) if d_real_scores else 0
        avg_d_fake = sum(d_fake_scores)/len(d_fake_scores) if d_fake_scores else 0

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f}")
        print(f"  D(real): {avg_d_real:.3f} | D(fake): {avg_d_fake:.3f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Adaptive learning rate adjustment
        if ADAPTIVE_LR and epoch > 0:  # Start adjusting after first epoch
            lr_adjusted = False

            # Check if discriminator is too strong
            if avg_d_real > TARGET_D_REAL + 0.05 and avg_d_fake < TARGET_D_FAKE - 0.05:
                # Discriminator is dominating - slow it down
                new_lr_d = current_lr_d * LR_ADJUSTMENT_RATE
                if new_lr_d >= MIN_LR_D:
                    current_lr_d = new_lr_d
                    for param_group in optimizer_D.param_groups:
                        param_group['lr'] = current_lr_d
                    print(f"  📉 Lowered LR_D to {current_lr_d:.6f} (discriminator too strong)")
                    lr_adjusted = True

            # Check if discriminator is too weak
            elif avg_d_real < TARGET_D_REAL - 0.1 and avg_d_fake > TARGET_D_FAKE + 0.1:
                # Generator is dominating - speed up discriminator
                new_lr_d = current_lr_d / LR_ADJUSTMENT_RATE
                if new_lr_d <= MAX_LR_D:
                    current_lr_d = new_lr_d
                    for param_group in optimizer_D.param_groups:
                        param_group['lr'] = current_lr_d
                    print(f"  📈 Raised LR_D to {current_lr_d:.6f} (discriminator too weak)")
                    lr_adjusted = True

            if not lr_adjusted:
                print(f"  ✅ LR_D stable at {current_lr_d:.6f} (balanced)")

        # Warning if discriminator is too strong
        if avg_d_real > 0.95 and avg_d_fake < 0.05:
            print(f"  ⚠️  WARNING: Discriminator very strong! D(real)={avg_d_real:.3f}, D(fake)={avg_d_fake:.3f}")

        # Warning if generator loss is exploding
        if avg_g_loss > 20:
            print(f"  ⚠️  WARNING: Generator loss very high ({avg_g_loss:.1f})! Training may be unstable")

        # Save models
        save_model_chunked(generator, os.path.join(CHECKPOINT_DIR, 'latest_generator.pth'))
        save_model_chunked(discriminator, os.path.join(CHECKPOINT_DIR, 'latest_discriminator.pth'))

        # Generate sample images
        generate_sample_images(generator, dataset, epoch + 1, DEVICE)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_model_chunked(generator, os.path.join(CHECKPOINT_DIR, f'generator_epoch_{epoch+1:03d}.pth'))
            save_model_chunked(discriminator, os.path.join(CHECKPOINT_DIR, f'discriminator_epoch_{epoch+1:03d}.pth'))

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    train()
