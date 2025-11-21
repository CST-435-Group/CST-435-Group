import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
import numpy as np

class ViewClassifier:
    def __init__(self, data_dir='GAN/datasets/military_vehicles_processed', output_file='view_classifications.json'):
        self.data_dir = data_dir
        self.output_file = output_file
        self.classifications = {}
        self.current_batch = 0
        self.images_per_batch = 20  # Show 20 at a time for easier viewing

        # Load existing classifications if they exist
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                self.classifications = json.load(f)

        # View options
        self.views = {
            '1': 'front',
            '2': 'side',
            '3': 'top',
            '4': 'back',
            '5': 'skip'  # Skip unclear images
        }

        # Collect all images (filter for t90 only)
        self.image_paths = []
        self.image_labels = []

        for class_name in sorted(os.listdir(data_dir)):
            # Only process t90
            if class_name.lower() != 't90':
                continue

            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in sorted(os.listdir(class_path)):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        self.image_paths.append(img_path)
                        self.image_labels.append(class_name)

        print(f"Found {len(self.image_paths)} images")
        print(f"Already classified: {len(self.classifications)}")

    def show_batch(self):
        """Show a batch of images for classification"""
        start_idx = self.current_batch * self.images_per_batch
        end_idx = min(start_idx + self.images_per_batch, len(self.image_paths))

        if start_idx >= len(self.image_paths):
            print("\nAll images processed!")
            self.save_classifications()
            return False

        batch_paths = self.image_paths[start_idx:end_idx]
        batch_labels = self.image_labels[start_idx:end_idx]

        # Create grid
        n_images = len(batch_paths)
        grid_size = int(np.ceil(np.sqrt(n_images)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        fig.suptitle(f'Batch {self.current_batch + 1} - Images {start_idx} to {end_idx-1}\n' +
                     'Press 1=Front, 2=Side, 3=Top, 4=Back, 5=Skip | Enter to save and continue | Q to quit',
                     fontsize=14)

        axes = axes.flatten() if n_images > 1 else [axes]

        self.current_classifications = {}

        for idx, (img_path, label) in enumerate(zip(batch_paths, batch_labels)):
            ax = axes[idx]

            # Load and display image
            try:
                img = Image.open(img_path).convert('RGB')
                ax.imshow(img)

                # Check if already classified
                existing = self.classifications.get(img_path, None)
                if existing:
                    title_color = 'green'
                    title = f"{idx}: {label}\n[{existing}]"
                else:
                    title_color = 'black'
                    title = f"{idx}: {label}"

                ax.set_title(title, fontsize=8, color=title_color)
                ax.axis('off')

                # Store for keyboard input
                self.current_classifications[idx] = {
                    'path': img_path,
                    'label': label,
                    'view': existing
                }
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading\n{img_path}',
                       ha='center', va='center', fontsize=6)
                ax.axis('off')

        # Hide unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')

        # Connect keyboard event
        fig.canvas.mpl_connect('key_press_event', lambda event: self.on_key_press(event, fig))

        plt.tight_layout()
        plt.show()

        return True

    def on_key_press(self, event, fig):
        """Handle keyboard input for classification"""
        key = event.key

        if key in ['1', '2', '3', '4', '5']:
            # Classify all unclassified images in current batch
            view = self.views[key]
            for idx, info in self.current_classifications.items():
                if info['view'] is None:  # Only classify if not already classified
                    img_path = info['path']
                    self.classifications[img_path] = view
                    info['view'] = view

            print(f"\nClassified unclassified images as: {view}")
            self.save_classifications()

        elif key == 'enter' or key == 'return':
            # Save and move to next batch
            self.save_classifications()
            self.current_batch += 1
            plt.close(fig)

        elif key == 'q':
            # Quit
            self.save_classifications()
            plt.close('all')

        elif key.isdigit():
            # Try to parse as image index for individual classification
            try:
                idx = int(key)
                if idx in self.current_classifications:
                    print(f"\nSelect view for image {idx}:")
                    print("1=Front, 2=Side, 3=Top, 4=Back, 5=Skip")
            except:
                pass

    def save_classifications(self):
        """Save classifications to file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.classifications, f, indent=2)

        # Count views
        view_counts = {}
        for view in self.classifications.values():
            view_counts[view] = view_counts.get(view, 0) + 1

        print(f"\nSaved {len(self.classifications)} classifications")
        print("View distribution:", view_counts)

    def run(self):
        """Run the classification tool"""
        print("\n=== Tank View Classifier ===")
        print("Instructions:")
        print("  1 = Front view")
        print("  2 = Side view")
        print("  3 = Top view")
        print("  4 = Back view")
        print("  5 = Skip/Unclear")
        print("  Enter = Save and next batch")
        print("  Q = Quit")
        print("\nPress a number to classify ALL unclassified images in the current view")
        print("Then press Enter to move to next batch\n")

        while self.show_batch():
            pass

        print("\n=== Classification Complete ===")
        self.create_reclassified_dataset()

    def create_reclassified_dataset(self):
        """Create new dataset with view labels"""
        output_dir = 'GAN/datasets/military_vehicles_with_views'
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nCreating reclassified dataset in {output_dir}/")

        for img_path, view in self.classifications.items():
            if view == 'skip':
                continue

            # Get original label
            original_label = os.path.basename(os.path.dirname(img_path))
            img_name = os.path.basename(img_path)

            # Create new label with view
            new_label = f"{original_label}_{view}"

            # Create directory
            new_class_dir = os.path.join(output_dir, new_label)
            os.makedirs(new_class_dir, exist_ok=True)

            # Copy image
            new_img_path = os.path.join(new_class_dir, img_name)
            img = Image.open(img_path)
            img.save(new_img_path)

        # Count new classes
        classes = os.listdir(output_dir)
        print(f"Created {len(classes)} new classes with view labels:")
        for cls in sorted(classes):
            count = len(os.listdir(os.path.join(output_dir, cls)))
            print(f"  {cls}: {count} images")

if __name__ == '__main__':
    classifier = ViewClassifier()
    classifier.run()
