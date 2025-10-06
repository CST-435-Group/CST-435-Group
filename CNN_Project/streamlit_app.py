"""
Fruit Image Classification - Streamlit App (Enhanced with 4 Tabs)
CST-435 Neural Networks Assignment

This app:
- Tab 1: Interactive demo with 10 random images
- Tab 2: Upload your own images for prediction
- Tab 3: Complete model analytics
- Tab 4: README documentation
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import json
import numpy as np
from pathlib import Path
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import io

# Page configuration
st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍎",
    layout="wide"
)

# --------------------------
# Define Model Architecture (must match training)
# --------------------------
class FruitCNN(nn.Module):
    def __init__(self, num_classes):
        super(FruitCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

# --------------------------
# Load Model and Metadata
# --------------------------
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Resolve base directory relative to this script to avoid issues with working dir during deployment
        base_dir = Path(__file__).resolve().parent

        # If deployment provides a MODEL_DIR env var, prefer it first
        model_dir_env = None
        try:
            import os
            model_dir_env = os.getenv('MODEL_DIR')
        except Exception:
            model_dir_env = None

        metadata_path_candidates = []
        checkpoint_path_candidates = []

        if model_dir_env:
            env_path = Path(model_dir_env)
            metadata_path_candidates.append(env_path / 'model_metadata.json')
            checkpoint_path_candidates.append(env_path / 'best_model.pth')

        # Try common locations for metadata and checkpoint (prefer models/ under the project)
        metadata_path_candidates.extend([base_dir / 'models' / 'model_metadata.json', base_dir / 'data' / 'model_metadata.json', base_dir / 'model_metadata.json'])
        checkpoint_path_candidates.extend([base_dir / 'models' / 'best_model.pth', base_dir / 'data' / 'best_model.pth', base_dir / 'best_model.pth'])

        metadata_path = None
        for p in metadata_path_candidates:
            if p.exists():
                metadata_path = p
                break

        checkpoint_path = None
        for p in checkpoint_path_candidates:
            if p.exists():
                checkpoint_path = p
                break

        if metadata_path is None or checkpoint_path is None:
            # Build helpful diagnostic info
            checked = [str(p) for p in (metadata_path_candidates + checkpoint_path_candidates)]
            # List files present under base_dir and models dir to aid debugging
            present_files = []
            try:
                for f in sorted([str(p) for p in base_dir.iterdir()]):
                    present_files.append(f)
            except Exception:
                present_files = []

            models_dir = base_dir / 'models'
            models_list = []
            if models_dir.exists() and models_dir.is_dir():
                try:
                    for f in sorted([str(p) for p in models_dir.iterdir()]):
                        models_list.append(f)
                except Exception:
                    models_list = []

            raise FileNotFoundError(
                "Could not find model files. Paths checked: {}. Files at project root: {}. Files in models/: {}".format(
                    ', '.join(checked), ', '.join(present_files) or '(<none>)', ', '.join(models_list) or '(<none>)'
                )
            )

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load model checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create model
        model = FruitCNN(num_classes=metadata['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, metadata, checkpoint
    except Exception as e:
        # Provide clearer guidance depending on error
        st.error(f"Error loading model: {e}")
        st.error("Ensure your model files exist. Expected files: 'models/model_metadata.json' and 'models/best_model.pth' or under 'data/' directory.")
        st.stop()

@st.cache_data
def load_dataset_info():
    """Load dataset information"""
    try:
        data_path = Path('data') / 'dataset_metadata.json'
        j = json.loads(data_path.read_text())

        # Normalize image paths to be OS-native and absolute where possible
        norm_paths = []
        img_root = Path.cwd()
        for p in j.get('image_paths', []):
            # replace Windows backslashes and strip surrounding whitespace
            p_str = str(p).replace('\\', '/').strip()

            candidate = Path(p_str)
            # If relative, try project root
            if not candidate.is_absolute():
                candidate = img_root / p_str

            # If not found, try searching by filename under preprocessed_images/
            if not candidate.exists():
                fname = Path(p_str).name
                found = None
                search_root = img_root / 'preprocessed_images'
                if search_root.exists():
                    # try to find first matching filename
                    for f in search_root.rglob(fname):
                        found = f
                        break
                if found:
                    candidate = found

            # Final fallback: keep original string (will raise later if missing)
            norm_paths.append(str(candidate))

        # Replace with normalized paths
        j['image_paths'] = norm_paths

        return j
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_data
def load_training_history():
    """Load training history"""
    try:
        with open('data/training_history.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        return None

def preprocess_image(image_path):
    """Preprocess image for model input"""
    img = Image.open(image_path).convert('L')  # Grayscale
    img_array = np.array(img) / 255.0  # Normalize
    img_array = (img_array - 0.5) / 0.5  # Standardize
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
    return img_tensor, img

def preprocess_uploaded_image(uploaded_file):
    """Preprocess uploaded image for model input"""
    # Open image from uploaded file
    img = Image.open(uploaded_file)
    
    # Convert to RGB first (handles various formats including PNG with alpha)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to grayscale
    img_gray = img.convert('L')
    
    # Resize to 128x128
    img_resized = img_gray.resize((128, 128), Image.Resampling.LANCZOS)
    
    # Normalize and standardize for model
    img_array = np.array(img_resized) / 255.0
    img_array = (img_array - 0.5) / 0.5
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor, img, img_resized

def predict(model, image_tensor):
    """Make prediction"""
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_class].item()
    return pred_class, confidence, probabilities[0].numpy()

@st.cache_data
def analyze_all_images(_model, image_paths, labels, fruit_names):
    """Run prediction on all images and return analytics"""
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    misclassified = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (img_path, true_label) in enumerate(zip(image_paths, labels)):
        # Update progress
        progress = (idx + 1) / len(image_paths)
        progress_bar.progress(progress)
        status_text.text(f"Analyzing image {idx + 1} of {len(image_paths)}...")
        
        # Make prediction
        img_tensor, _ = preprocess_image(img_path)
        pred_class, confidence, _ = predict(_model, img_tensor)
        
        all_predictions.append(pred_class)
        all_true_labels.append(true_label)
        all_confidences.append(confidence)
        
        # Track misclassifications
        if pred_class != true_label:
            misclassified.append({
                'image_path': img_path,
                'true_label': fruit_names[true_label],
                'predicted': fruit_names[pred_class],
                'confidence': confidence
            })
    
    progress_bar.empty()
    status_text.empty()
    
    return all_predictions, all_true_labels, all_confidences, misclassified

# --------------------------
# Tab 1: Interactive Demo
# --------------------------
def demo_tab(model, model_metadata, dataset_info):
    """Interactive demo with 10 random images"""
    fruit_names = model_metadata['fruit_names']
    image_paths = dataset_info['image_paths']
    labels = dataset_info['labels']
    
    st.header("🖼️ Image Classification Demo")
    st.markdown("View 10 random test images with their predictions")
    
    # Session state for random selection
    if 'current_batch' not in st.session_state:
        st.session_state.current_batch = 0
    
    # Button to load new batch
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("🎲 Load 10 Random Images", type="primary"):
            st.session_state.current_batch += 1
    with col2:
        st.metric("Batch #", st.session_state.current_batch + 1)
    
    # Select 10 random images
    random.seed(st.session_state.current_batch)
    selected_indices = random.sample(range(len(image_paths)), min(10, len(image_paths)))
    
    # Track accuracy for this batch
    correct_predictions = 0
    
    # Display images in a grid (2 rows of 5)
    st.markdown("---")
    
    for row in range(2):
        cols = st.columns(5)
        for col_idx in range(5):
            img_idx = row * 5 + col_idx
            if img_idx >= len(selected_indices):
                break
            
            idx = selected_indices[img_idx]
            img_path = image_paths[idx]
            true_label = labels[idx]
            true_fruit = fruit_names[true_label]
            
            # Make prediction
            img_tensor, img = preprocess_image(img_path)
            pred_class, confidence, probabilities = predict(model, img_tensor)
            pred_fruit = fruit_names[pred_class]
            
            is_correct = pred_class == true_label
            if is_correct:
                correct_predictions += 1
            
            # Display in column
            with cols[col_idx]:
                # Show image
                st.image(img)
                
                # Show prediction with color coding
                if is_correct:
                    st.success(f"✅ **{pred_fruit}**")
                    st.caption(f"Confidence: {confidence*100:.1f}%")
                else:
                    st.error(f"❌ **{pred_fruit}**")
                    st.caption(f"True: **{true_fruit}**")
                    st.caption(f"Confidence: {confidence*100:.1f}%")
                
                # Show top 3 predictions
                with st.expander("Top 3"):
                    top3_indices = np.argsort(probabilities)[-3:][::-1]
                    for i, class_idx in enumerate(top3_indices):
                        prob = probabilities[class_idx]
                        st.caption(f"{i+1}. {fruit_names[class_idx]}: {prob*100:.1f}%")
    
    # Batch accuracy
    st.markdown("---")
    batch_accuracy = (correct_predictions / len(selected_indices)) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Batch Accuracy", f"{batch_accuracy:.1f}%")
    with col2:
        st.metric("Correct", correct_predictions)
    with col3:
        st.metric("Incorrect", len(selected_indices) - correct_predictions)
    with col4:
        st.metric("Total in Batch", len(selected_indices))
    
    # Detailed Results Table
    st.markdown("---")
    st.subheader("📊 Detailed Results")
    
    results_data = []
    for img_idx in selected_indices:
        img_path = image_paths[img_idx]
        true_label = labels[img_idx]
        true_fruit = fruit_names[true_label]
        
        img_tensor, _ = preprocess_image(img_path)
        pred_class, confidence, _ = predict(model, img_tensor)
        pred_fruit = fruit_names[pred_class]
        is_correct = pred_class == true_label
        
        results_data.append({
            'Image': Path(img_path).name,
            'True Label': true_fruit,
            'Predicted': pred_fruit,
            'Confidence': f"{confidence*100:.1f}%",
            'Correct': '✅' if is_correct else '❌'
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df)

# --------------------------
# Tab 2: Upload & Predict
# --------------------------
def upload_tab(model, model_metadata):
    """Upload and predict custom images"""
    fruit_names = model_metadata['fruit_names']
    # Upload feature has been disabled. Show informational message instead.
    st.header("📤 Upload & Predict")
    st.warning("The upload feature has been disabled in this app.")
    st.markdown("If you need to classify images, please use the Demo tab or re-enable uploads in the source code.")
    
    st.markdown("### 🍎 Supported Fruits:")
    for i, fruit in enumerate(fruit_names, 1):
        st.markdown(f"{i}. **{fruit}**")

# --------------------------
# Tab 3: Analytics
# --------------------------
def analytics_tab(model, model_metadata, dataset_info):
    """Complete model analytics on all images"""
    fruit_names = model_metadata['fruit_names']
    image_paths = dataset_info['image_paths']
    labels = dataset_info['labels']
    
    st.header("📊 Complete Model Analytics")
    st.markdown("Comprehensive analysis of model performance on all images")
    
    # Initialize session state
    if 'analytics_complete' not in st.session_state:
        st.session_state.analytics_complete = False
    
    # Show appropriate button based on state
    if not st.session_state.analytics_complete:
        # Not analyzed yet - show info and button
        st.info(f"📊 Click the button below to analyze all {len(image_paths)} images")
        st.warning("⏱️ This may take a few minutes depending on dataset size")
        
    if st.button("🔍 Run Full Analysis", type="primary"):
            with st.spinner("Analyzing all images... Please wait..."):
                predictions, true_labels, confidences, misclassified = analyze_all_images(
                    model, image_paths, labels, fruit_names
                )
                
                # Store in session state
                st.session_state.predictions = predictions
                st.session_state.true_labels = true_labels
                st.session_state.confidences = confidences
                st.session_state.misclassified = misclassified
                st.session_state.analytics_complete = True
                st.rerun()
    
    else:
        # Analysis complete - show results and option to rerun
        col1, col2 = st.columns([4, 1])
        with col1:
            st.success("✅ Analysis complete! Results shown below.")
        with col2:
            if st.button("🔄 Rerun Analysis", type="secondary"):
                st.session_state.analytics_complete = False
                st.rerun()
        
        
    # Safely retrieve analytics results from session_state (may be missing)
    predictions = st.session_state.get('predictions', [])
    true_labels = st.session_state.get('true_labels', [])
    confidences = st.session_state.get('confidences', [])
    misclassified = st.session_state.get('misclassified', [])

    # Overall metrics
    st.markdown("---")
    st.subheader("🎯 Overall Performance")

    # Handle empty analytics gracefully
    if len(true_labels) > 0 and len(predictions) == len(true_labels):
        overall_acc = accuracy_score(true_labels, predictions)
        correct_count = sum(1 for t, p in zip(true_labels, predictions) if t == p)
        incorrect_count = len(true_labels) - correct_count
    else:
        overall_acc = 0.0
        correct_count = 0
        incorrect_count = 0

    avg_confidence = float(np.mean(confidences)) if len(confidences) > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Accuracy", f"{overall_acc*100:.2f}%")
    with col2:
        st.metric("Avg Confidence", f"{avg_confidence*100:.2f}%")
    with col3:
        st.metric("Correct", correct_count)
    with col4:
        st.metric("Incorrect", incorrect_count)

    # Confusion Matrix and Per-Class Metrics (only when we have results)
    if len(true_labels) > 0 and len(predictions) == len(true_labels):
        # Confusion Matrix
        st.markdown("---")
        st.subheader("🔢 Confusion Matrix")

        try:
            cm = confusion_matrix(true_labels, predictions)

            # Ensure we have a non-empty array to plot
            if cm is None or getattr(cm, 'size', 0) == 0:
                st.info("Confusion matrix is empty — no labeled predictions to display.")
            else:
                fig, ax = plt.subplots(figsize=(10, 8))

                # If the number of class names doesn't match matrix size, use inferred ticklabels
                if cm.shape[0] == len(fruit_names):
                    xt = yt = fruit_names
                else:
                    xt = yt = None

                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=xt, yticklabels=yt,
                            ax=ax, cbar_kws={'label': 'Count'})
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                ax.set_title('Confusion Matrix')
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not compute/plot confusion matrix: {e}")
            import traceback
            traceback.print_exc()

        # Per-Class Metrics
        st.markdown("---")
        st.subheader("📈 Per-Class Performance")

        try:
            report = classification_report(true_labels, predictions, 
                                           target_names=fruit_names, 
                                           output_dict=True)

            # Create DataFrame
            per_class_data = []
            for fruit in fruit_names:
                # Be defensive: some reports may not include every class when empty
                if fruit in report:
                    precision = report[fruit].get('precision', 0.0)
                    recall = report[fruit].get('recall', 0.0)
                    f1 = report[fruit].get('f1-score', 0.0)
                    support = int(report[fruit].get('support', 0))
                else:
                    precision = recall = f1 = 0.0
                    support = 0

                per_class_data.append({
                    'Fruit': fruit,
                    'Precision': f"{precision*100:.2f}%",
                    'Recall': f"{recall*100:.2f}%",
                    'F1-Score': f"{f1*100:.2f}%",
                    'Support': support
                })

            per_class_df = pd.DataFrame(per_class_data)
            st.dataframe(per_class_df)
        except Exception as e:
            st.error(f"Could not compute classification report: {e}")
            import traceback
            traceback.print_exc()

        # Confidence Distribution (only plot if we have confidences)
        if len(confidences) > 0:
            st.markdown("---")
            st.subheader("📊 Confidence Distribution")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(confidences, bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Prediction Confidences')
            ax.axvline(avg_confidence, color='red', linestyle='--', 
                       label=f'Mean: {avg_confidence:.3f}')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No confidence scores available yet. Run the analysis to generate confidence distribution.")
    else:
        st.info("No analytics results to display yet. Click 'Run Full Analysis' to compute confusion matrix and per-class metrics.")

    # Training History (if available)
    history = load_training_history()
    if history:
        st.markdown("---")
        st.subheader("📉 Training History")

        col1, col2 = st.columns(2)

        with col1:
            # Loss curve
            fig, ax = plt.subplots(figsize=(8, 5))
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
            ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            # Accuracy curve
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, [acc*100 for acc in history['train_acc']], 
                   'b-', label='Training Accuracy')
            ax.plot(epochs, [acc*100 for acc in history['val_acc']], 
                   'r-', label='Validation Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Training and Validation Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

    # Misclassification Analysis
    st.markdown("---")
    st.subheader("❌ Misclassification Analysis")

    st.metric("Total Misclassified", len(misclassified))

    if len(misclassified) > 0:
        # Show worst misclassifications (highest confidence but wrong)
        st.markdown("#### Top 10 Most Confident Mistakes")

        sorted_mistakes = sorted(misclassified, 
                                key=lambda x: x['confidence'], 
                                reverse=True)[:10]

        mistake_data = []
        for mistake in sorted_mistakes:
            mistake_data.append({
                'Image': Path(mistake['image_path']).name,
                'True Label': mistake['true_label'],
                'Predicted': mistake['predicted'],
                'Confidence': f"{mistake['confidence']*100:.1f}%"
            })

        mistake_df = pd.DataFrame(mistake_data)
        st.dataframe(mistake_df)

        # Show sample misclassifications
        st.markdown("#### Sample Misclassified Images")

        num_samples = min(10, len(sorted_mistakes))
        cols = st.columns(5)

        for i in range(num_samples):
            col_idx = i % 5
            mistake = sorted_mistakes[i]

            with cols[col_idx]:
                img = Image.open(mistake['image_path']).convert('L')
                st.image(img)
                st.caption(f"True: {mistake['true_label']}")
                st.caption(f"Pred: {mistake['predicted']}")
                st.caption(f"Conf: {mistake['confidence']*100:.1f}%")

# --------------------------
# Tab 4: README
# --------------------------
def readme_tab():
    """Display README documentation"""
    st.header("📖 README Documentation")
    
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            readme_content = f.read()
        st.markdown(readme_content)
    except FileNotFoundError:
        st.error("README.md not found in the project directory")
        st.info("Please ensure README.md exists in the same directory as this script")


# assignment_tab removed as requested by user


# --------------------------
# Main App with Sidebar
# --------------------------
def main():
    # Load model and data
    model, model_metadata, checkpoint = load_model()
    dataset_info = load_dataset_info()
    
    if dataset_info is None:
        st.stop()

    # Initialize analytics session state keys with safe defaults so UI doesn't crash
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = []
    if 'true_labels' not in st.session_state:
        st.session_state['true_labels'] = []
    if 'confidences' not in st.session_state:
        st.session_state['confidences'] = []
    if 'misclassified' not in st.session_state:
        st.session_state['misclassified'] = []
    if 'analytics_complete' not in st.session_state:
        st.session_state['analytics_complete'] = False
    
    fruit_names = model_metadata['fruit_names']
    
    # Title
    st.title("🍎 Fruit Image Classifier")
    st.markdown("### CNN Model for Fruit Classification - CST-435 Assignment")
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("📊 Model Information")
        st.metric("Test Accuracy", f"{model_metadata['test_accuracy']*100:.2f}%")
        st.metric("Best Val Accuracy", f"{model_metadata['best_val_accuracy']*100:.2f}%")
        st.metric("Number of Classes", model_metadata['num_classes'])
        
        st.markdown("### 🍎 Fruit Categories")
        for i, fruit in enumerate(fruit_names, 1):
            st.markdown(f"{i}. **{fruit}**")
        
        st.markdown("---")
        st.markdown("### 📈 Dataset Info")
        st.metric("Total Images", model_metadata['total_images'])
        st.metric("Training Images", model_metadata['train_size'])
        st.metric("Validation Images", model_metadata['val_size'])
        st.metric("Test Images", model_metadata['test_size'])
        
        st.markdown("---")
        st.markdown("### 🔧 Model Architecture")
        st.markdown("""
        - Input: 128×128 Grayscale
        - 3 Convolutional Blocks
        - 32 → 64 → 128 filters
        - Max Pooling after each block
        - 2 FC layers + Dropout
        - ~8.5M parameters
        """)
    
    # Create tabs (Upload tab removed)
    tab1, tab2, tab3 = st.tabs([
        "🖼️ Demo",
        "📊 Analytics",
        "📚 README"
    ])

    with tab1:
        demo_tab(model, model_metadata, dataset_info)

    with tab2:
        analytics_tab(model, model_metadata, dataset_info)

    with tab3:
        readme_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>CST-435 Neural Networks Assignment</strong></p>
        <p>Fruit Image Classification using Kaggle Dataset</p>
        <p>PyTorch CNN | Grayscale 128×128 | {} Classes</p>
    </div>
    """.format(len(fruit_names)), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
