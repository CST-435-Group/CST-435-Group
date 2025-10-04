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
        # Load metadata
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load model checkpoint
        checkpoint = torch.load('models/best_model.pth', map_location='cpu')
        
        # Create model
        model = FruitCNN(num_classes=metadata['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, metadata, checkpoint
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please run train_model.py first to train the model!")
        st.stop()

@st.cache_data
def load_dataset_info():
    """Load dataset information"""
    try:
        with open('data/dataset_metadata.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_data
def load_training_history():
    """Load training history"""
    try:
        with open('models/training_history.json', 'r') as f:
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
                st.image(img, use_container_width=True)
                
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
    st.dataframe(results_df, use_container_width=True)

# --------------------------
# Tab 2: Upload & Predict
# --------------------------
def upload_tab(model, model_metadata):
    """Upload and predict custom images"""
    fruit_names = model_metadata['fruit_names']
    
    st.header("📤 Upload & Predict")
    st.markdown("Upload your own fruit images to see what the model predicts!")
    
    # File uploader
    st.markdown("### Upload an Image")
    uploaded_files = st.file_uploader(
        "Drag and drop image(s) here or click to browse",
        type=['png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp'],
        accept_multiple_files=True,
        help="Upload fruit images to classify. Supports PNG, JPG, JPEG, BMP, GIF, WebP"
    )
    
    if uploaded_files:
        st.markdown("---")
        st.subheader(f"📊 Results ({len(uploaded_files)} image(s) uploaded)")
        
        # Process each uploaded file
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"### Image {idx + 1}: {uploaded_file.name}")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            try:
                # Preprocess image
                img_tensor, original_img, processed_img = preprocess_uploaded_image(uploaded_file)
                
                # Make prediction
                pred_class, confidence, probabilities = predict(model, img_tensor)
                pred_fruit = fruit_names[pred_class]
                
                # Display original image
                with col1:
                    st.markdown("**Original Image**")
                    st.image(original_img, use_container_width=True)
                    st.caption(f"Size: {original_img.size[0]}×{original_img.size[1]}")
                
                # Display processed image
                with col2:
                    st.markdown("**Processed (128×128 Grayscale)**")
                    st.image(processed_img, use_container_width=True)
                    st.caption("Ready for model input")
                
                # Display prediction
                with col3:
                    st.markdown("**Prediction**")
                    st.success(f"### 🍎 {pred_fruit}")
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # Top 3 predictions
                    st.markdown("**Top 3 Predictions:**")
                    top3_indices = np.argsort(probabilities)[-3:][::-1]
                    for i, class_idx in enumerate(top3_indices):
                        prob = probabilities[class_idx]
                        fruit = fruit_names[class_idx]
                        if i == 0:
                            st.markdown(f"🥇 **{fruit}**: {prob*100:.1f}%")
                        elif i == 1:
                            st.markdown(f"🥈 {fruit}: {prob*100:.1f}%")
                        else:
                            st.markdown(f"🥉 {fruit}: {prob*100:.1f}%")
                
                # Probability distribution chart
                st.markdown("**Probability Distribution**")
                fig, ax = plt.subplots(figsize=(10, 4))
                colors = ['green' if i == pred_class else 'lightblue' for i in range(len(fruit_names))]
                ax.barh(fruit_names, probabilities * 100, color=colors)
                ax.set_xlabel('Probability (%)')
                ax.set_title(f'Prediction Probabilities for "{uploaded_file.name}"')
                ax.set_xlim(0, 100)
                for i, v in enumerate(probabilities * 100):
                    ax.text(v + 1, i, f'{v:.1f}%', va='center')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                st.markdown("---")
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Upload one or more images to get started!")
        
        st.markdown("""
        ### 📝 Instructions:
        1. **Click** the upload box above or **drag and drop** image files
        2. Upload fruit images (supports multiple files)
        3. See the **original** and **processed** (128×128 grayscale) versions
        4. Get **instant predictions** with confidence scores
        5. View probability distribution across all fruit classes
        
        ### 📁 Supported Formats:
        - PNG, JPG, JPEG, BMP, GIF, WebP
        - Any image size (automatically resized)
        - Color or grayscale images
        
        ### 💡 Tips for Best Results:
        - Use clear, well-lit fruit images
        - Center the fruit in the frame
        - Avoid cluttered backgrounds
        - Try different angles and varieties
        
        ### 🍎 Supported Fruits:
        """)
        
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
        
        if st.button("🔍 Run Full Analysis", type="primary", use_container_width=True):
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
        
        
        predictions = st.session_state.predictions
        true_labels = st.session_state.true_labels
        confidences = st.session_state.confidences
        misclassified = st.session_state.misclassified
        
        # Overall metrics
        st.markdown("---")
        st.subheader("🎯 Overall Performance")
        
        overall_acc = accuracy_score(true_labels, predictions)
        avg_confidence = np.mean(confidences)
        correct_count = sum(1 for t, p in zip(true_labels, predictions) if t == p)
        incorrect_count = len(true_labels) - correct_count
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Accuracy", f"{overall_acc*100:.2f}%")
        with col2:
            st.metric("Avg Confidence", f"{avg_confidence*100:.2f}%")
        with col3:
            st.metric("Correct", correct_count)
        with col4:
            st.metric("Incorrect", incorrect_count)
        
        # Confusion Matrix
        st.markdown("---")
        st.subheader("🔢 Confusion Matrix")
        
        cm = confusion_matrix(true_labels, predictions)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=fruit_names, yticklabels=fruit_names,
                    ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Per-Class Metrics
        st.markdown("---")
        st.subheader("📈 Per-Class Performance")
        
        report = classification_report(true_labels, predictions, 
                                       target_names=fruit_names, 
                                       output_dict=True)
        
        # Create DataFrame
        per_class_data = []
        for fruit in fruit_names:
            per_class_data.append({
                'Fruit': fruit,
                'Precision': f"{report[fruit]['precision']*100:.2f}%",
                'Recall': f"{report[fruit]['recall']*100:.2f}%",
                'F1-Score': f"{report[fruit]['f1-score']*100:.2f}%",
                'Support': int(report[fruit]['support'])
            })
        
        per_class_df = pd.DataFrame(per_class_data)
        st.dataframe(per_class_df, use_container_width=True)
        
        # Confidence Distribution
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
            st.dataframe(mistake_df, use_container_width=True)
            
            # Show sample misclassifications
            st.markdown("#### Sample Misclassified Images")
            
            num_samples = min(10, len(sorted_mistakes))
            cols = st.columns(5)
            
            for i in range(num_samples):
                col_idx = i % 5
                mistake = sorted_mistakes[i]
                
                with cols[col_idx]:
                    img = Image.open(mistake['image_path']).convert('L')
                    st.image(img, use_container_width=True)
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

# --------------------------
# Main App with Sidebar
# --------------------------
def main():
    # Load model and data
    model, model_metadata, checkpoint = load_model()
    dataset_info = load_dataset_info()
    
    if dataset_info is None:
        st.stop()
    
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
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Demo", "📤 Upload & Predict", "📊 Analytics", "📖 README"])
    
    with tab1:
        demo_tab(model, model_metadata, dataset_info)
    
    with tab2:
        upload_tab(model, model_metadata)
    
    with tab3:
        analytics_tab(model, model_metadata, dataset_info)
    
    with tab4:
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
