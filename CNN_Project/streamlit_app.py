"""
Fruit Image Classification - Streamlit App
CST-435 Neural Networks Assignment

This app:
- Loads the trained CNN model
- Shows model accuracy
- Displays 10 test images at a time with predictions
- Shows correct/incorrect classifications
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

def preprocess_image(image_path):
    """Preprocess image for model input"""
    img = Image.open(image_path).convert('L')  # Grayscale
    img_array = np.array(img) / 255.0  # Normalize
    img_array = (img_array - 0.5) / 0.5  # Standardize
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
    return img_tensor, img

def predict(model, image_tensor):
    """Make prediction"""
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_class].item()
    return pred_class, confidence, probabilities[0].numpy()

# --------------------------
# Streamlit App
# --------------------------
def main():
    # Load model and data
    model, model_metadata, checkpoint = load_model()
    dataset_info = load_dataset_info()
    
    if dataset_info is None:
        st.stop()
    
    fruit_names = model_metadata['fruit_names']
    
    # Title and Header
    st.title("🍎 Fruit Image Classifier")
    st.markdown("### CNN Model for Fruit Classification")
    st.markdown("---")
    
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
        - 2 Fully Connected layers
        - Dropout (0.5) for regularization
        """)
    
    # Main content
    st.header("🖼️ Image Classification Demo")
    st.markdown("View 10 random test images with their predictions")
    
    # Get test images
    image_paths = dataset_info['image_paths']
    labels = dataset_info['labels']
    
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
                # Show image - FIXED: use_container_width instead of use_column_width
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
                with st.expander("Top 3 Predictions"):
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
    
    # Detailed Results
    st.markdown("---")
    st.header("📊 Detailed Results")
    
    # Create a table of results
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
    
    import pandas as pd
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
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
