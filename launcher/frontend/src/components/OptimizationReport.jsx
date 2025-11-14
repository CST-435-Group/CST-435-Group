import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { BookOpen, TrendingUp, AlertTriangle, CheckCircle, Code, Target } from 'lucide-react'

export default function OptimizationReport() {
  const [activeSection, setActiveSection] = useState('overview')

  const sections = [
    { id: 'overview', label: 'Overview', icon: BookOpen },
    { id: 'current', label: 'Current Model', icon: Target },
    { id: 'improvements', label: 'Improvements', icon: TrendingUp },
    { id: 'expected', label: 'Expected Results', icon: CheckCircle },
    { id: 'references', label: 'References', icon: Code }
  ]

  const markdownContent = {
    overview: `# Optimization in Deep Learning

Optimization lies at the core of deep learning. It determines how effectively a model learns patterns from data by minimizing the loss function. Various factors such as learning rate, architecture, and data quality, impact how well this process performs. Optimization algorithms determine how model parameters move in parameter space to reduce sensitivity to hyperparameters.

## Problem Statement

The current CNN model shows severe overfitting with **99% false positive rates** on new data, indicating the model is memorizing training examples rather than learning generalizable patterns. This report outlines systematic improvements to address this critical issue.`,

    current: `## Current CNN Model Overview

The existing Convolutional Neural Network (CNN) is designed for classifying fruit images into 5 categories. It consists of:

- Three convolutional layers followed by ReLU activation and max pooling
- Batch Normalization after each convolutional layer for stable gradient flow
- Dropout layers to prevent overfitting by randomly disabling neurons during training
- Two fully connected layers leading to a SoftMax output layer

### Current Configuration

\`\`\`python
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
dropout_rate = 0.5
num_epochs = 50
\`\`\`

## Factors Affecting Performance

While Adam adapts the learning rate for each parameter, it couples L2 regularization and weight decay which can unintentionally scale parameter updates. This coupling may cause over-regularization or under-regularization in certain layers, poor generalization on unseen data, and instability in long term convergence as the adaptive learning rates can shrink too aggressively.

These drawbacks are particularly prevalent in CNNs trained on moderately sized datasets like the fruit image data that was used in this project, where generalization is more important than fast early convergence.`,

    improvements: `## Proposed Optimization Improvements

### 1. Switch from Adam to AdamW ⭐ HIGH PRIORITY

**Current:**
\`\`\`python
optimizer = optim.Adam(model.parameters(), lr=0.001)
\`\`\`

**Improved:**
\`\`\`python
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
\`\`\`

**Why This Helps:** AdamW (Loshchilov & Hutter, 2019) fixes a fundamental flaw in Adam where weight decay is applied before the adaptive learning rate adjustments, leading to inconsistent regularization. By decoupling weight decay from the gradient updates, AdamW provides more consistent regularization across all parameters regardless of their gradient magnitudes. The weight_decay parameter (0.01) adds a penalty term that shrinks weights toward zero, preventing the model from becoming overly complex and reducing overfitting. Studies show AdamW consistently outperforms Adam on image classification tasks, particularly when the training dataset is limited.

---

### 2. Implement Data Augmentation ⭐ HIGH PRIORITY

**Problem:** The model is memorizing training examples rather than learning generalizable features, leading to 99% false positive rates on new images.

**Solution:**
\`\`\`python
train_transform = transforms.Compose([
    transforms.RandomRotation(15),           # Rotate ±15 degrees
    transforms.RandomHorizontalFlip(0.5),    # Flip 50% of images
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Shift position
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),       # Zoom variation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
\`\`\`

**Why This Helps:** Data augmentation artificially expands the training dataset by creating modified versions of existing images (Shorten & Khoshgoftaar, 2019). This forces the model to learn invariant features - characteristics that remain consistent across rotations, translations, and flips. For fruit classification, this is critical because real-world images will have fruits at different angles, positions, and scales. Without augmentation, the model may learn to recognize specific backgrounds or image orientations rather than actual fruit features. Research shows data augmentation can reduce overfitting by 10-30% on small to medium-sized datasets.

---

### 3. Add Label Smoothing

**Implementation:**
\`\`\`python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
\`\`\`

**Why This Helps:** Label smoothing (Szegedy et al., 2016) prevents the model from becoming overconfident in its predictions. Traditional one-hot encoding assigns 100% probability to the correct class and 0% to all others, encouraging the model to push logits to extreme values. With label smoothing of 0.1, the target for the correct class becomes 0.9, with the remaining 0.1 distributed among other classes. This regularization technique:
- Reduces the model's tendency to make 99% confident incorrect predictions
- Improves calibration (predicted probabilities better match actual accuracy)
- Acts as a soft regularizer that prevents overfitting
- Has been shown to improve generalization by 1-3% on image classification tasks

---

### 4. Early Stopping Mechanism

**Implementation:**
\`\`\`python
early_stopping_patience = 7
epochs_without_improvement = 0

# In training loop after validation:
if val_acc > best_val_acc:
    best_val_acc = val_acc
    epochs_without_improvement = 0
    torch.save(model.state_dict(), 'models/best_model.pth')
else:
    epochs_without_improvement += 1
    
if epochs_without_improvement >= early_stopping_patience:
    print(f"Early stopping triggered after {epoch+1} epochs")
    break
\`\`\`

**Why This Helps:** Early stopping (Prechelt, 1998) monitors validation performance and halts training when the model stops improving. This prevents the model from continuing to memorize training data after it has learned generalizable patterns. With a patience of 7 epochs, training continues for 7 epochs after validation accuracy peaks, giving the model opportunity to escape local minima while preventing excessive overfitting. This is particularly important for the fruit dataset where training for all 50 epochs likely causes severe overfitting, as evidenced by the high false positive rate.

---

### 5. Enhanced Dropout Regularization

**Modification:**
\`\`\`python
# In FruitCNN model definition
self.dropout1 = nn.Dropout(0.6)  # Increased from 0.5
self.dropout2 = nn.Dropout(0.6)  # Increased from 0.5
\`\`\`

**Why This Helps:** Dropout (Srivastava et al., 2014) randomly deactivates neurons during training, forcing the network to learn redundant representations. Increasing dropout from 0.5 to 0.6 means 60% of neurons are disabled during each training step, preventing co-adaptation where neurons become overly reliant on specific other neurons. This is especially effective in fully connected layers where overfitting tends to occur. Higher dropout rates are appropriate when:
- The model shows signs of severe overfitting (99% false positives)
- The dataset is relatively small (typical for fruit classification projects)
- The model has high capacity (multiple FC layers with 256 and 128 neurons)

---

### 6. Gradient Clipping

**Implementation:**
\`\`\`python
# In training loop after loss.backward():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
\`\`\`

**Why This Helps:** Gradient clipping (Pascanu et al., 2013) prevents exploding gradients by capping the norm of gradient vectors at a maximum value. During backpropagation, especially in deeper networks with batch normalization, gradients can occasionally spike to very large values. These spikes cause:
- Unstable training with sudden jumps in loss
- Parameter updates that overshoot optimal values
- Potential numerical overflow issues

By clipping gradients to a maximum norm of 1.0, the model maintains stable training dynamics while still allowing meaningful parameter updates. This is particularly valuable when using adaptive optimizers like AdamW.

---

### 7. Class Imbalance Handling

**Diagnosis and Solution:**
\`\`\`python
# After collecting labels, check distribution:
from collections import Counter
class_counts = Counter(labels)
print("Class distribution:", class_counts)

# If imbalanced, apply class weights:
class_weights = torch.tensor([1.0/count for count in class_counts.values()])
class_weights = class_weights / class_weights.sum() * len(class_counts)
criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device), 
    label_smoothing=0.1
)
\`\`\`

**Why This Helps:** Class imbalance occurs when some fruit categories have significantly more training examples than others. Without correction, the model learns to favor majority classes because predicting them more often reduces overall loss. Class weighting inversely weights the loss function - classes with fewer examples receive higher loss weights, forcing the model to pay equal attention to all categories. This prevents the model from:
- Defaulting to predicting only the most common classes
- Achieving high accuracy on common classes while failing on rare ones
- Developing biased decision boundaries

Research shows weighted loss functions can improve minority class accuracy by 15-40% on imbalanced datasets.

---

### 8. Learning Rate Adjustment

**Updated Configuration:**
\`\`\`python
# Lower initial learning rate
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

# More aggressive scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    patience=3, 
    factor=0.5,      # Reduce by 50% instead of default
    min_lr=1e-6      # Set minimum learning rate floor
)
\`\`\`

**Why This Helps:** A lower initial learning rate (0.0005 instead of 0.001) promotes more stable training by taking smaller steps in parameter space. This is particularly important when combining multiple regularization techniques, as aggressive updates can interfere with their effectiveness. The improved scheduler configuration:
- **patience=3**: Reduces learning rate after 3 epochs without improvement, allowing faster adaptation
- **factor=0.5**: More significant learning rate reductions help escape plateaus
- **min_lr=1e-6**: Prevents learning rate from becoming ineffectively small

This configuration balances exploration (finding good parameter regions) with exploitation (fine-tuning within those regions), leading to better convergence and generalization.`,

    expected: `## Expected Improvements

By implementing these optimization strategies in combination, we expect:

### 1. Reduced Overfitting
False positive rate should decrease from **99% to <10%**

### 2. Better Generalization
Test accuracy should more closely match validation accuracy (gap reduced from 40%+ to <5%)

### 3. Improved Calibration
Model confidence scores will better reflect actual prediction accuracy

### 4. Stable Training
Loss curves will be smoother with fewer erratic spikes

### 5. Faster Convergence
Early stopping will prevent unnecessary training epochs (typically stopping at 15-25 epochs instead of 50)

## Implementation Priority

### Phase 1 (Critical - Implement First)
1. ✅ **Data Augmentation** - Single biggest impact on overfitting
2. ✅ **AdamW with weight decay** - Better regularization foundation
3. ✅ **Label Smoothing** - Prevents overconfident wrong predictions

### Phase 2 (Important - Implement Second)
4. ✅ **Early Stopping** - Prevents training past optimal point
5. ✅ **Increased Dropout** - Additional regularization
6. ✅ **Lower Learning Rate** - More stable training

### Phase 3 (Fine-Tuning - Implement Third)
7. ✅ **Class Imbalance Check** - Only if needed
8. ✅ **Gradient Clipping** - For training stability

## Training Monitoring Checklist

During retraining, monitor these metrics to verify improvements:

- [ ] Validation loss stops decreasing around epoch 15-25
- [ ] Training and validation accuracy gap is <5%
- [ ] Test predictions show more distributed probabilities (not 99%+)
- [ ] Per-class recall is balanced (no single class dominates)
- [ ] Loss curves are smooth without sudden spikes`,

    references: `## References

**Loshchilov, I., & Hutter, F. (2019).** Decoupled weight decay regularization. *ICLR 2019*.
- Introduced AdamW optimizer that properly separates weight decay from gradient updates

**Pascanu, R., Mikolov, T., & Bengio, Y. (2013).** On the difficulty of training recurrent neural networks. *ICML 2013*.
- Established gradient clipping as standard practice for training stability

**Prechelt, L. (1998).** Early stopping - but when? *Neural Networks: Tricks of the Trade*.
- Comprehensive study on early stopping strategies and patience selection

**Shorten, C., & Khoshgoftaar, T. M. (2019).** A survey on image data augmentation for deep learning. *Journal of Big Data, 6*(1), 60.
- Extensive review of data augmentation techniques for computer vision

**Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).** Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(1), 1929-1958.
- Original dropout paper showing effectiveness on multiple architectures

**Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).** Rethinking the inception architecture for computer vision. *CVPR 2016*.
- Introduced label smoothing as regularization technique

## Additional Resources

### PyTorch Documentation
- [AdamW Optimizer](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [Data Augmentation Transforms](https://pytorch.org/vision/stable/transforms.html)
- [CrossEntropyLoss with Label Smoothing](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

### Best Practices
- [PyTorch Training Best Practices](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Debugging Deep Learning Models](https://cs231n.github.io/neural-networks-3/)

---

**Last Updated:** November 2025  
**Authors:** Susana, Soren`
  }

  const SectionIcon = sections.find(s => s.id === activeSection)?.icon || BookOpen

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
        <div className="flex items-center mb-4">
          <TrendingUp size={48} className="text-blue-600 mr-4" />
          <div>
            <h1 className="text-4xl font-bold text-gray-800">CNN Optimization Report</h1>
            <p className="text-gray-600 text-lg">Addressing 99% False Positive Rate Through Systematic Improvements</p>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-center mb-2">
              <AlertTriangle className="text-red-600 mr-2" size={20} />
              <span className="text-sm font-semibold text-gray-700">Current Issue</span>
            </div>
            <p className="text-2xl font-bold text-red-600">99% False Positives</p>
          </div>
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-center mb-2">
              <Target className="text-green-600 mr-2" size={20} />
              <span className="text-sm font-semibold text-gray-700">Target Improvement</span>
            </div>
            <p className="text-2xl font-bold text-green-600">&lt;10% Error Rate</p>
          </div>
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center mb-2">
              <CheckCircle className="text-blue-600 mr-2" size={20} />
              <span className="text-sm font-semibold text-gray-700">Improvements</span>
            </div>
            <p className="text-2xl font-bold text-blue-600">8 Strategies</p>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white rounded-2xl shadow-xl mb-8 overflow-hidden">
        <div className="flex overflow-x-auto">
          {sections.map((section) => {
            const Icon = section.icon
            return (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={`flex items-center px-6 py-4 font-semibold transition-colors whitespace-nowrap ${
                  activeSection === section.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-100'
                }`}
              >
                <Icon size={20} className="mr-2" />
                {section.label}
              </button>
            )
          })}
        </div>
      </div>

      {/* Content Area */}
      <div className="bg-white rounded-2xl shadow-xl p-8">
        <div className="flex items-center mb-6">
          <SectionIcon size={32} className="text-blue-600 mr-3" />
          <h2 className="text-3xl font-bold text-gray-800">
            {sections.find(s => s.id === activeSection)?.label}
          </h2>
        </div>

        <div className="prose prose-lg max-w-none">
          <ReactMarkdown
            components={{
                h1: ({node, ...props}) => <h1 className="text-3xl font-bold text-gray-800 mb-4 mt-8" {...props} />,
                h2: ({node, ...props}) => <h2 className="text-2xl font-bold text-gray-800 mb-3 mt-6" {...props} />,
                h3: ({node, ...props}) => <h3 className="text-xl font-bold text-gray-700 mb-2 mt-4" {...props} />,
                p: ({node, ...props}) => <p className="text-gray-700 mb-4 leading-relaxed" {...props} />,
                ul: ({node, ...props}) => <ul className="list-disc list-inside mb-4 text-gray-700" {...props} />,
                ol: ({node, ...props}) => <ol className="list-decimal list-inside mb-4 text-gray-700" {...props} />,
                li: ({node, ...props}) => <li className="mb-2" {...props} />,
                pre: ({node, ...props}) => (
                <pre className="bg-gray-50 border-2 border-blue-300 p-6 rounded-xl overflow-x-auto my-6 shadow-lg" {...props} />
                ),
                code: ({node, inline, ...props}) => 
                inline ? (
                    <code className="bg-blue-100 px-2 py-1 rounded text-sm font-mono text-blue-700 font-semibold border border-blue-300" {...props} />
                ) : (
                    <code className="block text-gray-800 text-base font-mono" {...props} />
                ),
                strong: ({node, ...props}) => <strong className="font-bold text-gray-900" {...props} />,
                hr: ({node, ...props}) => <hr className="my-8 border-t-2 border-gray-200" {...props} />,
            }}
            >
            {markdownContent[activeSection]}
        </ReactMarkdown>
        </div>
      </div>

      {/* Download Section */}
      {/* Download Button */}
    <div className="mt-8 text-center">
    <button 
        onClick={() => {
        const element = document.createElement('a')
        const file = new Blob([JSON.stringify(markdownContent, null, 2)], {type: 'application/json'})
        element.href = URL.createObjectURL(file)
        element.download = 'cnn-optimization-report.json'
        document.body.appendChild(element)
        element.click()
        document.body.removeChild(element)
        }}
        className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded-lg transition-colors"
    >
        Download Report
    </button>
    </div>
    </div>
  )
}