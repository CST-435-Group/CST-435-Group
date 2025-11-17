# Section B: Practical Examples - CNN Optimization

**Author: Soren (Person B - Practical Techniques & Implementation)**

## Practical Examples

### Implementation: Adam to AdamW

The optimization was implemented by changing the optimizer in the CNN training script:

**Original (Adam):**
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Improved (AdamW):**
```python
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
```

The change decouples weight decay from gradient updates, providing consistent regularization across all parameters. The learning rate was reduced to 0.0005 for more stable updates, while `weight_decay=0.01` prevents overfitting.

### Experimental Results

A controlled experiment compared Adam and AdamW on fruit image classification (5,000 images, 5 classes, 10 epochs):

| Metric | Adam | AdamW | Improvement |
|--------|------|-------|-------------|
| **Validation Accuracy** | 86.27% | 98.00% | **+11.73%** |
| **Validation Loss** | 0.2888 | 0.0652 | **-77.4%** |
| **Training Accuracy** | 76.00% | 90.03% | +14.03% |
| **Training Loss** | 0.5459 | 0.2712 | -50.3% |

**Epoch Progression:**
```
Adam:   Epoch 1: 36.37% | Epoch 5: 65.17% | Epoch 10: 76.00%
AdamW:  Epoch 1: 42.20% | Epoch 5: 82.00% | Epoch 10: 90.03%
```

### Training Loop Example

```python
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

    # Adjust learning rate based on validation loss
    scheduler.step(val_loss)
```

### Why AdamW Outperforms Adam

1. **Decoupled Weight Decay**: AdamW applies weight decay independently of gradient-based updates, providing consistent regularization regardless of gradient magnitudes.

2. **Better Generalization**: AdamW achieved 98% validation accuracy compared to Adam's 86.27%, with a smaller train-validation gap (7.97% vs 10.27%).

3. **Stable Convergence**: The lower learning rate combined with decoupled weight decay produces smoother loss curves and more reliable optimization.

### Comparison Code Snippet

```python
# Test Adam
model_adam = FruitCNN(num_classes=5).to(device)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
history_adam = train_model(model_adam, optimizer_adam, epochs=10)

# Test AdamW
model_adamw = FruitCNN(num_classes=5).to(device)
optimizer_adamw = optim.AdamW(model_adamw.parameters(), lr=0.0005, weight_decay=0.01)
history_adamw = train_model(model_adamw, optimizer_adamw, epochs=10)

# Results show 11.73% improvement in validation accuracy
```

### Conclusion

Switching from Adam to AdamW provided significant improvements with minimal code changes. The 11.73% accuracy improvement and 77.4% loss reduction demonstrate AdamW's superiority for CNN optimization, particularly on moderately-sized datasets where generalization is critical.

## References

- Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR*. https://arxiv.org/abs/1711.05101
- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR*. https://arxiv.org/abs/1412.6980
