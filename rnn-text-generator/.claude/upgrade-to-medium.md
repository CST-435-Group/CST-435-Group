# Instructions: Upgrade to Medium Model Configuration

## Context
After running the small model configuration (4.8M params) with the 145-book dataset (22.7M sequences), you may want to upgrade to the medium configuration for potentially better accuracy.

## Current Small Config (4.8M params)
```python
max_vocab_size=10000,
sequence_length=50,
embedding_dim=128,
lstm_units=256,
num_layers=2,
dropout_rate=0.25
```

**Data-to-param ratio:** 4.7x (safe)
**Expected accuracy:** 35-42% validation

---

## Medium Config Upgrade (7.5M params)

### Changes Required in `backend/app/train_optimal.py`

**Line ~80-87:** Update the LimitedVocabTextGenerator parameters:

```python
generator = LimitedVocabTextGenerator(
    max_vocab_size=10000,   # Keep same
    sequence_length=50,     # Keep same
    embedding_dim=192,      # *** INCREASE from 128 ***
    lstm_units=384,         # *** INCREASE from 256 ***
    num_layers=2,           # Keep same
    dropout_rate=0.25,      # Keep same
    vocab_file=VOCAB_PATH   # Keep same
)
```

**Line ~54:** Update the configuration header:
```python
print("CONFIGURATION V8 - MEDIUM (HIGHER CAPACITY)")
```

**Line ~62-68:** Update the architecture description:
```python
print("ARCHITECTURE (MEDIUM CONFIG):")
print("  ✓ Sequence length: 50 tokens")
print("  ✓ Embedding dim: 192 (increased from 128)")
print("  ✓ LSTM units: 384 (increased from 256)")
print("  ✓ LSTM layers: 2 with Layer Normalization")
print("  ✓ Dropout: 0.25")
print("  ✓ Parameters: ~7.5M (3.0x data-to-param ratio)")
print("  ✓ Expected: 38-48% validation accuracy")
```

---

## What to Expect with Medium Config

### Performance Comparison

| Metric | Small (4.8M) | Medium (7.5M) | Difference |
|--------|--------------|---------------|------------|
| **Parameters** | 4.8M | 7.5M | +56% capacity |
| **Data Ratio** | 4.7x | 3.0x | More aggressive |
| **Expected Val Acc** | 35-42% | 38-48% | +3-6% higher |
| **Training Speed** | Fast | Medium | ~15% slower |
| **Overfitting Risk** | Low | Medium | Watch train/val gap |

### Training Behavior

**Small Model:**
- Lower accuracy ceiling
- More stable training
- Train/Val gap: 2-3%
- Best for: Safe baseline, proving preprocessing works

**Medium Model:**
- Higher accuracy ceiling
- May have more variance
- Train/Val gap: 3-5% (acceptable)
- Best for: Maximum performance without overfitting

---

## When to Upgrade

### ✅ Upgrade to Medium if:
- Small model validation accuracy plateaus at 38-40%
- Train/Val gap is < 3% (not overfitting)
- You want to maximize accuracy
- Training is stable with no issues

### ❌ DON'T Upgrade if:
- Small model is already overfitting (train/val gap > 5%)
- Small model validation accuracy is < 35%
- You're still debugging preprocessing/data issues

---

## Alternative: Collect More Data First

If medium model still overfits (train/val gap > 5%), consider:

1. **Download more books** (target: 200+ books for safe 7.5M params)
2. **Add different text types** (news, Wikipedia, etc.)
3. **Target ratio:** 5-10x sequences-to-parameters

**For safe 7.5M params training:**
- Current: 22.7M sequences → 3.0x ratio
- Ideal: 37.5M+ sequences → 5.0x ratio
- Need: ~60% more data

---

## Implementation Steps

1. **Review small model results:**
   - Check final validation accuracy
   - Check train/val gap
   - Review generated text quality

2. **Make the changes above** in `train_optimal.py`

3. **Re-run training:**
   ```bash
   cd backend
   python app/train_optimal.py
   ```

4. **Compare results:**
   - Medium should be 3-6% better if working correctly
   - Watch for increased train/val gap
   - Generated text should be more coherent

---

## Rollback Plan

If medium model performs worse or overfits:

**Revert changes:**
- `embedding_dim=192` → `128`
- `lstm_units=384` → `256`

**Or try intermediate:**
- `embedding_dim=160`
- `lstm_units=320`
- This gives ~6M params at 3.8x ratio

---

## Expected Timeline

- **Small model training:** ~2-3 hours (30 epochs on RTX 3080 Ti)
- **Medium model training:** ~3-4 hours (30 epochs, larger model)
- **Total experiment time:** 5-7 hours

---

## Success Metrics

**Small Model Success:**
- ✅ Val accuracy: 35-42%
- ✅ Train/val gap: < 3%
- ✅ Generated text: Grammatically correct, somewhat coherent

**Medium Model Success:**
- ✅ Val accuracy: 38-48%
- ✅ Train/val gap: < 5%
- ✅ Generated text: More coherent, better context
- ✅ Improvement: +3-6% over small model

---

## Notes

- The 145-book dataset is borderline for 7.5M params (3.0x ratio)
- Small config is the safe starting point
- Only upgrade if small performs well
- Consider downloading 50-100 more books for safer medium training
- Your fixed preprocessing (contractions intact) should help both configs

**Created:** 2025-01-XX
**Dataset:** 145 books, 22.7M sequences, 10k vocab
**Current Config:** Small (4.8M params, 4.7x ratio)
