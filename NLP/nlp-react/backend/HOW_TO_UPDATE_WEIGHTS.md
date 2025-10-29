# How to Update Model Weights

## Your Situation

You have **retrained the DistilBERT model** with new weights and want to use those instead of the base HuggingFace model.

## What You Need

Your retrained model should be saved in one of these formats:

### Option 1: HuggingFace Format (Recommended)
A directory containing:
- `config.json`
- `pytorch_model.bin` (or `model.safetensors`)
- `tokenizer.json` + `tokenizer_config.json` + `vocab.txt`

### Option 2: PyTorch State Dict
A single `.pth` or `.pt` file containing just the model weights

---

## Step-by-Step Instructions

### If You Have HuggingFace Format (Full Model Directory)

**1. Copy your retrained model to the backend:**
```
NLP/nlp-react/backend/saved_model/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
├── vocab.txt
└── special_tokens_map.json
```

**2. The code is already set up!**

The model.py will automatically check for a local model first. Just make sure your files are in `saved_model/` directory.

---

### If You Have PyTorch State Dict Only

**1. Save your retrained weights:**

Put your `.pth` or `.pt` file here:
```
NLP/nlp-react/backend/saved_model/model_weights.pth
```

**2. Update model.py to load the weights:**

You'll need to modify the `__init__` method to:
- Load the base architecture from HuggingFace
- Load your custom weights from the .pth file

---

## Quick Fix: Update model.py to Load Local Weights

I'll update the code to check for local weights first, then fall back to HuggingFace if not found.

---

## Where to Put Your Retrained Model Files

### Current Directory Structure:
```
NLP/nlp-react/backend/
├── model.py              ← The code
├── saved_model/          ← PUT YOUR WEIGHTS HERE (currently empty)
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files...
└── requirements.txt
```

### After Adding Your Weights:
```
NLP/nlp-react/backend/
├── model.py
├── saved_model/          ← Your retrained model
│   ├── config.json       ← Model configuration
│   ├── pytorch_model.bin ← Your retrained weights
│   ├── tokenizer.json    ← Tokenizer (same as base)
│   ├── vocab.txt         ← Vocabulary (same as base)
│   └── ...other files
└── requirements.txt
```

---

## Testing After Update

```bash
# Test locally first
cd NLP/nlp-react/backend
python model.py

# Should see:
# Loading model: ./saved_model
# Model loaded on device: cpu
```

---

## For Railway Deployment

If you want Railway to use your retrained weights:

**Option A: Commit weights to git (if small enough)**
```bash
git add NLP/nlp-react/backend/saved_model/
git commit -m "Add retrained model weights"
git push
```

**Option B: Use HuggingFace Hub (if weights are large)**
1. Upload your model to HuggingFace
2. Change model_name to your HuggingFace repo: `"your-username/your-model-name"`

**Option C: Use Railway Volumes (for large files)**
Upload weights directly to Railway persistent storage

---

## What Format Are Your Weights In?

Tell me what you have and I'll help you integrate it:

1. **Full HuggingFace model directory?** (with config.json, pytorch_model.bin, etc.)
2. **Just a .pth/.pt file?** (state dict only)
3. **Something else?** (TensorFlow, ONNX, etc.)

Where are your retrained weights currently saved?
