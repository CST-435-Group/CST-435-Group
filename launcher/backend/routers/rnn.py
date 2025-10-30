"""
FastAPI Router for RNN Project (Text Generation)
Wraps existing RNN backend functionality with lazy loading
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import sys
from pathlib import Path
import json
import asyncio

# Add RNN project to path
rnn_project_path = Path(__file__).parent.parent.parent.parent / "rnn-text-generator" / "backend" / "app"
sys.path.insert(0, str(rnn_project_path))

try:
    from text_generator import TextGenerator
    from models import (
        GenerateRequest,
        GenerateResponse,
        ModelInfo,
        HealthResponse,
        TestMetrics,
        AvailableModel,
        AvailableModelsResponse,
        SwitchModelRequest,
        SwitchModelResponse
    )
except ImportError:
    print("Warning: Could not import RNN model. RNN endpoints will not work.")
    TextGenerator = None

# Create router
router = APIRouter()

# Global generator storage (lazy loading)
rnn_generator = None
current_model_name = None
MODEL_DIR = None


def load_rnn_generator():
    """Lazy load RNN text generator"""
    global rnn_generator, MODEL_DIR, current_model_name

    if rnn_generator is not None:
        return rnn_generator

    try:
        if TextGenerator is None:
            raise ImportError("TextGenerator not available")

        # Set model directory
        MODEL_DIR = Path(__file__).parent.parent.parent.parent / "rnn-text-generator" / "backend" / "saved_models"

        generator = TextGenerator()

        # Try to load default model (excluding broken models)
        model_files = list(MODEL_DIR.glob("model*.pt"))
        # Filter out non-functioning models
        model_files = [f for f in model_files if f.stem.lower() not in ['best_model', 'model_best']]

        if model_files:
            # Find corresponding tokenizer and config
            model_path = model_files[0]
            model_name = model_path.stem

            # Determine tokenizer path
            if model_name.startswith('model_optimal'):
                suffix = model_name.split('_')[-1]
                tokenizer_path = MODEL_DIR / f"tokenizer_optimal_{suffix}.pkl"
            else:
                tokenizer_path = MODEL_DIR / f"{model_name.replace('model', 'tokenizer')}.pkl"

            if model_path.exists() and tokenizer_path.exists():
                generator.load_model(str(model_path), str(tokenizer_path))
                rnn_generator = generator
                current_model_name = model_name
                print(f"[OK] RNN text generator loaded successfully with model: {model_name}")
                return generator

        print("[WARNING] No RNN model found. Please train a model first.")
        return None

    except Exception as e:
        print(f"[ERROR] Error loading RNN generator: {e}")
        import traceback
        traceback.print_exc()
        return None


@router.get("/")
async def rnn_info():
    """Get RNN project information"""
    return {
        "project": "RNN Text Generation",
        "description": "LSTM-based neural network for next-word prediction and text generation",
        "endpoints": {
            "/health": "Check if model is loaded",
            "/model/info": "Get model architecture information",
            "/generate": "Generate text from seed",
            "/model/test": "Evaluate model on test data",
            "/models/available": "List available trained models",
            "/models/switch": "Switch to different model",
            "/technical-report": "Get technical report markdown content"
        }
    }


@router.get("/health")
async def health_check():
    """Check if RNN generator is loaded"""
    return {
        "status": "ready" if rnn_generator is not None else "not_loaded",
        "model_loaded": rnn_generator is not None
    }


@router.post("/preload")
async def preload_model():
    """Preload the RNN model into memory"""
    global rnn_generator

    if rnn_generator is not None:
        return {
            "status": "already_loaded",
            "message": "RNN model is already loaded"
        }

    print("[LOADING] Preloading RNN model on user request...")
    generator = load_rnn_generator()

    if generator is None:
        raise HTTPException(status_code=500, detail="Failed to load RNN model")

    return {
        "status": "loaded",
        "message": "RNN model loaded successfully"
    }


@router.post("/unload")
async def unload_model():
    """Unload the RNN model from memory"""
    global rnn_generator

    if rnn_generator is None:
        return {
            "status": "not_loaded",
            "message": "RNN model was not loaded"
        }

    print("[UNLOADING] Unloading RNN model to free memory...")
    rnn_generator = None

    # Force garbage collection to free memory immediately
    import gc
    gc.collect()

    return {
        "status": "unloaded",
        "message": "RNN model unloaded successfully"
    }


@router.get("/model/info")
async def get_model_info():
    """Get model architecture information"""
    generator = load_rnn_generator()

    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Calculate total trainable parameters
        total_neurons = sum(p.numel() for p in generator.model.parameters() if p.requires_grad)

        return {
            "vocab_size": generator.vocab_size,
            "sequence_length": generator.sequence_length,
            "embedding_dim": generator.embedding_dim,
            "lstm_units": generator.lstm_units,
            "num_layers": generator.num_layers,
            "total_neurons": total_neurons,
            "current_model": current_model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


@router.post("/generate")
async def generate_text(request: GenerateRequest):
    """Generate text from seed using sampling or beam search"""
    generator = load_rnn_generator()

    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        generated = generator.generate_text(
            seed_text=request.seed_text,
            num_words=request.num_words,
            temperature=request.temperature,
            use_beam_search=request.use_beam_search,
            beam_width=request.beam_width,
            length_penalty=request.length_penalty,
            repetition_penalty=request.repetition_penalty,
            beam_temperature=request.beam_temperature,
            add_punctuation=request.add_punctuation,
            validate_grammar=request.validate_grammar
        )

        return {
            "generated_text": generated,
            "seed_text": request.seed_text,
            "num_words": request.num_words,
            "temperature": request.temperature,
            "use_beam_search": request.use_beam_search,
            "beam_width": request.beam_width if request.use_beam_search else None,
            "length_penalty": request.length_penalty if request.use_beam_search else None,
            "repetition_penalty": request.repetition_penalty if request.use_beam_search else None,
            "beam_temperature": request.beam_temperature if request.use_beam_search else None,
            "add_punctuation": request.add_punctuation,
            "validate_grammar": request.validate_grammar
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")


@router.post("/generate/stream")
async def generate_text_stream(request: GenerateRequest):
    """Generate text with streaming - sends each word as it's generated"""
    generator = load_rnn_generator()

    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async def stream_generator():
        """Async generator that yields words as they're generated"""
        import torch
        import numpy as np

        try:
            # Send initial metadata
            yield f"data: {json.dumps({'type': 'start', 'seed_text': request.seed_text})}\n\n"

            # Initialize grammar validator if requested
            grammar_validator = None
            if request.validate_grammar:
                from text_generator import get_grammar_validator
                grammar_validator = get_grammar_validator()
                if grammar_validator:
                    print("[INFO] Grammar validation enabled for streaming")

            # Use sampling method (beam search is harder to stream)
            generator.model.eval()
            generated_text = request.seed_text.lower()

            with torch.no_grad():
                for word_idx in range(request.num_words):
                    # Tokenize current text
                    token_list = generator.tokenizer.texts_to_sequences([generated_text])[0]

                    # Take last sequence_length tokens
                    token_list = token_list[-(generator.sequence_length):]

                    # Pad to model input size
                    if len(token_list) < generator.max_sequence_len - 1:
                        token_list = [0] * (generator.max_sequence_len - 1 - len(token_list)) + token_list
                    else:
                        token_list = token_list[-(generator.max_sequence_len - 1):]

                    # Convert to tensor
                    token_tensor = torch.LongTensor([token_list]).to(generator.device)

                    # Predict next word probabilities
                    output = generator.model(token_tensor)
                    predicted_probs = torch.softmax(output / request.temperature, dim=-1)
                    predicted_probs = predicted_probs.cpu().numpy()[0]

                    # Try multiple samples if grammar validation is enabled
                    max_attempts = 10 if grammar_validator else 1
                    word_accepted = False

                    for attempt in range(max_attempts):
                        # Sample from distribution
                        predicted_index = np.random.choice(
                            len(predicted_probs),
                            p=predicted_probs
                        )

                        # Convert index to word
                        if predicted_index in generator.tokenizer.idx_to_word:
                            word = generator.tokenizer.idx_to_word[predicted_index]
                            if word != '<PAD>':
                                # Create test text with new word
                                if word in ('s', 't', 'd', 'll', 're', 've', 'm') and generated_text and not generated_text.endswith("'"):
                                    test_word = "'" + word
                                else:
                                    test_word = " " + word

                                test_text = generated_text + test_word

                                # Validate grammar if enabled
                                if grammar_validator:
                                    is_valid, score = grammar_validator.validate_sequence(test_text, partial=True)
                                    if not is_valid and attempt < max_attempts - 1:
                                        # Try another word
                                        continue

                                # Word accepted
                                generated_text = test_text
                                word_accepted = True

                                # Send the word
                                yield f"data: {json.dumps({'type': 'token', 'word': test_word, 'index': word_idx, 'grammar_score': score if grammar_validator else 1.0})}\n\n"

                                # Small delay to prevent overwhelming the client
                                await asyncio.sleep(0.01)
                                break

                    if not word_accepted:
                        # If no word was accepted after max attempts, skip this position
                        continue

            # Apply punctuation post-processing if requested
            if request.add_punctuation:
                from text_generator import add_punctuation_postprocess
                print("[INFO] Applying punctuation post-processing...")
                formatted_text = add_punctuation_postprocess(generated_text)

                # Send punctuation update
                yield f"data: {json.dumps({'type': 'punctuation', 'formatted_text': formatted_text})}\n\n"
                generated_text = formatted_text

            # Send completion message
            yield f"data: {json.dumps({'type': 'done', 'full_text': generated_text})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/model/test")
async def test_model(use_beam_search: bool = True, beam_width: int = 5):
    """Test model and return metrics"""
    generator = load_rnn_generator()

    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Try to find test data
        data_dir = Path(__file__).parent.parent.parent.parent / "rnn-text-generator" / "backend" / "data"
        possible_files = [
            "training_text.txt",
            "kjv.txt",
            "web.txt",
            "net.txt",
            "asv.txt"
        ]

        test_data_path = None
        for filename in possible_files:
            path = data_dir / filename
            if path.exists():
                test_data_path = path
                break

        if test_data_path is None:
            txt_files = list(data_dir.glob("*.txt"))
            if txt_files:
                test_data_path = txt_files[0]

        if test_data_path is None or not test_data_path.exists():
            raise HTTPException(status_code=404, detail="No test data found")

        # Load and prepare test data
        with open(test_data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Use last 10% for testing
        words = text.split()
        test_size = len(words) // 10
        test_text = ' '.join(words[-test_size:])

        # Prepare sequences and evaluate
        X, y, _ = generator.prepare_sequences(test_text)
        metrics = generator.evaluate_model(X, y, use_beam_search=use_beam_search, beam_width=beam_width)

        return {
            "test_loss": metrics['test_loss'],
            "test_accuracy": metrics['test_accuracy'],
            "perplexity": metrics['perplexity'],
            "samples_tested": metrics['samples_tested'],
            "r_squared": metrics['r_squared']
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing model: {str(e)}")


@router.get("/models/available")
async def list_available_models():
    """List all available models (excluding broken models)"""
    # List of model names to exclude (broken/non-functioning models)
    excluded_models = ['best_model', 'model_best']

    generator = load_rnn_generator()

    if generator is None:
        # Try to list models even if generator not loaded
        try:
            model_dir = Path(__file__).parent.parent.parent.parent / "rnn-text-generator" / "backend" / "saved_models"
            models = []

            for model_file in model_dir.glob("*.pt"):
                model_name = model_file.stem
                # Skip excluded models
                if model_name.lower() in excluded_models:
                    continue
                models.append({
                    "name": model_name,
                    "display_name": model_name.replace('_', ' ').title()
                })

            return {
                "models": models,
                "current_model": None
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

    try:
        # Use generator's model directory
        models = []
        model_dir = Path(MODEL_DIR) if MODEL_DIR else Path(__file__).parent.parent.parent.parent / "rnn-text-generator" / "backend" / "saved_models"

        for model_file in model_dir.glob("*.pt"):
            model_name = model_file.stem
            # Skip excluded models
            if model_name.lower() in excluded_models:
                continue
            models.append({
                "name": model_name,
                "display_name": model_name.replace('_', ' ').title()
            })

        return {
            "models": models,
            "current_model": current_model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@router.post("/models/switch")
async def switch_model(request: SwitchModelRequest):
    """Switch to a different model"""
    global rnn_generator, current_model_name

    try:
        model_dir = Path(__file__).parent.parent.parent.parent / "rnn-text-generator" / "backend" / "saved_models"
        model_path = model_dir / f"{request.model_name}.pt"

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found")

        # Determine tokenizer path
        if request.model_name.startswith('model_optimal'):
            suffix = request.model_name.split('_')[-1]
            tokenizer_path = model_dir / f"tokenizer_optimal_{suffix}.pkl"
        else:
            tokenizer_path = model_dir / f"{request.model_name.replace('model', 'tokenizer')}.pkl"

        if not tokenizer_path.exists():
            raise HTTPException(status_code=404, detail=f"Tokenizer for '{request.model_name}' not found")

        # Load the model
        generator = TextGenerator()
        generator.load_model(str(model_path), str(tokenizer_path))
        rnn_generator = generator
        current_model_name = request.model_name

        return {
            "success": True,
            "message": f"Successfully switched to model: {request.model_name}",
            "model_name": request.model_name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error switching model: {str(e)}")


@router.get("/technical-report")
async def get_technical_report():
    """Get technical report markdown content"""
    try:
        report_path = Path(__file__).parent.parent.parent.parent / "rnn-text-generator" / "Technical_Report.md"

        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Technical report not found")

        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            "content": content,
            "filename": "Technical_Report.md"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading technical report: {str(e)}")


# Preload model on startup (optional)
@router.on_event("startup")
async def startup_event():
    """Preload model on startup"""
    # Uncomment to preload (uses more RAM)
    # load_rnn_generator()
    pass
