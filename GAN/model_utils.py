"""
Model Saving/Loading Utilities with Automatic Chunking
Handles large model files by splitting them into GitHub-friendly chunks (<100MB)

Automatically splits models that would exceed 90MB into smaller files
Reassembles chunks transparently when loading
"""

import os
import json
import torch
import io
from pathlib import Path
from typing import Dict, Any, List


MAX_CHUNK_SIZE_MB = 90  # Split files that would exceed this size
MAX_CHUNK_SIZE_BYTES = MAX_CHUNK_SIZE_MB * 1024 * 1024


def get_tensor_size_bytes(tensor: torch.Tensor) -> int:
    """Calculate the size of a tensor in bytes"""
    return tensor.element_size() * tensor.nelement()


def get_state_dict_size_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    """Calculate total size of a state dict in bytes"""
    return sum(get_tensor_size_bytes(tensor) for tensor in state_dict.values())


def split_state_dict(state_dict: Dict[str, torch.Tensor], max_chunk_bytes: int) -> List[Dict[str, torch.Tensor]]:
    """
    Split a state_dict into chunks that don't exceed max_chunk_bytes

    Args:
        state_dict: PyTorch state dictionary
        max_chunk_bytes: Maximum size per chunk in bytes

    Returns:
        List of state_dict chunks
    """
    chunks = []
    current_chunk = {}
    current_size = 0

    # Sort parameters by size to pack efficiently
    sorted_params = sorted(state_dict.items(), key=lambda x: get_tensor_size_bytes(x[1]), reverse=True)

    for param_name, param_tensor in sorted_params:
        param_size = get_tensor_size_bytes(param_tensor)

        # If single parameter exceeds max size, it goes in its own chunk
        if param_size > max_chunk_bytes:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = {}
                current_size = 0
            chunks.append({param_name: param_tensor})
            continue

        # If adding this parameter would exceed chunk size, start new chunk
        if current_size + param_size > max_chunk_bytes:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = {param_name: param_tensor}
            current_size = param_size
        else:
            current_chunk[param_name] = param_tensor
            current_size += param_size

    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def save_model_chunked(model_data: Dict[str, Any], save_path: str) -> None:
    """
    Save a model with automatic chunking if it exceeds size threshold

    Args:
        model_data: Dictionary containing model state dicts and metadata
        save_path: Path where model should be saved

    If model exceeds MAX_CHUNK_SIZE_BYTES:
        - Splits large state_dicts into chunks
        - Saves chunks as separate files: save_path.chunk_000, .chunk_001, etc.
        - Creates manifest file: save_path.manifest.json
    Otherwise:
        - Saves normally as a single .pth file
    """
    save_path = Path(save_path)

    # First, estimate total size by serializing to memory
    buffer = io.BytesIO()
    torch.save(model_data, buffer)
    total_size = buffer.tell()

    # If under threshold, save normally
    if total_size <= MAX_CHUNK_SIZE_BYTES:
        torch.save(model_data, save_path)
        print(f"  [SAVE] Model saved: {save_path.name} ({total_size / (1024*1024):.1f}MB)")
        return

    # Model exceeds threshold - need to chunk
    print(f"  [CHUNKING] Model size {total_size / (1024*1024):.1f}MB exceeds {MAX_CHUNK_SIZE_MB}MB, splitting...")

    # Separate metadata from state dicts
    metadata = {}
    state_dicts = {}

    def convert_to_serializable(obj):
        """Convert tensors and other non-serializable objects to serializable types"""
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    for key, value in model_data.items():
        if isinstance(value, dict) and any(isinstance(v, torch.Tensor) for v in value.values()):
            # This is a state dict
            state_dicts[key] = value
        else:
            # This is metadata - convert to serializable
            metadata[key] = convert_to_serializable(value)

    # Split each state dict that's too large
    chunked_state_dicts = {}
    for state_dict_name, state_dict in state_dicts.items():
        state_dict_size = get_state_dict_size_bytes(state_dict)

        if state_dict_size > MAX_CHUNK_SIZE_BYTES * 0.8:  # Leave some headroom
            chunks = split_state_dict(state_dict, MAX_CHUNK_SIZE_BYTES)
            chunked_state_dicts[state_dict_name] = {
                'chunked': True,
                'num_chunks': len(chunks),
                'chunks': chunks
            }
            print(f"    - {state_dict_name}: {state_dict_size / (1024*1024):.1f}MB -> {len(chunks)} chunks")
        else:
            chunked_state_dicts[state_dict_name] = {
                'chunked': False,
                'data': state_dict
            }

    # Create manifest
    manifest = {
        'version': '1.0',
        'is_chunked': True,
        'total_size_mb': total_size / (1024*1024),
        'metadata': metadata,
        'state_dicts': {}
    }

    # Save chunks
    chunk_files = []
    global_chunk_id = 0

    for state_dict_name, chunk_info in chunked_state_dicts.items():
        if chunk_info['chunked']:
            # Save each chunk to separate file
            chunk_ids = []
            for chunk_data in chunk_info['chunks']:
                chunk_filename = f"{save_path.stem}.chunk_{global_chunk_id:03d}{save_path.suffix}"
                chunk_path = save_path.parent / chunk_filename

                torch.save(chunk_data, chunk_path)
                chunk_size = os.path.getsize(chunk_path)

                chunk_ids.append(global_chunk_id)
                chunk_files.append({
                    'chunk_id': global_chunk_id,
                    'filename': chunk_filename,
                    'size_mb': chunk_size / (1024*1024),
                    'state_dict': state_dict_name
                })

                global_chunk_id += 1

            manifest['state_dicts'][state_dict_name] = {
                'chunked': True,
                'chunk_ids': chunk_ids
            }
        else:
            # Save unchunked state dict to a single chunk file
            chunk_filename = f"{save_path.stem}.chunk_{global_chunk_id:03d}{save_path.suffix}"
            chunk_path = save_path.parent / chunk_filename

            torch.save(chunk_info['data'], chunk_path)
            chunk_size = os.path.getsize(chunk_path)

            manifest['state_dicts'][state_dict_name] = {
                'chunked': False,
                'chunk_id': global_chunk_id
            }

            chunk_files.append({
                'chunk_id': global_chunk_id,
                'filename': chunk_filename,
                'size_mb': chunk_size / (1024*1024),
                'state_dict': state_dict_name
            })

            global_chunk_id += 1

    manifest['chunk_files'] = chunk_files
    manifest['total_chunks'] = len(chunk_files)

    # Save manifest
    manifest_path = save_path.parent / f"{save_path.stem}.manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"  [CHUNKED] Saved {len(chunk_files)} chunks (max {max(c['size_mb'] for c in chunk_files):.1f}MB)")
    print(f"  [MANIFEST] {manifest_path.name}")

    # Also save a small placeholder file to indicate chunking
    placeholder_data = {
        'is_chunked': True,
        'manifest_file': f"{save_path.stem}.manifest.json",
        'message': f'This model is split into {len(chunk_files)} chunks. Use load_model_chunked() to load it.'
    }
    torch.save(placeholder_data, save_path)


def load_model_chunked(load_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Load a model, automatically handling chunked files

    Args:
        load_path: Path to model file or manifest
        device: Device to load tensors to ('cpu', 'cuda', etc.)

    Returns:
        Complete model_data dictionary
    """
    load_path = Path(load_path)

    # Check if this is a chunked model by looking for manifest
    manifest_path = load_path.parent / f"{load_path.stem}.manifest.json"

    if not manifest_path.exists():
        # Not chunked, load normally
        model_data = torch.load(load_path, map_location=device)

        # Check if it's a placeholder for chunked model
        if isinstance(model_data, dict) and model_data.get('is_chunked'):
            manifest_path = load_path.parent / model_data['manifest_file']
        else:
            return model_data

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"  [LOADING] Chunked model: {manifest['total_chunks']} chunks, {manifest['total_size_mb']:.1f}MB total")

    # Start with metadata
    model_data = manifest['metadata'].copy()

    # Load and reassemble each state dict
    for state_dict_name, state_dict_info in manifest['state_dicts'].items():
        if state_dict_info['chunked']:
            # Load and merge chunks
            merged_state_dict = {}

            for chunk_id in state_dict_info['chunk_ids']:
                chunk_info = next(c for c in manifest['chunk_files'] if c['chunk_id'] == chunk_id)
                chunk_path = load_path.parent / chunk_info['filename']

                chunk_data = torch.load(chunk_path, map_location=device)
                merged_state_dict.update(chunk_data)

            model_data[state_dict_name] = merged_state_dict
        else:
            # Load single chunk
            chunk_id = state_dict_info['chunk_id']
            chunk_info = next(c for c in manifest['chunk_files'] if c['chunk_id'] == chunk_id)
            chunk_path = load_path.parent / chunk_info['filename']

            model_data[state_dict_name] = torch.load(chunk_path, map_location=device)

    print(f"  [LOADED] Model reassembled in RAM")

    return model_data


def clean_chunks(model_path: str) -> None:
    """
    Remove chunk files and manifest for a chunked model

    Args:
        model_path: Path to the model file
    """
    model_path = Path(model_path)
    manifest_path = model_path.parent / f"{model_path.stem}.manifest.json"

    if not manifest_path.exists():
        # Not a chunked model, just remove the single file
        if model_path.exists():
            os.remove(model_path)
        return

    # Load manifest to find all chunks
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Remove all chunk files
    for chunk_info in manifest['chunk_files']:
        chunk_path = model_path.parent / chunk_info['filename']
        if chunk_path.exists():
            os.remove(chunk_path)

    # Remove manifest
    os.remove(manifest_path)

    # Remove placeholder file if it exists
    if model_path.exists():
        os.remove(model_path)

    print(f"  [CLEANED] Removed {len(manifest['chunk_files'])} chunks and manifest")
