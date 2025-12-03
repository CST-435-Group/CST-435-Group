"""
Export trained PyTorch model to TensorFlow.js format for web deployment
"""

import torch
import onnx
import numpy as np
from stable_baselines3 import PPO
from pathlib import Path
import subprocess
import sys


def load_pytorch_model(model_path):
    """
    Load trained Stable-Baselines3 model.

    Args:
        model_path: Path to .zip model file

    Returns:
        Loaded model
    """
    print(f"[EXPORT] Loading model from {model_path}")
    model = PPO.load(model_path)
    print("[EXPORT] Model loaded successfully")
    return model


def extract_policy_network(model):
    """
    Extract the policy network from the full model.
    This is what makes decisions during gameplay.

    Args:
        model: Stable-Baselines3 model

    Returns:
        torch.nn.Module: Policy network
    """
    print("[EXPORT] Extracting policy network...")
    # Get the actor (policy) network from the model
    policy_net = model.policy
    policy_net.eval()  # Set to evaluation mode
    print("[EXPORT] Policy network extracted")
    return policy_net


def convert_to_onnx(policy_network, save_path, input_shape):
    """
    Convert PyTorch policy network to ONNX format.

    Args:
        policy_network: Extracted policy network
        save_path: Where to save .onnx file
        input_shape: Input tensor shape (e.g., [1, 84, 84, 3])
    """
    print(f"[EXPORT] Converting to ONNX format...")
    print(f"[EXPORT] Input shape: {input_shape}")

    # Create dummy input with correct shape
    # PyTorch expects (batch, channels, height, width) for images
    # But Stable-Baselines3 uses (batch, channels, height, width) directly
    dummy_input = torch.randn(*input_shape)

    # Export to ONNX
    torch.onnx.export(
        policy_network,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action_logits'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action_logits': {0: 'batch_size'}
        }
    )

    print(f"[EXPORT] ONNX model saved to {save_path}")

    # Verify the model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print("[EXPORT] ONNX model verified successfully")


def convert_onnx_to_tfjs(onnx_path, output_dir):
    """
    Convert ONNX model to TensorFlow.js format.

    Args:
        onnx_path: Path to .onnx file
        output_dir: Output directory for TensorFlow.js model
    """
    print(f"[EXPORT] Converting ONNX to TensorFlow.js...")

    # First convert ONNX to TensorFlow SavedModel
    tf_model_dir = output_dir / "tf_saved_model"
    tf_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXPORT] Step 1: Converting ONNX to TensorFlow SavedModel...")
    try:
        # Use onnx-tf to convert ONNX to TensorFlow
        result = subprocess.run(
            [sys.executable, "-m", "onnx_tf.backend.cli", "convert",
             "-i", str(onnx_path),
             "-o", str(tf_model_dir)],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"[EXPORT] Warning: onnx-tf conversion failed: {result.stderr}")
            print(f"[EXPORT] Trying alternative method...")

            # Alternative: Use Python API
            import onnx_tf.backend
            onnx_model = onnx.load(str(onnx_path))
            tf_rep = onnx_tf.backend.prepare(onnx_model)
            tf_rep.export_graph(str(tf_model_dir))

        print(f"[EXPORT] TensorFlow SavedModel created")
    except Exception as e:
        print(f"[EXPORT] Error in ONNX to TF conversion: {e}")
        raise

    # Then convert TensorFlow SavedModel to TensorFlow.js
    tfjs_model_dir = output_dir / "tfjs_model"
    tfjs_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXPORT] Step 2: Converting TensorFlow to TensorFlow.js...")
    try:
        result = subprocess.run(
            ["tensorflowjs_converter",
             "--input_format=tf_saved_model",
             "--output_format=tfjs_graph_model",
             "--signature_name=serving_default",
             "--saved_model_tags=serve",
             str(tf_model_dir),
             str(tfjs_model_dir)],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"[EXPORT] TensorFlow.js conversion failed: {result.stderr}")
            raise RuntimeError(f"tensorflowjs_converter failed: {result.stderr}")

        print(f"[EXPORT] TensorFlow.js model saved to {tfjs_model_dir}")
        print(result.stdout)
    except Exception as e:
        print(f"[EXPORT] Error in TF to TFJS conversion: {e}")
        raise


def test_exported_model(pytorch_policy, test_input):
    """
    Test that PyTorch model works with test input.

    Args:
        pytorch_policy: PyTorch policy network
        test_input: Sample input tensor

    Returns:
        Output prediction
    """
    print("[EXPORT] Testing exported model...")

    with torch.no_grad():
        output = pytorch_policy(test_input)
        # Get action probabilities
        if hasattr(output, 'distribution'):
            action_probs = output.distribution.probs
        elif isinstance(output, tuple):
            action_probs = torch.softmax(output[0], dim=-1)
        else:
            action_probs = torch.softmax(output, dim=-1)

        print(f"[EXPORT] Model output shape: {action_probs.shape}")
        print(f"[EXPORT] Action probabilities: {action_probs.cpu().numpy()}")

    print("[EXPORT] Model test successful!")
    return action_probs


if __name__ == "__main__":
    """
    Export pipeline:
    1. Load PyTorch model
    2. Extract policy network
    3. Convert to ONNX
    4. Convert to TensorFlow.js
    5. Test exported model
    """
    print("=" * 60)
    print("RL PLATFORMER MODEL EXPORT")
    print("=" * 60)

    # Paths
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / "models" / "platformer_agent.zip"
    export_dir = base_dir / "models" / "exported"
    onnx_path = export_dir / "model.onnx"

    # Create export directory
    export_dir.mkdir(parents=True, exist_ok=True)

    # Check if model exists
    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}")
        print("[ERROR] Please train a model first using train_agent.py")
        sys.exit(1)

    try:
        # Step 1: Load PyTorch model
        model = load_pytorch_model(str(model_path))

        # Step 2: Extract policy network
        policy_net = extract_policy_network(model)

        # Step 3: Create test input
        # Input shape: (batch, channels, height, width)
        # For RL: (1, 3, 84, 84) - batch=1, RGB channels=3, 84x84 image
        input_shape = (1, 3, 84, 84)
        test_input = torch.randn(*input_shape)

        # Step 4: Test model before export
        test_exported_model(policy_net, test_input)

        # Step 5: Convert to ONNX
        convert_to_onnx(policy_net, str(onnx_path), input_shape)

        # Step 6: Convert ONNX to TensorFlow.js
        convert_onnx_to_tfjs(onnx_path, export_dir)

        print("=" * 60)
        print("[SUCCESS] Model export complete!")
        print(f"[SUCCESS] TensorFlow.js model saved to:")
        print(f"[SUCCESS]   {export_dir / 'tfjs_model'}")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Copy the tfjs_model folder to your web app's public directory")
        print("2. Use @tensorflow/tfjs to load and run the model in the browser")
        print("3. Preprocess game frames to 84x84x3 before inference")

    except Exception as e:
        print("=" * 60)
        print(f"[ERROR] Export failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
