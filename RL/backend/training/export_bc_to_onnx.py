"""
Export Behavioral Cloning model to ONNX format
"""
import torch
import torch.onnx
import numpy as np
from train_behavioral_cloning import PolicyNetwork


def export_bc_to_onnx(checkpoint_path, output_path):
    """
    Export BC model to ONNX format for browser deployment

    Args:
        checkpoint_path: Path to .pth checkpoint file
        output_path: Where to save .onnx file
    """
    print(f"[BC-EXPORT] Loading BC model from {checkpoint_path}")

    # Load checkpoint (weights_only=False because checkpoint contains numpy arrays)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract model parameters
    input_size = checkpoint['input_size']
    hidden_sizes = checkpoint['hidden_sizes']
    num_actions = checkpoint['num_actions']

    print(f"[BC-EXPORT] Model architecture:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden layers: {hidden_sizes}")
    print(f"  Output actions: {num_actions}")

    # Create model
    model = PolicyNetwork(input_size, hidden_sizes, num_actions)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[BC-EXPORT] Model loaded successfully")

    # Create dummy input (batch_size=1, features=input_size)
    dummy_input = torch.randn(1, input_size)

    print(f"[BC-EXPORT] Exporting to ONNX...")
    print(f"[BC-EXPORT] Output path: {output_path}")

    # Export to ONNX
    # Using opset_version=13 for better compatibility with onnxruntime-web
    # (opset 18 is not supported by torch.onnx.export, max is 17)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['state_features'],
        output_names=['action_probs'],
        dynamic_axes={
            'state_features': {0: 'batch_size'},
            'action_probs': {0: 'batch_size'}
        }
    )

    print(f"[BC-EXPORT] SUCCESS: ONNX model saved to: {output_path}")

    # Verify the export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[BC-EXPORT] SUCCESS: ONNX model verified successfully")

    # Print model info
    print(f"\n[BC-EXPORT] Model Summary:")
    print(f"  Input: state_features [batch_size, {input_size}]")
    print(f"  Output: action_probs [batch_size, {num_actions}]")
    print(f"  Actions: [left, right, jump, sprint]")
    print(f"\n[BC-EXPORT] Normalization parameters:")
    print(f"  Mean: {checkpoint['normalize_mean']}")
    print(f"  Std: {checkpoint['normalize_std']}")
    print(f"\n[BC-EXPORT] Ready for deployment!")

    return output_path


if __name__ == "__main__":
    import sys

    checkpoint_path = "models/bc_agent.pth"
    output_path = "models/exported_onnx/model.onnx"

    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    # Create output directory
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    export_bc_to_onnx(checkpoint_path, output_path)
