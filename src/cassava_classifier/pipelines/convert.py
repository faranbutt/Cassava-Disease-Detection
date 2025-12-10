# src/cassava_classifier/pipelines/convert.py
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import DictConfig
from ..models.model import CassavaLightningModule

def convert_to_onnx(checkpoint_path: str, output_path: str, model_config: DictConfig):
    """
    Convert a trained PyTorch Lightning checkpoint to ONNX format.
    
    Args:
        checkpoint_path: Path to the .ckpt file
        output_path: Path where .onnx file will be saved
        model_config: Hydra config for the model
    """
    # Load model from checkpoint
    model = CassavaLightningModule.load_from_checkpoint(
        checkpoint_path,
        model_config=model_config,
        map_location='cpu'  # Ensure CPU loading for portability
    )
    model.eval()
    
    # Handle image division models (448px -> 4x224px)
    if model_config.get('divide_image', False):
        # For divided models, we need to handle the full 448px input
        dummy_input = torch.randn(1, 3, model_config.img_size, model_config.img_size)
    else:
        # Standard models use direct input size
        dummy_input = torch.randn(1, 3, model_config.img_size, model_config.img_size)
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✅ Model successfully converted to ONNX: {output_path}")
        return True
    except Exception as e:
        print(f"❌ ONNX conversion failed for {checkpoint_path}: {str(e)}")
        return False