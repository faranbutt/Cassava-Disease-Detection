# src/cassava_classifier/pipelines/convert.py
from pathlib import Path

import torch
import torch.serialization
from omegaconf import DictConfig
import omegaconf
from ..models.model import CassavaLightningModule


def convert_to_onnx(checkpoint_path: str, output_path: str, model_config: DictConfig):
    """
    Convert a trained PyTorch Lightning checkpoint to ONNX format.
    """
    # ✅ Fix: Allow OmegaConf in safe loading
    torch.serialization.add_safe_globals([type(model_config)])

    # Load model from checkpoint (weights_only=False for compatibility)
    model = CassavaLightningModule.load_from_checkpoint(
        checkpoint_path,
        model_config=model_config,
        map_location="cpu",
        weights_only=False  # ← Also add this for safety
    )
    model.eval()

    # Handle image division models (448px -> 4x224px)
    if model_config.get("divide_image", False):
        dummy_input = torch.randn(1, 3, model_config.img_size, model_config.img_size)
    else:
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
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"✅ Model successfully converted to ONNX: {output_path}")
        return True
    except Exception as e:
        print(f"❌ ONNX conversion failed for {checkpoint_path}: {str(e)}")
        return False
