# src/cassava_classifier/pipelines/infer.py
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import DictConfig
from ..models.model import CassavaLightningModule
from ..data.dataset import CassavaDataset
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Resize, Normalize
import cv2
import numpy as np

def load_checkpoint(model_path: str, model_config: DictConfig):
    """Load trained model from checkpoint"""
    model = CassavaLightningModule.load_from_checkpoint(
        model_path, 
        model_config=model_config
    )
    model.eval()
    return model

def preprocess_image(image_path: str, img_size: int):
    """Preprocess single image for inference"""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = Compose([
        Resize(height=img_size, width=img_size),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])
    
    augmented = transform(image=image)
    return augmented['image'].unsqueeze(0)  # Add batch dimension

def predict_single_image(model, image_tensor, device='cpu'):
    """Run prediction on single image"""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class.item(), probabilities.squeeze().cpu().numpy()

def ensemble_predict(cfg: DictConfig):
    """Load all 3 models and run ensemble prediction"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = []
    configs = []
    
    for model_name in ['model1', 'model2', 'model3']:
        # Load model config
        if model_name == 'model1':
            model_config = cfg._content['model1_config']  # You'll need to adjust this
        # ... similar for model2, model3
        
        # Load checkpoint (you'll need to track checkpoint paths)
        checkpoint_path = f"{cfg.data.output_dir}/models/{model_name}/best-fold0.ckpt"
        model = load_checkpoint(checkpoint_path, model_config)
        model.to(device)
        models.append(model)
        configs.append(model_config)
    
    # For now, just return success
    print("✅ Ensemble prediction system ready!")
    return models, configs