# src/cassava_classifier/pipelines/infer.py
from pathlib import Path

import cv2
import pandas as pd
import omegaconf
import torch
import torch.nn.functional as F
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score

from ..data.datamodule import CassavaDataModule
from ..models.model import CassavaLightningModule
from ..utils.preprocessing import clean_labels
import mlflow

torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([type(omegaconf.DictConfig({}))])

def load_checkpoint(model_path: str, model_config):
    """Load trained model from checkpoint"""
    model = CassavaLightningModule.load_from_checkpoint(
        model_path,
        model_config=model_config,
        map_location="cpu",
        weights_only=False  # ⚠️ Required to load non-weight data (e.g., OmegaConf)
    )
    model.eval()
    return model


def preprocess_image(image_path: str, img_size: int):
    """Preprocess single image for inference"""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = Compose(
        [
            Resize(height=img_size, width=img_size),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ]
    )

    augmented = transform(image=image)
    return augmented["image"].unsqueeze(0)  # Add batch dimension


def predict_single_image(model, image_tensor, device="cpu"):
    """Run prediction on single image"""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class.item(), probabilities.squeeze().cpu().numpy()

def ensemble_predict(cfg: DictConfig):
    print(f"\n{'='*50}")
    print("RUNNING ENSEMBLE PREDICTION")
    print(f"{'='*50}")

    # Load validation data
    train_csv = Path(cfg.data.dataroot) / "train.csv"
    df = pd.read_csv(train_csv)
    df = clean_labels(df, cfg.data.dataroot)

    # Use last fold's val split (for consistency)
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(
        n_splits=cfg.train.n_splits, shuffle=True, random_state=cfg.train.seed
    )
    _, val_idx = list(skf.split(df, df["label"]))[-1]
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Create dataloader
    data_module = CassavaDataModule(
        val_df,
        val_df,
        model_config=cfg.model,
        dataroot=cfg.data.dataroot,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
    )
    data_module.setup()
    val_loader = data_module.val_dataloader()

    # Load all 3 models - FIXED PATHS
    import glob
    import os
    
    model_dirs = [
        Path(cfg.data.output_dir) / "models" / "model1",
        Path(cfg.data.output_dir) / "models" / "model2",
        Path(cfg.data.output_dir) / "models" / "model3",
    ]
    
    model_paths = []
    for model_dir in model_dirs:
        checkpoints = glob.glob(f"{model_dir}/best-fold_*.ckpt")
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints in {model_dir}")
        latest = max(checkpoints, key=os.path.getctime)
        model_paths.append(latest)

    configs = [
        OmegaConf.load("configs/model/model1.yaml"),
        OmegaConf.load("configs/model/model2.yaml"),
        OmegaConf.load("configs/model/model3.yaml"),
    ]

    models = []
    for path, model_cfg in zip(model_paths, configs):
        model = load_checkpoint(path, model_cfg)  # ← Use fixed function
        model.to(device)
        model.eval()
        models.append(model)
        
    # Run ensemble
    all_preds = []
    all_targets = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for batch in val_loader:
        x, y = batch
        x = x.to(device)
        ensemble_probs = []

        for model, model_cfg in zip(models, configs):
            if x.shape[-1] != model_cfg.img_size:
                x_resized = F.interpolate(
                    x.float(), size=(model_cfg.img_size, model_cfg.img_size)
                )
            else:
                x_resized = x.float()

            with torch.no_grad():
                probs = torch.softmax(model(x_resized), dim=1)
                ensemble_probs.append(probs)

        avg_probs = torch.stack(ensemble_probs).mean(0)
        preds = torch.argmax(avg_probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.numpy())

    # Metrics
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="weighted")
    
    mlflow.set_tracking_uri(cfg.train.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.train.mlflow.experiment_name)
    
    with mlflow.start_run(run_name="ensemble_all_models"):
        mlflow.log_metrics({
            "ensemble_accuracy": acc,
            "ensemble_f1": f1
        })
        mlflow.log_params({
            "ensembled_models": ["model1", "model2", "model3"],
            "debug_mode": cfg.get("debug", False)
        })
    
    print("\nENSEMBLE RESULTS:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"{'='*50}")
    print(f"\n{'='*50}")
    