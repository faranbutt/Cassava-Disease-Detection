# src/cassava_classifier/pipelines/train.py
import sys
import os
import glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from omegaconf import DictConfig, OmegaConf 

# Absolute imports
from cassava_classifier.data.datamodule import CassavaDataModule
from cassava_classifier.models.model import CassavaLightningModule
from cassava_classifier.utils.preprocessing import clean_labels
from cassava_classifier.pipelines.convert import convert_to_onnx
from cassava_classifier.pipelines.infer import ensemble_predict


def train_fold(fold: int, train_df, val_df, model_config: DictConfig, cfg: DictConfig):
    data_module = CassavaDataModule(
        train_df, val_df,
        model_config=model_config,
        dataroot=cfg.data.dataroot,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers
    )

    model = CassavaLightningModule(model_config=model_config, lr=cfg.train.lr)

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.data.output_dir) / "models" / cfg.model_name,
        filename="best-fold{fold}",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices=1,
        precision=cfg.train.precision,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="val_loss", patience=cfg.train.patience),
            LearningRateMonitor()
        ],
        log_every_n_steps=cfg.train.log_every_n_steps,
        enable_progress_bar=True
    )

    trainer.fit(model, data_module)
    return checkpoint_callback.best_model_path


def train_model(cfg: DictConfig):
    # cfg.model is already the correct model config (model1, model2, or model3)
    model_config = cfg.model

    # Set seeds for reproducibility
    pl.seed_everything(cfg.train.seed, workers=True)

    # Load and clean data
    train_csv = Path(cfg.data.dataroot) / "train.csv"
    df = pd.read_csv(train_csv)
    df = clean_labels(df, cfg.data.dataroot)

    # Debug mode: use subset if enabled
    if cfg.get("debug", False):
        df = df.sample(n=cfg.debug_samples, random_state=cfg.train.seed).reset_index(drop=True)
        print(f"⚠️ DEBUG MODE: Using {len(df)} samples")

    # K-Fold Cross Validation
    skf = StratifiedKFold(
        n_splits=cfg.train.n_splits,
        shuffle=True,
        random_state=cfg.train.seed
    )
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label']), 1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        best_path = train_fold(fold, train_df, val_df, model_config, cfg)
        print(f"Fold {fold} best model: {best_path}")


def get_latest_checkpoint(model_dir: str) -> str:
    checkpoints = glob.glob(f"{model_dir}/best-foldfold=0*.ckpt")
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    return max(checkpoints, key=os.path.getctime)

def train_all_models_and_ensemble(cfg: DictConfig):
    model_configs = {
        "model1": "configs/model/model1.yaml",
        "model2": "configs/model/model2.yaml",
        "model3": "configs/model/model3.yaml"
    }
    
    for model_name, config_path in model_configs.items():
        print(f"\n{'='*50}")
        print(f"TRAINING {model_name}")
        print(f"{'='*50}")
        
        model_cfg = OmegaConf.load(config_path)
        OmegaConf.update(cfg, "model", model_cfg, merge=False)
        OmegaConf.update(cfg, "model_name", model_name, merge=False)
        
        train_model(cfg)
        
        # Convert to ONNX using correct path
        model_dir = Path(cfg.data.output_dir) / "models" / model_name
        checkpoint_path = get_latest_checkpoint(str(model_dir))
        onnx_path = model_dir / f"{model_name}.onnx"
        convert_to_onnx(str(checkpoint_path), str(onnx_path), model_cfg)
    
    ensemble_predict(cfg)