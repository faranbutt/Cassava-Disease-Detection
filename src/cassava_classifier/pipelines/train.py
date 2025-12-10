import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from omegaconf import DictConfig

# Absolute imports
from cassava_classifier.data.datamodule import CassavaDataModule
from cassava_classifier.models.model import CassavaLightningModule
from cassava_classifier.utils.preprocessing import clean_labels


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