# src/cassava_classifier/pipelines/train.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from omegaconf import DictConfig, OmegaConf  # ← Add OmegaConf

# Absolute imports
from cassava_classifier.data.datamodule import CassavaDataModule
from cassava_classifier.models.model import CassavaLightningModule
from cassava_classifier.utils.preprocessing import clean_labels
from cassava_classifier.pipelines.convert import convert_to_onnx


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


def train_all_models_and_ensemble(cfg: DictConfig):
    """
    Train all three models and convert them to ONNX format for production.
    """
    model_configs = {
        "model1": "configs/model/model1.yaml",
        "model2": "configs/model/model2.yaml", 
        "model3": "configs/model/model3.yaml"
    }
    
    trained_model_paths = {}
    
    for model_name, config_path in model_configs.items():
        print(f"\n{'='*60}")
        print(f"TRAINING {model_name}")
        print(f"{'='*60}")
        
        # Load model-specific config
        from omegaconf import OmegaConf
        model_specific_config = OmegaConf.load(config_path)
        
        # Update cfg with current model config
        cfg.model = model_specific_config
        cfg.model_name = model_name
        
        # Train the model
        train_model(cfg)
        
        # Get the best checkpoint path
        checkpoint_path = Path(cfg.data.output_dir) / "models" / model_name / "best-fold0.ckpt"
        if not checkpoint_path.exists():
            # Try alternative naming (Hydra might use different fold naming)
            checkpoint_path = list(Path(cfg.data.output_dir).glob(f"models/{model_name}/best-fold*.ckpt"))[0]
        
        trained_model_paths[model_name] = str(checkpoint_path)
        print(f"✅ {model_name} trained. Checkpoint: {checkpoint_path}")
        
        # Convert to ONNX
        onnx_path = Path(cfg.data.output_dir) / "models" / model_name / f"{model_name}.onnx"
        success = convert_to_onnx(str(checkpoint_path), str(onnx_path), model_specific_config)
        
        if success:
            print(f"✅ {model_name} converted to ONNX: {onnx_path}")
        else:
            print(f"⚠️  {model_name} ONNX conversion failed, continuing...")
    
    print(f"\n{'='*60}")
    print("ALL MODELS TRAINED AND CONVERTED TO ONNX")
    print(f"{'='*60}")
    
    # Return paths for ensemble (if needed later)
    return trained_model_paths