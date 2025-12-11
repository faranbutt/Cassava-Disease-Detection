# src/cassava_classifier/pipelines/train.py
import glob
import os
import sys
from pathlib import Path
import typing
import torch
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from cassava_classifier.data.datamodule import CassavaDataModule
from cassava_classifier.models.model import CassavaLightningModule
from cassava_classifier.pipelines.convert import convert_to_onnx
from cassava_classifier.pipelines.infer import ensemble_predict
from cassava_classifier.utils.preprocessing import clean_labels

# Ensure repo root is on path (optional, safe)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

torch.serialization.add_safe_globals([typing.Any])

def train_fold(fold: int, train_df, val_df, model_config: DictConfig, cfg: DictConfig):
    """Train a single fold, log to MLflow, save checkpoint, and register model."""
    # MLflow setup
    tracking_uri = cfg.train.mlflow.get("tracking_uri", "sqlite:///mlflow.db")
    artifact_location = cfg.train.mlflow.get("artifact_location", "./mlartifacts")
    experiment_name = cfg.train.mlflow.get("experiment_name", "cassava-disease-classification")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(name=experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)

    # Start run
    run_name = f"{cfg.model_name}_fold_{fold}"
    with mlflow.start_run(run_name=run_name):
        # Log params
        mlflow.log_params({
            "model_name": cfg.model_name,
            "fold": fold,
            "batch_size": cfg.train.batch_size,
            "epochs": cfg.train.epochs,
            "lr": cfg.train.lr,
            "img_size": model_config.img_size,
            "use_multidrop": model_config.get("use_multidrop", False),
            "use_attention": model_config.get("use_attention", False),
            "label_smoothing": model_config.get("label_smoothing", 0.0),
            "divide_image": model_config.get("divide_image", False)
        })

        # Attempt to log git commit hash for traceability
        try:
            import subprocess

            git_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("ascii")
                .strip()
            )
            mlflow.log_param("git_commit", git_hash)
        except Exception:
            mlflow.log_param("git_commit", "unknown")

        # Build datamodule & model
        data_module = CassavaDataModule(
            train_df,
            val_df,
            model_config=model_config,
            dataroot=cfg.data.dataroot,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
        )

        model = CassavaLightningModule(model_config=model_config, lr=cfg.train.lr)

        # Ensure checkpoint directory exists
        checkpoint_dir = Path(cfg.data.output_dir) / "models" / cfg.model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"best-fold_{fold}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

        trainer = Trainer(
            max_epochs=cfg.train.epochs,
            accelerator="auto",
            devices=1,
            precision=cfg.train.precision,
            callbacks=[
                checkpoint_callback,
                EarlyStopping(monitor="val_loss", patience=cfg.train.patience),
                LearningRateMonitor(),
            ],
            log_every_n_steps=cfg.train.log_every_n_steps,
            enable_progress_bar=True,
            logger=True,
        )

        # Fit
        trainer.fit(model, data_module)

        # Trainer may not have saved a checkpoint if no improvement; get best path carefully
        best_ckpt_path = checkpoint_callback.best_model_path
        if not best_ckpt_path:
            # Fall back to newest checkpoint file in the directory
            ckpts = glob.glob(str(checkpoint_dir / "best-fold_*.ckpt"))
            best_ckpt_path = max(ckpts, key=os.path.getctime) if ckpts else ""

        # Log metrics (best-effort)
        val_metrics = trainer.callback_metrics or {}
        train_metrics = trainer.logged_metrics or {}
        mlflow.log_metrics({
            "val_loss": float(val_metrics.get("val_loss", 0)) if val_metrics else 0.0,
            "val_acc": float(val_metrics.get("val_acc", 0)) if val_metrics else 0.0,
            "train_loss": float(train_metrics.get("train_loss_epoch", 0)) if train_metrics else 0.0,
            "train_acc": float(train_metrics.get("train_acc_epoch", 0)) if train_metrics else 0.0,
        })

        # Save and log a simple metrics plot (best-effort)
        plot_dir = Path("plots")
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"metrics_fold_{fold}_{cfg.model_name}.png"
        try:
            create_metrics_plot(trainer, str(plot_path))
            mlflow.log_artifact(str(plot_path), artifact_path="plots")
        except Exception as e:
            print(f"Warning: couldn't create/log metrics plot: {e}")

        # Log the checkpoint file as an artifact
        if best_ckpt_path and Path(best_ckpt_path).exists():
            try:
                mlflow.log_artifact(best_ckpt_path, artifact_path="checkpoints")
            except Exception as e:
                print(f"Warning: failed to log checkpoint artifact: {e}")
        else:
            print("Warning: No checkpoint file found to log as artifact.")

        # Attempt to register the best model in MLflow Model Registry
        # Load the best checkpoint into a fresh module and log it via mlflow.pytorch
        if best_ckpt_path and Path(best_ckpt_path).exists():
            try:
                best_model = CassavaLightningModule.load_from_checkpoint(
                    best_ckpt_path, model_config=model_config
                )
                registered_model_name = f"{experiment_name}-{cfg.model_name}"
                # Log + register model
                mlflow.pytorch.log_model(
                    pytorch_model=best_model,
                    artifact_path="model",
                    registered_model_name=registered_model_name,
                )
                client = MlflowClient()
                print("Available registered models:")
                for m in client.list_registered_models():
                    print(" -", m.name)
                print(f"✅ Logged and registered model as: {registered_model_name}")
            except Exception as e:
                print(f"Warning: failed to log/register model to MLflow: {e}")
        else:
            print("Skipping model registry step (no checkpoint).")

        # Return the checkpoint path
        return best_ckpt_path


def create_metrics_plot(trainer, plot_path):
    """Create training/validation metrics plot"""
    try:
        # Get all logged metrics from trainer
        # We'll manually track them since trainer.logged_metrics only has last value
        # Instead, we'll use the fact that we can access the logger's history if available
        # For simplicity, let's extract from trainer's callback_metrics (which may be empty)
        # Better approach: Track metrics during training via a custom callback
        
        # For now, fallback to using the single value we have
        metrics = trainer.logged_metrics or {}
        
        # Create plot
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        
        # Training Loss (epoch)
        train_loss_epoch = metrics.get("train_loss_epoch", 0.0)
        ax[0,0].plot([1], [train_loss_epoch], 'b-', label='Train Loss')
        ax[0,0].set_title('Training Loss')
        ax[0,0].legend()
        
        # Validation Loss
        val_loss = metrics.get("val_loss", 0.0)
        ax[0,1].plot([1], [val_loss], 'r-', label='Val Loss')
        ax[0,1].set_title('Validation Loss')
        ax[0,1].legend()
        
        # Training Accuracy (epoch)
        train_acc_epoch = metrics.get("train_acc_epoch", 0.0)
        ax[1,0].plot([1], [train_acc_epoch], 'b-', label='Train Acc')
        ax[1,0].set_title('Training Accuracy')
        ax[1,0].legend()
        
        # Validation Accuracy
        val_acc = metrics.get("val_acc", 0.0)
        ax[1,1].plot([1], [val_acc], 'r-', label='Val Acc')
        ax[1,1].set_title('Validation Accuracy')
        ax[1,1].legend()
        
        # Set fixed axis limits for visibility
        ax[0,0].set_ylim(0, 3)  # Adjust based on your loss range
        ax[0,1].set_ylim(0, 3)
        ax[1,0].set_ylim(0, 1)
        ax[1,1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create plot: {e}")


def train_model(cfg: DictConfig):
    """Train the model specified in cfg.model using K-fold CV."""
    model_config = cfg.model
    pl.seed_everything(cfg.train.seed, workers=True)

    train_csv = Path(cfg.data.dataroot) / "train.csv"
    df = pd.read_csv(train_csv)
    df = clean_labels(df, cfg.data.dataroot)

    # Debug subset
    if cfg.get("debug", False):
        df = df.sample(n=min(cfg.debug_samples, len(df)), random_state=cfg.train.seed).reset_index(drop=True)
        print(f"⚠️ DEBUG MODE: Using {len(df)} samples")

    skf = StratifiedKFold(
        n_splits=cfg.train.n_splits, shuffle=True, random_state=cfg.train.seed
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"]), 1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        best_path = train_fold(fold, train_df, val_df, model_config, cfg)
        print(f"Fold {fold} best model: {best_path}")


def get_latest_checkpoint(model_dir: str) -> str:
    """Return newest best-fold_*.ckpt in model_dir."""
    checkpoints = glob.glob(os.path.join(model_dir, "best-fold_*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    return max(checkpoints, key=os.path.getctime)


def train_all_models_and_ensemble(cfg: DictConfig):
    """Train model1, model2, model3 (as per configs) and run ONNX conversion + ensemble."""
    model_configs = {
        "model1": "configs/model/model1.yaml",
        "model2": "configs/model/model2.yaml",
        "model3": "configs/model/model3.yaml",
    }

    for model_name, config_path in model_configs.items():
        print(f"\n{'='*50}")
        print(f"TRAINING {model_name}")
        print(f"{'='*50}")

        model_cfg = OmegaConf.load(config_path)
        OmegaConf.update(cfg, "model", model_cfg, merge=False)
        OmegaConf.update(cfg, "model_name", model_name, merge=False)

        train_model(cfg)

        # Convert to ONNX unless in debug mode
        if not cfg.get("debug", False):
            model_dir = Path(cfg.data.output_dir) / "models" / model_name
            try:
                checkpoint_path = get_latest_checkpoint(str(model_dir))
                onnx_path = model_dir / f"{model_name}.onnx"
                convert_to_onnx(str(checkpoint_path), str(onnx_path), model_cfg)
            except Exception as e:
                print(f"Warning: ONNX conversion skipped/failed for {model_name}: {e}")
        else:
            print("⚠️ Skipping ONNX export in debug mode")

    # Ensemble evaluation (skip in debug to avoid tiny-sample instability)
    if not cfg.get("debug", False):
        ensemble_predict(cfg)
    else:
        print("⚠️ Skipping ensemble in debug mode")
