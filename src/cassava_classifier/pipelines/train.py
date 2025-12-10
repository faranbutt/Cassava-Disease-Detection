# src/cassava_classifier/pipelines/train.py
import glob
import os
import sys
import mlflow
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.model_selection import StratifiedKFold
from cassava_classifier.data.datamodule import CassavaDataModule
from cassava_classifier.models.model import CassavaLightningModule
from cassava_classifier.pipelines.convert import convert_to_onnx
from cassava_classifier.pipelines.infer import ensemble_predict
from cassava_classifier.utils.preprocessing import clean_labels
from pathlib import Path
import matplotlib.pyplot as plt


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def train_fold(fold: int, train_df, val_df, model_config: DictConfig, cfg: DictConfig):
    # Initialize MLflow
    tracking_uri = "sqlite:///mlflow.db"
    artifact_location = "./mlartifacts"
    
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = cfg.train.mlflow.experiment_name
    client = mlflow.MlflowClient()
    
    # Get or create experiment with artifact location
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        # Create experiment with custom artifact location
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location
        )
    else:
        experiment_id = experiment.experiment_id
        # Note: artifact_location is fixed once experiment is created

    # Use the experiment
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"{cfg.model_name}_fold_{fold}"):
        # Log hyperparameters
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
        
        # Add git commit hash (if available)
        try:
            import subprocess
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            mlflow.log_param("git_commit", git_hash)
        except:
            mlflow.log_param("git_commit", "unknown")
        
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
            logger=True  # Enable PyTorch Lightning logging
        )

        # Train and get metrics
        trainer.fit(model, data_module)
        
        # Get final metrics
        val_metrics = trainer.callback_metrics
        train_metrics = trainer.logged_metrics
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            "val_loss": float(val_metrics.get("val_loss", 0)),
            "val_acc": float(val_metrics.get("val_acc", 0)),
            "train_loss": float(train_metrics.get("train_loss_epoch", 0)),
            "train_acc": float(train_metrics.get("train_acc_epoch", 0))
        })
        
        # Create and log plots
        plot_path = f"plots/metrics_fold_{fold}_{cfg.model_name}.png"
        create_metrics_plot(trainer, plot_path)
        mlflow.log_artifact(plot_path)
        
        return checkpoint_callback.best_model_path


def create_metrics_plot(trainer, plot_path):
    """Create training/validation metrics plot"""
    try:
        # Get logged metrics
        metrics = trainer.logged_metrics
        
        # Create plot
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        ax[0,0].plot([metrics.get("train_loss_step", 0)], 'b-', label='Train Loss')
        ax[0,0].set_title('Training Loss')
        ax[0,0].legend()
        
        ax[0,1].plot([metrics.get("val_loss", 0)], 'r-', label='Val Loss')
        ax[0,1].set_title('Validation Loss')
        ax[0,1].legend()
        
        # Accuracy
        ax[1,0].plot([metrics.get("train_acc_step", 0)], 'b-', label='Train Acc')
        ax[1,0].set_title('Training Accuracy')
        ax[1,0].legend()
        
        ax[1,1].plot([metrics.get("val_acc", 0)], 'r-', label='Val Acc')
        ax[1,1].set_title('Validation Accuracy')
        ax[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create plot: {e}")


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
        df = df.sample(n=cfg.debug_samples, random_state=cfg.train.seed).reset_index(
            drop=True
        )
        print(f"⚠️ DEBUG MODE: Using {len(df)} samples")

    # K-Fold Cross Validation
    skf = StratifiedKFold(
        n_splits=cfg.train.n_splits, shuffle=True, random_state=cfg.train.seed
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"]), 1):
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

        # Convert to ONNX using correct path
        model_dir = Path(cfg.data.output_dir) / "models" / model_name
        checkpoint_path = get_latest_checkpoint(str(model_dir))
        onnx_path = model_dir / f"{model_name}.onnx"
        convert_to_onnx(str(checkpoint_path), str(onnx_path), model_cfg)

    ensemble_predict(cfg)
