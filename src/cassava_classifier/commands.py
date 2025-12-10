# src/cassava_classifier/commands.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import hydra
from omegaconf import DictConfig
import torch
from cassava_classifier.pipelines.train import train_model  # ← Absolute import

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m cassava_classifier.commands model_name=model1")
        return
    if cfg.get("run_full", False):
        train_all_models_and_ensemble(cfg)
    else:
        train_model(cfg)

if __name__ == "__main__":
    main()