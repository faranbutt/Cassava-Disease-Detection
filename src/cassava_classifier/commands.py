# working/Cassava-Disease-Detection/src/cassava_classifier/commands.py
import os
import sys

import hydra
import torch
from omegaconf import DictConfig

from cassava_classifier.pipelines.train import (
    train_all_models_and_ensemble,
    train_model,
)

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Absolute imports

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.get("run_full", False):
        train_all_models_and_ensemble(cfg)
    elif cfg.get("predict", False):
        from cassava_classifier.pipelines.infer import (
            load_checkpoint,
            predict_single_image,
            preprocess_image,
        )

        # Load model and predict
        model_config = cfg.model
        model = load_checkpoint(cfg.predict.model_path, model_config)
        image_tensor = preprocess_image(cfg.predict.image_path, model_config.img_size)
        pred_class, probs = predict_single_image(model, image_tensor)
        print(f"Predicted class: {pred_class}")
        print(f"Probabilities: {probs}")
    else:
        train_model(cfg)


if __name__ == "__main__":
    main()
