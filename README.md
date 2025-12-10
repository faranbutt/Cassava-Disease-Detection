# Cassava Leaf Disease Classification

This repository implements a high-performance solution for the **Cassava Leaf Disease Classification** challenge, inspired by 3rd-place Kaggle competition approaches. The system combines **Vision Transformers with advanced techniques** including image division, attention weighting, multi-dropout, and label smoothing to achieve robust classification of cassava leaf diseases.

## 🌿 Project Overview

Cassava is a vital crop for over 800 million people worldwide, but it is threatened by several devastating diseases. Early and accurate diagnosis is crucial for effective disease management. This project provides:

- **Ensemble of 3 specialized Vision Transformer models**
- **Image division technique** for high-resolution analysis
- **Attention-based feature weighting** across image quadrants
- **Multi-dropout regularization** for improved generalization
- **Label smoothing** to handle noisy labels

The pipeline supports full MLOps workflow: data management, training, validation, inference, and production deployment.

## 🛠️ Technical Setup

### Prerequisites
- Python 3.9+
- Git
- CUDA-enabled GPU (recommended)

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/cassava-leaf-disease-classifier.git
   cd cassava-leaf-disease-classifier

2. Install dependencies using uv
```
# Install core dependencies
uv add torch torchvision timm pytorch-lightning pandas scikit-learn opencv-python-headless albumentations hydra-core omegaconf

# Install development dependencies
uv add --group dev black isort flake8 pre-commit

# Install the package in editable mode
uv pip install -e .

# Install pre-commit hooks
uv run pre-commit install
```