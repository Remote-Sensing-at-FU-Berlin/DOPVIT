# ViT Tree Species Classification from DOP

This project uses a Vision Transformer to classify digital orthophotos of forest canopy in 10m by 10m patches.
To train you need labeled data, RGB+NIR with at least 20cm resolution is recommended.

This project contains:
- CNN and ViT models for species classification
- Full training loop with early stopping
- Explainability tools (Grad-CAM, occlusion)
- Modular architecture for experiments
- Reproducible scripts and documentation

## Structure
- `models/` – CNN, ViT wrapper, model factory
- `training/` – train, eval, early stopping
- `explainability/` – gradcam + occlusion
- `scripts/` – runnable training and explainability scripts
- `docs/` – flowchart + documentation

## Requirements
PyTorch, torchvision, numpy, matplotlib

## Train ViT
```bash
python scripts/train_vit.py
