# PCOS Detection from Ovarian Ultrasound Images

Bachelor's thesis project investigating deep learning approaches for automatic classification of ovarian ultrasound images into:

- Dominant Follicle
- Normal
- Polycystic Ovary (PCO)

The project is designed as a reproducible experiment framework for comparing convolutional neural networks and foundation models on the same dataset using a unified training and evaluation pipeline.

---

## Features

### Dataset preparation

- Dataset auditing
- Exact duplicate detection (SHA-256)
- Near-duplicate detection (perceptual hashing)
- Reproducible train/validation/test splits
- Stratified sampling
- Image preprocessing and augmentation

### Training

- Transfer learning with ResNet-50
- GPU and CPU support
- Mixed-precision training
- Automatic checkpointing
- Experiment management
- Reproducible configuration files

### Evaluation

- Accuracy
- Balanced accuracy
- Precision
- Recall
- Macro F1
- Weighted F1
- Confusion matrix
- ROC curves
- Precision–Recall curves
- Confidence analysis

### Explainability

- Grad-CAM visualizations

---

## Project structure

```text
dataset/
│
├── raw/
├── splits/
│
experiments/
│
├── <timestamp>_<experiment_name>/
│   ├── config.json
│   ├── history.json
│   ├── metrics.json
│   ├── predictions.csv
│   ├── best_checkpoint.pt
│   ├── logs.txt
│   ├── status.json
│   ├── figures/
│   └── xai/
│
reports/
│
├── comparisons/
│
src/
│
├── data/
├── evaluation/
├── experiments/
├── models/
├── training/
├── visualisation/
└── xai/
```

---

## Current baseline

Implemented:

- Frozen ResNet-50
- ImageNet pretrained weights
- Cross-entropy loss
- AdamW optimizer
- Automatic experiment tracking

Each experiment stores:

- configuration
- training history
- best checkpoint
- evaluation metrics
- predictions
- visualizations
- Grad-CAM explanations

---

## Planned models

- ResNet-50 (baseline) - done
- DINOv2
- OpenUS
- USFM

---

## Installation

```bash
git clone <repository>

cd PCOS-Detection

python -m venv .venv

source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Training

```bash
python -m src.training.run_resnet_experiment \
    --epochs 15 \
    --experiment-name resnet50_frozen_baseline
```

---

## Evaluation

Training automatically produces:

- learning curves
- confusion matrix
- confidence analysis
- ROC curves
- Precision–Recall curves
- experiment logs

Grad-CAM explanations can be generated afterwards:

```bash
python -m src.xai.gradcam_resnet \
    --experiment-dir experiments/<experiment>
```

---

## License

MIT License