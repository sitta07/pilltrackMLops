# PillTrack MLOps - Automated Pill & Box Classification Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)]
[![ArcFace](https://img.shields.io/badge/ArcFace-Face%20Recognition-purple.svg)]
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-green.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-orange.svg)](https://mlflow.org/)
[![AWS](https://img.shields.io/badge/AWS-Cloud-orange.svg)](https://aws.amazon.com/)



**Production-grade ML system for pharmaceutical pill and box recognition using deep learning, DVC orchestration, and incremental learning strategies.**

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [CI/CD Workflows](#cicd-workflows)
- [Training Commands](#training-commands)
- [Google Colab Training](#google-colab-training)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Inference & Deployment](#inference--deployment)
- [Troubleshooting](#troubleshooting)

---

## Overview

**PillTrack** is an MLOps system for pharmaceutical image recognition with:

- Separate models for pills and boxes
- Automatic incremental learning (add new drugs without full retraining)
- DVC-based reproducible pipelines
- MLflow experiment tracking
- GitHub Actions CI/CD automation
- Google Colab training support

---

## Features

| Feature | Details |
|---------|---------|
| **ConvNeXt + ArcFace** | Modern architecture with metric learning |
| **Incremental Learning** | Add new drug classes in minutes |
| **Full Retraining** | Retrain from scratch when needed |
| **Data Pipeline** | Preprocessing → Split → Augmentation → Training |
| **MLflow Tracking** | Experiment management & hyperparameter tuning |
| **Cloud Ready** | Google Colab with automatic GPU detection |
| **Version Control** | DVC for data & model versioning |

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/sitta07/pilltrackMLops.git
cd pilltrackMLops
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run full box training pipeline
dvc repro train_box

# Or just augment and finetune for new class
dvc repro augment_box
python src/finetune.py --type box

# Evaluate
dvc repro evaluate_box
```

---

## Project Structure

```
pilltrackMLops/
├── data/
│   ├── raw_box/                    # Original box images
│   ├── raw_pill/                   # Original pill images
│   ├── processed_box/              # After preprocessing
│   ├── dataset_split_box/          # train/val/test split
│   ├── dataset_train_final_box/    # Augmented data
│   └── dataset_train_final_pill/
├── src/
│   ├── data/
│   │   ├── augment.py              # Data augmentation
│   │   ├── remove_background.py    # Preprocessing
│   │   └── split_data.py
│   ├── models/
│   │   └── architecture.py         # FocalLoss, ArcFace, PillModel
│   ├── train.py                    # Training entry point
│   ├── finetune.py                 # Incremental learning
│   ├── evaluate.py                 # Model evaluation
│   └── inference.py                # Inference utilities
├── experiments/
│   └── arcface_lite_v1/
│       ├── box/
│       │   ├── best_model.pth      # Trained model
│       │   └── class_mapping.json  # Class indices
│       └── pill/
├── metrics/
│   ├── box_metrics.json
│   └── pill_metrics.json
├── train_box_colab.ipynb           # Google Colab notebook
├── dvc.yaml                        # DVC pipeline
├── params.yaml                     # Configuration
├── requirements.txt                # Dependencies
└── README.md
```

---

## CI/CD Workflows

### GitHub Actions Automation

The system supports two training modes via Git tags:

#### 1. Full Retraining (Major Update)

**Scenario**: Complete model retraining from scratch
- When changing model architecture
- When data structure changes significantly
- When incremental learning cannot handle updates

**What happens**: Runs `dvc repro train_box/pill` completely

**For Pills**:
```bash
git tag major-pill-production-v1.0
git push origin major-pill-production-v1.0
```

**For Boxes**:
```bash
git tag major-box-production-v1.1
git push origin major-box-production-v1.1
```

#### 2. Incremental Learning (Add New Class) - Most Common

**Scenario**: Adding new drug class (e.g., Panadol) without full retraining
- Fastest way to add new drugs
- Preserves existing model knowledge
- Takes 5-10 minutes instead of hours

**What happens**:
- Skips base model training (uses `dvc commit -f`)
- Loads existing model
- Finetunes only the classification head
- Auto-updates `best_model.pth`

**For Pills**:
```bash
git tag pill-production-v1.5
git push origin pill-production-v1.5
```

**For Boxes**:
```bash
git tag box-production-v1.1
git push origin box-production-v1.1
```

---

## Training Commands


### Manual Training

#### Full Retraining Pipeline


**Box Model**:
```bash
dvc repro train_box
```

**Pill Model**:
```bash
dvc repro train_pill
```

#### Step-by-Step Execution

**Box**:
```bash
dvc repro preprocess_box      # Remove background
dvc repro split_box           # Split train/val/test
dvc repro augment_box         # Generate augmented data
dvc repro train_box           # Train model
dvc repro evaluate_box        # Evaluate on test set
```

**Pill**:
```bash
dvc repro preprocess_pill
dvc repro split_pill
dvc repro augment_pill
dvc repro train_pill
dvc repro evaluate_pill
```

#### Incremental Learning (Add New Class)

**Box**:
```bash
dvc repro augment_box         # Augment new data only
dvc commit train_box -f       # Skip base training
python src/finetune.py --type box    # Finetune head
dvc repro evaluate_box        # Evaluate
```

**Pill**:
```bash
dvc repro augment_pill
dvc commit train_pill -f
python src/finetune.py --type pill
dvc repro evaluate_pill
```

---

## Google Colab Training

**Train on GPU without local setup!**

1. Download: `train_box_colab.ipynb`
2. Upload to [Google Colab](https://colab.research.google.com)
3. Run cells sequentially:
   - Setup (mounts Drive, installs packages)
   - Config (customize hyperparameters if needed)
   - Training (runs full pipeline)
   - Results (view MLflow metrics)

**Features**:
- Automatic GPU/MPS detection
- MLflow tracking with artifact logging
- Real-time progress with tqdm
- Auto-download to Google Drive

---

## Model Architecture

```
Input Image (3, 224, 224)
    ↓
ConvNeXt-Small (Pretrained)
    ↓
BatchNorm → Dropout → Linear (→ 512-dim)
    ↓
BatchNorm (Embedding Layer)
    ↓
ArcFace Head (Margin-based Loss)
    ↓
Classification Logits → Softmax
```

**Loss Functions**:
- **ArcFace**: Margin-based loss for metric learning
- **Focal Loss**: Handles class imbalance with dynamic weighting

---

## Configuration

Edit `params.yaml` to customize:

```yaml
split:
  val_ratio: 0.2
  test_ratio: 0.1
  seed: 42

augment:
  box:
    target_per_class: 350
    aug_params:
      rotate_limit: 15              # Slight rotation only
      p_vertical_flip: 0.0          # No vertical flip
      p_horizontal_flip: 0.5        # Can flip left-right
      perspective_scale: 0.1        # 3D perspective

  pill:
    target_per_class: 50
    aug_params:
      rotate_limit: 180             # Full rotation
      p_vertical_flip: 0.5          # Can flip any direction
      p_horizontal_flip: 0.5
      p_brightness: 0.2

train:
  box:
    model_name: convnext_small
    img_size: 224
    batch_size: 16
    epochs: 10
    lr: 0.0001
    dropout: 0.5
    embed_dim: 512
    weight_decay: 1e-4
    focal:
      gamma: 2.0
      alpha: 0.25
```

---

## Inference & Deployment

### Test Finetuned Model

**Box**:
```bash
python src/inference.py --image data/raw_box/drug_name/test.jpg \
  --model_dir experiments/arcface_finetuned/box --arch convnext_small
```

**Pill**:
```bash
python src/inference.py --image data/raw_pill/drug_name/test.jpg \
  --model_dir experiments/arcface_finetuned/pill --arch convnext_small
```

### Test Base Model

**Box**:
```bash
python src/inference.py --image data/raw_box/drug_name/test.jpg \
  --model_dir experiments/arcface_lite_v1/box --arch convnext_small
```

**Pill**:
```bash
python src/inference.py --image data/raw_pill/drug_name/test.jpg \
  --model_dir experiments/arcface_lite_v1/pill --arch convnext_small
```

### Cloud Data Sync

```bash
# Push data and models to DVC remote (S3, etc.)
dvc push

# Pull from remote
dvc pull
```

### Production Deployment

**Required**: Verify `.env`, `best_model.pth`, and `class_mapping.json` are present

```bash
python src/deploy.py --version v1.0.0 \
  --path experiments/arcface_finetuned \
  --note "Initial release: supports 10 drug types"
```

---

## MLflow Tracking

```bash
# View experiments
mlflow ui

# Log custom metrics
mlflow log_metric("accuracy", 0.95)
mlflow log_params({"lr": 0.0001})
mlflow log_artifact("model.pth")
```

Check results in `metrics/box_metrics.json` and `metrics/pill_metrics.json`

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch_size` in `params.yaml` |
| Model not found | Verify path to `best_model.pth` exists |
| DVC pipeline fails | Run `dvc status` to check dependencies |
| Colab GPU not detected | Restart kernel or select GPU runtime |
| Import errors | Run `pip install -r requirements.txt` |

---

## Examples

### Add New Drug in 5 Minutes

```bash
# 1. Add images to data/raw_box/new_drug/
# 2. Augment
dvc repro augment_box

# 3. Skip training, use existing model
dvc commit train_box -f

# 4. Finetune
python src/finetune.py --type box --epochs 3

# 5. Done!
```

### Hyperparameter Tuning

```bash
# Edit params.yaml with different values:
# - lr: [0.0001, 0.0002, 0.0005]
# - batch_size: [16, 32]
# - dropout: [0.3, 0.5]

# View all experiments
mlflow ui
```

---

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing`
3. Commit: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing`
5. Open Pull Request

---

## Author

**Sitta Boonkaew**  
AI Engineer Intern @ AI SmartTech

---

## License

MIT License

Copyright (c) 2026 Sitta Boonkaew 

**IMPORTANT**: This codebase is the intellectual property of Sitta Boonkaew. Any modifications, redistributions, or derivative works must:
- Maintain the original copyright notice and license
- Clearly indicate any modifications made
- Include attribution to the original author
- Preserve this license notice in all copies or substantial portions of the software

**You are NOT permitted to remove or obscure the original author's name or contribution from the code or documentation.**

For questions or licensing inquiries, please contact: sittasahathum@gmail.com (66 65 273 2611)

---

**Version**: 1.0.0 | **Status**: Production Ready 