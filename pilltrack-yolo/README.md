# PillTrack YOLOv8 Training Pipeline
Structured YOLOv8 training and deployment pipeline for PillTrack object detection. Project structure: dataset/ (versioned datasets), configs/train_config.yaml (training config), scripts/ (train.py, predict.py, export.py), models/ (production models), runs/ (auto-generated results). To train: update `dataset_folder` in configs/train_config.yaml then run `python scripts/train.py`. To predict: `python scripts/predict.py`. To export ONNX: `python scripts/export.py`. Each dataset version creates a separate experiment folder under runs/detect/.

```bash
python scripts/train.py
python scripts/predict.py
python scripts/export.py

```