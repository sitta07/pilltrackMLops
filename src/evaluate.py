# src/evaluate.py
import os
import sys
import argparse
import yaml
import json
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

# Setup Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.architecture import PillModel
from src.data.dataset import get_transforms

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Evaluator")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--type", required=True, choices=['pill', 'box'])
    parser.add_argument("--threshold", type=float, default=80.0, help="Minimum accuracy required")
    args = parser.parse_args()

    # Load Config
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    cfg = params['train'][args.type]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load Mapping
    map_path = os.path.join(args.model_dir, "class_mapping.json")
    with open(map_path, 'r') as f:
        mapping = json.load(f)
    
    # Load Model
    model = PillModel(
        num_classes=len(mapping),
        model_name=cfg['model_name'],
        embed_dim=cfg['embed_dim'],
        dropout=cfg['dropout']
    ).to(device)

    weight_path = os.path.join(args.model_dir, "best_model.pth")
    checkpoint = torch.load(weight_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # Prepare Data
    transform = get_transforms(cfg['img_size'], is_train=False)
    test_ds = datasets.ImageFolder(args.test_dir, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False)

    # Evaluate
    correct = 0
    total = 0
    
    logger.info(f"🧪 Starting Evaluation on {len(test_ds)} test images...")
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs, labels) # dummy label for forward
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    logger.info(f"📊 Test Accuracy: {acc:.2f}%")

    # 🔥 The Gatekeeper Logic
    # บันทึกผลลงไฟล์ (เผื่อใช้ดูย้อนหลัง)
    metrics = {"accuracy": acc, "passed": acc >= args.threshold}
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    if acc < args.threshold:
        logger.error(f"❌ FAILED: Accuracy {acc:.2f}% is lower than threshold {args.threshold}%")
        sys.exit(1) # ส่งสัญญาณให้ DVC/CI หยุดทำงานทันที!
    else:
        logger.info(f"✅ PASSED: Model is ready for production/S3.")

if __name__ == "__main__":
    main()