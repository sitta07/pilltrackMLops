import os
import sys
import yaml
import json
import argparse
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import numpy as np

# Setup Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.architecture import PillModel, FocalLoss
from src.data.dataset import get_transforms

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("FineTuner")

def load_params(param_path):
    with open(param_path, 'r') as f: return yaml.safe_load(f)

def get_updated_mapping(data_dir, old_mapping_path):
    """
    🔥 หัวใจสำคัญ: อ่าน Map เก่า + สแกน Folder ใหม่ -> สร้าง Map ใหม่แบบต่อท้าย (Append Only)
    """
    # 1. โหลด Map เก่า
    if os.path.exists(old_mapping_path):
        with open(old_mapping_path, 'r') as f:
            old_map_str = json.load(f) 
            old_map = {int(k): v for k, v in old_map_str.items()}
            
        name_to_idx = {v: k for k, v in old_map.items()}
        next_idx = max(old_map.keys()) + 1
        logger.info(f"📜 Found old mapping with {len(old_map)} classes.")
    else:
        logger.warning("⚠️ No old mapping found! Starting from scratch.")
        old_map = {}
        name_to_idx = {}
        next_idx = 0

    # 2. สแกนหา Class ใหม่
    all_folders = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    new_classes = []
    
    for cls_name in all_folders:
        if cls_name not in name_to_idx:
            # เจอของใหม่! ให้ ID ต่อท้ายไปเลย
            name_to_idx[cls_name] = next_idx
            old_map[next_idx] = cls_name
            new_classes.append(cls_name)
            next_idx += 1
            
    if new_classes:
        logger.info(f"🆕 Detected {len(new_classes)} NEW classes: {new_classes}")
    else:
        logger.info("✅ No new classes detected.")

    return old_map, name_to_idx, new_classes

def model_surgery(old_model_path, total_classes, device, cfg):
    """ 🏥 ผ่าตัดโมเดล: ขยายขนาด Head ให้รองรับ Class ใหม่ """
    checkpoint = torch.load(old_model_path, map_location=device)
    old_state = checkpoint # บางทีอาจเป็น dict หรือ state_dict เลย
    if "model_state_dict" in old_state:
        old_state = old_state["model_state_dict"]
    
    # เช็คขนาด Head เดิม
    if 'head.weight' in old_state:
        old_classes_count = old_state['head.weight'].shape[0]
    else:
        old_classes_count = total_classes 

    logger.info(f"🏥 Surgery: Expanding model from {old_classes_count} -> {total_classes} classes")

    # สร้างร่างใหม่
    new_model = PillModel(
        num_classes=total_classes,
        model_name=cfg['model_name'],
        embed_dim=cfg['embed_dim'],
        dropout=cfg['dropout']
    ).to(device)

    # ถ่ายโอนวิญญาณ (Weights)
    new_state = new_model.state_dict()
    for k, v in old_state.items():
        if k in new_state:
            if 'head.weight' not in k:
                # Copy Backbone/FC ปกติ
                if v.shape == new_state[k].shape:
                    new_state[k] = v
            else:
                # Copy ArcFace Head เฉพาะส่วนเดิม
                n_old = v.shape[0]
                if n_old <= total_classes:
                    new_state[k][:n_old] = v

    new_model.load_state_dict(new_state)
    return new_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--old_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--type", required=True, choices=['pill', 'box'])
    args = parser.parse_args()

    params = load_params("params.yaml")
    cfg = params['train'][args.type]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Manage Mapping
    old_map_path = os.path.join(args.old_model_dir, "class_mapping.json")
    idx_to_class, name_to_idx, new_classes = get_updated_mapping(args.data_dir, old_map_path)
    
    if not new_classes and not os.path.exists(old_map_path):
         logger.error("❌ No new classes and no old mapping. Cannot proceed.")
         return

    # Save NEW Mapping ทันที
    with open(os.path.join(args.output_dir, "class_mapping.json"), "w") as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f, indent=4)

    # 2. Model Surgery
    old_weights = os.path.join(args.old_model_dir, "best_model.pth")
    model = model_surgery(old_weights, len(idx_to_class), device, cfg)

    # 3. Freeze Backbone ❄️
    for p in model.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True # Unlock Head
    logger.info("❄️ Backbone Frozen. Training Head Only.")

    # 4. Data Loading (Force ID Match)
    transform = get_transforms(cfg['img_size'], is_train=True)
    full_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    
    # ⚠️ Hack: Override Targets ให้ตรงกับ ID ของเรา
    new_targets = []
    for path, _ in full_dataset.samples:
        # ดึงชื่อ Folder จาก Path
        class_name = os.path.basename(os.path.dirname(path))
        correct_idx = name_to_idx[class_name]
        new_targets.append(correct_idx)
    
    full_dataset.targets = new_targets
    full_dataset.samples = [(s[0], new_targets[i]) for i, s in enumerate(full_dataset.samples)]

    # Filter: New Classes + 10% Old (Replay Buffer)
    indices = []
    for i, label in enumerate(full_dataset.targets):
        class_name = idx_to_class[label]
        if class_name in new_classes:
            indices.append(i) # เอาของใหม่หมด
        elif np.random.rand() < 0.1: 
            indices.append(i) # สุ่มของเก่ามา 10%

    if not indices:
        logger.warning("⚠️ No data selected for fine-tuning (maybe no new classes?). Using all data.")
        indices = range(len(full_dataset))

    subset = Subset(full_dataset, indices)
    loader = DataLoader(subset, batch_size=cfg['batch_size'], shuffle=True)
    logger.info(f"🎯 Fine-tuning on {len(subset)} samples.")

    # 5. Fast Training Loop (5 Epochs)
    optimizer = optim.AdamW(model.head.parameters(), lr=cfg['lr'])
    criterion = FocalLoss()

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs, labels)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Ep {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

    # 6. Save
    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
    logger.info(f"✅ Fine-tuning Done! Saved to {args.output_dir}")

if __name__ == "__main__":
    main()