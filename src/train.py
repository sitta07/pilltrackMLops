import os
import sys
import yaml
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import logging
import numpy as np
import random

# Setup Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.architecture import PillModel, FocalLoss
from src.data.dataset import get_transforms

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Trainer")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--type", required=True, choices=['pill', 'box'])
    args = parser.parse_args()

    # 1. Load Config
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    # ดึง Config ตาม Type (pill หรือ box)
    cfg = params['train'][args.type]
    
    set_seed(cfg.get('seed', 42))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"🚀 Start Training [{args.type.upper()}] on {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Data Loaders
    # ใช้ get_transforms จาก dataset.py (ถ้ามี) หรือใช้ standard transform
    train_tfm = get_transforms(cfg['img_size'], is_train=True)
    val_tfm = get_transforms(cfg['img_size'], is_train=False)

    train_ds = datasets.ImageFolder(args.train_dir, transform=train_tfm)
    val_ds = datasets.ImageFolder(args.val_dir, transform=val_tfm)
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg['batch_size'], 
        shuffle=True, 
        num_workers=cfg.get('num_workers', 0),
        drop_last=True  
    )

    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg.get('num_workers', 0))

    # 3. Save Class Mapping (Auto-Generated) 📝
    class_map = {i: c for i, c in enumerate(train_ds.classes)}
    with open(os.path.join(args.output_dir, "class_mapping.json"), "w") as f:
        json.dump(class_map, f, indent=4)
    logger.info(f"📊 Found {len(class_map)} classes. Mapping saved.")

    # 4. Initialize Model
    model = PillModel(
        num_classes=len(class_map),
        model_name=cfg['model_name'],
        embed_dim=cfg['embed_dim'],
        dropout=cfg['dropout']
    ).to(device)

    # 5. Loss & Optimizer
    criterion = FocalLoss(gamma=cfg['focal']['gamma'], alpha=cfg['focal']['alpha'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    
    # Scheduler (Optional: ลด LR เมื่อ Loss นิ่ง)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 6. Training Loop
    best_acc = 0.0
    
    for epoch in range(cfg['epochs']):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{cfg['epochs']}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, labels) # ArcFace Forward ต้องการ labels
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate Acc (คร่าวๆ จาก Logits)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix(loss=loss.item())

        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)

        # --- VAL ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs, labels)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, pred = outputs.max(1)
                val_correct += pred.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Scheduler Step
        scheduler.step(avg_val_loss)

        logger.info(f"Ep {epoch+1} | Tr_Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | Val_Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")

        # Save Best
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            logger.info(f"    🌟 New Best Saved! (Acc: {val_acc:.2f}%)")

    logger.info("🏆 Training Completed Successfully!")

if __name__ == "__main__":
    main()