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

# 🔥 Helper Function: คำนวณ Accuracy
def calculate_accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.max(1)
        correct = pred.eq(target).sum().item()
        return correct * 100.0 / batch_size

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
    
    cfg = params['train'][args.type]
    
    set_seed(cfg.get('seed', 42))

    # 🚀 GPU Setup (RTX 5060 Ti Ready)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍏 Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")
    
    logger.info(f"🚀 Start Training [{args.type.upper()}] on {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Data Loaders
    train_tfm = get_transforms(cfg['img_size'], is_train=True)
    val_tfm = get_transforms(cfg['img_size'], is_train=False)

    train_ds = datasets.ImageFolder(args.train_dir, transform=train_tfm)
    val_ds = datasets.ImageFolder(args.val_dir, transform=val_tfm)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg['batch_size'], 
        shuffle=True, 
        num_workers=cfg.get('num_workers', 4), # แนะนำให้เพิ่ม worker ถ้า CPU ไหว
        drop_last=True,
        pin_memory=True # ช่วยให้ส่งข้อมูลเข้า GPU เร็วขึ้น
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg['batch_size'], 
        shuffle=False, 
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True
    )

    # 3. Save Class Mapping
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 🔥 Early Stopping Params
    best_val_loss = float('inf')
    patience = cfg.get('patience', 3)  # อ่านจาก config ถ้าไม่มีใช้ 7
    early_stop_counter = 0
    best_acc = 0.0

    # 6. Training Loop
    for epoch in range(cfg['epochs']):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        
        # 🟢 Progress Bar ที่โชว์ Loss และ Acc สดๆ
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{cfg['epochs']}")
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, labels) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # คำนวณ Acc
            acc = calculate_accuracy(outputs, labels)
            
            # เก็บค่าสะสม
            running_loss += loss.item()
            running_acc += acc
            
            # 🔥 อัปเดตตัวเลขในหลอดโหลด
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc:.2f}%"})

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)

        # --- VAL ---
        model.eval()
        val_loss = 0.0
        val_acc_accum = 0.0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs, labels)
                loss = criterion(outputs, labels)
                
                acc = calculate_accuracy(outputs, labels)
                
                val_loss += loss.item()
                val_acc_accum += acc
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc_accum / len(val_loader)
        
        # Scheduler Step
        scheduler.step(avg_val_loss)

        logger.info(f"✅ Ep {epoch+1} Result | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.2f}%")

        # --- SAVE BEST & EARLY STOPPING ---
        
        # 1. Save Best Model (เก็บตัวที่ Acc ดีที่สุดไว้ใช้งาน)
        if avg_val_acc >= best_acc:
            best_acc = avg_val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            logger.info(f"    💾 Model Saved! (New Best Acc: {best_acc:.2f}%)")
        
        # 2. Early Stopping Check (เช็คจาก Loss เพื่อกัน Overfitting)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0  # Reset Counter
        else:
            early_stop_counter += 1
            logger.warning(f"    ⏳ EarlyStopping Counter: {early_stop_counter}/{patience}")
            
            if early_stop_counter >= patience:
                logger.info("🛑 Early stopping triggered! Training finished.")
                break

    logger.info("🏆 Training Process Finished.")

if __name__ == "__main__":
    main()