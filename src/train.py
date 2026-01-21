# src/train.py (แทรกไว้บนสุดเลย)
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import json
import argparse
import logging
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score

# Import Modules ที่เราแยกไว้
from src.data.dataset import create_dataloaders
from src.models.architecture import PillModel, FocalLoss

# ============================================================
# ⚙️ SYSTEM SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Trainer")

def set_seed(seed):
    """ ล็อคค่าสุ่มเพื่อให้ผลการเทรนเหมือนเดิมทุกครั้ง (Reproducibility) """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # สำหรับ Mac M1/M2 (MPS)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def load_params(param_path):
    with open(param_path, 'r') as f:
        return yaml.safe_load(f)

# ============================================================
# 🔄 TRAINING & VALIDATION LOOPS
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    # Progress Bar แบบ Cleanๆ
    loop = tqdm(loader, desc="🔥 Train", leave=False, ncols=100)
    
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward Pass (ArcFace)
        # ส่ง Labels ไปด้วย เพื่อให้ ArcFace Margin Product ทำงาน
        outputs = model(imgs, labels) 
        
        # Calculate Loss
        loss = criterion(outputs, labels)
        
        # Backward Pass
        loss.backward()
        optimizer.step()
        
        # Metrics Tracking
        total_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        loop.set_postfix(loss=f"{loss.item():.4f}")
        
    return total_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    
    # ดึง Center Weights ของ ArcFace มาใช้คำนวณ Cosine Sim แบบเพียวๆ
    # เพื่อจำลองตอนใช้งานจริง (Inference) ที่เราจะเทียบ Cosine Distance
    class_weights = F.normalize(model.head.weight)
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # 1. Calc Loss (Training Objective) - ยังต้องใช้ Margin
            outputs_margin = model(imgs, labels)
            loss = criterion(outputs_margin, labels)
            total_loss += loss.item() * imgs.size(0)
            
            # 2. Calc Metric (Real-world Objective) - ใช้ Clean Cosine Similarity
            embeddings = model(imgs, labels=None) # ขอแค่ Embeddings
            embeddings_norm = F.normalize(embeddings)
            
            # Dot Product (Cosine Sim) กับ Class Centers
            logits_clean = F.linear(embeddings_norm, class_weights)
            
            preds = torch.argmax(logits_clean, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # คำนวณ Precision (กันเหนียวเผื่อ Class ไม่ Balance)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return total_loss / total, correct / total, precision

# ============================================================
# 🚀 MAIN ORCHESTRATOR
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train Pill/Box Classification Model")
    parser.add_argument("--train_dir", required=True, help="Path to training data")
    parser.add_argument("--val_dir", required=True, help="Path to validation data")
    parser.add_argument("--output_dir", required=True, help="Root output directory")
    parser.add_argument("--type", required=True, choices=['pill', 'box'], help="Select config type")
    args = parser.parse_args()

    # 1. Load Config & Setup
    params = load_params("params.yaml")
    
    # เช็คว่ามี Config Type นี้จริงไหม
    if args.type not in params['train']:
        raise ValueError(f"❌ Unknown train type: '{args.type}'. Please check 'train' section in params.yaml")
        
    cfg = params['train'][args.type]  # 🔥 โหลด Config แยกตาม Pill/Box ตรงนี้
    
    set_seed(cfg['seed'])
    
    # Auto-Detect Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # สร้าง Folder ปลายทาง: output_dir/type (เช่น experiments/v1/pill)
    final_output_dir = os.path.join(args.output_dir, args.type)
    os.makedirs(final_output_dir, exist_ok=True)
    
    logger.info(f"🚀 Start Training [{args.type.upper()}] on {device}")
    logger.info(f"📂 Output Dir: {final_output_dir}")
    logger.info(f"⚙️  Model: {cfg['model_name']} | Epochs: {cfg['epochs']} | BS: {cfg['batch_size']}")

    # 2. Prepare Data
    train_loader, val_loader, classes = create_dataloaders(args.train_dir, args.val_dir, cfg)
    num_classes = len(classes)
    logger.info(f"📊 Found {num_classes} classes")
    
    # Save Class Mapping (สำคัญมากตอนเอาไป Deploy!)
    class_map_path = os.path.join(final_output_dir, "class_mapping.json")
    with open(class_map_path, "w") as f:
        json.dump({i: name for i, name in enumerate(classes)}, f, indent=4)

    # 3. Initialize Model
    model = PillModel(
        num_classes=num_classes,
        model_name=cfg['model_name'],
        embed_dim=cfg['embed_dim'],
        dropout=cfg['dropout']
    ).to(device)
    
    criterion = FocalLoss(gamma=cfg['focal']['gamma'], alpha=cfg['focal']['alpha'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    # 4. Training Loop
    best_acc = 0.0
    
    for epoch in range(cfg['epochs']):
        # Train
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        v_loss, v_acc, v_prec = validate(model, val_loader, criterion, device)
        
        # Log Result
        logger.info(
            f"Ep {epoch+1:02d}/{cfg['epochs']} | "
            f"Tr_Loss: {t_loss:.4f} Acc: {t_acc:.4f} | "
            f"Val_Loss: {v_loss:.4f} Acc: {v_acc:.4f} Prec: {v_prec:.4f}"
        )

        # Save Checkpoint (Last Model)
        last_path = os.path.join(final_output_dir, "last_model.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': v_acc,
            'config': cfg
        }, last_path)

        # Save Best Model
        if v_acc >= best_acc:
            best_acc = v_acc
            best_path = os.path.join(final_output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path) # เซฟแค่ state_dict เพียวๆ จะได้โหลดง่าย
            logger.info(f"   🌟 New Best Saved! (Acc: {best_acc:.4f})")
            
    # 5. Finalize for DVC Metrics
    # DVC ชอบอ่านไฟล์ JSON เพื่อทำ Plot
    metrics_path = os.path.join(final_output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"best_val_acc": best_acc, "last_val_acc": v_acc}, f, indent=4)
        
    logger.info("🏆 Training Completed Successfully!")

if __name__ == "__main__":
    main()