# src/data/split_data.py
import os
import shutil
import argparse
import yaml
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ฟังก์ชันโหลด Config
def load_params(param_path):
    with open(param_path, 'r') as f:
        return yaml.safe_load(f)

def main(input_path, output_path):
    # 1. Load Config from params.yaml
    params = load_params("params.yaml")
    split_conf = params["split"]
    
    seed = split_conf["seed"]
    train_ratio = split_conf["train_size"]
    val_ratio = split_conf["val_size"]
    test_ratio = split_conf["test_size"]

    # เช็คว่าบวกกันได้ 1.0 หรือเปล่า
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"❌ Ratios must sum to 1.0 (Got {train_ratio + val_ratio + test_ratio})")

    input_path = Path(input_path)
    output_path = Path(output_path)

    # 2. Reset Output Directory (Idempotency Key 🔑)
    # ต้องลบของเก่าทิ้งก่อนสร้างใหม่ เพื่อให้ผลลัพธ์เหมือนเดิมทุกครั้งที่รัน
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # สร้าง folder รอ: train, val, test
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    print(f"🚀 STARTING STRATIFIED SPLIT")
    print(f"   🎯 Ratios: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")

    # 3. Process each class
    classes = [d.name for d in input_path.iterdir() if d.is_dir()]
    
    for class_name in tqdm(classes, desc="Splitting Classes"):
        src_dir = input_path / class_name
        images = list(src_dir.glob("*.png"))
        
        if not images:
            continue

        # --- Logic การแบ่งแบบ Stratified (แบ่ง 2 รอบ) ---
        # รอบ 1: แยก Train ออกจาก (Val + Test)
        train_imgs, temp_imgs = train_test_split(
            images, 
            train_size=train_ratio, 
            random_state=seed,
            shuffle=True
        )

        # รอบ 2: แยก Val กับ Test ออกจากก้อนที่เหลือ
        # ต้องคำนวณ Ratio ใหม่เทียบกับก้อนที่เหลือ
        val_relative = val_ratio / (val_ratio + test_ratio)
        
        val_imgs, test_imgs = train_test_split(
            temp_imgs, 
            train_size=val_relative, 
            random_state=seed,
            shuffle=True
        )

        # 4. Copy Files ไปลงหลุม
        splits_map = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }

        for split_name, img_list in splits_map.items():
            dest_dir = output_path / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in img_list:
                shutil.copy2(img_path, dest_dir / img_path.name)

    print(f"✅ Splitting Done! Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    main(args.input, args.output)