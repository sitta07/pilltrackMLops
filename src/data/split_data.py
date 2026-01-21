# src/data/split_data.py (Updated for 3-way split)
import os
import argparse
import yaml
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main(input_dir, output_dir):
    # Load Params
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    split_conf = params.get("split", {})
    val_ratio = split_conf.get("val_ratio", 0.2)
    test_ratio = split_conf.get("test_ratio", 0.1) # 🔥 อ่านค่า test ratio
    seed = split_conf.get("seed", 42)

    # Cleanup
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    
    # สร้าง 3 โฟลเดอร์
    subsets = ['train', 'val', 'test']
    for sub in subsets:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for cls in tqdm(classes, desc="Splitting Data"):
        cls_path = os.path.join(input_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png'))]
        
        if not images: continue

        # 🔥 Logic: Split 2 รอบ
        # รอบ 1: แยก Test ออกมาก่อน (เช่น 10%)
        # รอบ 2: เอาที่เหลือมาแยก Train/Val
        
        train_val_imgs, test_imgs = train_test_split(
            images, test_size=test_ratio, random_state=seed, shuffle=True
        )
        
        # คำนวณ Ratio ใหม่สำหรับ Val เทียบกับก้อนที่เหลือ
        # เช่น ต้องการ Val 20% จากทั้งหมด -> แต่ตอนนี้เหลือแค่ 90%
        # adjusted_val_ratio = 0.2 / (1 - 0.1) = 0.222...
        remaining_ratio = 1.0 - test_ratio
        if remaining_ratio == 0: remaining_ratio = 1.0 # กัน Error
        
        adjusted_val_ratio = val_ratio / remaining_ratio
        
        train_imgs, val_imgs = train_test_split(
            train_val_imgs, test_size=adjusted_val_ratio, random_state=seed, shuffle=True
        )

        # Copy Files
        for sub, imgs in zip(subsets, [train_imgs, val_imgs, test_imgs]):
            dst_dir = os.path.join(output_dir, sub, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for img in imgs:
                shutil.copy2(os.path.join(cls_path, img), os.path.join(dst_dir, img))

    print(f"✅ Split Done (Train/Val/Test)! Output: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)