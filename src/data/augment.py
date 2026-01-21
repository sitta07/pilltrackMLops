# src/data/augment.py (Configurable Version)
import os
import argparse
import yaml
import cv2
import albumentations as A
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Augment")

def get_augmentor(cfg_type, params):
    """
    สร้าง Pipeline ตาม Config ที่ส่งมา (ไม่มี Hardcode ตัวเลขแล้ว)
    """
    # ดึงค่าจาก yaml ถ้าไม่มีให้ใช้ค่า default
    p = params.get('aug_params', {})
    
    if cfg_type == 'pill':
        return A.Compose([
            A.Rotate(
                limit=p.get('rotate_limit', 180), 
                p=p.get('p_rotate', 0.8)
            ),
            A.VerticalFlip(p=p.get('p_vertical_flip', 0.5)),
            A.HorizontalFlip(p=p.get('p_horizontal_flip', 0.5)),
            A.RandomBrightnessContrast(p=p.get('p_brightness', 0.2)),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Blur(blur_limit=p.get('blur_limit', 3), p=0.2),
        ])
    
    elif cfg_type == 'box':
        return A.Compose([
            A.Rotate(
                limit=p.get('rotate_limit', 15), 
                p=p.get('p_rotate', 0.5)
            ),
            A.HorizontalFlip(p=p.get('p_horizontal_flip', 0.5)),
            # VerticalFlip จะทำงานก็ต่อเมื่อใน yaml ตั้งค่า p > 0 (ซึ่งของ box เราตั้ง 0.0)
            A.VerticalFlip(p=p.get('p_vertical_flip', 0.0)), 
            
            A.RandomBrightnessContrast(p=p.get('p_brightness', 0.2)),
            
            A.ShiftScaleRotate(
                shift_limit=p.get('shift_limit', 0.05),
                scale_limit=p.get('scale_limit', 0.1),
                rotate_limit=0, # Rotation แยกไปทำข้างบนแล้ว
                p=0.5
            ),
            A.Perspective(
                scale=(0.01, p.get('perspective_scale', 0.1)), 
                p=0.3
            )
        ])
    
    else:
        raise ValueError(f"Unknown type: {cfg_type}")

def process_single_image(args):
    img_path, output_folder, augmentor, num_aug = args # รับ augmentor ตัวเต็มมาเลย
    try:
        image = cv2.imread(img_path)
        if image is None: return 0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        
        # Save Original
        cv2.imwrite(os.path.join(output_folder, filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Generate Augmented
        count = 0
        for i in range(num_aug):
            # 🔥 ใช้ augmentor ที่ config มาแล้ว
            augmented = augmentor(image=image)['image']
            save_name = f"{name}_aug_{i}{ext}"
            cv2.imwrite(os.path.join(output_folder, save_name), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
            count += 1
            
        return count
    except Exception as e:
        return 0

def augment_dataset(input_dir, output_dir, config, aug_type):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tasks = []
    # ส่ง config ทั้งก้อนไปสร้าง Augmentor
    augmentor = get_augmentor(aug_type, config)
    
    target_count = config.get('target_per_class', 200)

    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for cls in classes:
        cls_input = os.path.join(input_dir, cls)
        cls_output = os.path.join(output_dir, cls)
        os.makedirs(cls_output, exist_ok=True)
        
        files = [f for f in os.listdir(cls_input) if f.lower().endswith(('.jpg', '.png'))]
        if not files: continue

        current_count = len(files)
        if current_count >= target_count:
            aug_per_img = 0 
        else:
            needed = target_count - current_count
            aug_per_img = max(1, int(needed / current_count)) + 1

        for f in files:
            # ส่ง augmentor ตัวเดิมไปให้ทุก process (Albumentations thread-safe ระดับนึง แต่ถ้า process แยกต้อง pickle ได้)
            tasks.append((
                os.path.join(cls_input, f),
                cls_output,
                augmentor, 
                aug_per_img
            ))

    logger.info(f"🚀 Augmenting [{aug_type.upper()}] - {len(tasks)} images...")
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_image, tasks), total=len(tasks)))
    
    logger.info(f"✅ Augmentation Complete. Total generated: {sum(results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--type", required=True, choices=['pill', 'box'])
    args = parser.parse_args()

    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)
        
    augment_cfg = params.get('augment', {}).get(args.type, {})
    
    augment_dataset(args.input, args.output, augment_cfg, args.type)