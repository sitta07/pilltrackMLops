# src/data/augment.py
import os
import random
import yaml
import argparse
import logging
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
from itertools import cycle

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Augmentor")

def load_params(param_path):
    with open(param_path, 'r') as f:
        return yaml.safe_load(f)

def get_random_geometry_params(img_size, config):
    w, h = img_size
    return {
        "angle": random.uniform(-config["rotation"], config["rotation"]),
        "translate": (
            random.uniform(-config["translate"], config["translate"]) * w,
            random.uniform(-config["translate"], config["translate"]) * h,
        ),
        "scale": random.uniform(config["scale_min"], config["scale_max"]),
        "shear": random.uniform(-config["shear"], config["shear"])
    }

def apply_geometry(img, params):
    return F.affine(
        img,
        angle=params["angle"],
        translate=params["translate"],
        scale=params["scale"],
        shear=params["shear"],
        fill=0
    )

def augment_image(img_rgba, geom_config, color_config):
    try:
        # Create Color Jitter Transform dynamically
        color_aug = transforms.ColorJitter(
            brightness=color_config["brightness"],
            contrast=color_config["contrast"],
            saturation=color_config["saturation"]
        )

        img_np = np.array(img_rgba)
        rgb = Image.fromarray(img_np[:, :, :3])
        alpha = Image.fromarray(img_np[:, :, 3])

        # 1. Geometry
        geom_params = get_random_geometry_params(img_rgba.size, geom_config)
        rgb = apply_geometry(rgb, geom_params)
        alpha = apply_geometry(alpha, geom_params)

        # 2. Color (RGB Only)
        rgb = color_aug(rgb)

        # 3. Alpha Clean
        alpha_np = np.array(alpha)
        alpha_np = (alpha_np > 50).astype(np.uint8) * 255

        return Image.fromarray(
            np.dstack((np.array(rgb), alpha_np)),
            mode="RGBA"
        )
    except Exception as e:
        return img_rgba

def main(input_path, output_path, config_type):
    # Load Params
    params = load_params("params.yaml")
    
    # เลือก Config ตาม Type ที่ส่งมา (pill หรือ box)
    if config_type not in params["augment"]:
        raise ValueError(f"❌ Unknown config type: {config_type}. Check params.yaml")
        
    aug_config = params["augment"][config_type]
    target_count = aug_config["target_per_class"]
    geom_conf = aug_config["geometry"]
    color_conf = aug_config["color"]

    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Cleaning output dir
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"🚀 STARTING AUGMENTATION [{config_type.upper()}]")
    print(f"🎯 Target per class: {target_count}")

    classes = [d for d in input_path.iterdir() if d.is_dir()]
    stats_list = []

    for class_dir in tqdm(classes, desc="Augmenting"):
        class_name = class_dir.name
        save_dir = output_path / class_name
        save_dir.mkdir(parents=True, exist_ok=True)

        originals = list(class_dir.glob("*.png"))
        num_orig = len(originals)
        
        # Stat Init
        current_stat = {'Class': class_name, 'Original': num_orig, 'Final': 0}

        if num_orig == 0:
            stats_list.append(current_stat)
            continue

        # 1. Copy Originals
        for img_path in originals:
            shutil.copy2(img_path, save_dir / img_path.name)

        # 2. Augment if needed
        needed = target_count - num_orig
        if needed > 0:
            img_cycler = cycle(originals)
            generated = 0
            while generated < needed:
                src_path = next(img_cycler)
                try:
                    img = Image.open(src_path).convert("RGBA")
                    aug_img = augment_image(img, geom_conf, color_conf)
                    
                    save_name = f"{src_path.stem}_aug_{generated}.png"
                    aug_img.save(save_dir / save_name)
                    generated += 1
                except Exception:
                    pass
            current_stat['Final'] = num_orig + generated
        else:
            current_stat['Final'] = num_orig
            
        stats_list.append(current_stat)

    # Export Report
    pd.DataFrame(stats_list).to_csv(output_path / "augment_stats.csv", index=False)
    print(f"✅ Done! Stats saved to {output_path}/augment_stats.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--type", required=True, choices=['pill', 'box'], help="Use config for 'pill' or 'box'")
    args = parser.parse_args()

    main(args.input, args.output, args.type)