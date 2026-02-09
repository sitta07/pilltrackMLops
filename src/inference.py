import os
import sys
import json
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F

# Setup Path ให้ Python หาไฟล์อื่นเจอ
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.architecture import PillModel
from src.data.dataset import get_transforms

def load_mapping(model_dir):
    map_path = os.path.join(model_dir, "class_mapping.json")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"❌ Mapping file not found at {map_path}")
    
    with open(map_path, 'r') as f:
        mapping = json.load(f)
    
    # Convert keys to int because JSON stores keys as strings
    return {int(k): v for k, v in mapping.items()}

def predict(image_path, model_dir, arch, device):
    # 1. Load Mapping
    idx_to_class = load_mapping(model_dir)
    num_classes = len(idx_to_class)
    print(f"📖 Loaded mapping with {num_classes} classes.")

    # 2. Load Model
    # 🔥 แก้แล้ว: ใส่ dropout=0.5 เข้าไป (ค่านี้ไม่มีผลตอน eval แต่ต้องใส่ให้ครบ)
    model = PillModel(
        num_classes=num_classes,
        model_name=arch,
        embed_dim=512,  # ค่า Default
        dropout=0.5     
    ).to(device)

    weight_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"❌ Weights not found at {weight_path}")

    # Load Weights
    checkpoint = torch.load(weight_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval() # 👈 สำคัญมาก! บรรทัดนี้จะปิดการทำงานของ Dropout เอง
    print(f"✅ Model loaded from {weight_path}")

    # 3. Preprocess Image
    transform = get_transforms(img_size=224, is_train=False) 
    
    # Open Image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")
        
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        dummy_label = torch.tensor([0]).to(device)
        outputs = model(input_tensor, dummy_label)
        
        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = probs.max(1)
        
        pred_idx = pred_idx.item()
        confidence = confidence.item() * 100

    # 5. Result
    class_name = idx_to_class.get(pred_idx, "Unknown")
    
    return class_name, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model_dir", required=True, help="Folder containing .pth and .json")
    parser.add_argument("--arch", default="convnext_small", help="Model architecture")
    args = parser.parse_args()

    # 🔥 UPDATE: เพิ่มการเช็ค CUDA สำหรับ RTX 5060 Ti
    # -----------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍏 Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")
    # -----------------------------------------------------------    
    try:
        print("-" * 50)
        print(f"🔍 Inspecting: {args.image}")
        name, conf = predict(args.image, args.model_dir, args.arch, device)
        
        print("\n🎉 Prediction Result:")
        print(f"💊 Class: {name}")
        print(f"📊 Confidence: {conf:.2f}%")
        print("-" * 50)
    except Exception as e:
        print(f"\n❌ Error: {e}")