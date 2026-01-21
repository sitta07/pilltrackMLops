# src/inference.py
import sys
import os
import argparse
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Fix path import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.architecture import PillModel

class PillPredictor:
    def __init__(self, model_dir, device=None):
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # 1. Load Class Mapping
        map_path = os.path.join(model_dir, "class_mapping.json")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"❌ Class mapping not found at {map_path}")
            
        with open(map_path, "r") as f:
            self.idx_to_class = json.load(f)  # {"0": "class_A", "1": "class_B"}
        
        # 2. Load Config & Model Weights
        # เราจะโหลด state_dict เพียวๆ แล้ว init model ใหม่
        weights_path = os.path.join(model_dir, "best_model.pth")
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # หมายเหตุ: ปกติเราควรเซฟ config ไว้ใน json แยกจะได้ไม่ต้อง hardcode
        # แต่อันนี้ขอสมมติว่าใช้ config เดียวกับตอนเทรนไปก่อน
        self.model = PillModel(
            num_classes=len(self.idx_to_class),
            model_name="convnext_tiny",  # ⚠️ ต้องตรงกับตอนเทรน
            embed_dim=512,
            dropout=0.0
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # 3. Setup Transform (ต้องเหมือน Val Transform)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        print(f"✅ Model loaded from {model_dir} (Classes: {len(self.idx_to_class)})")

    def predict(self, image_path):
        if not os.path.exists(image_path):
            return {"error": "Image not found"}
            
        # Prepare Image
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get Embeddings
            embedding = self.model(img_tensor) # shape: [1, 512]
            embedding_norm = F.normalize(embedding)
            
            # Get Class Center Weights
            centers = F.normalize(self.model.head.weight) # shape: [num_classes, 512]
            
            # Calculate Cosine Similarity
            # [1, 512] x [512, num_classes] = [1, num_classes]
            logits = torch.matmul(embedding_norm, centers.T)
            
            # Convert to Probability
            probs = F.softmax(logits * 10, dim=1) # *10 เพื่อ scale ให้ค่าต่างกันชัดขึ้น (Temperature Scaling)
            
            conf, pred_idx = torch.max(probs, 1)
            pred_class = self.idx_to_class[str(pred_idx.item())]
            
        return {
            "class": pred_class,
            "confidence": f"{conf.item()*100:.2f}%",
            "score": logits.max().item()
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--model_dir", required=True, help="Path to folder containing best_model.pth")
    args = parser.parse_args()

    predictor = PillPredictor(args.model_dir)
    result = predictor.predict(args.image)
    
    print("\n" + "="*30)
    print(f"🖼️  Image: {os.path.basename(args.image)}")
    print(f"💊 Prediction: {result['class']}")
    print(f"📊 Confidence: {result['confidence']}")
    print("="*30 + "\n")