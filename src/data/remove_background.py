# src/data/remove_background.py
import os
import argparse
from PIL import Image
from tqdm.auto import tqdm
import torch
from transparent_background import Remover

# กำหนดนามสกุลไฟล์ที่รองรับ
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def process_background_removal(input_root, output_root, model_type='base'):
    # 1. Check Input
    if not os.path.exists(input_root):
        raise FileNotFoundError(f"❌ Input folder not found: {input_root}")

    # 2. Setup Device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if torch.cuda.is_available(): device = 'cuda'
    
    print(f"⚡ Using device: {device.upper()}")
    print(f"⏳ Loading AI model ({model_type})...")
    
    # Load Model ครั้งเดียวใช้นานๆ
    remover = Remover(mode=model_type, device=device)

    # 3. Scan Files
    all_files = []
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(VALID_EXTENSIONS):
                all_files.append(os.path.join(root, file))

    if not all_files:
        print("❌ No valid images found")
        return

    print(f"🔥 Found {len(all_files)} images. Starting processing...")

    # 4. Processing Loop
    success = 0
    error_count = 0

    pbar = tqdm(all_files, desc="⚡ Removing Background", unit="img")
    for file_path in pbar:
        try:
            # สร้าง path ปลายทางโดยคง structure เดิม
            rel_path = os.path.relpath(os.path.dirname(file_path), input_root)
            output_dir = os.path.join(output_root, rel_path)
            os.makedirs(output_dir, exist_ok=True)

            # Process
            img = Image.open(file_path).convert("RGB")
            out = remover.process(img)

            # Save as PNG
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_dir, base_name + ".png")
            out.save(output_path)

            success += 1
            pbar.set_postfix(file=base_name[:10], status="OK")

        except Exception as e:
            error_count += 1
            pbar.set_postfix(file=base_name[:10], status="ERR")
            # print(f"⚠️ Error {file_path}: {e}")

    print("\n" + "=" * 40)
    print(f"🎉 Processing Complete!")
    print(f"✅ Success: {success}")
    print(f"❌ Error: {error_count}")
    print(f"📂 Output: {output_root}")
    print("=" * 40)

if __name__ == "__main__":
    # ส่วนสำคัญ: รับค่าจาก Command Line (DVC จะส่งค่ามาทางนี้)
    parser = argparse.ArgumentParser(description="Remove background from images")
    parser.add_argument("--input", required=True, help="Path to raw images")
    parser.add_argument("--output", required=True, help="Path to save processed images")
    parser.add_argument("--model", default="base", help="Model type (base, fast, etc.)")
    
    args = parser.parse_args()
    
    process_background_removal(args.input, args.output, args.model)