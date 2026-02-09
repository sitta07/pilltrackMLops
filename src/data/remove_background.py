import os
import argparse
from PIL import Image
from tqdm.auto import tqdm
import torch
from transparent_background import Remover
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Preprocess")

# กำหนดนามสกุลไฟล์ที่รองรับ
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def process_background_removal(input_root, output_root, model_type='base'):
    # 1. Check Input
    if not os.path.exists(input_root):
        raise FileNotFoundError(f"❌ Input folder not found: {input_root}")

    # 2. Setup Device
    # -----------------------------------------------------------
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
   
    logger.info(f"⚡ Using device: {device.upper()}")
    
    # 3. Scan Files First (เพื่อดูว่าต้องทำกี่รูป)
    files_to_process = []
    skipped_count = 0
    
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(VALID_EXTENSIONS):
                input_path = os.path.join(root, file)
                
                # Construct Output Path
                rel_path = os.path.relpath(os.path.dirname(input_path), input_root)
                output_dir = os.path.join(output_root, rel_path)
                
                base_name = os.path.splitext(file)[0]
                output_path = os.path.join(output_dir, base_name + ".png")
                
                # 🔥 LAZY CHECK: เช็คก่อนเลยว่ามีไฟล์ปลายทางหรือยัง
                if os.path.exists(output_path):
                    skipped_count += 1
                else:
                    files_to_process.append((input_path, output_dir, output_path))

    if not files_to_process:
        logger.info(f"✨ All {skipped_count} images are already processed. Nothing to do!")
        return

    logger.info(f"🔥 Found {len(files_to_process)} NEW images to process (Skipped {skipped_count} existing).")

    # 4. Load Model (โหลดเฉพาะเมื่อมีงานต้องทำ)
    logger.info(f"⏳ Loading AI model ({model_type})...")
    remover = Remover(mode=model_type, device=device)

    # 5. Processing Loop
    success = 0
    error_count = 0

    pbar = tqdm(files_to_process, desc="⚡ Removing Background", unit="img")
    for input_path, output_dir, output_path in pbar:
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Process
            img = Image.open(input_path).convert("RGB")
            out = remover.process(img)

            # Save as PNG
            out.save(output_path)

            success += 1
            pbar.set_postfix(status="OK")

        except Exception as e:
            error_count += 1
            pbar.set_postfix(status="ERR")
            logger.error(f"⚠️ Error processing {input_path}: {e}")

    print("\n" + "=" * 40)
    print(f"🎉 Processing Complete!")
    print(f"✅ Processed New: {success}")
    print(f"⏩ Skipped Old: {skipped_count}")
    print(f"❌ Error: {error_count}")
    print(f"📂 Output: {output_root}")
    print("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove background from images")
    parser.add_argument("--input", required=True, help="Path to raw images")
    parser.add_argument("--output", required=True, help="Path to save processed images")
    parser.add_argument("--model", default="base", help="Model type (base, fast, etc.)")
    
    args = parser.parse_args()
    
    process_background_removal(args.input, args.output, args.model)