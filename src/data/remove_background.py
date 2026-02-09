import os
import argparse
from PIL import Image
from tqdm.auto import tqdm
import torch
from transparent_background import Remover
import logging
import numpy as np

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Preprocess")

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def process_background_removal(input_root, output_root, model_type='base'):
    # 1. Check Input
    if not os.path.exists(input_root):
        raise FileNotFoundError(f"❌ Input folder not found: {input_root}")

    # 2. Setup Device (Initial Attempt)
    if torch.cuda.is_available():
        device_name = 'cuda'
        logger.info(f"🚀 Attempting to use GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device_name = 'mps'
    else:
        device_name = 'cpu'

    # 3. Load Model & Compatibility Test (🔥 HERO LOGIC)
    try:
        logger.info(f"⏳ Loading AI model ({model_type}) on {device_name.upper()}...")
        remover = Remover(mode=model_type, device=device_name)
        
        # ------------------------------------------------------------------
        # 🧪 TEST RUN: ลองรันภาพเปล่าๆ 1 รูป เพื่อดูว่า Kernel พังไหม?
        # ------------------------------------------------------------------
        if device_name == 'cuda':
            logger.info("🧪 Testing GPU compatibility for RTX 50 Series...")
            # สร้างภาพจำลองเล็กๆ
            dummy_img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
            remover.process(dummy_img) # ถ้าพัง มันจะเด้งไปเข้า except ทันที
            logger.info("✅ GPU Kernel Compatible! We are flying! 🦅")

    except RuntimeError as e:
        error_msg = str(e)
        if "no kernel image is available" in error_msg or "CUDA error" in error_msg:
            logger.warning("⚠️ GPU KERNEL ERROR DETECTED (Library mismatch with RTX 50 Series).")
            logger.warning("🔄 Activating Smart Fallback: Switching to CPU for preprocessing...")
            logger.warning("Note: This will be slower, but DVC will cache it, so run only once!")
            
            # Fallback to CPU
            device_name = 'cpu'
            remover = Remover(mode=model_type, device='cpu')
        else:
            raise e # ถ้าเป็น error อื่นที่ไม่ใช่เรื่องการ์ดจอ ให้บึ้มไปตามปกติ

    # 4. Scan Files
    files_to_process = []
    skipped_count = 0
    
    print("🔍 Scanning files...")
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(VALID_EXTENSIONS):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(os.path.dirname(input_path), input_root)
                output_dir = os.path.join(output_root, rel_path)
                base_name = os.path.splitext(file)[0]
                output_path = os.path.join(output_dir, base_name + ".png")
                
                if os.path.exists(output_path):
                    skipped_count += 1
                else:
                    files_to_process.append((input_path, output_dir, output_path))

    if not files_to_process:
        logger.info(f"✨ All {skipped_count} images are already processed. Nothing to do!")
        return

    logger.info(f"🔥 Found {len(files_to_process)} NEW images to process (Skipped {skipped_count}).")
    logger.info(f"⚡ Running on: {device_name.upper()}")

    # 5. Processing Loop
    success = 0
    error_count = 0

    pbar = tqdm(files_to_process, desc="⚡ Removing Background", unit="img")
    for input_path, output_dir, output_path in pbar:
        try:
            os.makedirs(output_dir, exist_ok=True)
            img = Image.open(input_path).convert("RGB")
            
            # Process
            out = remover.process(img)

            # Save
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