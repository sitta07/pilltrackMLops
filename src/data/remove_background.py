import os
import argparse
from PIL import Image
from tqdm.auto import tqdm
import logging
import io
from rembg import remove, new_session

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Preprocess")

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def process_background_removal(input_root, output_root, model_name='u2net'):
    # 1. Check Input
    if not os.path.exists(input_root):
        raise FileNotFoundError(f"❌ Input folder not found: {input_root}")

    # 2. Setup Session (นี่คือทีเด็ด! มันจะหา GPU เองอัตโนมัติ)
    logger.info(f"⏳ Initializing rembg session with model: {model_name}...")
    logger.info("🚀 ONNX Runtime will automatically detect and use RTX 5060 Ti if available.")
    try:
        # สร้าง session ครั้งเดียวแล้วใช้ซ้ำ (เร็วมาก)
        # provider จะลองใช้ CUDA ก่อน ถ้าไม่ได้จะถอยไป CPU เอง
        session = new_session(model_name=model_name)
    except Exception as e:
        logger.error(f"❌ Failed to initialize rembg session: {e}")
        return

    # 3. Scan Files
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

    # 4. Processing Loop
    success = 0
    error_count = 0

    pbar = tqdm(files_to_process, desc="⚡ Removing Background (rembg)", unit="img")
    for input_path, output_dir, output_path in pbar:
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Read image as bytes
            with open(input_path, 'rb') as i:
                input_data = i.read()
            
            # 🔥 Process! (บรรทัดมหัศจรรย์)
            output_data = remove(input_data, session=session)

            # Save bytes as PNG
            img = Image.open(io.BytesIO(output_data)).convert("RGBA")
            img.save(output_path, format='PNG')

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
    parser = argparse.ArgumentParser(description="Remove background using rembg (GPU accelerated)")
    parser.add_argument("--input", required=True, help="Path to raw images")
    parser.add_argument("--output", required=True, help="Path to save processed images")
    # u2net คือโมเดลมาตรฐานที่คุณภาพดีมาก
    parser.add_argument("--model", default="u2net", help="Model name (u2net, u2netp, isnet-general-use)")
    
    args = parser.parse_args()
    
    process_background_removal(args.input, args.output, args.model)