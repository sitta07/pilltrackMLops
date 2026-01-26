import boto3
import json
import os
import argparse
import hashlib
import sys
from dotenv import load_dotenv

# =========================================================
# ⚙️ CONFIGURATION & SETUP
# =========================================================
load_dotenv() # โหลดค่าจาก .env

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
REGION = os.getenv("AWS_REGION", "ap-southeast-1")

# ชื่อไฟล์จริงตามที่พี่แคปหน้าจอมา
MODEL_FILENAME = "best_model.pth"
MAPPING_FILENAME = "class_mapping.json"

# =========================================================
# 🛠️ HELPER FUNCTIONS
# =========================================================

def calculate_md5(file_path):
    """
    Checkpoint 1: คำนวณค่า Digital Signature (MD5) ของไฟล์
    เพื่อให้มั่นใจว่าไฟล์ต้นฉบับกับปลายทางคือตัวเดียวกัน 100%
    """
    if not os.path.exists(file_path):
        return None
    
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        # อ่านทีละ 4KB กัน Memory เต็มกรณีไฟล์ใหญ่
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def upload_file(s3_client, local_path, s3_key):
    """
    Function สำหรับส่งไฟล์ขึ้น S3 พร้อม Error Handling
    """
    try:
        print(f"   ⬆️ Uploading: {os.path.basename(local_path)} -> .../{s3_key.split('/')[-1]}")
        s3_client.upload_file(local_path, BUCKET_NAME, s3_key)
        return True
    except Exception as e:
        print(f"   ❌ Error Uploading {local_path}: {e}")
        return False

# =========================================================
# 🚀 MAIN DEPLOYMENT LOGIC
# =========================================================

def deploy_system(version, base_experiment_path, note=""):
    print("="*60)
    print(f"🚀 STARTING DEPLOYMENT PIPELINE")
    print(f"📦 Version Tag:  {version}")
    print(f"📂 Source Path:  {base_experiment_path}")
    print(f"☁️  Target Bucket: {BUCKET_NAME}")
    print("="*60)

    # 1. Initialize S3 Client
    try:
        s3 = boto3.client('s3', region_name=REGION)
    except Exception as e:
        print(f"❌ Error: Cannot connect to AWS. Check your .env file.\n{e}")
        sys.exit(1)

    # 2. Define Local Paths (กำหนดที่อยู่ไฟล์ในเครื่อง)
    # Structure: experiments/arcface_finetuned/{type}/{filename}
    files = {
        "pill_model": os.path.join(base_experiment_path, "pill", MODEL_FILENAME),
        "pill_map":   os.path.join(base_experiment_path, "pill", MAPPING_FILENAME),
        "box_model":  os.path.join(base_experiment_path, "box", MODEL_FILENAME),
        "box_map":    os.path.join(base_experiment_path, "box", MAPPING_FILENAME)
    }

    # 3. Validation Phase (ตรวจของก่อนส่ง)
    print("\n🔍 Phase 1: Validating Files...")
    missing_files = []
    for key, path in files.items():
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        print("❌ CRITICAL ERROR: Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("⛔ Deployment Aborted.")
        sys.exit(1)
    else:
        print("   ✅ All required files found.")

    # 4. Integrity Check Phase (สร้าง Checkpoint)
    print("\n🔐 Phase 2: Generating Checksums (MD5)...")
    checksums = {}
    for key, path in files.items():
        md5_hash = calculate_md5(path)
        checksums[key] = md5_hash
        print(f"   - {key}: {md5_hash}")

    # 5. Prepare Manifest (เตรียมใบปะหน้า)
    version_data = {
        "version": version,
        "release_note": note,
        "structure": ["pill", "box"],
        "files": {
            "pill": {
                "model": {"filename": MODEL_FILENAME, "md5": checksums["pill_model"]},
                "map":   {"filename": MAPPING_FILENAME, "md5": checksums["pill_map"]}
            },
            "box": {
                "model": {"filename": MODEL_FILENAME, "md5": checksums["box_model"]},
                "map":   {"filename": MAPPING_FILENAME, "md5": checksums["box_map"]}
            }
        },
        "timestamp": "auto-generated-by-server"
    }

    # เขียนไฟล์ version.json ชั่วคราว
    with open("version.json", "w", encoding='utf-8') as f:
        json.dump(version_data, f, ensure_ascii=False, indent=2)

    # 6. Upload Phase (ส่งของจริง)
    print("\n☁️  Phase 3: Uploading to S3...")
    
    # เราจะส่งไป 2 ที่: 1. โฟลเดอร์รุ่น (v1.x) และ 2. โฟลเดอร์ล่าสุด (latest)
    targets = [f"releases/{version}", "releases/latest"]

    for target_folder in targets:
        print(f"\n   📂 Target: /{target_folder}")
        
        # Upload Pill Files
        upload_file(s3, files["pill_model"], f"{target_folder}/pill/{MODEL_FILENAME}")
        upload_file(s3, files["pill_map"],   f"{target_folder}/pill/{MAPPING_FILENAME}")
        
        # Upload Box Files
        upload_file(s3, files["box_model"],  f"{target_folder}/box/{MODEL_FILENAME}")
        upload_file(s3, files["box_map"],    f"{target_folder}/box/{MAPPING_FILENAME}")
        
        # Upload Manifest
        upload_file(s3, "version.json",      f"{target_folder}/version.json")

    # 7. Cleanup
    if os.path.exists("version.json"):
        os.remove("version.json")

    print("\n" + "="*60)
    print(f"🎉 DEPLOYMENT SUCCESSFUL!")
    print(f"✅ Version {version} is now live at 'releases/latest/'")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deploy PillTrack Models to S3')
    parser.add_argument('--version', required=True, help='Version Tag (e.g. v1.1.0)')
    parser.add_argument('--path', required=True, help='Path to experiments folder (e.g. experiments/arcface_finetuned)')
    parser.add_argument('--note', default="No release notes", help='Release notes for this version')
    
    args = parser.parse_args()
    
    deploy_system(args.version, args.path, args.note)