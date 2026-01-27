import boto3
import json
import os
import argparse
import hashlib
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# ⚙️ CONFIGURATION
# =========================================================
REGION = os.getenv("AWS_REGION", "ap-southeast-1")
BUCKET_NAME = None  # จะถูก Override ด้วยค่าจาก Argument

MODEL_FILENAME = "best_model.pth"
MAPPING_FILENAME = "class_mapping.json"

# =========================================================
# 🛠️ HELPER FUNCTIONS
# =========================================================

def calculate_md5(file_path):
    if not os.path.exists(file_path): return None
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""): hash_md5.update(chunk)
    return hash_md5.hexdigest()

def upload_file(s3_client, local_path, s3_key):
    # ✅ ใช้ BUCKET_NAME ที่ถูกเซ็ตค่าจาก Argument แล้ว
    try:
        print(f"   ⬆️ Uploading: {os.path.basename(local_path)} -> {s3_key}")
        s3_client.upload_file(local_path, BUCKET_NAME, s3_key)
        return True
    except Exception as e:
        print(f"   ❌ Error Uploading {local_path}: {e}")
        return False

# =========================================================
# 🚀 MAIN LOGIC
# =========================================================

def deploy_system(version, base_experiment_path, target_type, bucket_arg, note=""):
    # ✅ Override Global Variable ด้วยค่าที่รับมา
    global BUCKET_NAME
    BUCKET_NAME = bucket_arg

    print("="*60)
    print(f"🚀 STARTING SMART DEPLOYMENT PIPELINE")
    print(f"📦 Version Tag:  {version}")
    print(f"📂 Source Path:  {base_experiment_path}")
    print(f"☁️  Target Bucket: {BUCKET_NAME}") # ต้องไม่ว่างเปล่าแล้ว
    print("="*60)

    if not BUCKET_NAME:
        print("❌ CRITICAL ERROR: S3 Bucket Name is missing! Check your secrets/args.")
        sys.exit(1)

    try:
        s3 = boto3.client('s3', region_name=REGION)
    except Exception as e:
        print(f"❌ Error: AWS Connection Failed.\n{e}")
        sys.exit(1)

    available_targets = []
    potential_targets = ["pill", "box"] if target_type == "both" else [target_type]
    deployment_manifest = {}

    print("\n🔍 Phase 1: Validating Target Files...")
    for t in potential_targets:
        target_dir = os.path.join(base_experiment_path, t)
        model_p = os.path.join(target_dir, MODEL_FILENAME)
        map_p = os.path.join(target_dir, MAPPING_FILENAME)

        if os.path.exists(model_p) and os.path.exists(map_p):
            print(f"   ✅ {t.upper()} found: Ready.")
            available_targets.append(t)
            deployment_manifest[t] = {
                "model": model_p, "map": map_p,
                "md5_model": calculate_md5(model_p), "md5_map": calculate_md5(map_p)
            }
        else:
            print(f"   ⏩ {t.upper()} not found: Skipping.")

    if not available_targets:
        print("\n❌ CRITICAL ERROR: No valid models found!")
        sys.exit(1)

    # Prepare Manifest
    version_data = {
        "version": version, "release_note": note,
        "deployed_targets": available_targets, "files": {},
        "timestamp": datetime.now().isoformat()
    }
    
    for t in available_targets:
        version_data["files"][t] = {
            "model": {"filename": MODEL_FILENAME, "md5": deployment_manifest[t]["md5_model"]},
            "map":   {"filename": MAPPING_FILENAME, "md5": deployment_manifest[t]["md5_map"]}
        }

    with open("version.json", "w", encoding='utf-8') as f:
        json.dump(version_data, f, ensure_ascii=False, indent=2)

    print("\n☁️  Phase 2: Uploading to S3...")
    s3_folders = [f"releases/{version}", "releases/latest"]

    for folder in s3_folders:
        print(f"\n   📂 Destination: /{folder}")
        for t in available_targets:
            upload_file(s3, deployment_manifest[t]["model"], f"{folder}/{t}/{MODEL_FILENAME}")
            upload_file(s3, deployment_manifest[t]["map"],   f"{folder}/{t}/{MAPPING_FILENAME}")
        upload_file(s3, "version.json", f"{folder}/version.json")

    if os.path.exists("version.json"): os.remove("version.json")
    print(f"\n✅ DEPLOYMENT SUCCESSFUL! ({', '.join(available_targets).upper()})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', required=True)
    parser.add_argument('--path', required=True)
    parser.add_argument('--type', default='both')
    # ✅ เพิ่ม argument รับ bucket name
    parser.add_argument('--bucket', required=True, help='S3 Bucket Name')
    parser.add_argument('--note', default="Auto-deployed")
    
    args = parser.parse_args()
    deploy_system(args.version, args.path, args.type, args.bucket, args.note)