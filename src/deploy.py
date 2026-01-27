import boto3
import json
import os
import argparse
import hashlib
import sys
from datetime import datetime
from dotenv import load_dotenv

# =========================================================
# ⚙️ CONFIGURATION & SETUP
# =========================================================
load_dotenv()

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
REGION = os.getenv("AWS_REGION", "ap-southeast-1")

MODEL_FILENAME = "best_model.pth"
MAPPING_FILENAME = "class_mapping.json"

# =========================================================
# 🛠️ HELPER FUNCTIONS
# =========================================================

def calculate_md5(file_path):
    if not os.path.exists(file_path):
        return None
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def upload_file(s3_client, local_path, s3_key):
    try:
        print(f"   ⬆️ Uploading: {os.path.basename(local_path)} -> {s3_key}")
        s3_client.upload_file(local_path, BUCKET_NAME, s3_key)
        return True
    except Exception as e:
        print(f"   ❌ Error Uploading {local_path}: {e}")
        return False

# =========================================================
# 🚀 MAIN DEPLOYMENT LOGIC
# =========================================================

def deploy_system(version, base_experiment_path, target_type, note=""):
    print("="*60)
    print(f"🚀 STARTING SMART DEPLOYMENT PIPELINE")
    print(f"📦 Version Tag:  {version}")
    print(f"📂 Source Path:  {base_experiment_path}")
    print(f"🎯 Target Mode:  {target_type.upper()}")
    print("="*60)

    # 1. Initialize S3
    try:
        s3 = boto3.client('s3', region_name=REGION)
    except Exception as e:
        print(f"❌ Error: AWS Connection Failed.\n{e}")
        sys.exit(1)

    # 2. Define Targets to Process
    # ถ้าส่ง type มาเป็น 'both' จะเช็คทั้งคู่ แต่ถ้าเจาะจง 'box' หรือ 'pill' ก็จะเช็คแค่นั้น
    available_targets = []
    potential_targets = ["pill", "box"] if target_type == "both" else [target_type]
    
    deployment_manifest = {}

    print("\n🔍 Phase 1: Validating Target Files...")
    for t in potential_targets:
        target_dir = os.path.join(base_experiment_path, t)
        model_p = os.path.join(target_dir, MODEL_FILENAME)
        map_p = os.path.join(target_dir, MAPPING_FILENAME)

        if os.path.exists(model_p) and os.path.exists(map_p):
            print(f"   ✅ {t.upper()} found: Ready for deployment.")
            available_targets.append(t)
            deployment_manifest[t] = {
                "model": model_p,
                "map": map_p,
                "md5_model": calculate_md5(model_p),
                "md5_map": calculate_md5(map_p)
            }
        else:
            print(f"   ⏩ {t.upper()} not found or incomplete: Skipping.")

    if not available_targets:
        print("\n❌ CRITICAL ERROR: No valid models found to deploy!")
        print(f"   Checked in: {base_experiment_path}")
        sys.exit(1)

    # 3. Prepare Version Metadata
    version_data = {
        "version": version,
        "release_note": note,
        "deployed_targets": available_targets,
        "files": {},
        "timestamp": datetime.now().isoformat()
    }

    for t in available_targets:
        version_data["files"][t] = {
            "model": {"filename": MODEL_FILENAME, "md5": deployment_manifest[t]["md5_model"]},
            "map":   {"filename": MAPPING_FILENAME, "md5": deployment_manifest[t]["md5_map"]}
        }

    with open("version.json", "w", encoding='utf-8') as f:
        json.dump(version_data, f, ensure_ascii=False, indent=2)

    # 4. Upload Phase
    print("\n☁️  Phase 2: Uploading to S3...")
    s3_folders = [f"releases/{version}", "releases/latest"]

    for folder in s3_folders:
        print(f"\n   📂 Destination: /{folder}")
        for t in available_targets:
            # Upload Model & Map
            upload_file(s3, deployment_manifest[t]["model"], f"{folder}/{t}/{MODEL_FILENAME}")
            upload_file(s3, deployment_manifest[t]["map"],   f"{folder}/{t}/{MAPPING_FILENAME}")
        
        # Upload version.json
        upload_file(s3, "version.json", f"{folder}/version.json")

    # 5. Cleanup
    if os.path.exists("version.json"):
        os.remove("version.json")

    print("\n" + "="*60)
    print(f"🎉 DEPLOYMENT SUCCESSFUL!")
    print(f"✅ Targets deployed: {', '.join(available_targets).upper()}")
    print(f"📍 Location: 's3://{BUCKET_NAME}/releases/latest/'")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Smart Deploy PillTrack Models')
    parser.add_argument('--version', required=True)
    parser.add_argument('--path', required=True, help='Base experiment path')
    parser.add_argument('--type', default='both', choices=['pill', 'box', 'both'], help='Target to deploy')
    parser.add_argument('--note', default="Auto-deployed via CI/CD")
    
    args = parser.parse_args()
    deploy_system(args.version, args.path, args.type, args.note)