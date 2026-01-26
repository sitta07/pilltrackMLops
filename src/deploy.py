import boto3
import json
import os
import argparse
from dotenv import load_dotenv

# โหลดค่า Config จาก .env
load_dotenv()
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
REGION = os.getenv("AWS_REGION", "ap-southeast-1")

# ================= CONFIG ชื่อไฟล์ =================
# ⚠️ แก้ให้ตรงกับไฟล์จริงใน Screenshot
MODEL_FILENAME = "best_model.pth"       
MAPPING_FILENAME = "class_mapping.json" 
# ===============================================

def upload_file(local_path, s3_key):
    """ฟังก์ชันช่วย Upload และแจ้งสถานะ"""
    s3 = boto3.client('s3', region_name=REGION)
    try:
        # เช็คก่อนว่ามีไฟล์จริงไหม
        if not os.path.exists(local_path):
             print(f"   ⚠️ Warning: File not found {local_path} (Skipping)")
             return False

        print(f"   ⬆️ Uploading: {local_path} -> s3://{BUCKET_NAME}/{s3_key}")
        s3.upload_file(local_path, BUCKET_NAME, s3_key)
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def deploy_to_s3(version, base_experiment_path, note=""):
    print(f"🚀 Starting Deployment: Version {version}")
    print(f"📦 Source Folder: {base_experiment_path}")
    print("-" * 50)

    # 1. กำหนด path ของไฟล์ในเครื่องเรา (Local)
    # path: experiments/arcface_finetuned/pill/best_model.pth
    local_pill_model = os.path.join(base_experiment_path, "pill", MODEL_FILENAME)
    local_pill_map = os.path.join(base_experiment_path, "pill", MAPPING_FILENAME)
    
    local_box_model = os.path.join(base_experiment_path, "box", MODEL_FILENAME)
    local_box_map = os.path.join(base_experiment_path, "box", MAPPING_FILENAME)

    # เช็คของหลัก (Model) ก่อนส่ง ถ้าไม่มีให้ Error เลย
    if not os.path.exists(local_pill_model) or not os.path.exists(local_box_model):
        print(f"❌ Critical Error: หาไฟล์ {MODEL_FILENAME} ไม่เจอ!")
        print(f"   - Checked: {local_pill_model}")
        print(f"   - Checked: {local_box_model}")
        return

    # 2. เตรียมข้อมูล Version Info
    version_data = {
        "version": version,
        "release_note": note,
        "models": ["pill", "box"],
        "files": [MODEL_FILENAME, MAPPING_FILENAME]
    }
    
    with open("version.json", "w", encoding='utf-8') as f:
        json.dump(version_data, f, ensure_ascii=False, indent=2)

    # 3. เริ่ม Upload (Loop เดียวจบ ทั้ง Archive และ Latest)
    targets = [version, "latest"] 
    
    for target in targets:
        print(f"\n📂 Updating target: /releases/{target}/")
        
        # --- PILL ---
        upload_file(local_pill_model, f"releases/{target}/pill/{MODEL_FILENAME}")
        upload_file(local_pill_map,   f"releases/{target}/pill/{MAPPING_FILENAME}")
        
        # --- BOX ---
        upload_file(local_box_model,  f"releases/{target}/box/{MODEL_FILENAME}")
        upload_file(local_box_map,    f"releases/{target}/box/{MAPPING_FILENAME}")
        
        # --- INFO ---
        upload_file("version.json", f"releases/{target}/version.json")

    # ลบไฟล์ขยะ
    os.remove("version.json")
    print("-" * 50)
    print(f"✅ Deployment Complete! รุ่น {version} พร้อมใช้งานที่ 'releases/latest/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', required=True, help='เช่น v1.0.0')
    parser.add_argument('--path', required=True, help='Path ไปยังโฟลเดอร์ experiment หลัก')
    parser.add_argument('--note', default="", help='รายละเอียดการอัปเดต')
    
    args = parser.parse_args()
    deploy_to_s3(args.version, args.path, args.note)