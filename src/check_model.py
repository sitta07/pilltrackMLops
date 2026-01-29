import json
import os
import datetime

# Path ของโมเดลล่าสุดที่พึ่งรัน Full Train เสร็จ [cite: 2026-01-29]
MODEL_DIR = "experiments/arcface_lite_v1/box"
MAPPING_FILE = os.path.join(MODEL_DIR, "class_mapping.json")
MODEL_FILE = os.path.join(MODEL_DIR, "best_model.pth")

def check_latest_model():
    print("🔍 [Checking Model Status]")
    
    # 1. เช็คเวลาที่ไฟล์ถูกสร้าง/แก้ไขล่าสุด
    if os.path.exists(MODEL_FILE):
        mtime = os.path.getmtime(MODEL_FILE)
        last_modified = datetime.datetime.fromtimestamp(mtime)
        print(f"📅 Last Trained: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # เช็คว่าเป็นของวันนี้จริงไหม [cite: 2026-01-29]
        if last_modified.date() == datetime.date(2026, 1, 29):
            print("✅ Status: This is the LATEST model from today's run!")
        else:
            print("⚠️ Warning: This might be an OLD model.")
    else:
        print("❌ Error: Model file not found!")

    print("-" * 30)

    # 2. เช็ค Class Mapping ที่เรียนรู้ไป 8 คลาส [cite: 2026-01-29]
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r') as f:
            mapping = json.load(f)
        
        print(f"📊 Total Classes Found: {len(mapping)}")
        print("💊 Class List:")
        # เรียงตาม Index เพื่อความดูง่าย
        for name, idx in sorted(mapping.items(), key=lambda x: x[1]):
            print(f"  [{idx}] : {name}")
    else:
        print("❌ Error: Class mapping file not found!")

if __name__ == "__main__":
    check_latest_model()