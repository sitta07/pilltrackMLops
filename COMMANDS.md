# Pill & Box Classification Pipeline Commands

รวมคำสั่งสำหรับรัน Pipeline ทั้งหมด ตั้งแต่ Training ไปจนถึง Inference
ระบบใช้ **DVC** ในการคุม Pipeline และรองรับ **Incremental Learning** (เพิ่มยาใหม่ไม่ต้องเทรนใหม่หมด)

---

## 1. Training Workflows (BOX)

### A. เพิ่มยาตัวใหม่ (Daily/Weekly Update)
ใช้เมื่อใส่โฟลเดอร์ยาใหม่ลงใน `data/raw_box` และต้องการอัปเดตโมเดลด่วน (Fast Finetune)

```bash
# 1. เตรียมข้อมูล (Preprocess -> Split -> Augment)
dvc repro augment_box

# 2. ⚠️ สำคัญ: บอก DVC ว่าไม่ต้อง Full Train โมเดลตัวแม่ใหม่ (ข้ามไปเลย)
dvc commit train_box

# 3. สั่ง Finetune เฉพาะ Head (เร็ว + ไม่ลืมความรู้เก่า)
dvc repro finetune_box
```

### B. ล้างกระดานเทรนใหม่หมด (Major Update)
ใช้เมื่อต้องการ Re-train ตั้งแต่ Backbone (นาน) หรือเปลี่ยน Architecture โมเดล
```bash
dvc repro train_box
```
(Note: ระบบจะทำ Preprocess -> Split -> Augment -> Full Train ให้เองตามลำดับ)

## 2. Training Workflows (PILL)

สำหรับโมเดลเม็ดยา (Pipeline แยกต่างหาก)
```bash
dvc repro train_pill
```

## 3. Testing & Inference

ทดสอบโมเดลกับรูปภาพเดี่ยวๆ

ทดสอบ Box Model (ตัว Finetuned)
```bash
python src/inference.py --image data/raw_box/ชื่อยา/รูปทดสอบ.jpg --model_dir experiments/arcface_finetuned/box --arch convnext_small
```

ทดสอบ Pill Model
```bash
python src/inference.py --image data/raw/ชื่อยา/รูปทดสอบ.jpg --model_dir experiments/arcface_lite_v1/pill --arch convnext_tiny
```

## 4. Config Guide (params.yaml)

แก้ค่าต่างๆ ได้ที่ไฟล์ params.yaml โดยไม่ต้องแตะโค้ด
augment: ปรับจำนวนรูปที่จะปั๊ม (Target per class), องศาการหมุน
split: ปรับสัดส่วน Train/Val (Default 0.2)
train: ปรับ Learning Rate, Epochs, Model Name
augment:
  box:
    target_per_class: 300
train:
  box:
    epochs: 5

