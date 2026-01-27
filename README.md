# Pill & Box Classification Pipeline Manual Commands

รวมคำสั่งสำหรับรัน Pipeline ทั้งหมด ตั้งแต่ Training ไปจนถึง Inference
ระบบใช้ **DVC** ในการคุม Pipeline และรองรับ **Incremental Learning** (เพิ่มยาใหม่ไม่ต้องเทรนใหม่หมด)

---

## 1. Training Workflows (BOX) 
(Note: ระบบจะทำ Preprocess -> Split -> Augment -> Full Train ให้เองตามลำดับ)

### A. เพิ่มยาตัวใหม่ (Daily/Weekly Update)
```bash
dvc repro augment_box # 1. เตรียมข้อมูล (Preprocess -> Split -> Augment)

dvc commit train_box # 2. สำคัญ: บอก DVC ว่าไม่ต้อง Full Train โมเดลตัวแม่ใหม่ (ข้ามไปเลย)

dvc repro evaluate_box  # 3. สั่ง Finetune & Evaluate (ทำเสร็จตรวจผลให้ด้วย)
```

### B. เทรนใหม่หมด (Major Update)
```bash
dvc repro train_box 

```
## 2. Training Workflows (PILL)

### A. เพิ่มยาตัวใหม่ (Daily/Weekly Update)

```bash
dvc repro augment_pill # 1. เตรียมข้อมูล

dvc commit train_pill # 2. ⚠️ ล็อค Base Model (ห้ามเทรนใหม่)

dvc repro evaluate_pill # 3. Finetune & Evaluate
```

### B. เทรนใหม่หมด (Major Update)
```bash
dvc repro train_pill
```
### ทดสอบ Finetuned Model

**Box Finetuned Model:**
```bash
python src/inference.py \
    --image data/raw_box/ชื่อยา/รูปทดสอบ.jpg \
    --model_dir experiments/arcface_finetuned/box \
    --arch convnext_small
```
**Pill Finetuned Model:**
```bash
python src/inference.py \
    --image data/raw/ชื่อยา/รูปทดสอบ.jpg \
    --model_dir experiments/arcface_finetuned/pill \
    --arch convnext_tiny
```

### ทดสอบ Base Model (โมเดลตัวหลัก)
**Box Base Model:**
```bash
python src/inference.py \
    --image data/raw_box/ชื่อยา/รูปทดสอบ.jpg \
    --model_dir experiments/arcface_lite_v1/box \
    --arch convnext_small
```
**Pill Base Model:**

```bash
python src/inference.py \
    --image data/raw/ชื่อยา/รูปทดสอบ.jpg \
    --model_dir experiments/arcface_lite_v1/pill \
    --arch convnext_tiny
```


## 3. Cloud Data Sync (DVC <-> S3)
สำหรับการเก็บรักษา Data, Cache และ Model Versioning (Back-up & Share)
เก็บงานขึ้น Cloud (หลัง Train เสร็จ)

```bash
dvc push # ส่งไฟล์ข้อมูลและโมเดลทั้งหมดขึ้น S3 (Buckket โซน dvc-store)
```

ดึงงานจาก Cloud (เปลี่ยนเครื่อง/ทีมงานอื่น)

```bash
dvc pull # ดึงข้อมูลล่าสุดมาลงเครื่อง
```

## 4. Production Deployment 
เมื่อโมเดลผ่านการเทรนและทดสอบแล้ว ใช้คำสั่งนี้เพื่อปล่อยของไปที่ S3 โซน releases (เพื่อให้เครื่องลูกค้าโหลดไปใช้)

⚠️ ข้อควรระวัง:

ตรวจสอบไฟล์ .env ว่าใส่ AWS Key ครบถ้วน

ตรวจสอบว่าในโฟลเดอร์มีไฟล์ best_model.pth และ class_mapping.json ครบทั้ง Pill และ Box

```bash
python src/deploy.py \
    --version v1.0.0 \
    --path experiments/arcface_finetuned \
    --note "Initial Release: รองรับยา 10 ชนิดแรก"
```

## 5. Infrastructure Management (Terraform) 
ใช้สำหรับสร้างหรือแก้ไข S3 Bucket และ IAM User (นานๆ ใช้ที)
```bash
cd infra

# ตรวจสอบว่าต้องแก้อะไรบ้าง
terraform plan

# ลงมือแก้จริง (ต้องยืนยัน yes)
terraform apply
```
