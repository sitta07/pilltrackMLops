# Pill & Box Classification Pipeline Commands

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


