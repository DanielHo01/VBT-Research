"""
Quick sanity check: 50 images, 5 epochs.
Goal: verify script works + model can detect barbell.
"""
import os
import shutil
import random
from ultralytics import YOLO
import torch

os.environ["YOLO_VERBOSE"] = "False"

# ── Build tiny test set (50 train, 20 valid) ──
SRC_TRAIN = "D:/EasyVBT-Research/datasets/merged/train"
SRC_VALID = "D:/EasyVBT-Research/datasets/merged/valid"
TEST_DIR  = "D:/EasyVBT-Research/datasets/quick_test"

for d in [f"{TEST_DIR}/train/images", f"{TEST_DIR}/train/labels",
          f"{TEST_DIR}/valid/images", f"{TEST_DIR}/valid/labels"]:
    os.makedirs(d, exist_ok=True)

# Pick 50 train images
all_imgs = sorted([f for f in os.listdir(f"{SRC_TRAIN}/images") if f.endswith(('.jpg','.png'))])
random.seed(42)
selected = random.sample(all_imgs, 50)

for f in selected:
    shutil.copy2(f"{SRC_TRAIN}/images/{f}", f"{TEST_DIR}/train/images/{f}")
    lbl = f.rsplit('.',1)[0] + '.txt'
    shutil.copy2(f"{SRC_TRAIN}/labels/{lbl}", f"{TEST_DIR}/train/labels/{lbl}")

# Pick 20 valid images
valid_imgs = sorted([f for f in os.listdir(f"{SRC_VALID}/images") if f.endswith(('.jpg','.png'))])
for f in valid_imgs[:20]:
    shutil.copy2(f"{SRC_VALID}/images/{f}", f"{TEST_DIR}/valid/images/{f}")
    lbl = f.rsplit('.',1)[0] + '.txt'
    shutil.copy2(f"{SRC_VALID}/labels/{lbl}", f"{TEST_DIR}/valid/labels/{lbl}")

# YAML
yaml = """path: D:/EasyVBT-Research/datasets/quick_test
train: train/images
val: valid/images
nc: 12
names:
  '0': barbell
  '1': plate_0.5kg
  '2': plate_1.5kg
  '3': plate_10kg
  '4': plate_15kg
  '5': plate_1kg
  '6': plate_2.5kg
  '7': plate_20kg
  '8': plate_25kg
  '9': plate_2kg
  '10': plate_5kg
  '11': zacisk
"""
with open(f"{TEST_DIR}/data.yaml", 'w') as fp:
    fp.write(yaml)

print(f"Quick test set: 50 train, 20 valid")

# ── Train ──
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

model = YOLO('yolov8n.pt')

results = model.train(
    data=f'{TEST_DIR}/data.yaml',
    epochs=5,
    imgsz=640,
    batch=8,
    device=0,
    workers=0,           # Windows fix
    patience=999,        # 不早停
    project='datasets/quick_test_output',
    name='quick_check',
    exist_ok=True,
    pretrained=True,
    amp=False,           # GTX 1650 关 AMP
    optimizer='SGD',
    lr0=0.02,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    warmup_epochs=1.0,
    mosaic=0.5,
    verbose=True,
    seed=42,
)

# ── Eval ──
print("\n=== Quick test results ===")
results_csv = "datasets/quick_test_output/quick_check/results.csv"
if os.path.exists(results_csv):
    import pandas as pd
    df = pd.read_csv(results_csv)
    cols = df.columns.tolist()
    print(f"Columns: {cols[:6]}")
    print(f"\nLast epoch:")
    last = df.iloc[-1]
    # 找 mAP 列
    map50_col = next((c for c in df.columns if 'mAP50(B)' in c), None)
    if map50_col:
        map50 = float(last[map50_col])
        print(f"  mAP50 = {map50:.4f}")
        if map50 > 0.05:
            print("  ✅ Model CAN detect barbell. Ready for full training.")
        else:
            print("  ⚠️  mAP too low. Check labels/model.")
    else:
        print("  mAP50 column not found, check results manually.")
        print(df.tail())
else:
    print("No results.csv - check output dir manually")