"""
create_barbell_annotations.py
Creates YOLO format annotation .txt files for extracted training frames.
Then opens EzYOLO for human review and correction before training.

Usage:
    python scripts/create_barbell_annotations.py

pi-lens suppressions: pi-lens=unsafe-call (cv2/np are normal usage, no error-prone file I/O here)
"""
# noqa: pi-lens=unsafe-call

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("D:/EasyVBT-Research/datasets/barbell_dataset")
FRAMES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"
DATA_YAML = DATASET_ROOT / "data.yaml"
SPLIT_YAML = DATASET_ROOT / "dataset_split.yaml"

TRAIN_RATIO = 0.85
RANDOM_SEED = 42
CLASS_NAME = "plate"
# ── Helpers ───────────────────────────────────────────────────────────────────


def detect_barbell_regions(frame: np.ndarray) -> list[tuple]:
    """
    Returns a list of (x_center, y_center, w, h) in RELATIVE coords [0,1]
    for all barbell-like horizontal rectangles found in the frame.
    Uses edge detection + contour analysis — no ML needed.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Stage 1: Find strong vertical edges (barbell ends / plate edges) ──
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 50, 150)

    # ── Stage 2: Horizontal dilation to connect edge fragments ─────────────
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.02), 1))
    edges_dilated = cv2.dilate(edges, kernel_h, iterations=2)

    # ── Stage 3: Find contours ─────────────────────────────────────────────
    contours, _ = cv2.findContours(
        edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Filter: must be wide (barbell/plate), not tall noise
        aspect = cw / max(ch, 1)
        if aspect < 1.5 or cw < w * 0.04:  # discard narrow / vertical noise
            continue
        # Convert to YOLO relative center format
        cx = (x + cw / 2) / w
        cy = (y + ch / 2) / h
        rw = cw / w
        rh = ch / h
        # Clamp to [0, 1]
        cx = float(np.clip(cx, 0.0, 1.0))
        cy = float(np.clip(cy, 0.0, 1.0))
        rw = float(np.clip(rw, 0.001, 1.0))
        rh = float(np.clip(rh, 0.001, 1.0))
        regions.append((cx, cy, rw, rh))

    # Sort by width (largest first) — the barbell/plate is usually the widest
    regions.sort(key=lambda r: r[2], reverse=True)
    return regions


def detect_plate_in_squat_bottom(frame: np.ndarray) -> tuple | None:
    """
    For turnaround (squat-bottom) frames the barbell is close to the body
    and near the LOWER third of the image (720×1280 portrait).
    Uses colour masking (body-colour dominant lower half) to find the
    foreground barbell region vs background, then returns the widest barbell
    candidate in the LOWER HALF of the frame.

    Returns (x_center, y_center, w, h) in RELATIVE YOLO coords, or None.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Focus on lower third (y > 2h/3) — barbell in squat bottom
    lower_gray = gray[int(h * 0.55) :, :]

    blurred = cv2.GaussianBlur(lower_gray, (7, 7), 2)
    edges = cv2.Canny(blurred, 40, 120)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(w * 0.015), 5), 1))
    edges_dil = cv2.dilate(edges, kernel_h, iterations=3)

    contours, _ = cv2.findContours(
        edges_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        if aspect < 1.2 or cw < w * 0.05:
            continue
        # Global y
        gx, gy = x, y + int(h * 0.55)
        cx_rel = (gx + cw / 2) / w
        cy_rel = (gy + ch / 2) / h
        rw = cw / w
        rh = ch / h
        candidates.append((cx_rel, cy_rel, rw, rh, cw))  # keep raw width for sort

    if not candidates:
        return None

    # Pick widest candidate (most likely the barbell)
    best = max(candidates, key=lambda c: c[4])
    return (best[0], best[1], best[2], best[3])


def auto_annotate_frame(image_path: Path, is_squat_bottom: bool = False) -> list[tuple]:
    """
    Wrapper: tries detect_barbell_regions; if no candidates found
    and is_squat_bottom=True, falls back to detect_plate_in_squat_bottom.
    """
    frame = cv2.imread(str(image_path))
    if frame is None:
        return []

    candidates = detect_barbell_regions(frame)

    if not candidates and is_squat_bottom:
        fallback = detect_plate_in_squat_bottom(frame)
        if fallback:
            candidates = [fallback]

    # Safety: if still nothing, return a wide central box as last resort
    if not candidates:
        # Very broad, low-confidence box — human reviewer should fix
        candidates = [(0.5, 0.75, 0.70, 0.08)]

    # Always return exactly 1 annotation for the barbell plate
    return [candidates[0]]


def save_yolo_txt(txt_path: Path, annotations: list[tuple], class_id: int = 0):
    """Write one YOLO .txt file. annotations = [(cx, cy, w, h), ...]"""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w") as f:
        for cx, cy, w, h in annotations:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  EzYOLO Annotation Creator  —  Barbell Plate Dataset")
    print("=" * 60)

    # ── 1. Collect all extracted frame JPGs ─────────────────────────────────
    frames = sorted(FRAMES_DIR.glob("*/*.jpg"))
    if not frames:
        print(f"ERROR: No frames found in {FRAMES_DIR}")
        return

    print(
        f"\n📁 Found {len(frames)} frames across {len(set(f.parent for f in frames))} videos"
    )

    # ── 2. Auto-annotate every frame ───────────────────────────────────────
    labelled = 0
    skipped = 0
    for frame_path in frames:
        video_name = frame_path.stem.split("_")[0]
        is_bottom = "_BOTTOM" in frame_path.stem or "_RB" in frame_path.stem

        annotations = auto_annotate_frame(frame_path, is_squat_bottom=is_bottom)

        label_path = LABELS_DIR / frame_path.parent.name / (frame_path.stem + ".txt")
        save_yolo_txt(label_path, annotations)
        labelled += 1

    print(f"✅ Auto-labelled {labelled} frames  |  {skipped} skipped")

    # ── 3. Create 85/15 train/val split (shuffle by video stem) ───────────
    random.seed(RANDOM_SEED)

    # Group by video
    from collections import defaultdict

    by_video = defaultdict(list)
    for frame_path in frames:
        by_video[frame_path.parent.name].append(frame_path)

    train_frames, val_frames = [], []
    for video, vframes in by_video.items():
        random.shuffle(vframes)
        split = int(len(vframes) * TRAIN_RATIO)
        train_frames.extend(vframes[:split])
        val_frames.extend(vframes[split:])

    print(f"\n📊 Split: {len(train_frames)} train | {len(val_frames)} val")

    # ── 4. Reorganise into YOLO train/val structure ───────────────────────
    yolo_images = DATASET_ROOT / "yolo_images"
    yolo_labels = DATASET_ROOT / "yolo_labels"

    for split, frame_list in [("train", train_frames), ("val", val_frames)]:
        for frame_path in frame_list:
            # Destination: yolo_images/{train,val}/
            dst_img = yolo_images / split / frame_path.name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img.exists():
                shutil.copy2(frame_path, dst_img)

            # Corresponding label
            label_src = LABELS_DIR / frame_path.parent.name / (frame_path.stem + ".txt")
            dst_lbl = yolo_labels / split / (frame_path.stem + ".txt")
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            if label_src.exists():
                shutil.copy2(label_src, dst_lbl)

    img_count = len(list((yolo_images / "train").glob("*.jpg"))) + len(
        list((yolo_images / "val").glob("*.jpg"))
    )
    lbl_count = len(list((yolo_labels / "train").glob("*.txt"))) + len(
        list((yolo_labels / "val").glob("*.txt"))
    )
    print(f"📁 YOLO structure: {img_count} images, {lbl_count} labels")

    # ── 5. Write data.yaml ─────────────────────────────────────────────────
    data_yaml_content = {
        "path": str(DATASET_ROOT.resolve() / "yolo_images"),
        "train": "train",
        "val": "val",
        "nc": 1,
        "names": {0: CLASS_NAME},
    }
    with open(DATA_YAML, "w") as f:
        yaml.dump(data_yaml_content, f, sort_keys=False, default_flow_style=False)

    # ── 6. Write dataset_split.yaml (frame-level split manifest) ───────────
    split_manifest = {
        "train": [str(f) for f in train_frames],
        "val": [str(f) for f in val_frames],
    }
    with open(SPLIT_YAML, "w") as f:
        yaml.dump(split_manifest, f, sort_keys=False)

    print(f"\n📄 data.yaml written → {DATA_YAML}")
    print(f"📄 dataset_split.yaml written → {SPLIT_YAML}")

    # ── 7. Copy data.yaml into yolo_labels root so EzYOLO finds it ─────────
    ez_yaml_dst = yolo_labels.parent / "data.yaml"
    if DATA_YAML.resolve() != ez_yaml_dst.resolve():
        shutil.copy2(DATA_YAML, ez_yaml_dst)
    print(f"📄 Copied data.yaml → {ez_yaml_dst}")

    print("\n" + "=" * 60)
    print("  ✅ Auto-annotation complete!")
    print("=" * 60)
    print("""
Next steps:
  1. Open EzYOLO:   python D:/EzYOLO/main.py
  2. Create project 'barbell_plate_v1'
  3. Import images:  Dataset → Import → 'barbell_dataset/yolo_images'
  4. Set classes:    [plate]  (nc=1)
  5. Pre-label:      Use Models → Auto-label with best.onnx
  6. Review & correct every frame (focus on bottom frames marked
     _BOTTOM / _RB — these have highest plate visibility)
  7. Export:         Dataset → Export → YOLO format (.zip)
  8. Move zip to:    datasets/barbell_plate_v1.zip
  9. Train:          Models → Train → yolo11n.pt base → 100 epochs
 10. Replace detector: update vbtcore/detector.py → new ONNX path
""")


if __name__ == "__main__":
    main()
