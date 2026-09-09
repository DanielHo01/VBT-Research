"""
verify_labels.py — 校验 OWL-ViT 伪标签的 YOLO 格式标注
======================================================

用法:
    python3 scripts/verify_labels.py                           # 默认 datasets/owlvit_pseudo
    python3 scripts/verify_labels.py --images-dir /path/to/images
"""
import argparse
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'validation' / 'dataset_benchmark'))
import config as cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    default_dir = cfg.datasets_dir() / 'owlvit_pseudo'
    ap.add_argument('--images-dir', default=str(default_dir / 'images'))
    ap.add_argument('--labels-dir', default=str(default_dir / 'labels'))
    args = ap.parse_args()

    img_dir = args.images_dir
    label_dir = args.labels_dir

    if not os.path.isdir(img_dir):
        print(f"[错误] 图片目录不存在: {img_dir}")
        return 1

    files = sorted(os.listdir(img_dir))
    print(f"Verifying {len(files)} crops:\n")

    all_ok = True
    for f in files:
        img = cv2.imread(os.path.join(img_dir, f))
        if img is None:
            print(f"  BAD {f}: 无法读取图片")
            all_ok = False
            continue
        H, W = img.shape[:2]

        label_path = os.path.join(label_dir, f.replace('.jpg', '.txt'))
        if not os.path.exists(label_path):
            print(f"  BAD {f}: 缺少标签文件")
            all_ok = False
            continue

        with open(label_path) as lf:
            line = lf.read().strip()
        parts = [float(x) for x in line.split()]
        if len(parts) != 5:
            print(f"  BAD {f}: 标签应为 5 列 (class cx cy w h)，实际 {len(parts)} 列: {line}")
            all_ok = False
            continue
        cls, cx, cy, bw, bh = parts

        # Check range
        ok = all(0 <= v <= 1 for v in [cx, cy, bw, bh])
        ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 999

        # Draw bbox
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)
        img_draw = img.copy()
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        status = "OK" if ok else "BAD"
        print(f"  {status} {f}: crop={W}x{H} label=({cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f}) "
              f"ratio={ratio:.2f} bbox=[{x1},{y1},{x2},{y2}]")
        if not ok:
            all_ok = False

    print(f"\n{'All labels OK!' if all_ok else 'SOME LABELS HAVE ISSUES!'}")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
