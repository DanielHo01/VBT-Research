"""verify_labels.py — M3 标注质量检查（自标数据进训练前的门）
================================================================
相对 v2（Phase 1.5 crop 版）的重写：面向全帧 YOLO 标签。

检查项：
  硬错（exit 1）：文件缺失/解析失败/坐标越界/非单类/退化框。
  警告（exit 0）：长宽比超 3（片应近圆）、框过小、空标签图
                  （允许，属负样本，但列出来复核）、近重复框（IoU>0.9）。

用法：
    python scripts/verify_labels.py --labels-dir datasets/interactive_labels
    python scripts/verify_labels.py --labels-dir datasets/interactive_labels ^
        --render-dir datasets/interactive_labels/_qa --max-render 24
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from label_common import (box_iou, load_manifest, parse_label_text,  # noqa: E402
                          validate_box_norm, yolo_to_pixels)

IMG_EXTS = (".jpg", ".jpeg", ".png")


def verify(labels_dir: Path) -> dict:
    manifest = load_manifest(labels_dir / "manifest.json")
    img_dir, lbl_dir = labels_dir / "images", labels_dir / "labels"
    report: dict = {"errors": [], "warnings": [], "n_images": 0,
                    "n_boxes": 0, "empty": [], "per_video": Counter(),
                    "ratios": []}
    label_files = sorted(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []
    if not label_files:
        report["errors"].append(f"无标签文件: {lbl_dir}")
        return report
    for lbl_p in label_files:
        stem = lbl_p.stem
        img_p = next((img_dir / f"{stem}{e}" for e in IMG_EXTS
                      if (img_dir / f"{stem}{e}").exists()), None)
        if img_p is None:
            report["errors"].append(f"{stem}: 缺图像文件")
            continue
        try:
            boxes = parse_label_text(lbl_p.read_text(encoding="utf-8"))
        except ValueError as e:
            report["errors"].append(f"{stem}: {e}")
            continue
        W = H = 1000  # 归一化检查不需要真实尺寸
        px = []
        for i, b in enumerate(boxes):
            issues = validate_box_norm(b)
            hard = [x for x in issues if "越界" in x or "非有限" in x or "class" in x]
            soft = [x for x in issues if x not in hard]
            for h in hard:
                report["errors"].append(f"{stem}#{i}: {h}")
            for s in soft:
                report["warnings"].append(f"{stem}#{i}: {s}")
            _, cx, cy, w, h = yolo_to_pixels(
                f"{b[0]} {b[1]} {b[2]} {b[3]} {b[4]}", W, H)
            px.append((cx, cy, w, h))
            if b[3] > 0 and b[4] > 0:
                report["ratios"].append(max(b[3], b[4]) / min(b[3], b[4]))
        for i in range(len(px)):
            for j in range(i + 1, len(px)):
                if box_iou(px[i], px[j]) > 0.9:
                    report["warnings"].append(f"{stem}: 框{i}与框{j}近重复")
        report["n_images"] += 1
        report["n_boxes"] += len(boxes)
        if not boxes:
            report["empty"].append(stem)
        meta = manifest["images"].get(img_p.name)
        if meta:
            report["per_video"][meta.get("video", "?")] += 1
    return report


def render_contact_sheet(labels_dir: Path, render_dir: Path,
                         max_render: int, seed: int = 7) -> int:
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        print("缺 cv2，跳过渲染")
        return -1
    import random  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    img_dir, lbl_dir = labels_dir / "images", labels_dir / "labels"
    files = sorted(lbl_dir.glob("*.txt"))
    picks = random.Random(seed).sample(files, min(max_render, len(files)))
    cells = []
    for lbl_p in picks:
        img_p = next((img_dir / f"{lbl_p.stem}{e}" for e in IMG_EXTS
                      if (img_dir / f"{lbl_p.stem}{e}").exists()), None)
        if img_p is None:
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        H, W = img.shape[:2]
        for line in lbl_p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            _, cx, cy, w, h = yolo_to_pixels(line, W, H)
            cv2.rectangle(img, (int(cx - w / 2), int(cy - h / 2)),
                          (int(cx + w / 2), int(cy + h / 2)), (0, 255, 0), 2)
        cv2.putText(img, lbl_p.stem[:28], (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cells.append(cv2.resize(img, (360, 240)))
    if not cells:
        return 0
    cols = 4
    rows = (len(cells) + cols - 1) // cols
    canvas = np.zeros((rows * 240, cols * 360, 3), dtype=np.uint8)
    for i, c in enumerate(cells):
        r, col = divmod(i, cols)
        canvas[r * 240:(r + 1) * 240, col * 360:(col + 1) * 360] = c
    render_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(render_dir / "contact_sheet.jpg"), canvas)
    return len(cells)


def main() -> int:
    ap = argparse.ArgumentParser(description="M3 标注质量检查")
    ap.add_argument("--labels-dir", default="datasets/interactive_labels")
    ap.add_argument("--render-dir", default=None)
    ap.add_argument("--max-render", type=int, default=24)
    args = ap.parse_args()

    report = verify(Path(args.labels_dir))
    print(f"图像 {report['n_images']} / 框 {report['n_boxes']} / "
          f"空标签 {len(report['empty'])}")
    if report["ratios"]:
        import numpy as np  # noqa: PLC0415
        r = np.array(report["ratios"])
        print(f"长宽比: 中位 {np.median(r):.2f} / >2 占比 {np.mean(r > 2):.3f}")
    if report["per_video"]:
        print("按视频:")
        for v, c in sorted(report["per_video"].items()):
            print(f"  {v}: {c} 帧")
    for w in report["warnings"][:30]:
        print(f"  WARN {w}")
    if len(report["warnings"]) > 30:
        print(f"  ... 还有 {len(report['warnings']) - 30} 条警告")
    for e in report["errors"][:30]:
        print(f"  ERR {e}")
    if args.render_dir:
        n = render_contact_sheet(Path(args.labels_dir), Path(args.render_dir),
                                 args.max_render)
        print(f"contact sheet: {n} 张 → {args.render_dir}")
    if report["errors"]:
        print(f"\n硬错 {len(report['errors'])} 条：修完再进训练")
        return 1
    print("\n通过（警告请复核 contact sheet）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
