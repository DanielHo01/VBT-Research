"""audit_public.py — M3 公开数据源审计（先审后合的"审"）
==============================================================
合入训练之前，对每个公开源做三道 gate：

  Gate 1 语义：类名映射（keep/drop/unlisted，规则见 public_registry.py）。
              unlisted 类强制人工复核——未知语义的框不许进训练。
  Gate 2 形状：kept 框的面积/长宽比分布。整杠 smell =
              中位面积 >8% 且 中位长宽比 >1.8（片应小而圆）。
  Gate 3 目检：audit_montage.jpg（kept 绿 / drop 红 / unlisted 黄），
              人眼过一遍 24 张抽样——这是真正的最终门。

 verdict：PASS（可合）/ REVIEW（人工定）/ REJECT（语义不对，整源弃用）。

用法：
    python scripts/audit_public.py --src datasets/public/weightlifting-plates-v11
    python scripts/audit_public.py --src <dir> --source barbell-tracking  # 用注册表类规则
    python scripts/audit_public.py --src <dir> --names plate,barbell      # 无 data.yaml 时手动给类名
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from label_common import parse_label_text  # noqa: E402
from public_registry import SOURCES, mapping_table  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = ("train", "valid", "val", "test")


def parse_data_yaml_names(src: Path) -> list[str] | None:
    """轻量解析 data.yaml 的 names（不依赖 pyyaml）。"""
    for cand in [src / "data.yaml", src / "data.yml"]:
        if not cand.exists():
            continue
        text = cand.read_text(encoding="utf-8", errors="ignore")
        # 内联式 names: ['a', 'b'] / [a, b]
        m = re.search(r"names\s*:\s*\[(.*?)\]", text, re.S)
        if m:
            return [s.strip().strip("'\"") for s in m.group(1).split(",")
                    if s.strip()]
        # 块式 0: name
        m = re.search(r"names\s*:\s*\n((?:\s*\d+\s*:.*\n?)+)", text)
        if m:
            pairs = re.findall(r"(\d+)\s*:\s*['\"]?([^'\"\n]+)['\"]?", m.group(1))
            if pairs:
                return [n for _, n in sorted(pairs, key=lambda p: int(p[0]))]
    return None


def collect_pairs(src: Path) -> tuple[dict[str, list[tuple[Path, Path | None]]], dict]:
    """递归配对 images/labels；按父目录名归 split。返回 (split→[(img,lbl)], 统计)。"""
    imgs = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS]
    by_split: dict[str, list] = {}
    stats = {"n_images": len(imgs), "n_paired": 0, "n_no_label": 0,
             "n_bad_label": 0}
    for img in imgs:
        split = "flat"
        for part in img.parts:
            if part.lower() in SPLIT_NAMES:
                split = part.lower()
                break
        # 找同名 txt：同目录优先，其次 images→labels 映射
        lbl = img.with_suffix(".txt")
        if not lbl.exists():
            alt = Path(str(img).replace("images", "labels")).with_suffix(".txt")
            lbl = alt if alt.exists() else None
        if lbl is None:
            stats["n_no_label"] += 1
        else:
            try:
                parse_label_text(lbl.read_text(encoding="utf-8", errors="ignore"))
                stats["n_paired"] += 1
            except ValueError:
                stats["n_bad_label"] += 1
                lbl = None
        by_split.setdefault(split, []).append((img, lbl))
    return by_split, stats


def audit_boxes(by_split, names: list[str] | None,
                keep: list[str], drop: list[str]) -> dict:
    """框级审计：映射 + kept 形状统计。"""
    table = mapping_table(names, keep, drop) if names else []
    decision = {i: d for i, _, d in table}
    box_counter: Counter = Counter()
    kept_areas, kept_aspects = [], []
    per_class: Counter = Counter()
    for split, pairs in by_split.items():
        for img, lbl in pairs:
            if lbl is None:
                continue
            try:
                boxes = parse_label_text(
                    lbl.read_text(encoding="utf-8", errors="ignore"))
            except ValueError:
                continue
            for cls, cx, cy, bw, bh in boxes:
                d = decision.get(cls, "unlisted")
                box_counter[d] += 1
                cname = names[cls] if names and 0 <= cls < len(names) else f"id{cls}"
                per_class[f"{cname}→{d}"] += 1
                if d == "keep" and bw > 0 and bh > 0:
                    kept_areas.append(bw * bh)
                    kept_aspects.append(max(bw, bh) / min(bw, bh))
    out = {"mapping": [{"id": i, "name": n, "decision": d} for i, n, d in table],
           "boxes": dict(box_counter), "per_class": dict(per_class)}
    if kept_areas:
        a = np.array(kept_areas)
        r = np.array(kept_aspects)
        out["kept_shape"] = {
            "n": len(a),
            "median_area": round(float(np.median(a)), 5),
            "median_aspect": round(float(np.median(r)), 3),
            "frac_aspect_gt2": round(float(np.mean(r > 2.0)), 4),
            "frac_tiny_lt0005": round(float(np.mean(a < 0.005)), 4),
        }
    return out


def verdict(box_audit: dict, stats: dict) -> tuple[str, list[str]]:
    notes = []
    boxes = Counter(box_audit.get("boxes", {}))
    kept = boxes.get("keep", 0)
    if kept == 0:
        return "REJECT", ["kept 框为 0：该源无可用语义"]
    shape = box_audit.get("kept_shape", {})
    if shape.get("median_area", 0) > 0.08 and shape.get("median_aspect", 0) > 1.8:
        notes.append(f"整杠 smell：中位面积 {shape['median_area']} + "
                     f"中位长宽比 {shape['median_aspect']}（片应小而圆）")
    if boxes.get("unlisted", 0):
        notes.append(f"unlisted 框 {boxes['unlisted']} 个：类语义未知，需复核映射")
    if kept < 50:
        notes.append(f"kept 仅 {kept} 框：量级太小，合入价值低")
    if stats["n_bad_label"]:
        notes.append(f"坏标签文件 {stats['n_bad_label']} 个")
    if not box_audit.get("mapping"):
        notes.append("无类名表（缺 data.yaml 且未传 --names）：映射不可信")
    return ("REVIEW" if notes else "PASS"), notes


def render_montage(by_split, names, keep, drop, out_path: Path,
                   per_split: int = 8, seed: int = 7) -> int:
    """抽样渲染：kept 绿 / drop 红 / unlisted 黄。返回渲染张数（cv2 缺失返回 -1）。"""
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return -1
    import random
    from public_registry import match_class  # noqa: PLC0415
    rng = random.Random(seed)
    picks = []
    for split, pairs in sorted(by_split.items()):
        ok = [(i, l) for i, l in pairs if l is not None]
        picks += [(split, i, l) for i, l in rng.sample(ok, min(per_split, len(ok)))]
    if not picks:
        return 0
    colors = {"keep": (0, 255, 0), "drop": (0, 0, 255), "unlisted": (0, 255, 255)}
    cells = []
    for split, img_p, lbl_p in picks:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        H, W = img.shape[:2]
        try:
            boxes = parse_label_text(lbl_p.read_text(encoding="utf-8", errors="ignore"))
        except ValueError:
            continue
        for cls, cx, cy, bw, bh in boxes:
            cname = names[cls] if names and 0 <= cls < len(names) else f"id{cls}"
            d = match_class(cname, keep, drop)
            x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
            x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[d], 2)
        cv2.putText(img, f"{split}:{img_p.name[:24]}", (8, 24),
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
    cv2.imwrite(str(out_path), canvas)
    return len(cells)


def main() -> int:
    ap = argparse.ArgumentParser(description="M3 公开数据源审计")
    ap.add_argument("--src", required=True, help="数据集解压目录")
    ap.add_argument("--source", default=None, help="注册表名（取类规则）")
    ap.add_argument("--names", default=None, help="逗号类名（无 data.yaml 时）")
    ap.add_argument("--keep", default=None, help="覆盖 keep 规则（逗号）")
    ap.add_argument("--drop", default=None, help="覆盖 drop 规则（逗号）")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        print(f"目录不存在: {src}")
        return 1
    keep = args.keep.split(",") if args.keep else []
    drop = args.drop.split(",") if args.drop else []
    if args.source:
        if args.source not in SOURCES:
            print(f"未知源 {args.source}")
            return 1
        cfg = SOURCES[args.source]
        keep = keep or cfg["keep"]
        drop = drop or cfg["drop"]
        print(f"注册表规则 {args.source}: keep={keep} drop={drop}")

    names = parse_data_yaml_names(src)
    if names:
        print(f"data.yaml 类: {names}")
    elif args.names:
        names = [n.strip() for n in args.names.split(",")]
        print(f"手动类名: {names}")
    else:
        print("警告：无类名表（缺 data.yaml 且未传 --names），映射将全为 unlisted")

    by_split, stats = collect_pairs(src)
    print(f"图像 {stats['n_images']}（配对 {stats['n_paired']} / 无标签 "
          f"{stats['n_no_label']} / 坏标签 {stats['n_bad_label']}），"
          f"splits: {sorted(by_split)}")
    box_audit = audit_boxes(by_split, names, keep, drop)
    print(f"框映射: {box_audit['boxes']}")
    for row in box_audit["mapping"]:
        print(f"  id={row['id']} {row['name']} → {row['decision']}")
    if box_audit.get("kept_shape"):
        print(f"kept 形状: {box_audit['kept_shape']}")
    v, notes = verdict(box_audit, stats)
    print(f"\nverdict: {v}")
    for n in notes:
        print(f"  - {n}")

    n_render = 0
    montage_path = src.parent / f"{src.name}_montage.jpg"
    if not args.no_render:
        n_render = render_montage(by_split, names, keep, drop, montage_path)
        print(f"montage: {montage_path}（{n_render} 张）"
              if n_render >= 0 else "montage: 缺 cv2，跳过渲染")
    report = {"source_dir": str(src), "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "registry_source": args.source, "keep": keep, "drop": drop,
              "names": names, "stats": stats, "box_audit": box_audit,
              "verdict": v, "verdict_notes": notes, "montage": str(montage_path),
              "n_rendered": n_render}
    out_json = src.parent / f"{src.name}_audit.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [f"# Audit: {src.name} → {v}", f"- 生成: {report['generated']}",
          f"- 图像: {stats['n_images']}（配对 {stats['n_paired']}）",
          f"- 框映射: {box_audit['boxes']}", f"- kept 形状: {box_audit.get('kept_shape', {})}"]
    md += [f"- {n}" for n in notes]
    md.append(f"- montage: {montage_path.name}（必看！人眼终审）")
    (src.parent / f"{src.name}_audit.md").write_text("\n".join(md), encoding="utf-8")
    print(f"报告: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
