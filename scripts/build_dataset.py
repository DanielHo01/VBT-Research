"""build_dataset.py — M3 训练集组装（多源合并 + 自标加权）
================================================================
合什么：
  自标（datasets/interactive_labels，manifest 驱动）
    → 按视频划 train/val（防泄漏），train 部分重复 K 倍
      （--repeat-self auto：使自标占 train 约 25%，见 label_common）。
  公开（--public 可重复，每源须先过 audit_public.py）
    → 注册表 keep/drop → v2a 策略 --drop-name-contains（默认 end,cap：
      End/杆眼框尺寸语义≠片，暂不合入，见 M3_TRAINING_PLAN.md）
    → 全进 train（公开集自带 valid/test 也合并；我们的 val 是自留视频）。
    空标签图保留（负样本）；缺标签文件的孤图跳过并计数（不臆造负样本）。

产出：out/train|val/images|labels + data.yaml + dataset_card.json。

用法：
    python scripts/build_dataset.py --public datasets/public/weightlifting-plates-v11 ^
        --public datasets/public/weight-plate-detector --val-videos 30kg_1.03_0.89_0.76_0.65.mp4,130kg_0.55_0.47_0.42.mp4
    python scripts/build_dataset.py --public ... --auto-val 4   # 自动建议 val 视频（ deterministic）
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_public import collect_pairs, parse_data_yaml_names  # noqa: E402
from label_common import (load_manifest, parse_label_text, plan_repeat_factor,  # noqa: E402
                          split_by_video, suggest_val_split)
from public_registry import SOURCES, match_class  # noqa: E402
from verify_labels import verify as verify_self  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png")


def copy_self(out: Path, labels_dir: Path, records: list[dict],
              split: str, repeat: int) -> tuple[int, int]:
    """拷自标数据；repeat>1 时 train 复制 _r{k} 份。返回 (图数, 框数)。"""
    img_d, lbl_d = out / split / "images", out / split / "labels"
    img_d.mkdir(parents=True, exist_ok=True)
    lbl_d.mkdir(parents=True, exist_ok=True)
    n_img = n_box = 0
    for r in records:
        src_img = Path(r["img_path"])
        src_lbl = src_img.with_suffix(".txt")
        if not src_lbl.exists():  # labels/ 与 images/ 分目录布局
            alt = labels_dir / "labels" / f"{src_img.stem}.txt"
            src_lbl = alt if alt.exists() else None
        if src_lbl is None:
            print(f"  跳过（缺标签）: {src_img.name}")
            continue
        boxes = parse_label_text(src_lbl.read_text(encoding="utf-8"))
        reps = repeat if split == "train" else 1
        for k in range(reps):
            stem = src_img.stem if k == 0 else f"{src_img.stem}_r{k}"
            shutil.copy2(src_img, img_d / f"{stem}{src_img.suffix}")
            shutil.copy2(src_lbl, lbl_d / f"{stem}.txt")
            n_img += 1
            n_box += len(boxes)
    return n_img, n_box


def merge_public(out: Path, pub_dir: Path, reg_key: str,
                 drop_contains: list[str]) -> dict:
    """合入一个公开源 → train。返回统计。"""
    cfg = SOURCES[reg_key]
    names = parse_data_yaml_names(pub_dir)
    by_split, stats = collect_pairs(pub_dir)
    img_d, lbl_d = out / "train" / "images", out / "train" / "labels"
    img_d.mkdir(parents=True, exist_ok=True)
    lbl_d.mkdir(parents=True, exist_ok=True)
    stat = {"registry": reg_key, "names": names, "n_src_images": stats["n_images"],
            "n_kept_images": 0, "n_kept_boxes": 0, "n_negatives": 0,
            "n_skipped_orphan": 0, "dropped_by_policy": 0}
    for split, pairs in by_split.items():
        for img_p, lbl_p in pairs:
            if lbl_p is None:
                stat["n_skipped_orphan"] += 1
                continue
            boxes = parse_label_text(lbl_p.read_text(encoding="utf-8", errors="ignore"))
            kept = []
            for cls, cx, cy, bw, bh in boxes:
                cname = names[cls] if names and 0 <= cls < len(names) else f"id{cls}"
                if match_class(cname, cfg["keep"], cfg["drop"]) != "keep":
                    continue
                if any(s in cname.lower() for s in drop_contains):
                    stat["dropped_by_policy"] += 1
                    continue
                kept.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stem = f"{pub_dir.name}__{split}__{img_p.stem}"
            shutil.copy2(img_p, img_d / f"{stem}{img_p.suffix}")
            (lbl_d / f"{stem}.txt").write_text(
                "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            stat["n_kept_images"] += 1
            stat["n_kept_boxes"] += len(kept)
            if not kept:
                stat["n_negatives"] += 1
    return stat


def main() -> int:
    ap = argparse.ArgumentParser(description="M3 训练集组装")
    ap.add_argument("--labels-dir", default="datasets/interactive_labels")
    ap.add_argument("--public", action="append", default=[],
                    help="公开源目录（可重复，须先过 audit）")
    ap.add_argument("--registry-map", default="",
                    help="'目录名=注册表key,...'（目录名与注册表不一致时）")
    ap.add_argument("--out", default="datasets/plate_v2")
    ap.add_argument("--val-videos", default="",
                    help="逗号分隔 / @文件（一行一个）")
    ap.add_argument("--auto-val", type=int, default=0,
                    help="自动建议 N 条 val 视频（确定性采样）")
    ap.add_argument("--repeat-self", default="auto",
                    help="auto（占 train 25%%）或整数")
    ap.add_argument("--target-self-ratio", type=float, default=0.25)
    ap.add_argument("--drop-name-contains", default="end,cap",
                    help="v2a 策略：类名含这些子串的框丢弃；none=关闭")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    labels_dir, out = Path(args.labels_dir), Path(args.out)
    if not args.skip_verify:
        print("== 自标校验 ==")
        rep = verify_self(labels_dir)
        print(f"图像 {rep['n_images']} / 框 {rep['n_boxes']} / "
              f"硬错 {len(rep['errors'])} / 警告 {len(rep['warnings'])}")
        if rep["errors"]:
            for e in rep["errors"][:10]:
                print(f"  ERR {e}")
            print("自标有硬错，--skip-verify 可跳过（不推荐）")
            return 1

    manifest = load_manifest(labels_dir / "manifest.json")
    records = []
    for img_name, m in manifest["images"].items():
        p = labels_dir / "images" / img_name
        if p.exists():
            records.append({"video": m["video"], "frame": m["frame"],
                            "img_path": str(p)})
    if not records:
        print("自标为空：先跑 interactive_label.py")
        return 1

    if args.val_videos.startswith("@"):
        val_videos = Path(args.val_videos[1:]).read_text(encoding="utf-8").split()
    elif args.val_videos:
        val_videos = [v.strip() for v in args.val_videos.split(",") if v.strip()]
    elif args.auto_val:
        val_videos = suggest_val_split([r["video"] for r in records],
                                       args.auto_val, args.seed)
        print(f"auto-val 建议（请目检轻/重/易/难覆盖）: {val_videos}")
    else:
        print("需 --val-videos 或 --auto-val（按视频留 val，防泄漏）")
        return 1
    train_self, val_self = split_by_video(records, val_videos)
    if not train_self or not val_self:
        print(f"划分失败：train {len(train_self)} / val {len(val_self)}")
        return 1
    print(f"自标划分：train {len(train_self)} 帧 / val {len(val_self)} 帧")

    reg_map = dict(kv.split("=", 1) for kv in args.registry_map.split(",") if "=" in kv)
    drop_contains = [] if args.drop_name_contains.lower() == "none" else [
        s.strip().lower() for s in args.drop_name_contains.split(",") if s.strip()]
    if out.exists():
        shutil.rmtree(out)
    pub_stats = []
    n_pub_imgs = 0
    for pub in args.public:
        pub_dir = Path(pub)
        key = reg_map.get(pub_dir.name, pub_dir.name)
        if key not in SOURCES:
            print(f"未知注册表源 {key}（目录 {pub}），用 --registry-map 映射")
            return 1
        audit_json = pub_dir.parent / f"{pub_dir.name}_audit.json"
        if not audit_json.exists():
            print(f"警告：{pub} 未见 audit 报告（先审后合！）——继续但请补审")
        st = merge_public(out, pub_dir, key, drop_contains)
        n_pub_imgs += st["n_kept_images"]
        pub_stats.append(st)
        print(f"  {pub}: {st['n_kept_images']} 图 / {st['n_kept_boxes']} 框 "
              f"（负样本 {st['n_negatives']}，策略丢弃 {st['dropped_by_policy']}，"
              f"孤图跳过 {st['n_skipped_orphan']}）")

    if args.repeat_self == "auto":
        repeat = plan_repeat_factor(n_pub_imgs, len(train_self),
                                    args.target_self_ratio)
    else:
        repeat = max(1, int(args.repeat_self))
    print(f"自标重复倍数 K={repeat}（train 自标有效帧 ≈ {len(train_self) * repeat}）")
    n_tr_img, n_tr_box = copy_self(out, labels_dir, train_self, "train", repeat)
    n_va_img, n_va_box = copy_self(out, labels_dir, val_self, "val", 1)

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: val/images\n"
        f"nc: 1\nnames: ['plate']\n", encoding="utf-8")
    card = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "self": {"labels_dir": str(labels_dir), "train_frames": len(train_self),
                 "val_frames": len(val_self), "val_videos": sorted({r["video"] for r in val_self}),
                 "repeat": repeat, "train_images_eff": n_tr_img, "train_boxes_eff": n_tr_box,
                 "val_images": n_va_img, "val_boxes": n_va_box},
        "public": pub_stats,
        "policy": {"drop_name_contains": drop_contains},
        "seed": args.seed,
    }
    (out / "dataset_card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n组装完成：train 图 {n_pub_imgs + n_tr_img}（公开 {n_pub_imgs} + 自标×{repeat} {n_tr_img}）"
          f" / val 图 {n_va_img}\n  {out / 'data.yaml'}\n  {out / 'dataset_card.json'}")
    print("下一步：python scripts/train_plate_v2.py --data "
          f"{out / 'data.yaml'} --name plate_v2a")
    return 0


if __name__ == "__main__":
    sys.exit(main())
