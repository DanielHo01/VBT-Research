"""m3_merge_datasets.py — 多源数据集合并 + 审计（M3 数据闭环）
==================================================================
用途：把用户从 Roboflow Universe（或其他来源）下载的 YOLO 格式数据集 zip，
合并成单一训练集，并把五花八门的类名映射到统一 schema：

    统一类（data.yaml）:
      0 = plate          杠铃片（含按重量分色的 plate_25_red 等）
      1 = barbell_end    杠端"杆眼"（End / barbell end / barbell-cap）
      2 = barbell        整根杠铃（辅助：bar 级跟踪 / 负样本）
      3 = person         人（辅助：动作区域先验）

其他类（dumbbell、架子等）默认忽略（计入审计报告，可加白名单）。

输入约定（见 m3_data/README_INTAKE.md）:
  /home/user/m3_data/downloads/*.zip   每个 zip 是 Roboflow "YOLOv8/v11" 格式
                                       导出（含 data.yaml + train/valid/test 三目录）
  /home/user/m3_data/extra/*           可选：手标数据目录（同类格式，不进 zip 也行）

输出:
  /home/user/m3_data/merged/           合并后数据集（YOLO 格式 + data.yaml）
  /home/user/m3_data/merged/audit.json 审计报告（每源的类分布/尺寸/来源/忽略类清单）

用法:
  python3 scripts/m3_merge_datasets.py [--out /home/user/m3_data/merged]
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
M3 = Path.home() / "m3_data"
DOWNLOADS = M3 / "downloads"
EXTRA = M3 / "extra"
OUT = M3 / "merged"

# ── 类名 → 统一 schema 的映射表（关键词优先，可扩展）─────────────
CLASS_MAP = [
    (0, "plate",  re.compile(r"plate", re.I)),          # 含 plate_25_red / Weight plates…
    (1, "barbell_end", re.compile(r"barbell[-_ ]?end|^end$|barbell[-_ ]?cap|cap$", re.I)),
    (2, "barbell", re.compile(r"barbell|^bar$", re.I)),
    (3, "person",  re.compile(r"^person$|people", re.I)),
]
UNIFIED_NAMES = ["plate", "barbell_end", "barbell", "person"]

IMPORT_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_names(data_yaml: Path) -> list[str]:
    """从 Roboflow 导出的 data.yaml 里提取 names 列表（无 pyyaml 依赖）。"""
    text = data_yaml.read_text(errors="ignore")
    m = re.search(r"names\s*:\s*(.+)", text)
    if not m:
        return []
    payload = m.group(1).strip()
    if payload.startswith("["):                      # 行内列表（Roboflow 常用格式）
        try:
            return [str(x) for x in ast.literal_eval(payload)]
        except (ValueError, SyntaxError):
            return [x.strip().strip("'\"") for x in payload.strip("[]").split(",")]
    # 多行块格式：逐行找 '- xxx' 直到下一个顶层键
    names = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("-"):
            names.append(s[1:].strip().strip("'\""))
        elif s and not s.startswith("#") and s.endswith(":"):
            break
    return names


def map_class(name: str) -> int | None:
    for cls_id, _, pat in CLASS_MAP:
        if pat.search(name):
            return cls_id
    return None


def iter_yolo_pairs(images_dir: Path) -> list[tuple[Path, Path]]:
    """收集 (image, label) 对：images/ 下所有图片，labels/ 下相对同名 txt。"""
    pairs = []
    for img in images_dir.rglob("*"):
        if img.suffix.lower() not in IMPORT_IMAGE_EXTS:
            continue
        rel = img.relative_to(images_dir)
        label = images_dir.parent / "labels" / rel.with_suffix(".txt")
        pairs.append((img, label))
    return pairs


def process_source(name: str, root: Path, audit_src: dict) -> tuple[int, int]:
    """把单个源的 train/valid/test 图片+标注复制进 merged（文件名加前缀防冲突）。"""
    n_img = n_label = 0
    for split in ("train", "valid", "test"):
        src_dir = root / split
        if not (src_dir / "images").exists():
            continue
        for img, label in iter_yolo_pairs(src_dir / "images"):
            if not label.exists():
                audit_src.setdefault("missing_labels", []).append(str(img.name))
                continue
            n_img += 1
            out_name = f"{name}__{img.name}"
            shutil.copy2(img, OUT / "images" / out_name)
            new_lines = []
            for line in label.read_text(errors="ignore").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    cls_old = int(float(parts[0]))
                except ValueError:
                    continue
                old_names = audit_src["class_names"]
                if cls_old >= len(old_names):
                    audit_src.setdefault("bad_class_idx", 0)
                    audit_src["bad_class_idx"] += 1
                    continue
                new_cls = map_class(old_names[cls_old])
                if new_cls is None:
                    audit_src.setdefault("ignored_classes", set()).add(old_names[cls_old])
                    continue
                parts[0] = str(new_cls)
                new_lines.append(" ".join(parts))
            # 全被忽略 → 只留图片（作无标注负样本可另行过滤；此处照抄但标注留空）
            (OUT / "labels" / (out_name.rsplit(".", 1)[0] + ".txt")).write_text(
                "\n".join(new_lines) + ("\n" if new_lines else ""))
            n_label += 1
    return n_img, n_label


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)

    audit: dict = {"sources": {}, "unified": UNIFIED_NAMES}
    total_img = total_label = 0

    # ── Roboflow zip 源 ─────────────────────────────────
    zips = sorted(DOWNLOADS.glob("*.zip")) if DOWNLOADS.exists() else []
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            work = M3 / "work" / z.stem
            zf.extractall(work)
        yaml_path = next(work.rglob("data.yaml"), None)
        if yaml_path is None:
            audit["sources"][z.stem] = {"error": "zip 内无 data.yaml（需 Roboflow YOLO 格式导出）"}
            continue
        names = parse_names(yaml_path)
        src_audit = {"file": z.name, "class_names": names,
                     "ignored_classes": set(), "missing_labels": []}
        n_img, n_label = process_source(z.stem, work, src_audit)
        src_audit.update(images=n_img, labels=n_label,
                         ignored_classes=sorted(src_audit["ignored_classes"]))
        audit["sources"][z.stem] = src_audit
        total_img += n_img
        total_label += n_label
        print(f"[zip] {z.name}: {n_img} 图 / {n_label} 标 | 类 {names}")

    # ── 额外目录源（手标数据等）──────────────────────────
    if EXTRA.exists():
        for d in sorted(EXTRA.iterdir()):
            if not d.is_dir():
                continue
            yaml_path = d / "data.yaml"
            names = parse_names(yaml_path) if yaml_path.exists() else []
            src_audit = {"file": str(d), "class_names": names,
                         "ignored_classes": set(), "missing_labels": []}
            n_img, n_label = process_source(d.name, d, src_audit)
            src_audit.update(images=n_img, labels=n_label,
                             ignored_classes=sorted(src_audit["ignored_classes"]))
            audit["sources"][d.name] = src_audit
            total_img += n_img
            total_label += n_label
            print(f"[dir] {d.name}: {n_img} 图 / {n_label} 标 | 类 {names}")

    # ── 汇总类分布 ──────────────────────────────────────
    class_counts = {n: 0 for n in UNIFIED_NAMES}
    for lbl in (out / "labels").glob("*.txt"):
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if parts and parts[0].isdigit():
                class_counts[UNIFIED_NAMES[int(parts[0])]] += 1

    audit["totals"] = {"images": total_img, "labels": total_label,
                       "class_counts": class_counts}
    (out / "data.yaml").write_text(
        "train: ./images\nval: ./images\nnc: 4\nnames: ['plate', 'barbell_end', 'barbell', 'person']\n")
    (out / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"\n合并完成: {total_img} 图 / {total_label} 标 → {out}")
    print("类分布:", class_counts)
    print("审计报告:", out / "audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
