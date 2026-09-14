"""
scripts/holdout_intake.py — 留出集 intake 检查与覆盖度审计
==========================================================
落地 docs/HOLDOUT.md 第三节（覆盖清单）与第七节（intake 检查单）。

留出集的意义在于「不被污染」，因此本工具只做**校验与报告**，
不会修改任何视频或真值文件。

用法：
    # 完整 intake 检查（命名/索引/真值/体积/覆盖度）
    python3 scripts/holdout_intake.py --check

    # 只看覆盖度缺口
    python3 scripts/holdout_intake.py --coverage

    # 从视频文件名反向生成索引条目骨架（仍需人工补采集元信息）
    python3 scripts/holdout_intake.py --scaffold

退出码：0 = 全部通过；1 = 存在阻塞问题。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HOLDOUT = REPO / "validation" / "holdout"
INDEX = HOLDOUT / "holdout_index.json"
GT_CSV = HOLDOUT / "ground_truth.csv"
VIDEOS = HOLDOUT / "raw_videos"

# 单条视频与总量体积上限（docs/HOLDOUT.md 第六节）
MAX_SINGLE_MB = 15
MAX_TOTAL_MB = 100

# 文件名规范：{负荷}kg_{rep1}_{rep2}_....mp4
NAME_RE = re.compile(r"^(\d+(?:\.\d+)?)kg((?:_\d+\.\d+)+)\.mp4$")

# 覆盖清单（docs/HOLDOUT.md 第三节）
COVERAGE_SPEC = {
    "gym": {"min_distinct": 2, "label": "场景（不同健身房）"},
    "angle": {"required": {"side_near", "side_far", "landscape"}, "label": "机位"},
    "lighting": {"required": {"bright", "dim", "backlit"}, "min_each": 2, "label": "光照"},
    "load_band": {"required": {"light", "medium", "heavy"}, "label": "负荷档位"},
    "background": {"required": {"plate_stack", "clean"}, "label": "背景难度"},
    "device": {"min_distinct": 2, "label": "设备（手机型号）"},
}

REQUIRED_FIELDS = ["video_id", "load_kg", "gt_reps_mcv"]
META_FIELDS = ["gym", "camera", "angle", "lighting", "outer_plate", "date"]


def load_index() -> list[dict]:
    if not INDEX.exists():
        return []
    try:
        with open(INDEX, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"✗ holdout_index.json 解析失败: {e}")
        sys.exit(1)


def load_gt() -> dict[str, list[float]]:
    if not GT_CSV.exists():
        return {}
    out: dict[str, list[float]] = {}
    with open(GT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row.get("filename", "").strip()
            if not fn:
                continue
            reps = []
            for i in range(1, 9):
                v = row.get(f"rep{i}", "0") or "0"
                try:
                    fv = float(v)
                except ValueError:
                    fv = 0.0
                if fv > 0:
                    reps.append(fv)
            out[fn] = reps
    return out


def parse_name(video_id: str) -> tuple[float, list[float]] | None:
    m = NAME_RE.match(video_id)
    if not m:
        return None
    load = float(m.group(1))
    reps = [float(x) for x in m.group(2).lstrip("_").split("_")]
    return load, reps


def load_band(load_kg: float) -> str:
    if load_kg <= 40:
        return "light"
    if load_kg >= 100:
        return "heavy"
    return "medium"


def check() -> int:
    idx = load_index()
    gt = load_gt()
    files = sorted(p for p in VIDEOS.glob("*.mp4"))

    problems: list[str] = []
    warnings: list[str] = []

    print("=" * 78)
    print(" 留出集 intake 检查")
    print("=" * 78)
    print(f"  视频文件 : {len(files)}")
    print(f"  索引条目 : {len(idx)}")
    print(f"  真值行数 : {len(gt)}")
    print()

    if not files and not idx:
        print("留出集为空 —— 这是当前的已知状态（框架已就绪，等待采集）。")
        print()
        print("下一步：")
        print("  1. 按 docs/HOLDOUT.md 第三节的覆盖清单采集视频")
        print("  2. 视频放入 validation/holdout/raw_videos/")
        print("  3. python3 scripts/holdout_intake.py --scaffold  生成索引骨架")
        print("  4. 人工补齐 gym/camera/angle/lighting/outer_plate/date")
        print("  5. python3 scripts/holdout_intake.py --check     复验")
        return 0

    idx_by_id = {e.get("video_id", ""): e for e in idx}
    file_names = {p.name for p in files}

    # ── 文件 ↔ 索引 ↔ 真值 三方一致性 ──────────────────────────
    for name in sorted(file_names - set(idx_by_id)):
        problems.append(f"视频存在但索引缺失: {name}")
    for vid in sorted(set(idx_by_id) - file_names):
        problems.append(f"索引存在但视频缺失: {vid}")
    for vid in sorted(set(idx_by_id) - set(gt)):
        problems.append(f"索引存在但 ground_truth.csv 缺行: {vid}")

    total_mb = 0.0
    for p in files:
        mb = p.stat().st_size / 1024 / 1024
        total_mb += mb
        if mb > MAX_SINGLE_MB:
            warnings.append(f"单条超限 {mb:.1f}MB > {MAX_SINGLE_MB}MB: {p.name}")
    if total_mb > MAX_TOTAL_MB:
        warnings.append(
            f"总量 {total_mb:.1f}MB > {MAX_TOTAL_MB}MB —— 按 HOLDOUT.md 第六节应切 GitHub Release 分发"
        )

    # ── 逐条校验 ──────────────────────────────────────────────
    for e in idx:
        vid = e.get("video_id", "<无 video_id>")

        for f in REQUIRED_FIELDS:
            if f not in e:
                problems.append(f"{vid}: 缺必填字段 `{f}`")

        missing_meta = [f for f in META_FIELDS if not e.get(f)]
        if missing_meta:
            warnings.append(f"{vid}: 缺采集元信息 {missing_meta}（覆盖度审计需要）")

        parsed = parse_name(vid)
        if parsed is None:
            problems.append(
                f"{vid}: 文件名不符合规范 {{负荷}}kg_{{rep1}}_{{rep2}}....mp4"
            )
            continue
        name_load, name_reps = parsed

        # 文件名 vs 索引
        if "load_kg" in e and abs(float(e["load_kg"]) - name_load) > 1e-6:
            problems.append(
                f"{vid}: load_kg 索引({e['load_kg']}) ≠ 文件名({name_load})"
            )
        idx_reps = e.get("gt_reps_mcv", [])
        if idx_reps and [round(x, 2) for x in idx_reps] != [
            round(x, 2) for x in name_reps
        ]:
            problems.append(
                f"{vid}: gt_reps_mcv 索引{idx_reps} ≠ 文件名{name_reps}"
            )

        # 索引 vs CSV
        if vid in gt:
            csv_reps = gt[vid]
            if [round(x, 2) for x in csv_reps] != [round(x, 2) for x in name_reps]:
                problems.append(
                    f"{vid}: ground_truth.csv{csv_reps} ≠ 文件名{name_reps}"
                )

        if not 1 <= len(name_reps) <= 8:
            problems.append(f"{vid}: rep 数 {len(name_reps)} 超出 1–8 范围")

        for r in name_reps:
            if not 0.1 <= r <= 2.0:
                warnings.append(f"{vid}: 真值 {r} m/s 超出常见生理范围 [0.1, 2.0]")

    # ── 报告 ──────────────────────────────────────────────────
    if problems:
        print(f"✗ 阻塞问题 {len(problems)} 项：")
        for p in problems:
            print(f"    - {p}")
        print()
    if warnings:
        print(f"⚠ 警告 {len(warnings)} 项：")
        for w in warnings:
            print(f"    - {w}")
        print()
    if not problems and not warnings:
        print("✓ intake 检查全部通过")
        print()

    print(f"体积：{total_mb:.1f} MB / 上限 {MAX_TOTAL_MB} MB")
    print()
    coverage_report(idx)

    return 1 if problems else 0


def coverage_report(idx: list[dict] | None = None) -> None:
    if idx is None:
        idx = load_index()

    print("-" * 78)
    print(" 覆盖度审计（docs/HOLDOUT.md 第三节）")
    print("-" * 78)

    if not idx:
        print("  留出集为空，覆盖度 0%。所有维度均为缺口。")
        for key, spec in COVERAGE_SPEC.items():
            print(f"    ✗ {spec['label']}")
        return

    # 派生 load_band
    for e in idx:
        if "load_kg" in e:
            e["load_band"] = load_band(float(e["load_kg"]))

    gaps = 0
    for key, spec in COVERAGE_SPEC.items():
        values = [e.get(key) for e in idx if e.get(key)]
        distinct = set(values)

        if "required" in spec:
            missing = spec["required"] - distinct
            if missing:
                gaps += 1
                print(f"    ✗ {spec['label']}: 缺 {sorted(missing)}")
            elif "min_each" in spec:
                thin = {
                    v: values.count(v)
                    for v in spec["required"]
                    if values.count(v) < spec["min_each"]
                }
                if thin:
                    gaps += 1
                    print(
                        f"    ⚠ {spec['label']}: 数量不足（需各 ≥{spec['min_each']}）{thin}"
                    )
                else:
                    print(f"    ✓ {spec['label']}")
            else:
                print(f"    ✓ {spec['label']}")
        elif "min_distinct" in spec:
            if len(distinct) < spec["min_distinct"]:
                gaps += 1
                print(
                    f"    ✗ {spec['label']}: 只有 {len(distinct)} 种"
                    f"（需 ≥{spec['min_distinct']}）{sorted(distinct)}"
                )
            else:
                print(f"    ✓ {spec['label']}: {len(distinct)} 种")

    # 轻负荷必须含小铁片（非 450mm），验证标定查表
    light = [e for e in idx if e.get("load_band") == "light"]
    small = [e for e in light if e.get("outer_plate") not in (None, "", "20kg", "25kg", "45lb")]
    if light and not small:
        gaps += 1
        print("    ✗ 轻负荷未含外层小铁片（10/15kg 非 450mm）—— 标定查表无验证")

    print()
    print(f"  覆盖缺口：{gaps} 项" + ("（达标）" if gaps == 0 else "，每缺一格记一笔技术债"))


def scaffold() -> None:
    """从视频文件名反向生成索引骨架，采集元信息留空待人工补齐。"""
    files = sorted(p for p in VIDEOS.glob("*.mp4"))
    if not files:
        print("raw_videos/ 下没有 mp4，无需生成。")
        return

    existing = {e.get("video_id") for e in load_index()}
    added = []
    idx = load_index()
    for p in files:
        if p.name in existing:
            continue
        parsed = parse_name(p.name)
        if parsed is None:
            print(f"⚠ 跳过（命名不合规）: {p.name}")
            continue
        load, reps = parsed
        idx.append(
            {
                "video_id": p.name,
                "load_kg": load,
                "gt_reps_mcv": reps,
                "fps_override": None,
                "gym": "",
                "camera": "",
                "angle": "",
                "lighting": "",
                "background": "",
                "device": "",
                "outer_plate": "",
                "date": "",
            }
        )
        added.append(p.name)

    if not added:
        print("没有新视频需要加入索引。")
        return

    INDEX.write_text(
        json.dumps(idx, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"✓ 已为 {len(added)} 条视频生成索引骨架：")
    for a in added:
        print(f"    - {a}")
    print()
    print("请人工补齐每条的：gym / camera / angle / lighting / background / device / outer_plate / date")
    print("字段取值参考 docs/HOLDOUT.md 第三节覆盖清单。")


def main() -> None:
    ap = argparse.ArgumentParser(description="留出集 intake 检查与覆盖度审计")
    ap.add_argument("--check", action="store_true", help="完整 intake 检查")
    ap.add_argument("--coverage", action="store_true", help="只看覆盖度缺口")
    ap.add_argument("--scaffold", action="store_true", help="从文件名生成索引骨架")
    args = ap.parse_args()

    if args.scaffold:
        scaffold()
        return
    if args.coverage:
        coverage_report()
        return
    # 默认等同 --check
    sys.exit(check())


if __name__ == "__main__":
    main()
