"""
sensitivity_check.py — 参数敏感性回归（过拟合哨兵）
====================================================
背景：M1 的成绩里既有 bug 修复/架构机制（不可调参的部分），也有若干
阈值常数（可调参的部分）。为防"在 34 视频上调参 → 报喜给同一批视频"
的过拟合，本脚本对关键阈值做单变量扫描，输出平台宽度。

判读标准：
  - 某参数在 ±40% 扰动下 rep 计数不变 → 平台宽，健壮
  - 成绩悬崖式变化 → 该参数过拟合，需改为自适应推导或收集更多数据

用法: PYTHONPATH=<deps> python3 scripts/sensitivity_check.py [--quick]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from vbtcore.detector import PlateDetector  # noqa: E402
from vbtcore.engine import DetectFitTracker  # noqa: E402
from vbtcore.segment import segment_reps  # noqa: E402

BENCH = REPO / "validation" / "dataset_benchmark" / "raw_videos"
MODEL = str(REPO / "models" / "best.onnx")

# 覆盖不同负荷/机位的代表性子集（quick 模式减半）
VIDEOS = [
    ("30kg_1.03_0.89_0.76_0.65.mp4", 4),
    ("105kg_0.62_0.58_0.53_0.44.mp4", 4),
    ("110kg_0.45_0.48_0.33.mp4", 3),
    ("130kg_0.52_0.59_0.58_0.60_0.51_0.49_0.54_0.43.mp4", 8),
    ("102.5kg_0.51_0.49_0.42_0.30.mp4", 4),
    ("102.5kg_0.53_0.48_0.49_0.31.mp4", 4),
]

SWEEPS = {
    # 纯分段层（用缓存轨迹，快）
    "rom_keep_ratio": {"values": [0.30, 0.45, 0.60], "layer": "segment"},
    # 跟踪层（需重跑，慢）
    "hold_max_frames": {"values": [25, 40, 55], "layer": "track"},
    "max_step_factor": {"values": [0.20, 0.25, 0.35], "layer": "track"},
    # M1.5 regrind（只扫偏离值，省算力；基准 = 默认参数已缓存）
    "regrind_enabled": {"values": [False], "layer": "track"},
    "regrind_arm_disp": {"values": [1.0, 2.0], "layer": "track"},
    "regrind_min_correction": {"values": [0.1, 0.35], "layer": "track"},
}


def main():
    quick = "--quick" in sys.argv
    videos = VIDEOS[:3] if quick else VIDEOS
    det = PlateDetector(MODEL)

    # 缓存默认参数轨迹
    print("缓存默认参数轨迹 ...")
    tracks = {}
    for vid, gt in videos:
        tr = DetectFitTracker(det, redet_every=15)
        y, hs, diag, fps = tr.process(str(BENCH / vid))
        tracks[vid] = (y, hs, fps)

    base_counts = {}
    for vid, gt in videos:
        y, hs, fps = tracks[vid]
        mpp = 0.45 / np.median(hs)
        base_counts[vid] = len(segment_reps(y, fps, mpp).reps)

    for pname, cfg in SWEEPS.items():
        print(f"\n=== {pname} {cfg['values']} ===")
        for val in cfg["values"]:
            row = []
            for vid, gt in videos:
                if cfg["layer"] == "segment":
                    y, hs, fps = tracks[vid]
                    mpp = 0.45 / np.median(hs)
                    n = len(segment_reps(y, fps, mpp, rom_keep_ratio=val).reps)
                else:
                    tr = DetectFitTracker(det, redet_every=15, **{pname: val})
                    y, hs, diag, fps = tr.process(str(BENCH / vid))
                    mpp = 0.45 / np.median(hs)
                    n = len(segment_reps(y, fps, mpp).reps)
                flag = "" if n == base_counts[vid] else " ←变化"
                row.append(f"{vid[:10]}:{n}/{gt}{flag}")
            print(f"  {pname}={val}: " + "  ".join(row))
    print(
        "\n判读: 大多数视频计数在扰动下不变 → 参数平台宽（健壮）；"
        "若有视频悬崖式变化 → 过拟合风险，见 docs/TECH_ROUTE.md 第九节"
    )


if __name__ == "__main__":
    main()
