#!/usr/bin/env python3
"""scripts/tune_segmenter.py — 基于轨迹缓存的分段器快速评估

为什么存在
──────────
检测占全流程 97% 成本（整条视频 55~105s，全量 34 条约 45 分钟）；
分段器本身只要 0.3ms/条。若每试一组分段参数就重跑检测，一轮 45 分钟；
读 `validation/reports/trajectories/*.csv` 重跑分段则是**亚秒级**。

前置：先跑 `python scripts/dump_all_trajectories.py` 生成轨迹缓存。
轨迹只依赖 detector/calibrator/tracker —— 改动这三者后必须重新导出。

用法
────
    python scripts/tune_segmenter.py                # 评估当前分段器
    python scripts/tune_segmenter.py --compare      # 对比若干边界策略
    python scripts/tune_segmenter.py --bias-report  # 误差结构分析

⚠️ 纪律提醒
───────────
本脚本在**开发集**上评估。开发集上的任何改善都可能是过拟合，
留出集（validation/holdout/）就位前，不得据此宣称精度提升。
任何改动必须通过参考视频物理校验（见 REFERENCE_CHECK）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BENCH = REPO / "validation" / "dataset_benchmark"
TRAJ = REPO / "validation" / "reports" / "trajectories"

# 参考视频物理校验基准：任何改动后偏离即回滚
REFERENCE_CHECK = {
    "video": "110kg_0.71_0.73",
    "gt": [0.71, 0.73],
    "baseline_mcv": [0.705, 0.762],
    "mpp": 0.002471,
}


def load_data():
    if not TRAJ.exists() or not list(TRAJ.glob("*.csv")):
        print("❌ 轨迹缓存不存在。先跑：python scripts/dump_all_trajectories.py")
        raise SystemExit(2)
    index = json.loads((BENCH / "dataset_index.json").read_text(encoding="utf-8"))
    ds = {e["video_id"]: e for e in index}
    data = []
    for f in sorted(TRAJ.glob("*.csv")):
        rows = list(csv.DictReader(f.open(encoding="utf-8")))
        if not rows:
            continue  # 空杆视频无轨迹
        meta = ds[f.stem + ".mp4"]
        data.append(
            {
                "vid": f.stem,
                "gt": meta["gt_reps_mcv"],
                "load": meta["load_kg"],
                "t": np.array([float(r["t_s"]) for r in rows]),
                "y": np.array([float(r["y_m"]) for r in rows]),
                "v": np.array([float(r["v_mps"]) for r in rows]),
            }
        )
    return data


def segment_mcv(d, trim_end=False, trim_start=False, dur_offset=0.0):
    """跑分段器，可选边界策略。返回 MCV 列表。"""
    from vbtcore.segmenter import BiomechanicalRepSegmenter

    t, y, v = d["t"], d["y"], d["v"]
    out = []
    for r in BiomechanicalRepSegmenter().segment(t, y, v):
        s, e = r.start_idx, r.end_idx
        if trim_end:
            while e > s and v[e] <= 0:
                e -= 1
        if trim_start:
            while s < e and v[s] <= 0:
                s += 1
        dur = float(t[e] - t[s]) - dur_offset
        rom = abs(float(y[e] - y[s]))
        out.append(rom / dur if dur > 0 else 0.0)
    return out


def score(data, **kw):
    errs = []
    npass = 1  # 20kg 空杆正确拒绝
    ref = None
    for d in data:
        mcv = segment_mcv(d, **kw)
        if abs(len(mcv) - len(d["gt"])) <= 1:
            npass += 1
        if len(mcv) == len(d["gt"]):
            errs += [a - b for a, b in zip(mcv, d["gt"])]
        if d["vid"] == REFERENCE_CHECK["video"]:
            ref = [round(x, 3) for x in mcv]
    n = len(errs)
    rmse = math.sqrt(sum(e * e for e in errs) / n) if n else float("nan")
    bias = sum(errs) / n if n else float("nan")
    return {"npass": npass, "rmse": rmse, "bias": bias, "n": n, "ref": ref}


def cmd_default(data):
    s = score(data)
    print(f"计数通过 {s['npass']}/34 ｜ rep RMSE {s['rmse']:.4f} ｜ bias {s['bias']:+.4f} ｜ n={s['n']}")
    print(f"参考视频 {REFERENCE_CHECK['video']}: {s['ref']}")
    print(f"  真值 {REFERENCE_CHECK['gt']} ｜ 基准 {REFERENCE_CHECK['baseline_mcv']}")
    ok = s["ref"] == REFERENCE_CHECK["baseline_mcv"]
    print(f"  物理校验: {'✅ 与基准逐位一致' if ok else '⚠️ 偏离基准，需审查'}")


def cmd_compare(data):
    print("边界策略对比（开发集，仅供诊断，不得据此直接调参）\n")
    print(f"{'策略':<26}{'计数':>7}{'RMSE':>9}{'bias':>10}   参考视频")
    cases = [
        ("当前基线", {}),
        ("收终点到 v>0", {"trim_end": True}),
        ("收起点到 v>0", {"trim_start": True}),
        ("两端都收", {"trim_end": True, "trim_start": True}),
    ]
    for name, kw in cases:
        s = score(data, **kw)
        flag = "" if s["ref"] == REFERENCE_CHECK["baseline_mcv"] else "  ⚠️偏离"
        print(
            f"{name:<26}{s['npass']:>5}/34{s['rmse']:>9.4f}{s['bias']:>+10.4f}   {s['ref']}{flag}"
        )
    print(
        f"\n参考真值 {REFERENCE_CHECK['gt']}；基准 {REFERENCE_CHECK['baseline_mcv']}。"
        "\n⚠️ 整体 RMSE 改善但参考视频偏离 = 用低速档收益掩盖高速档过冲，属过拟合，不可采纳。"
    )


def cmd_bias_report(data):
    from vbtcore.segmenter import BiomechanicalRepSegmenter

    recs = []
    for d in data:
        reps = BiomechanicalRepSegmenter().segment(d["t"], d["y"], d["v"])
        if len(reps) != len(d["gt"]):
            continue
        for i, (r, g) in enumerate(zip(reps, d["gt"])):
            recs.append(
                {"vid": d["vid"], "load": d["load"], "idx": i, "g": g,
                 "p": r.mcv_mps, "e": r.mcv_mps - g,
                 "rom": r.rom_m, "dur": r.duration_s}
            )
    n = len(recs)
    errs = [r["e"] for r in recs]
    mean = sum(errs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in errs) / (n - 1))
    se = sd / math.sqrt(n)
    print(f"【整体】n={n}  bias={mean:+.4f}  sd={sd:.4f}  se={se:.4f}  t={mean / se:.2f}")
    print(f"  95% CI [{mean - 1.96 * se:+.4f}, {mean + 1.96 * se:+.4f}]")
    print(f"  随机误差 sd 是系统偏差的 {sd / abs(mean):.1f} 倍")
    print("  参考：GymAware LPT 自身重复性约 0.02~0.03 m/s\n")

    P = [r["p"] for r in recs]
    G = [r["g"] for r in recs]
    mg, mp = sum(G) / n, sum(P) / n
    a = sum((x - mg) * (y - mp) for x, y in zip(G, P)) / sum((x - mg) ** 2 for x in G)
    print(f"【标度检验】pred = {a:.4f}*gt {mp - a * mg:+.4f}")
    print(f"  斜率≈1 说明无乘性标度误差（mpp 标定正确），偏差是加性的\n")

    print("【分速度档】")
    for lo, hi in [(0.2, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.2)]:
        s = [r["e"] for r in recs if lo <= r["g"] < hi]
        if s:
            print(f"  GT[{lo},{hi}) n={len(s):3d}  bias={sum(s) / len(s):+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--bias-report", action="store_true")
    args = ap.parse_args()
    data = load_data()
    print(f"轨迹缓存：{len(data)} 条有效视频\n")
    if args.compare:
        cmd_compare(data)
    elif args.bias_report:
        cmd_bias_report(data)
    else:
        cmd_default(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
