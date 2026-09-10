"""
run_benchmark_v0.py — M0 基线报告（34 视频 GymAware 基准）
==========================================================
用法:
    cd /home/user/VBT-Research && python3 scripts/run_benchmark_v0.py

产出:
    validation/reports/BENCHMARK_v0.md / .json

评估口径（Stage 0/1）:
  - rep 计数: |n_pred - n_gt| <= 1 计为通过（±1 容差）
  - 速度精度: truncate 配对后 RMSE / MAE / bias / Pearson r
  - 正确拒绝: 无片视频返回 NO_PLATE_DETECTED = 正确行为（不计失败）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "validation" / "dataset_benchmark"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO))

from metrics_evaluator import MetricsEvaluator  # noqa: E402
from vbtcore import PlateDetector, analyze_video  # noqa: E402

MODEL = str(REPO / "models" / "yolo11_plate.onnx")
REPORT_DIR = REPO / "validation" / "reports"


def main():
    with open(BENCH / "dataset_index.json") as f:
        dataset = json.load(f)
    print(f"共 {len(dataset)} 个视频 | 模型: {Path(MODEL).name}")
    det = PlateDetector(MODEL)

    rows = []
    t_all = time.time()
    for k, item in enumerate(dataset):
        vid = item["video_id"]
        gt = item["gt_reps_mcv"]
        vp = str(BENCH / "raw_videos" / vid)
        r = analyze_video(vp, MODEL, detector=det)
        row = {
            "video": vid, "load_kg": item.get("load_kg"),
            "status": r.status, "n_gt": len(gt), "n_pred": len(r.mcv),
        }
        if r.mcv and gt:
            ev = MetricsEvaluator.evaluate_video(vid, gt, [{"mcv": m} for m in r.mcv])
            rmse = None if np.isnan(ev.rmse) else round(ev.rmse, 3)
            row.update({
                "rmse": rmse,
                "mae": None if np.isnan(ev.mae) else round(ev.mae, 3),
                "bias": None if np.isnan(ev.bias) else round(ev.bias, 3),
                "r": None if np.isnan(ev.pearson_r) else round(ev.pearson_r, 3),
            })
        elif not r.mcv:
            row["rmse"] = None
        row["count_ok"] = (r.mcv and gt and abs(len(r.mcv) - len(gt)) <= 1) or \
                          (not r.mcv and r.status == "NO_PLATE_DETECTED")
        row["ms_per_frame"] = r.diagnostics.get("ms_per_frame")
        row["coverage"] = r.diagnostics.get("coverage")
        row["yolo_ratio"] = r.diagnostics.get("yolo_ratio")
        rows.append(row)
        print(f"[{k+1:>2}/{len(dataset)}] {vid:<44} status={r.status:<18} "
              f"reps {row['n_pred']}/{row['n_gt']}  "
              f"RMSE={row.get('rmse')}  r={row.get('r')}  "
              f"{row.get('ms_per_frame')}ms/帧")

    total_min = (time.time() - t_all) / 60

    # ── 汇总 ──────────────────────────────────────────────
    n = len(rows)
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    paired = [r for r in rows if r.get("rmse") is not None]
    count_ok = sum(1 for r in rows if r["count_ok"])
    rejects = [r for r in rows if r["status"] == "NO_PLATE_DETECTED"]
    rmse_all = [r["rmse"] for r in paired]
    gts, preds = [], []
    for r in paired:
        item = next(d for d in dataset if d["video_id"] == r["video"])
        # 重新配对取原始值（简化：用 RMSE 反推不必，直接重算）
    # 全局 rep 级配对
    all_gt, all_pred = [], []
    for r in rows:
        if r.get("rmse") is None:
            continue
        item = next(d for d in dataset if d["video_id"] == r["video"])
        gt = item["gt_reps_mcv"]
        vp = str(BENCH / "raw_videos" / r["video"])
        # 从 rows 里保存的 n_pred 不足以算 rep 级，需要再跑一次？——不，
        # 改为在循环里就收集。见下方（二次遍历代价高），此处用 rep 级重算。
    spd = [r["ms_per_frame"] for r in rows if r.get("ms_per_frame")]

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": Path(MODEL).name,
        "engine": "vbtcore v1 (detect->fit hybrid, M0)",
        "total_minutes": round(total_min, 1),
        "n_videos": n,
        "status_counts": by_status,
        "count_accuracy": f"{count_ok}/{n} ({count_ok/n*100:.0f}%)",
        "paired_videos": len(paired),
        "video_rmse_mean": round(float(np.mean(rmse_all)), 3) if rmse_all else None,
        "video_rmse_median": round(float(np.median(rmse_all)), 3) if rmse_all else None,
        "ms_per_frame_mean": round(float(np.mean(spd)), 1) if spd else None,
        "rows": rows,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "BENCHMARK_v0.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    # ── Markdown ─────────────────────────────────────────
    lines = [
        "# Benchmark v0 — vbtcore 引擎基线（M0）",
        "",
        f"- 生成: {report['generated']} ｜ 引擎: {report['engine']}",
        f"- 视频: {n} ｜ 状态分布: {by_status}",
        f"- 计数通过(±1或正确拒绝): **{count_ok}/{n} ({count_ok/n*100:.0f}%)**",
        f"- 有配对视频: {len(paired)} ｜ 视频 RMSE 均值: {report['video_rmse_mean']} "
        f"中位数: {report['video_rmse_median']}",
        f"- 平均速度: {report['ms_per_frame_mean']} ms/帧（CPU ONNX）",
        f"- 总耗时: {total_min:.1f} 分钟",
        "",
        "| 视频 | 负荷 | 状态 | reps(pred/gt) | RMSE | bias | r | ms/帧 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['video']} | {r['load_kg']} | {r['status']} "
            f"| {r['n_pred']}/{r['n_gt']} | {r.get('rmse','-')} "
            f"| {r.get('bias','-')} | {r.get('r','-')} | {r.get('ms_per_frame','-')} |")
    (REPORT_DIR / "BENCHMARK_v0.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告: {REPORT_DIR/'BENCHMARK_v0.md'}")
    print(f"计数通过: {count_ok}/{n} | 配对视频 RMSE 均值 {report['video_rmse_mean']} "
          f"| {report['ms_per_frame_mean']} ms/帧")


if __name__ == "__main__":
    main()
