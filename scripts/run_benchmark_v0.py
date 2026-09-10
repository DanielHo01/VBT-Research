"""
run_benchmark_v0.py — M0 基线报告（34 视频 GymAware 基准）
==========================================================
用法:
    cd /home/user/VBT-Research && python3 scripts/run_benchmark_v0.py [--tag v1] [--only 50kg,105kg] [--engine LABEL]

产出:
    validation/reports/BENCHMARK_{tag}.md / .json（默认 tag=v0，保持 M0 行为）

评估口径（Stage 0/1）:
  - rep 计数: |n_pred - n_gt| <= 1 计为通过（±1 容差）
  - 速度精度: truncate 配对后 RMSE / MAE / bias / Pearson r
  - 正确拒绝: 无片视频返回 NO_PLATE_DETECTED = 正确行为（不计失败）
"""
from __future__ import annotations

import argparse
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

# 已知"只有杆"视频（无 45cm 片）——NO_PLATE_DETECTED = 正确拒绝
BAR_ONLY = {"20kg_0.87_0.88_0.89_0.91.mp4"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v0",
                    help="报告后缀：输出 BENCHMARK_{tag}.md/json")
    ap.add_argument("--only", default=None,
                    help="只跑文件名包含任一子串的视频（逗号分隔），如 --only 50kg,105kg_0.69")
    ap.add_argument("--engine", default="vbtcore v1 (detect->fit hybrid, M0)",
                    help="报告中的引擎标识")
    ap.add_argument("--no-regrind", action="store_true",
                    help="关闭 M1.5 底部重锚定（烧蚀实验）")
    args = ap.parse_args()

    with open(BENCH / "dataset_index.json") as f:
        dataset = json.load(f)
    if args.only:
        subs = [s.strip() for s in args.only.split(",") if s.strip()]
        dataset = [d for d in dataset
                   if any(s in d["video_id"] for s in subs)]
        print(f"--only {subs} → {len(dataset)} 个视频")
    print(f"共 {len(dataset)} 个视频 | 模型: {Path(MODEL).name} | tag={args.tag}")
    det = PlateDetector(MODEL)

    rows = []
    t_all = time.time()
    for k, item in enumerate(dataset):
        vid = item["video_id"]
        gt = item["gt_reps_mcv"]
        vp = str(BENCH / "raw_videos" / vid)
        r = analyze_video(vp, MODEL, detector=det,
                          regrind_enabled=not args.no_regrind)
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
        is_bar_only = vid in BAR_ONLY
        row["bar_only"] = is_bar_only
        if r.mcv and gt:
            row["count_ok"] = abs(len(r.mcv) - len(gt)) <= 1
        elif not r.mcv and r.status == "NO_PLATE_DETECTED":
            row["count_ok"] = is_bar_only          # 只有杆=正确拒绝；有片却拒=假拒绝
            row["false_reject"] = not is_bar_only
        else:
            row["count_ok"] = False
        row["ms_per_frame"] = r.diagnostics.get("ms_per_frame")
        row["coverage"] = r.diagnostics.get("coverage")
        row["yolo_ratio"] = r.diagnostics.get("yolo_ratio")
        row["regrind"] = [r.diagnostics.get("n_regrind_snap", 0),
                          r.diagnostics.get("n_regrind_micro", 0),
                          r.diagnostics.get("n_regrind_reject", 0)]
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
    rg_tot = [sum(r["regrind"][i] for r in rows) for i in range(3)]

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": Path(MODEL).name,
        "engine": args.engine,
        "tag": args.tag,
        "only": args.only,
        "regrind": not args.no_regrind,
        "total_minutes": round(total_min, 1),
        "n_videos": n,
        "status_counts": by_status,
        "count_accuracy": f"{count_ok}/{n} ({count_ok/n*100:.0f}%)",
        "paired_videos": len(paired),
        "video_rmse_mean": round(float(np.mean(rmse_all)), 3) if rmse_all else None,
        "video_rmse_median": round(float(np.median(rmse_all)), 3) if rmse_all else None,
        "ms_per_frame_mean": round(float(np.mean(spd)), 1) if spd else None,
        "regrind_totals": {"snap": rg_tot[0], "micro": rg_tot[1],
                           "reject": rg_tot[2]},
        "rows": rows,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"BENCHMARK_{args.tag}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    # ── Markdown ─────────────────────────────────────────
    lines = [
        f"# Benchmark {args.tag} — {args.engine}",
        "",
        f"- 生成: {report['generated']} ｜ 引擎: {report['engine']}",
        f"- 视频: {n} ｜ 状态分布: {by_status}",
        f"- 计数通过(±1或正确拒绝): **{count_ok}/{n} ({count_ok/n*100:.0f}%)**",
        f"- 假拒绝（有片却 NO_PLATE）: {sum(1 for r in rows if r.get('false_reject'))} 条 "
        f"（检测器域差，M3 数据闭环目标）",
        f"- 有配对视频: {len(paired)} ｜ 视频 RMSE 均值: {report['video_rmse_mean']} "
        f"中位数: {report['video_rmse_median']}",
        f"- 平均速度: {report['ms_per_frame_mean']} ms/帧（CPU ONNX）",
        f"- 总耗时: {total_min:.1f} 分钟",
        f"- regrind 触发统计(snap/微偏/拒绝): "
        f"{rg_tot[0]}/{rg_tot[1]}/{rg_tot[2]}",
        "",
        "| 视频 | 负荷 | 状态 | reps(pred/gt) | RMSE | bias | r | ms/帧 | regrind |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['video']} | {r['load_kg']} | {r['status']} "
            f"| {r['n_pred']}/{r['n_gt']} | {r.get('rmse','-')} "
            f"| {r.get('bias','-')} | {r.get('r','-')} | {r.get('ms_per_frame','-')} "
            f"| {'/'.join(str(x) for x in r['regrind'])} |")
    (REPORT_DIR / f"BENCHMARK_{args.tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告: {REPORT_DIR/f'BENCHMARK_{args.tag}.md'}")
    print(f"计数通过: {count_ok}/{n} | 配对视频 RMSE 均值 {report['video_rmse_mean']} "
          f"| {report['ms_per_frame_mean']} ms/帧")


if __name__ == "__main__":
    main()
