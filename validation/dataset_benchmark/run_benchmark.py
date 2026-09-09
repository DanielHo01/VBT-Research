"""
run_benchmark.py
主入口：遍历 dataset_index.json，对比多组算法管线，
      输出终端表格 + CSV + Bland-Altman 图 + 1:1 散点图。

用法（视频留在本地，不上传）：
    python run_benchmark.py                          # 仓库内相对路径
    python run_benchmark.py --model yolo11_plate
    python run_benchmark.py --videos-dir /path/to/raw_videos
    python run_benchmark.py --limit 5 --no-plot

依赖：
    pip install opencv-python onnxruntime numpy scipy pandas matplotlib
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 确保 algorithms 包可导入
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from algorithms.pipelines import run_pipeline
from metrics_evaluator import MetricsEvaluator, aggregate_results


# ── 配置 ────────────────────────────────────────────────────

RAW_VIDEOS = cfg.raw_videos_dir()
INDEX_FILE = cfg.dataset_index_path()
OUTPUT_DIR = cfg.output_dir()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PIPELINES = [
    ("baseline_sg",       "Baseline SG (15×3)"),
    ("subpixel_spline",   "Subpixel Spline"),
    ("kalman_sg",         "Kalman + SG"),
    ("global_smoothing",  "Global Smoothing"),
    ("associator",        "Associator + α-β + SG (生产管线)"),
]

ALGO_KEYS = [p[0] for p in PIPELINES]


# ── 主流程 ──────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多管线基准对比（本地）")
    p.add_argument('--videos-dir', type=Path, default=None)
    p.add_argument('--index', type=Path, default=None)
    p.add_argument('--out-dir', type=Path, default=None)
    p.add_argument('--model', type=str, default=None,
                   help='ONNX 模型：路径或 barbell_v4/plate_v1/yolo11_plate')
    p.add_argument('--plate-diameter', type=float, default=0.45)
    p.add_argument('--scale-factor', type=float, default=1.0)
    p.add_argument('--conf-threshold', type=float, default=0.25)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--no-plot', action='store_true', help='跳过绘图')
    return p.parse_args()


def main():
    args = parse_args()
    videos_dir = (args.videos_dir or RAW_VIDEOS).resolve()
    index_file = (args.index or INDEX_FILE).resolve()
    out_dir = (args.out_dir or OUTPUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        short = {
            'barbell_v4': cfg.REPO_ROOT / 'models' / 'barbell_v4.onnx',
            'plate_v1': cfg.REPO_ROOT / 'models' / 'plate_v1.onnx',
            'yolo11_plate': cfg.REPO_ROOT / 'models' / 'yolo11_plate.onnx',
        }
        model_path = short.get(args.model, Path(args.model).expanduser())
    else:
        model_path = cfg.default_model_path()

    if not model_path.exists():
        print(f"[错误] 模型不存在: {model_path}")
        return 1
    if not videos_dir.exists():
        print(f"[错误] 视频目录不存在: {videos_dir}")
        return 1

    print("=" * 72)
    print("  EasyVBT 算法基准验证  vs  GymAware Ground Truth")
    print("=" * 72)

    # 加载数据集
    with open(index_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if args.limit > 0:
        dataset = dataset[:args.limit]

    print(f"\n📋 数据集: {len(dataset)} 个视频")
    print(f"   视频目录: {videos_dir}")
    print(f"   模型:     {model_path}")

    # 预热 ONNX 模型（首次加载较慢）
    print("\n🧠 预热 ONNX 模型 ...")
    from algorithms.common import YoloPlateDetector
    _ = YoloPlateDetector(str(model_path))
    print("   模型加载完成\n")

    # ── 逐视频评估 ──────────────────────────────────────────
    # results[algo_key][video_id] = VideoEvalResult
    results: dict[str, dict] = {k: {} for k in ALGO_KEYS}

    for item in dataset:
        video_id    = item["video_id"]
        gt_mcvs     = item["gt_reps_mcv"]
        video_path  = videos_dir / video_id

        if not video_path.exists():
            print(f"  ⏭ 跳过不存在: {video_id}")
            continue

        print(f"\n▶  {video_id}  (GT reps: {len(gt_mcvs)})")

        for algo_key, algo_label in PIPELINES:
            t0 = time.time()
            try:
                pred_reps = run_pipeline(
                    video_path, algo_type=algo_key,
                    model_path=str(model_path),
                    scale_factor=args.scale_factor,
                    conf_threshold=args.conf_threshold,
                )
            except Exception as e:
                print(f"    [{algo_key}] ERROR: {e}")
                pred_reps = []

            eval_res = MetricsEvaluator.evaluate_video(
                video_id=video_id,
                gt_mcvs=gt_mcvs,
                pred_reps=pred_reps,
                strategy="truncate",
            )
            results[algo_key][video_id] = eval_res

            elapsed = time.time() - t0
            if eval_res.n_paired > 0:
                rmse_s = f"{eval_res.rmse:.4f}"
                bias_s = f"{eval_res.bias:+.4f}"
                r_s    = f"{eval_res.pearson_r:.3f}"
                icc_s  = f"{eval_res.icc:.3f}"
                flag = "  ✅" if eval_res.rmse < 0.063 else "  ⚠️"
                print(f"    [{algo_key:<18}] "
                      f"Reps: {eval_res.n_pred:>2}/{len(gt_mcvs):>2}  "
                      f"RMSE={rmse_s}  Bias={bias_s}  r={r_s}  ICC={icc_s}  "
                      f"{flag} ({elapsed:.1f}s)")
            else:
                print(f"    [{algo_key:<18}] ❌ 无有效Reps ({elapsed:.1f}s)")

    # ── 全局聚合 ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  全局聚合结果")
    print("=" * 72)

    agg_results: dict[str, dict] = {}
    for algo_key in ALGO_KEYS:
        video_results = list(results[algo_key].values())
        agg = aggregate_results(video_results)
        agg_results[algo_key] = agg

    # 终端表格
    header = (f"  {'算法':<22} | {'RMSE':>8} | {'MAE':>8} | {'Bias':>8} "
              "| {'Pearson r':>10} | {'ICC':>7} | {'LoA 上':>8} | {'LoA 下':>8}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    best_algo   = None
    best_algo_k = None
    best_rmse   = 999.0

    for algo_key, algo_label in PIPELINES:
        agg = agg_results.get(algo_key, {})
        if not agg:
            continue

        rmse   = agg['rmse']
        mae    = agg['mae']
        bias   = agg['bias']
        r      = agg['pearson_r']
        icc    = agg.get('icc', 0.0) or 0.0
        loa_up = agg['loa_upper']
        loa_lo = agg['loa_lower']
        n_rep  = agg['total_reps']
        n_vid  = agg['total_videos']

        # 计算 ICC 单独汇总（跨视频）
        all_gt   = np.array(agg['all_gt'])
        all_pred = np.array(agg['all_pred'])
        icc_total = MetricsEvaluator.compute_icc(all_gt, all_pred)

        row = (f"  {algo_label:<22} | {rmse:>8.4f} | {mae:>8.4f} | {bias:>+8.4f} "
               f"| {r:>10.4f} | {icc_total:>7.4f} | {loa_up:>+8.4f} | {loa_lo:>+8.4f}")
        print(row)

        if rmse < best_rmse:
            best_rmse   = rmse
            best_algo   = algo_label
            best_algo_k = algo_key

    print("=" * 72)
    target = 0.063
    if best_rmse <= target:
        print(f"  ⭐ WINNER: [{best_algo}]  RMSE={best_rmse:.4f} m/s  ✅ < {target} 达标！")
    else:
        print(f"  ⭐ WINNER: [{best_algo}]  RMSE={best_rmse:.4f} m/s  ⚠️  距 {target} 仍有差距")
    print("=" * 72)

    # ── 导出 CSV ─────────────────────────────────────────────
    rows = []
    for item in dataset:
        video_id = item["video_id"]
        gt_mcvs  = item["gt_reps_mcv"]
        for algo_key, algo_label in PIPELINES:
            vr = results[algo_key].get(video_id)
            if vr is None:
                continue
            for rep in vr.paired:
                rows.append({
                    "video_id":     video_id,
                    "load_kg":      item["load_kg"],
                    "algorithm":    algo_label,
                    "gt_mcv":       rep.gt_mcv,
                    "pred_mcv":     rep.pred_mcv,
                    "error":        rep.error,
                    "video_rmse":   vr.rmse,
                    "video_bias":   vr.bias,
                })

    if rows:
        df = pd.DataFrame(rows)
        csv_path = out_dir / "per_rep_results.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n📄 CSV 导出: {csv_path}")

    # ── 图表绘制 ─────────────────────────────────────────────
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.lines import Line2D
            plot_results(agg_results, best_algo, best_rmse, out_dir)
        except Exception as e:
            print(f"  ⚠ 绘图失败（跳过）: {e}")

    print(f"\n✅ 基准验证完成！结果目录: {out_dir}")
    # 兼容 __main__ 下旧接口调用
    return best_algo_k, best_rmse


# ── 绘图 ────────────────────────────────────────────────────

def plot_results(agg_results: dict, best_algo: str, best_rmse: float,
                 out_dir: Path | None = None):
    """绘制 1:1 散点图 + Bland-Altman 图"""
    if out_dir is None:
        out_dir = OUTPUT_DIR

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches  # noqa: F401
    from matplotlib.lines import Line2D  # noqa: F401

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"EasyVBT vs GymAware — Benchmark Results  (Best: {best_algo}, RMSE={best_rmse:.4f} m/s)",
                 fontsize=14, fontweight="bold")

    # ── 左：1:1 散点图 ───────────────────────────────────
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(ALGO_KEYS)))
    algo_labels = [p[1] for p in PIPELINES]

    for i, algo_key in enumerate(ALGO_KEYS):
        agg = agg_results.get(algo_key, {})
        if not agg or len(agg.get('all_gt', [])) == 0:
            continue
        gts   = np.array(agg['all_gt'])
        preds = np.array(agg['all_pred'])
        ax1.scatter(gts, preds, label=f"{algo_labels[i]}  RMSE={agg['rmse']:.3f}",
                    alpha=0.65, s=40, color=colors[i])

    # 1:1 理想线
    all_gts = np.concatenate([np.array(agg_results[k]['all_gt'])
                               for k in ALGO_KEYS if agg_results.get(k)])
    all_preds_max = max(np.concatenate([np.array(agg_results[k]['all_pred'])
                                         for k in ALGO_KEYS if agg_results.get(k)]).max(), 1.5)
    lims = [0, min(all_preds_max * 1.1, 2.0)]
    ax1.plot(lims, lims, 'k--', lw=1.5, label='Ideal 1:1')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel("GymAware MCV (m/s)", fontsize=11)
    ax1.set_ylabel("EasyVBT Predicted MCV (m/s)", fontsize=11)
    ax1.set_title("1:1 Scatter Plot", fontsize=12)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # ── 右：Bland-Altman ───────────────────────────────────
    ax2 = axes[1]
    for i, algo_key in enumerate(ALGO_KEYS):
        agg = agg_results.get(algo_key, {})
        if not agg or len(agg.get('all_gt', [])) == 0:
            continue
        gts    = np.array(agg['all_gt'])
        preds  = np.array(agg['all_pred'])
        means  = (gts + preds) / 2.0
        diffs  = preds - gts
        bias   = np.mean(diffs)
        sd     = np.std(diffs, ddof=1)
        loa_up = bias + 1.96 * sd
        loa_lo = bias - 1.96 * sd

        ax2.scatter(means, diffs, label=f"{algo_labels[i]} (bias={bias:+.3f})",
                     alpha=0.65, s=40, color=colors[i])

        # BA 参考线
        ax2.axhline(bias,    color=colors[i], lw=1.2, ls='-')
        ax2.axhline(loa_up,  color=colors[i], lw=0.8, ls='--', alpha=0.7)
        ax2.axhline(loa_lo,  color=colors[i], lw=0.8, ls='--', alpha=0.7)

    ax2.axhline(0, color='black', lw=1.5, ls='-', label='Zero bias')
    ax2.set_xlabel("Mean of GymAware & EasyVBT (m/s)", fontsize=11)
    ax2.set_ylabel("Difference (EasyVBT − GymAware) (m/s)", fontsize=11)
    ax2.set_title("Bland-Altman Plot", fontsize=12)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    scatter_path = out_dir / "benchmark_results.png"
    ba_path      = out_dir / "bland_altman.png"
    fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
    ax2.figure.savefig(ba_path, dpi=300, bbox_inches="tight")

    print(f"📊 图表保存:")
    print(f"   {scatter_path}")
    print(f"   {ba_path}")


if __name__ == "__main__":
    main()
