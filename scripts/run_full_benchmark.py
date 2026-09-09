"""
run_full_benchmark.py — 34 视频基准测试（单模型，可移植、可复现）
=================================================================

本地验证，视频不出本机：脚本只读取本地视频目录，结果写入本地 results/。

用法:
    # 默认：models/barbell_v4.onnx + associator 管线 + 全部视频
    python3 scripts/run_full_benchmark.py

    # 指定模型 / 只跑前 N 个 / 调标定参数
    python3 scripts/run_full_benchmark.py --model yolo11_plate
    python3 scripts/run_full_benchmark.py --limit 5
    python3 scripts/run_full_benchmark.py --scale-factor 1.15   # 与旧结果对比时用
    python3 scripts/run_full_benchmark.py --conf-threshold 0.35

    # 视频在别处：环境变量（推荐）或 --videos-dir
    VBT_VIDEOS_DIR=/path/to/raw_videos python3 scripts/run_full_benchmark.py
    python3 scripts/run_full_benchmark.py --videos-dir /path/to/raw_videos

输出（默认 validation/dataset_benchmark/results/）:
    benchmark_<model>.json   逐视频指标 + diagnostics + 逐 rep 配对
    benchmark_<model>.csv    逐 rep 明细
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# 让 validation/dataset_benchmark 包可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'validation' / 'dataset_benchmark'))

import config as cfg  # noqa: E402
from algorithms.pipelines import pipeline_associator  # noqa: E402
from metrics_evaluator import MetricsEvaluator  # noqa: E402

# 分级通过标准（m/s）。旧代码只用 err<=2.0 一刀切，几乎无意义。
PASS_TIERS = [0.05, 0.10, 0.20, 0.50]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VBT 基准测试（本地，不上传视频）")
    p.add_argument('--videos-dir', type=Path, default=None,
                   help=f'原始视频目录（默认 {cfg.raw_videos_dir()}，可用 VBT_VIDEOS_DIR 覆盖）')
    p.add_argument('--index', type=Path, default=None,
                   help=f'dataset_index.json（默认 {cfg.dataset_index_path()}）')
    p.add_argument('--model', type=str, default=None,
                   help='ONNX 模型：路径，或 barbell_v4 / plate_v1 / yolo11_plate 快捷名')
    p.add_argument('--out-dir', type=Path, default=None,
                   help='结果输出目录（默认 config.output_dir()）')
    p.add_argument('--pipeline', type=str, default='associator',
                   choices=['associator', 'baseline_sg', 'subpixel_spline',
                            'kalman_sg', 'global_smoothing'],
                   help='管线（默认 associator，即生产管线）')
    p.add_argument('--plate-diameter', type=float, default=0.45,
                   help='杠铃片直径（米）。IWF=0.45, IPF=0.4318')
    p.add_argument('--scale-factor', type=float, default=1.0,
                   help='标定修正系数（默认 1.0；旧代码魔数为 1.15）')
    p.add_argument('--conf-threshold', type=float, default=0.25,
                   help='检测器置信度阈值（默认 0.25）')
    p.add_argument('--limit', type=int, default=0,
                   help='只跑前 N 个视频（0 = 全部）')
    return p.parse_args()


def resolve_model(model: str | None) -> Path:
    """快捷名 -> 仓库内路径；否则按给定路径解析。"""
    shortnames = {
        'barbell_v4': cfg.REPO_ROOT / 'models' / 'barbell_v4.onnx',
        'plate_v1':   cfg.REPO_ROOT / 'models' / 'plate_v1.onnx',
        'yolo11_plate': cfg.REPO_ROOT / 'models' / 'yolo11_plate.onnx',
    }
    if model is None:
        return cfg.default_model_path()
    if model in shortnames:
        return shortnames[model]
    return Path(model).expanduser()


def main() -> int:
    args = parse_args()
    model_path = resolve_model(args.model)
    if not model_path.exists():
        print(f"[错误] 模型不存在: {model_path}")
        return 1

    videos_dir = (args.videos_dir or cfg.raw_videos_dir()).resolve()
    index_path = (args.index or cfg.dataset_index_path()).resolve()
    out_dir = (args.out_dir or cfg.output_dir()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not videos_dir.exists():
        print(f"[错误] 视频目录不存在: {videos_dir}")
        print("       把 34 个视频放到该目录，或用 VBT_VIDEOS_DIR / --videos-dir 指定本地路径。")
        return 1

    with open(index_path, encoding='utf-8') as f:
        dataset = json.load(f)
    if args.limit > 0:
        dataset = dataset[:args.limit]

    # 只保留实际存在的视频，并统计缺失
    missing = [it['video_id'] for it in dataset if not (videos_dir / it['video_id']).exists()]
    dataset = [it for it in dataset if (videos_dir / it['video_id']).exists()]
    if missing:
        print(f"[提示] {len(missing)} 个视频缺失（跳过）：")
        for vid in missing[:5]:
            print(f"       - {vid}")
        if len(missing) > 5:
            print(f"       ... 共 {len(missing)} 个")

    print("=" * 72)
    print(f"  模型:   {model_path}")
    print(f"  管线:   {args.pipeline}")
    print(f"  视频:   {len(dataset)} 个（目录 {videos_dir}）")
    print(f"  标定:   plate={args.plate_diameter}m  scale_factor={args.scale_factor}  conf>={args.conf_threshold}")
    print(f"  输出:   {out_dir}")
    print("=" * 72)

    video_results = []
    rows_rep = []
    t0 = time.time()

    for it in dataset:
        vid = it['video_id']
        gt = it.get('gt_reps_mcv', [])
        vp = videos_dir / vid

        try:
            result = pipeline_associator(
                str(vp),
                plate_diameter_m=args.plate_diameter,
                scale_factor=args.scale_factor,
                conf_threshold=args.conf_threshold,
                model_path=str(model_path),
                verbose=False,
            )
            reps = result.get('reps', [])
            diag = result.get('diagnostics', {})
            ev = MetricsEvaluator.evaluate_video(vid, gt, reps, strategy='truncate')
            ev_best = MetricsEvaluator.evaluate_video(vid, gt, reps, strategy='best_pair')
            scale = diag.get('scale', 0.0)
        except Exception as e:
            print(f"  ✗ {vid}  ERROR: {e}")
            video_results.append({'video_id': vid, 'error': str(e)})
            continue

        n_pair = ev.n_paired
        print(f"  {'✓' if n_pair else '✗'} {vid:<44} "
              f"GT={len(gt):>2} pred={ev.n_pred:>2} "
              f"RMSE={ev.rmse:6.3f} bias={ev.bias:+6.3f} scale={scale:<10} "
              f"cov={diag.get('coverage', 0.0)*100:3.0f}%")

        video_results.append({
            'video_id': vid,
            'load_kg': it.get('load_kg'),
            'n_gt': len(gt), 'n_pred': ev.n_pred,
            'rmse_truncate': None if np.isnan(ev.rmse) else ev.rmse,
            'bias_truncate': None if np.isnan(ev.bias) else ev.bias,
            'rmse_best_pair': None if np.isnan(ev_best.rmse) else ev_best.rmse,
            'paired': [{'gt': r.gt_mcv, 'pred': r.pred_mcv, 'error': r.error}
                       for r in ev.paired],
            'diagnostics': diag,
        })

        for i, r in enumerate(ev.paired):
            rows_rep.append({
                'video_id': vid,
                'load_kg': it.get('load_kg'),
                'rep_idx': i + 1,
                'gt_mcv': r.gt_mcv,
                'pred_mcv': r.pred_mcv,
                'error': r.error,
            })

    elapsed = time.time() - t0

    # ── 汇总 ────────────────────────────────────────────────
    errs = [v['rmse_truncate'] for v in video_results
            if v.get('rmse_truncate') is not None]
    valid = len(errs)
    print()
    print("=" * 72)
    print(f"  汇总（{elapsed:.1f}s）")
    print("=" * 72)
    if valid == 0:
        print("  没有任何可评估视频（无预测或无视频）。")
        return 1

    errs = np.array(errs)
    print(f"  可评估视频: {valid}/{len(video_results)}")
    print(f"  RMSE mean   = {errs.mean():.4f} m/s")
    print(f"  RMSE median = {np.median(errs):.4f} m/s")
    print(f"  全部配对 rep 数: {len(rows_rep)}")
    print()
    for tier in PASS_TIERS:
        n_pass = int((errs <= tier).sum())
        print(f"  通过率 (视频级 RMSE ≤ {tier:.2f}): {n_pass}/{valid} ({n_pass/valid*100:.0f}%)")

    # rep 数量完全匹配的视频
    exact = sum(1 for v in video_results
                if v.get('n_gt') == v.get('n_pred'))
    print(f"  rep 数量完全匹配: {exact}/{len(video_results)}")

    # 最差视频
    worst = sorted([v for v in video_results if v.get('rmse_truncate') is not None],
                   key=lambda v: -v['rmse_truncate'])[:5]
    if worst:
        print("\n  最差视频（优先后续诊断）:")
        for v in worst:
            print(f"    {v['video_id']:<44} RMSE={v['rmse_truncate']:.4f} "
                  f"(GT={v['n_gt']}, pred={v['n_pred']}, "
                  f"best_pair={v['rmse_best_pair']:.4f})")

    # ── 导出 ────────────────────────────────────────────────
    tag = model_path.stem
    meta = {
        'model': str(model_path),
        'pipeline': args.pipeline,
        'plate_diameter_m': args.plate_diameter,
        'scale_factor': args.scale_factor,
        'conf_threshold': args.conf_threshold,
        'videos_dir': str(videos_dir),
        'n_videos': len(dataset),
        'elapsed_s': round(elapsed, 1),
        'rmse_mean': float(errs.mean()),
        'rmse_median': float(np.median(errs)),
        'pass_tiers': {f'<={t}': int((errs <= t).sum()) for t in PASS_TIERS},
        'videos': video_results,
    }
    json_path = out_dir / f'benchmark_{tag}.json'
    csv_path = out_dir / f'benchmark_{tag}.csv'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if rows_rep:
        import pandas as pd
        pd.DataFrame(rows_rep).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n  已保存: {json_path}")
    if rows_rep:
        print(f"  已保存: {csv_path}")
    print("\n  提示: 逐视频错误定位请运行 scripts/diagnose_benchmark.py")
    return 0


if __name__ == '__main__':
    sys.exit(main())
