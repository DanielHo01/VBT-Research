"""
diagnose_benchmark.py — 逐视频错误定位（找到 RMSE 大的原因）
=============================================================

对每个视频输出：
  - 逐 rep 配对（GT vs 预测 vs 误差）
  - 检测质量（置信度分布、覆盖率）
  - 标定质量（scale 稳定性）
  - 错误来源标记（rep 数量不匹配 / 配对错位 / 标定不稳 / 检测弱）

用法:
    python3 scripts/diagnose_benchmark.py                # 全部视频
    python3 scripts/diagnose_benchmark.py --limit 5
    python3 scripts/diagnose_benchmark.py --model yolo11_plate
    python3 scripts/diagnose_benchmark.py --videos-dir /path/to/raw_videos

输出（validation/dataset_benchmark/results/）:
    diagnosis_<model>.csv   逐 rep 明细
    diagnosis_<model>.json   逐视频诊断 + 标记
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'validation' / 'dataset_benchmark'))

import config as cfg  # noqa: E402
from algorithms.pipelines import pipeline_associator  # noqa: E402
from metrics_evaluator import MetricsEvaluator  # noqa: E402

COVERAGE_FLAG = 0.60        # 覆盖率低于此值 -> 跟踪差
CONF_FLAG = 0.50            # 中位置信度低于此值 -> 检测弱
SCALE_SPREAD_FLAG = 1.30    # 标定高度 p90/p10 比 > 此值 -> 标定不稳
ALIGN_FLAG = 0.10           # truncate vs best_pair 的 RMSE 差 > 此值 -> 配对可能错位


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="逐视频诊断")
    p.add_argument('--videos-dir', type=Path, default=None, help='原始视频目录')
    p.add_argument('--index', type=Path, default=None, help='dataset_index.json')
    p.add_argument('--model', type=str, default=None,
                   help='ONNX 模型：路径或 barbell_v4/plate_v1/yolo11_plate')
    p.add_argument('--out-dir', type=Path, default=None, help='输出目录')
    p.add_argument('--plate-diameter', type=float, default=0.45)
    p.add_argument('--scale-factor', type=float, default=1.0)
    p.add_argument('--conf-threshold', type=float, default=0.25)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--top', type=int, default=5,
                   help='错误列表里最多详列的视频数')
    return p.parse_args()


def resolve_model(model: str | None) -> Path:
    short = {
        'barbell_v4': cfg.REPO_ROOT / 'models' / 'barbell_v4.onnx',
        'plate_v1': cfg.REPO_ROOT / 'models' / 'plate_v1.onnx',
        'yolo11_plate': cfg.REPO_ROOT / 'models' / 'yolo11_plate.onnx',
    }
    if model is None:
        return cfg.default_model_path()
    if model in short:
        return short[model]
    return Path(model).expanduser()


def flags_for(vid_result: dict) -> list[str]:
    flags = []
    d = vid_result
    if d['n_gt'] != d['n_pred']:
        flags.append(f"REP_COUNT_MISMATCH(gt={d['n_gt']}, pred={d['n_pred']})")
    if d.get('rmse_truncate') is not None and d.get('rmse_best_pair') is not None:
        if d['rmse_truncate'] - d['rmse_best_pair'] > ALIGN_FLAG:
            flags.append(f"ALIGNMENT(trunc={d['rmse_truncate']:.3f}, "
                         f"best={d['rmse_best_pair']:.3f})")
    cov = d.get('coverage', 1.0)
    if cov < COVERAGE_FLAG:
        flags.append(f"LOW_COVERAGE({cov*100:.0f}%)")
    conf = d.get('conf_median', 1.0)
    if conf is not None and conf < CONF_FLAG:
        flags.append(f"WEAK_DETECTIONS(conf={conf:.2f})")
    spread = d.get('scale_spread')
    if spread is not None and spread > SCALE_SPREAD_FLAG:
        flags.append(f"UNSTABLE_SCALE(p90/p10={spread:.2f})")
    if d.get('n_gt') and d.get('n_pred') == 0:
        flags.append("NO_PREDICTIONS")
    return flags


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
        return 1

    with open(index_path, encoding='utf-8') as f:
        dataset = json.load(f)
    if args.limit > 0:
        dataset = dataset[:args.limit]

    per_video, rep_rows, all_flags = [], [], []

    for it in dataset:
        vid = it['video_id']
        gt = it.get('gt_reps_mcv', [])
        vp = videos_dir / vid
        if not vp.exists():
            print(f"  ⏭ {vid} 缺失，跳过")
            continue

        traces: dict = {}
        try:
            result = pipeline_associator(
                str(vp),
                plate_diameter_m=args.plate_diameter,
                scale_factor=args.scale_factor,
                conf_threshold=args.conf_threshold,
                model_path=str(model_path),
                traces=traces,
            )
            reps = result['reps']
            diag = result['diagnostics']
            ev_tr = MetricsEvaluator.evaluate_video(vid, gt, reps, strategy='truncate')
            ev_bp = MetricsEvaluator.evaluate_video(vid, gt, reps, strategy='best_pair')
        except Exception as e:
            print(f"  ✗ {vid} ERROR: {e}")
            per_video.append({'video_id': vid, 'error': str(e)})
            continue

        # 检测/标定特征
        confs = traces.get('confs', np.array([]))
        heights = traces.get('heights', np.array([]))
        valid_conf = confs[confs > 0]
        valid_h = heights[heights > 0]
        conf_median = float(np.median(valid_conf)) if valid_conf.size else None
        if valid_h.size >= 10:
            h_p10, h_p90 = np.percentile(valid_h, 10), np.percentile(valid_h, 90)
            scale_spread = float(h_p90 / h_p10) if h_p10 > 0 else None
        else:
            scale_spread = None

        entry = {
            'video_id': vid,
            'load_kg': it.get('load_kg'),
            'n_gt': len(gt),
            'n_pred': len(reps),
            'rmse_truncate': None if np.isnan(ev_tr.rmse) else ev_tr.rmse,
            'bias_truncate': None if np.isnan(ev_tr.bias) else ev_tr.bias,
            'rmse_best_pair': None if np.isnan(ev_bp.rmse) else ev_bp.rmse,
            'coverage': diag.get('coverage', 0.0),
            'scale': diag.get('scale', 0.0),
            'scale_spread': scale_spread,
            'conf_median': conf_median,
            'assoc_state': diag.get('assoc_state'),
            'paired': [{'rep_idx': i + 1, 'gt': r.gt_mcv, 'pred': r.pred_mcv,
                        'error': r.error}
                       for i, r in enumerate(ev_tr.paired)],
        }
        entry['flags'] = flags_for(entry)
        per_video.append(entry)
        all_flags.extend(entry['flags'])

        for i, r in enumerate(ev_tr.paired):
            rep_rows.append({
                'video_id': vid,
                'load_kg': it.get('load_kg'),
                'rep_idx': i + 1,
                'gt_mcv': r.gt_mcv,
                'pred_mcv': r.pred_mcv,
                'error': r.error,
                'flags': ';'.join(entry['flags']),
            })

        flag_s = ('  [' + ', '.join(entry['flags']) + ']') if entry['flags'] else ''
        print(f"  {'⚠' if entry['flags'] else '✓'} {vid:<44} "
              f"GT={entry['n_gt']:>2} pred={entry['n_pred']:>2} "
              f"RMSE={entry['rmse_truncate'] if entry['rmse_truncate'] is not None else '-':>7} "
              f"conf={conf_median if conf_median is not None else '-':>5} "
              f"cov={entry['coverage']*100:3.0f}% "
              f"spread={scale_spread if scale_spread else '-':>5}{flag_s}")

    # ── 汇总 ────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  诊断汇总")
    print("=" * 72)
    valid = [v for v in per_video if v.get('rmse_truncate') is not None]
    if valid:
        errs = np.array([v['rmse_truncate'] for v in valid])
        print(f"  可评估视频: {len(valid)}  RMSE mean={errs.mean():.4f}  median={np.median(errs):.4f}")
    flagged = [v for v in per_video if v.get('flags')]
    print(f"  有问题的视频: {len(flagged)}/{len(per_video)}")
    from collections import Counter
    for flag, n in Counter(all_flags).most_common():
        print(f"    {n:>3}x  {flag}")

    print("\n  问题视频 Top（按 RMSE 排序）:")
    for v in sorted([v for v in per_video if v.get('flags')],
                    key=lambda x: -(x.get('rmse_truncate') or 999))[:args.top]:
        print(f"    {v['video_id']:<44} RMSE={v['rmse_truncate']}  {', '.join(v['flags'])}")

    # ── 导出 ────────────────────────────────────────────────
    tag = model_path.stem
    with open(out_dir / f'diagnosis_{tag}.json', 'w', encoding='utf-8') as f:
        json.dump({'model': str(model_path),
                   'plate_diameter_m': args.plate_diameter,
                   'scale_factor': args.scale_factor,
                   'videos': per_video}, f, ensure_ascii=False, indent=2)
    if rep_rows:
        import pandas as pd
        pd.DataFrame(rep_rows).to_csv(out_dir / f'diagnosis_{tag}.csv',
                                      index=False, encoding='utf-8-sig')
    print(f"\n  已保存: {out_dir / f'diagnosis_{tag}.json'}")
    if rep_rows:
        print(f"  已保存: {out_dir / f'diagnosis_{tag}.csv'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
