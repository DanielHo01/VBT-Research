"""
diagnose_plate_tracking.py — 杠铃片检测/跟踪层专项诊断
=====================================================

回答"plate 层到底哪里不可靠"：
  1. 多片歧义：每帧检出几个 plate？帧间 top-1 是否会切换到另一个片？
  2. 轨迹跳变：相邻帧 |Δcy| 超物理可能（30fps 下 >60px ≈ 4.8m/s）的比例
  3. 尺寸漂移：同一块片在固定机位下像素尺寸应恒定，|Δh| 变化比例
  4. 静态目标锁定率：top-1 有多少帧锁定在"几乎不动"的物体上
     （架子上的片堆/杠铃架 → 典型的错误关联）
  5. 标定可用性：可用于标定的帧比例（ratio<=1.4 且 conf>=min_conf）

用法:
    python3 scripts/diagnose_plate_tracking.py --videos 110kg_0.71_0.73.mp4 ...
    python3 scripts/diagnose_plate_tracking.py --limit 6
    python3 scripts/diagnose_plate_tracking.py --all --stride 2
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'validation' / 'dataset_benchmark'))

import config as cfg  # noqa: E402
from algorithms.common import YoloPlateDetector  # noqa: E402

# 物理约束：30fps 下 4.8 m/s 的杠速对应约 60px/帧（scale≈0.0027 m/px）
JUMP_PX = 60.0
# 同一块片在固定机位下尺寸应稳定；超过 20% 变化视为换了目标/检测劣化
SIZE_CHANGE_FRAC = 0.20
# 静态物体判定：该簇内 y 标准差 < 20px 且出现帧数 >= 20
STATIC_STD_PX = 20.0
STATIC_MIN_FRAMES = 20


def parse_args():
    p = argparse.ArgumentParser(description='杠铃片检测/跟踪层诊断')
    p.add_argument('--model', default=None, help='ONNX 模型或快捷名')
    p.add_argument('--videos', nargs='*', default=None, help='视频文件名（默认取前 N 个）')
    p.add_argument('--limit', type=int, default=6, help='未指定 --videos 时取前 N 个')
    p.add_argument('--stride', type=int, default=1, help='跳帧采样（默认每帧）')
    p.add_argument('--conf', type=float, default=0.25, help='检测置信度阈值')
    p.add_argument('--top-k', type=int, default=5, help='每帧最多取几个检测')
    p.add_argument('--videos-dir', default=None)
    return p.parse_args()


def resolve_model(name):
    short = {
        'barbell_v4': cfg.REPO_ROOT / 'models' / 'barbell_v4.onnx',
        'plate_v1': cfg.REPO_ROOT / 'models' / 'plate_v1.onnx',
        'yolo11_plate': cfg.REPO_ROOT / 'models' / 'yolo11_plate.onnx',
    }
    if name is None:
        return cfg.default_model_path()
    return short.get(name, Path(name).expanduser())


def cluster_tracks(dets_by_frame: list[list[dict]], link_px: float = 60.0):
    """
    把逐帧检测简单聚成"空间轨迹簇"（贪心最近邻）。
    返回 list[dict]: {id, members: [(frame, cx, cy, h)], std_y, n_frames}
    """
    tracks = []
    for fi, dets in enumerate(dets_by_frame):
        for d in dets:
            best, best_d = None, link_px
            for t in tracks:
                if fi - t['last_frame'] > 1:      # 只连接相邻帧
                    continue
                dist = np.hypot(d['cx'] - t['cx'], d['cy'] - t['cy'])
                if dist < best_d:
                    best, best_d = t, dist
            if best is None:
                tracks.append({'cx': d['cx'], 'cy': d['cy'], 'last_frame': fi,
                               'members': [(fi, d['cx'], d['cy'], d['h'])]})
            else:
                c, cy = d['cx'], d['cy']
                best['cx'] = 0.5 * best['cx'] + 0.5 * c    # 平滑
                best['cy'] = 0.5 * best['cy'] + 0.5 * cy
                best['last_frame'] = fi
                best['members'].append((fi, c, cy, d['h']))

    for t in tracks:
        ys = [m[2] for m in t['members']]
        t['std_y'] = float(np.std(ys)) if len(ys) > 1 else 999.0
        t['n_frames'] = len({m[0] for m in t['members']})
        t['h_med'] = float(np.median([m[3] for m in t['members']]))
    return tracks


def analyze_video(detector, vpath: Path, stride: int, conf_th: float, top_k: int):
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = 0
    dets_by_frame: list[list[dict]] = []
    top1: list[dict | None] = []

    fi = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if fi % stride == 0:
            dets = detector.detect_all(frame, conf_threshold=conf_th, top_k=top_k)
            dets_by_frame.append(dets)
            top1.append(dets[0] if dets else None)
        fi += 1
    cap.release()
    n_frames = len(dets_by_frame) if dets_by_frame else 0
    if n_frames == 0:
        return None

    n_det_frames = sum(1 for d in dets_by_frame if d)
    multi_frames = sum(1 for d in dets_by_frame if len(d) >= 2)
    dets_per_frame = np.mean([len(d) for d in dets_by_frame])

    # 跳变 & 尺寸漂移（只看相邻都有检测的帧）
    jumps, size_changes = [], []
    prev = None
    for d in top1:
        if d is None:
            prev = None
            continue
        if prev is not None:
            jumps.append(abs(d['cy'] - prev['cy']))
            if prev['h'] > 0:
                size_changes.append(abs(d['h'] - prev['h']) / prev['h'])
        prev = d
    jumps = np.array(jumps) if jumps else np.array([0.0])
    size_changes = np.array(size_changes) if size_changes else np.array([0.0])

    # 静态目标锁定：把 top-1 序列聚成簇，看多少帧落在"不动"的簇上
    tracks = cluster_tracks([[d] for d in top1 if d])
    static_ids = {i for i, t in enumerate(tracks)
                  if t['std_y'] < STATIC_STD_PX and t['n_frames'] >= STATIC_MIN_FRAMES}
    # 移到哪个簇：贪心最近
    frames_on_static = 0
    for d in top1:
        if d is None:
            continue
        for i, t in enumerate(tracks):
            if (d['cx'], d['cy']) in [(m[1], m[2]) for m in t['members']]:
                if i in static_ids:
                    frames_on_static += 1
                break

    # 标定可用性
    calib_ok = 0
    for d in top1:
        if d is None:
            continue
        ratio = max(d['w'], d['h']) / min(d['w'], d['h']) if min(d['w'], d['h']) > 0 else 999
        if ratio <= 1.4 and d['score'] >= 0.5:
            calib_ok += 1

    heights = np.array([d['h'] for d in top1 if d]) if n_det_frames else np.array([0.0])

    return {
        'video': vpath.name,
        'n_frames': n_frames,
        'det_rate': n_det_frames / n_frames,
        'dets_per_frame': float(dets_per_frame),
        'multi_rate': multi_frames / n_frames,
        'jump_rate': float((jumps > JUMP_PX).mean()),
        'jump_p95': float(np.percentile(jumps, 95)),
        'size_change_rate': float((size_changes > SIZE_CHANGE_FRAC).mean()),
        'static_lock_rate': frames_on_static / max(n_det_frames, 1),
        'n_tracks': len(tracks),
        'n_static_tracks': len(static_ids),
        'calib_frame_rate': calib_ok / n_frames,
        'h_p10_p90': (float(np.percentile(heights, 10)), float(np.percentile(heights, 90)))
                     if n_det_frames else (0.0, 0.0),
    }


def main() -> int:
    args = parse_args()
    model_path = resolve_model(args.model)
    videos_dir = Path(args.videos_dir) if args.videos_dir else cfg.raw_videos_dir()

    if args.videos:
        names = args.videos
    else:
        idx_path = cfg.dataset_index_path()
        with open(idx_path, encoding='utf-8') as f:
            idx = json.load(f)
        names = [it['video_id'] for it in idx[:args.limit]]

    detector = YoloPlateDetector(str(model_path))
    print(f"模型: {model_path}")
    print(f"阈值: conf>={args.conf}  top_k={args.top_k}  stride={args.stride}  "
          f"跳变阈值={JUMP_PX}px\n")

    rows = []
    for name in names:
        vp = videos_dir / name
        if not vp.exists():
            print(f"  ⏭ 缺失: {name}")
            continue
        r = analyze_video(detector, vp, args.stride, args.conf, args.top_k)
        if r is None:
            continue
        rows.append(r)
        print(f"{r['video']}")
        print(f"  检测率={r['det_rate']*100:3.0f}%  每帧检测数={r['dets_per_frame']:.2f}  "
              f"多片帧={r['multi_rate']*100:3.0f}%  轨迹簇={r['n_tracks']}(静态{r['n_static_tracks']})")
        print(f"  ⚠ 跳变率(|Δcy|>{JUMP_PX:.0f}px)={r['jump_rate']*100:3.0f}%  p95跳变={r['jump_p95']:.0f}px  "
              f"尺寸突变={r['size_change_rate']*100:3.0f}%")
        print(f"  ⚠ 静态目标锁定率={r['static_lock_rate']*100:3.0f}%  "
              f"标定可用帧={r['calib_frame_rate']*100:3.0f}%  "
              f"h范围={r['h_p10_p90'][0]:.0f}-{r['h_p10_p90'][1]:.0f}px")

    if rows:
        print("\n" + "=" * 72)
        print("  汇总（均值）")
        print("=" * 72)
        for k, label in [('det_rate', '检测率'), ('dets_per_frame', '每帧检测数'),
                         ('multi_rate', '多片帧比例'), ('jump_rate', '跳变率'),
                         ('size_change_rate', '尺寸突变率'),
                         ('static_lock_rate', '静态锁定率'),
                         ('calib_frame_rate', '标定可用帧')]:
            vals = [r[k] for r in rows]
            print(f"  {label:<12}: {np.mean(vals):.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
