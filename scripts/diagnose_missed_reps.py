"""
scripts/diagnose_missed_reps.py — 少计视频的帧级诊断
=====================================================
针对「计数少于真值」的视频，导出帧级轨迹并定位 rep 在哪一层被丢弃。

诊断分三层，逐层回答「rep 去哪了」：
  L1 轨迹层：y(t)/v(t) 是否真实反映了动作？（检测/跟踪是否漏了）
  L2 FSM 层：状态机是否进入了 CONCENTRIC？（状态跃迁是否卡住）
  L3 门禁层：候选 rep 是否被 min_rom_m / min_dur_s 拒绝？（阈值是否过严）

用法：
    python3 scripts/diagnose_missed_reps.py --video 105kg_0.69_0.64_0.65_0.60_0.56_0.45
    python3 scripts/diagnose_missed_reps.py --all-missed
    python3 scripts/diagnose_missed_reps.py --all-missed --dump-csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BENCH = REPO / "validation" / "dataset_benchmark"
OUT_DIR = REPO / "validation" / "reports" / "diagnostics"

# 当前已知的 3 条少计视频
MISSED = [
    "105kg_0.69_0.64_0.65_0.60_0.56_0.45",
    "105kg_0.60_0.54_0.55_0.54_0.51_0.37",
    "110kg_0.61_0.52_0.55_0.51_0.38",
]


def extract_trajectory(video_id: str, model: str) -> dict:
    """跑一遍流水线，抓出帧级 t/y/v 序列（复刻 pipeline.analyze_video 的前两阶段）。"""
    import cv2

    from vbtcore import PlateDetector
    from vbtcore.calibrator import StaticPlateCalibrator
    from vbtcore.tracker import DenseVisualTracker

    vp = BENCH / "raw_videos" / f"{video_id}.mp4"
    det = PlateDetector(model)

    cap = cv2.VideoCapture(str(vp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    calib = StaticPlateCalibrator(real_diameter_m=0.45, min_static_frames=20, max_cv=0.015)
    tracker = None
    mpp = None

    ts: list[float] = []
    ys: list[float] = []
    vs: list[float] = []
    det_h: list[float] = []      # 每帧最大检测框高（像素），None → nan
    det_conf: list[float] = []
    n_nodet = 0
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t_s = pts_ms / 1000.0 if pts_ms > 0 else idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if mpp is None:
            res = det.detect(frame, conf_thresh=0.45)
            if res:
                best = max(res, key=lambda b: b.w * b.h)
                calib.add_sample(float(best.h))
                if calib.is_ready():
                    mpp = calib.lock_scale()
                    bbox = (
                        best.cx - best.w / 2, best.cy - best.h / 2,
                        best.cx + best.w / 2, best.cy + best.h / 2,
                    )
                    tracker = DenseVisualTracker(
                        mpp=mpp, initial_bbox=bbox, initial_gray=gray, initial_time_s=t_s
                    )
            idx += 1
            continue

        res = det.detect(frame, conf_thresh=0.40)
        if res:
            best = max(res, key=lambda b: b.w * b.h)
            bbox = (
                best.cx - best.w / 2, best.cy - best.h / 2,
                best.cx + best.w / 2, best.cy + best.h / 2,
            )
            y_m, v_mps = tracker.step_keyframe(gray, bbox, t_s)
            det_h.append(float(best.h))
            det_conf.append(float(best.conf))
        else:
            y_m, v_mps = tracker.step_interframe(gray, t_s)
            det_h.append(float("nan"))
            det_conf.append(float("nan"))
            n_nodet += 1

        ts.append(t_s)
        ys.append(-y_m)
        vs.append(-v_mps)
        idx += 1

    cap.release()
    return {
        "video_id": video_id,
        "fps": fps,
        "mpp": mpp,
        "t": np.array(ts),
        "y": np.array(ys),
        "v": np.array(vs),
        "det_h": np.array(det_h),
        "det_conf": np.array(det_conf),
        "n_frames": idx,
        "n_nodet": n_nodet,
    }


def trace_fsm(t, y, v, min_rom_m=0.12, min_dur_s=0.20, v_thresh=0.08, v_band=0.02):
    """复刻 BiomechanicalRepSegmenter.segment 的 squat_bench 分支，记录每个候选与其判定。"""
    n = len(t)
    candidates = []
    transitions = []
    state = "IDLE"
    rep_start = 0

    for i in range(1, n - 2):
        vi = float(v[i])
        vn = float(v[i + 1])
        prev = state

        if state == "IDLE":
            if vi < -v_thresh:
                state = "ECCENTRIC"
        elif state == "ECCENTRIC":
            if vi >= -v_band and vn > v_band:
                state = "CONCENTRIC"
                rep_start = i
        elif state == "CONCENTRIC":
            if vi <= v_band and vn < v_band:
                dur = float(t[i]) - float(t[rep_start])
                rom = abs(float(y[i]) - float(y[rep_start]))
                accepted = dur >= min_dur_s and rom >= min_rom_m
                reason = []
                if dur < min_dur_s:
                    reason.append(f"dur {dur:.3f}<{min_dur_s}")
                if rom < min_rom_m:
                    reason.append(f"rom {rom:.4f}<{min_rom_m}")
                candidates.append({
                    "start_idx": rep_start, "end_idx": i,
                    "start_t": float(t[rep_start]), "end_t": float(t[i]),
                    "dur": dur, "rom": rom,
                    "mcv": rom / dur if dur > 0 else 0.0,
                    "accepted": accepted,
                    "reject_reason": "; ".join(reason),
                })
                state = "IDLE"

        if state != prev:
            transitions.append((i, float(t[i]), prev, state))

    return candidates, transitions, state


def find_true_valleys(t, y, v, v_thresh=0.08):
    """
    用「位移谷底」独立找出真实 rep 次数，作为 FSM 的对照。
    谷底 = 局部极小且前后有足够幅度的上下行程（不依赖 FSM 状态）。
    """
    n = len(y)
    valleys = []
    win = 8
    for i in range(win, n - win):
        seg = y[i - win:i + win + 1]
        if y[i] <= seg.min() + 1e-9:
            # 谷底前后各找一段峰值，确认是真正的下-上往返
            left_peak = y[max(0, i - 60):i + 1].max()
            right_peak = y[i:min(n, i + 60)].max()
            depth = min(left_peak - y[i], right_peak - y[i])
            if depth > 0.12:
                if not valleys or i - valleys[-1][0] > 20:
                    valleys.append((i, float(t[i]), float(depth)))
                elif depth > valleys[-1][2]:
                    valleys[-1] = (i, float(t[i]), float(depth))
    return valleys


def diagnose(video_id: str, model: str, dump_csv: bool = False) -> dict:
    ds = {d["video_id"]: d for d in json.load(open(BENCH / "dataset_index.json"))}
    gt = ds[f"{video_id}.mp4"]["gt_reps_mcv"]

    print("=" * 96)
    print(f" {video_id}")
    print("=" * 96)

    tr = extract_trajectory(video_id, model)
    t, y, v = tr["t"], tr["y"], tr["v"]

    dur_total = t[-1] - t[0]
    print(f"  帧数 {tr['n_frames']} ｜ 时长 {dur_total:.1f}s ｜ mpp {tr['mpp']:.6f} "
          f"｜ 无检测帧 {tr['n_nodet']} ({tr['n_nodet']/max(len(t),1)*100:.1f}%)")
    print(f"  y 行程 {y.max()-y.min():.3f} m ｜ v 范围 [{v.min():.2f}, {v.max():.2f}] m/s")

    # ── L1 轨迹层：独立谷底检测 ────────────────────────────
    valleys = find_true_valleys(t, y, v)
    print(f"\n  [L1 轨迹层] 独立谷底检测找到 {len(valleys)} 个往返 ｜ 真值 {len(gt)} reps")
    if valleys:
        gaps = [round(valleys[i+1][1]-valleys[i][1], 2) for i in range(len(valleys)-1)]
        print(f"    谷底时刻: {[round(x[1],2) for x in valleys]}")
        print(f"    间隔: {gaps}")
    if len(valleys) < len(gt):
        print(f"    ⚠ 轨迹里就只有 {len(valleys)} 个往返 → 丢失发生在**检测/跟踪层**")
    else:
        print("    ✓ 轨迹完整包含全部 rep → 丢失发生在**分段层**")

    # ── L2/L3 FSM 与门禁 ──────────────────────────────────
    cands, transitions, final_state = trace_fsm(t, y, v)
    acc = [c for c in cands if c["accepted"]]
    rej = [c for c in cands if not c["accepted"]]
    print(f"\n  [L2 FSM 层] 状态跃迁 {len(transitions)} 次 ｜ 产生候选 {len(cands)} 个 ｜ 终态 {final_state}")
    print(f"  [L3 门禁层] 接受 {len(acc)} ｜ 拒绝 {len(rej)}")

    if cands:
        print(f"\n    {'#':>3} {'起止帧':>12} {'起止秒':>14} {'dur':>6} {'rom':>7} {'mcv':>6}  判定")
        for k, c in enumerate(cands, 1):
            mark = "✓" if c["accepted"] else "✗"
            note = "" if c["accepted"] else f"  ← {c['reject_reason']}"
            print(f"    {k:>3} {c['start_idx']:>5}-{c['end_idx']:<6} "
                  f"{c['start_t']:>6.2f}-{c['end_t']:<7.2f} "
                  f"{c['dur']:>6.3f} {c['rom']:>7.4f} {c['mcv']:>6.3f}  {mark}{note}")

    # ── 对照：谷底 vs 候选，找出被 FSM 完全跳过的往返 ──────
    cand_windows = [(c["start_t"], c["end_t"]) for c in cands]
    skipped = []
    for vi, vt, vd in valleys:
        covered = any(s - 1.5 <= vt <= e + 1.5 for s, e in cand_windows)
        if not covered:
            skipped.append((vi, vt, vd))
    if skipped:
        print(f"\n    ⚠ FSM 完全跳过的往返 {len(skipped)} 个（轨迹有谷底但没产生候选）:")
        for vi, vt, vd in skipped:
            print(f"        frame {vi:>5} t={vt:>6.2f}s 深度 {vd:.3f} m")

    print(f"\n  结论：真值 {len(gt)} ｜ 轨迹往返 {len(valleys)} ｜ FSM 候选 {len(cands)} "
          f"｜ 门禁通过 {len(acc)}")

    if dump_csv:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        p = OUT_DIR / f"{video_id}_trajectory.csv"
        with open(p, "w", encoding="utf-8") as f:
            f.write("idx,t_s,y_m,v_mps,det_h_px,det_conf\n")
            for i in range(len(t)):
                f.write(f"{i},{t[i]:.4f},{y[i]:.5f},{v[i]:.5f},"
                        f"{tr['det_h'][i]:.2f},{tr['det_conf'][i]:.3f}\n")
        print(f"  → 轨迹已导出 {p.relative_to(REPO)}")

    return {
        "video_id": video_id, "gt": len(gt), "valleys": len(valleys),
        "candidates": len(cands), "accepted": len(acc), "rejected": len(rej),
        "skipped_by_fsm": len(skipped), "nodet_ratio": tr["n_nodet"]/max(len(t),1),
        "rejects": rej, "skipped": skipped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--all-missed", action="store_true")
    ap.add_argument("--model", default=str(REPO / "models" / "best.onnx"))
    ap.add_argument("--dump-csv", action="store_true")
    args = ap.parse_args()

    targets = MISSED if args.all_missed else ([args.video] if args.video else MISSED)

    results = []
    for vid in targets:
        results.append(diagnose(vid, args.model, args.dump_csv))
        print()

    print("=" * 96)
    print(" 汇总")
    print("=" * 96)
    print(f"{'视频':<46}{'真值':>5}{'轨迹往返':>9}{'候选':>6}{'通过':>6}{'FSM跳过':>8}{'无检测%':>8}")
    for r in results:
        print(f"{r['video_id']:<46}{r['gt']:>5}{r['valleys']:>9}{r['candidates']:>6}"
              f"{r['accepted']:>6}{r['skipped_by_fsm']:>8}{r['nodet_ratio']*100:>7.1f}%")


if __name__ == "__main__":
    main()
