#!/usr/bin/env python3
"""scripts/dump_all_trajectories.py — 导出全部 34 条视频的帧级轨迹

用途
────
把 stride=1 检测 + 标定 + 跟踪的结果（每帧 t/y/v）一次性落盘成 CSV。
之后调分段器参数时**直接读 CSV 重跑分段即可，无需再跑检测**。

为什么值得做
────────────
检测是全流程 97% 的成本（单帧 ONNX 推理 30~53ms，整条视频 55~105s）。
分段器本身只要 0.3ms/条。若每试一组分段参数就重跑全量检测，
一轮要 45 分钟；读 CSV 重跑分段则是**亚秒级**，可以做网格搜索。

轨迹只取决于检测/标定/跟踪，与分段器参数无关，所以缓存是安全的 ——
但若改动了 detector / calibrator / tracker，必须重新导出。

用法
────
    python scripts/dump_all_trajectories.py                 # 全部 34 条
    python scripts/dump_all_trajectories.py --workers 4     # 并行
    python scripts/dump_all_trajectories.py --smoke         # 仅 5 条冒烟集

输出
────
    validation/reports/trajectories/<video_id>.csv
      列：idx,t_s,y_m,v_mps,det_h_px,det_conf
    validation/reports/trajectories/_manifest.json
      记录导出时的 git sha 与各视频 mpp/fps，便于判断缓存是否过期。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BENCH = REPO / "validation" / "dataset_benchmark"
OUT_DIR = REPO / "validation" / "reports" / "trajectories"

SMOKE = [
    "20kg_0.87_0.88_0.89_0.91",
    "30kg_1.03_0.89_0.76_0.65",
    "80kg_0.88_0.88_0.94_0.90",
    "110kg_0.71_0.73",
    "140kg_0.41",
]


def extract(video_id: str, model: str) -> dict:
    """复刻 pipeline 前两阶段，逐帧记录 t/y/v。与 diagnose_missed_reps 同源。"""
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

    rows: list[tuple] = []
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

        # 与 scripts/diagnose_missed_reps.py 逐行同源：同样的 conf_thresh 与符号约定
        res = det.detect(frame, conf_thresh=0.40)
        if res:
            best = max(res, key=lambda b: b.w * b.h)
            bbox = (
                best.cx - best.w / 2, best.cy - best.h / 2,
                best.cx + best.w / 2, best.cy + best.h / 2,
            )
            y_m, v_mps = tracker.step_keyframe(gray, bbox, t_s)
            h_px = f"{float(best.h):.2f}"
            conf = f"{float(best.conf):.3f}"
        else:
            y_m, v_mps = tracker.step_interframe(gray, t_s)
            h_px = ""
            conf = ""
            n_nodet += 1

        # y/v 取反：图像坐标向下为正，物理量向上为正
        rows.append((idx, t_s, -y_m, -v_mps, h_px, conf))

        idx += 1

    cap.release()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{video_id}.csv"
    with out.open("w", encoding="utf-8") as fh:
        fh.write("idx,t_s,y_m,v_mps,det_h_px,det_conf\n")
        for i, t, y, v, hp, cf in rows:
            fh.write(f"{i},{t:.9f},{y:.9f},{v:.9f},{hp},{cf}\n")

    return {
        "video_id": video_id,
        "n_frames": idx,
        "n_traj": len(rows),
        "n_nodet": n_nodet,
        "mpp": mpp,
        "fps": fps,
    }


def _job(args):
    vid, model = args
    try:
        return extract(vid, model)
    except Exception as exc:  # noqa: BLE001
        return {"video_id": vid, "error": repr(exc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO / "models" / "best.onnx"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    index = json.loads((BENCH / "dataset_index.json").read_text(encoding="utf-8"))
    vids = [e["video_id"].replace(".mp4", "") for e in index]
    if args.smoke:
        vids = [v for v in vids if v in SMOKE]

    print(f"导出 {len(vids)} 条轨迹 → {OUT_DIR}  (workers={args.workers})")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_job, (v, args.model)): v for v in vids}
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if "error" in r:
                print(f"[{k:2d}/{len(vids)}] ✗ {r['video_id']:<46} {r['error']}")
            else:
                print(
                    f"[{k:2d}/{len(vids)}] ✓ {r['video_id']:<46} "
                    f"frames={r['n_frames']:4d} traj={r['n_traj']:4d} "
                    f"nodet={r['n_nodet']:3d} mpp={r['mpp']}"
                )

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = "unknown"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "_manifest.json").write_text(
        json.dumps(
            {
                "git_sha": sha,
                "note": "轨迹仅依赖 detector/calibrator/tracker；改动这三者后必须重新导出。",
                "videos": sorted(results, key=lambda r: r["video_id"]),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    ok = sum(1 for r in results if "error" not in r)
    print(f"\n完成 {ok}/{len(vids)}  manifest → {OUT_DIR / '_manifest.json'}")
    return 0 if ok == len(vids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
