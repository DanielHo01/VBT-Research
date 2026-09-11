"""mine_frames.py — M3 失败帧挖掘（专标失败帧，不标废帧）
==============================================================
用当前检测器扫全部视频，按"失败证据"给每帧打分，输出待标队列：

    优先级（高→低）：
      edge_only / no_pass  工作片疑似漏检（有框但全被锚定先验拒绝，
                           实测 6 条假拒绝 100% 死于 edge_x）
      zero_raw             整帧零检出（可疑沉默）
      low_conf             最高置信 <0.35（勉强检出）
      ok                   有过先验候选（低优先级，仍采样少量保正样本多样性）

    同视频内按 min-gap 去重 + per-video 上限 + min-per-video 保底
    （好视频也留几帧，保证各负荷/背景都有正样本）。

用法（Windows）：
    python scripts/mine_frames.py --bench-dir validation/dataset_benchmark ^
        --model models/yolo11_plate.onnx --out datasets/mining/queue_r0.json

输出：
    datasets/mining/queue_r0.json  （interactive_label.py --mine 直接消费）
    datasets/mining/queue_r0.md    （人工可读的挖掘报告）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from label_common import save_mining_queue  # noqa: E402
from vbtcore.detector import PlateDetector  # noqa: E402
from vbtcore.engine import anchor_score  # noqa: E402


def reject_reason(d, H: int, W: int, m: float = 0.03,
                  h_range: tuple = (0.03, 0.15),
                  max_ratio: float = 1.6) -> str:
    """与 vbtcore.engine.anchor_score 同规则的拒绝原因（挖掘诊断用）。"""
    if not (W * m < d.cx < W * (1 - m)):
        return "edge_x"
    if not (H * m < d.cy < H * (1 - m)):
        return "edge_y"
    if d.ratio > max_ratio:
        return "ratio"
    hr = d.h / H
    if hr > h_range[1]:
        return "size_big"
    if hr < h_range[0]:
        return "size_small"
    return "pass"


def score_frame(dets, H: int, W: int, low_conf: float = 0.35) -> tuple[float, list[str], dict]:
    """失败证据打分 → (分数, 原因, 明细)。分数越高越值得标。"""
    reasons: list[str] = []
    if not dets:
        return 2.5, ["zero_raw"], {"n_raw": 0, "max_conf": 0.0, "n_pass": 0}
    hist = Counter(reject_reason(d, H, W) for d in dets)
    n_pass = hist.get("pass", 0)
    max_conf = max(d.conf for d in dets)
    detail = {"n_raw": len(dets), "max_conf": round(max_conf, 3),
              "n_pass": n_pass, "reject_hist": dict(hist)}
    if n_pass == 0:
        reasons.append("no_pass")
        if set(hist) <= {"edge_x", "edge_y"}:
            reasons.append("edge_only")
        return 3.0, reasons, detail
    if max_conf < low_conf:
        reasons.append("low_conf")
        return 1.5, reasons, detail
    return 0.5, ["ok"], detail


def mine_video(det: PlateDetector, video_path: str, stride: int,
               conf: float, per_video: int, min_gap: int,
               min_per_video: int, low_conf: float) -> list[dict]:
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scored = []
    fi = 0
    while fi < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break
        dets = det.detect(frame, conf)
        s, reasons, detail = score_frame(dets, H, W, low_conf)
        scored.append({"frame": fi, "score": s, "reasons": reasons,
                       "detail": detail})
        fi += stride
    cap.release()
    # 高分优先 + 最小间隔贪心
    scored.sort(key=lambda r: -r["score"])
    picked: list[dict] = []
    for r in scored:
        if len(picked) >= per_video:
            break
        if all(abs(r["frame"] - p["frame"]) >= min_gap for p in picked):
            picked.append(r)
    # 保底：好视频也均匀留几帧（正样本多样性）
    if len(picked) < min_per_video and scored:
        by_frame = sorted(scored, key=lambda r: r["frame"])
        step = max(1, len(by_frame) // min_per_video)
        for r in by_frame[::step]:
            if len(picked) >= min_per_video:
                break
            if all(abs(r["frame"] - p["frame"]) >= min_gap // 2 for p in picked):
                picked.append(r)
    picked.sort(key=lambda r: r["frame"])
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description="M3 失败帧挖掘")
    ap.add_argument("--bench-dir", default="validation/dataset_benchmark")
    ap.add_argument("--model", default="models/yolo11_plate.onnx")
    ap.add_argument("--out", default="datasets/mining/queue_r0.json")
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--per-video", type=int, default=12)
    ap.add_argument("--min-gap", type=int, default=45)
    ap.add_argument("--min-per-video", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--low-conf", type=float, default=0.35)
    ap.add_argument("--only", default=None, help="逗号分隔的视频名子串过滤")
    args = ap.parse_args()

    bench = Path(args.bench_dir)
    raw_dir = bench / "raw_videos"
    videos = sorted(p for p in raw_dir.glob("*.mp4"))
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        videos = [p for p in videos if any(k in p.name for k in keys)]
    if not videos:
        print(f"无视频: {raw_dir}")
        return 1

    print(f"模型: {args.model} ｜ 视频: {len(videos)} ｜ stride={args.stride}")
    det = PlateDetector(args.model)
    t0 = time.time()
    items: list[dict] = []
    per_video_rows = []
    for vi, vp in enumerate(videos):
        picked = mine_video(det, str(vp), args.stride, args.conf,
                            args.per_video, args.min_gap,
                            args.min_per_video, args.low_conf)
        reason_hist: Counter = Counter()
        for r in picked:
            reason_hist[r["reasons"][0]] += 1
            items.append({"video": vp.name, "frame": r["frame"],
                          "score": r["score"], "reasons": r["reasons"],
                          "detail": r["detail"]})
        per_video_rows.append((vp.name, len(picked), dict(reason_hist)))
        print(f"  [{vi + 1}/{len(videos)}] {vp.name}: {len(picked)} 帧 "
              f"{dict(reason_hist)}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "bench_dir": str(bench),
        "params": {"stride": args.stride, "per_video": args.per_video,
                   "min_gap": args.min_gap, "min_per_video": args.min_per_video,
                   "conf": args.conf, "low_conf": args.low_conf},
        "n_videos": len(videos),
        "n_items": len(items),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "items": items,
    }
    save_mining_queue(out_path, payload)

    total_hist: Counter = Counter()
    for it in items:
        total_hist[it["reasons"][0]] += 1
    md = [f"# 挖掘队列 {out_path.stem}",
          f"- 生成: {payload['generated']} ｜ 模型: {args.model}",
          f"- 视频: {len(videos)} ｜ 待标帧: **{len(items)}** "
          f"｜ 原因分布: {dict(total_hist)} ｜ 耗时: {payload['elapsed_min']} 分钟",
          "",
          "| 视频 | 帧数 | 主因 |",
          "|---|---|---|"]
    for name, cnt, hist in per_video_rows:
        md.append(f"| {name} | {cnt} | {hist} |")
    md_path = out_path.with_suffix(".md")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n待标 {len(items)} 帧，原因分布 {dict(total_hist)}")
    print(f"队列: {out_path}\n报告: {md_path}")
    print("下一步: python scripts/interactive_label.py --mine "
          f"{out_path} --bench-dir {bench}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
