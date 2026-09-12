#!/usr/bin/env python3
"""
scripts/coverage_audit.py — Phase 0 检测覆盖率审计

用法:
    python scripts/coverage_audit.py \\
        --input-dir validation/dataset_benchmark/raw_videos/ \\
        --output-dir validation/reports/coverage_phase0/ \\
        --model models/best.onnx \\
        --tag phase0

产出:
    {output_dir}/{video_id}_coverage.json       — 每视频详细结果
    {output_dir}/COVERAGE_PHASE0_{tag}.json    — 全汇总
    {output_dir}/COVERAGE_PHASE0_{tag}.md      — 人类可读报告
    {output_dir}/{video_id}_overlay.jpg        — 标注漏检/跳变帧
    {output_dir}/{video_id}_missed/           — 漏检帧截图目录
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Phase 0 detector
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vbtcore.detector_phase0 import (
    detect_video_phase0,
    model_hash,
    write_phase0_json,
)

# ─── 常量 ────────────────────────────────────────────────────────────────────

THRESH_CONF_CURVE = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
THRESH_SCALE_JUMP = 0.30  # |Δh / h_prev| > 0.30
THRESH_POS_JUMP = 0.50  # |Δcx| > 0.50 * h_prev
THRESH_GAP = 3  # 连续漏检 ≥ 3 帧 = Track Loss
THRESH_CRITICAL_GAP = 5  # 连续漏检 ≥ 5 帧 = Critical Gap（目标在画面内时的模型失败）
THRESH_JUMP_RATE = 0.005  # JumpRate ≤ 0.5% = 优质检测


# ─── 指标计算 ────────────────────────────────────────────────────────────────


def compute_conf_curve(raw_json_path: str) -> dict[str, float]:
    """
    用原始 YOLO 输出（row[4] 概率）扫描不同 conf 阈值，
    返回 VDR(conf) 曲线。读取 detector_phase0 输出的 per-frame 原始 conf。
    """
    try:
        with open(raw_json_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 raw JSON: {raw_json_path}: {exc}") from exc

    frames = data.get("frames", [])
    total = len(frames)
    # 从 JSON 读各帧 primary conf（第一条 candidate）
    confs = []
    for fr in frames:
        cands = fr.get("candidates")
        if cands:
            confs.append(cands[0].get("conf", 0.0))
        else:
            confs.append(0.0)

    curve = {}
    for t in THRESH_CONF_CURVE:
        n = sum(1 for c in confs if c >= t)
        curve[f"VDR@{t:.2f}"] = round(n / total, 6) if total else 0.0
    return curve


def compute_jump_frames(frames_data: list[dict]) -> tuple[list[dict], dict]:
    """
    检测跳变帧。

    跳变帧定义（任一满足）：
      |Δcx| > 0.50 * h_prev     （横向大跳）
      |Δh / h_prev| > 0.30     （尺度大跳）

    返回: (跳变帧列表, 跳变统计摘要)
    """
    jumps = []
    prev = None

    for fr in frames_data:
        if not fr["candidates"] or fr["primary_idx"] < 0:
            prev = None
            continue

        primary = fr["candidates"][fr["primary_idx"]]
        item = {
            "frame_idx": fr["frame_idx"],
            "pts_s": fr["pts_s"],
            "cx": primary["cx"],
            "cy": primary["cy"],
            "h": primary["h"],
            "conf": primary["conf"],
            "jump_type": None,
        }

        if prev is not None:
            d_cx = abs(item["cx"] - prev["cx"])
            d_h = abs(item["h"] - prev["h"])
            h_prev = prev["h"]

            if h_prev > 0:
                ratio_h = d_h / h_prev
                ratio_cx = d_cx / h_prev
            else:
                ratio_h = 0.0
                ratio_cx = 0.0

            if ratio_cx > THRESH_POS_JUMP:
                item["jump_type"] = "POS"
                item["d_cx_px"] = round(d_cx, 1)
                item["ratio_cx"] = round(ratio_cx, 3)
            if ratio_h > THRESH_SCALE_JUMP:
                item["jump_type"] = (
                    "SCALE" if item["jump_type"] is None else "POS+SCALE"
                )
                item["d_h_px"] = round(d_h, 1)
                item["ratio_h"] = round(ratio_h, 3)

            if item["jump_type"] is not None:
                jumps.append(item)

        prev = item

    summary = {
        "total": len(jumps),
        "POS": sum(1 for j in jumps if "POS" in (j.get("jump_type") or "")),
        "SCALE": sum(1 for j in jumps if "SCALE" in (j.get("jump_type") or "")),
        "POS+SCALE": sum(1 for j in jumps if j.get("jump_type") == "POS+SCALE"),
    }
    return jumps, summary


def parse_gap_annotations(
    annotation_path: str | Path | None,
) -> dict[str, list[tuple[int, int]]]:
    """
    解析缺口相位标注文件。
    支持 JSON:  {"video_id": [[start, end], ...], ...}
    支持 CSV:   video_id,start,end  (无头行)
    返回 {video_id: [(start, end), ...]}
    """
    if annotation_path is None or not Path(annotation_path).exists():
        return {}

    p = Path(annotation_path)
    data: dict[str, list[tuple[int, int]]] = {}

    try:
        if p.suffix == ".json":
            with open(p) as f:
                raw = json.load(f)
            for vid, segs in raw.items():
                if isinstance(segs, list):
                    data[vid] = [
                        tuple(seg)
                        for seg in segs
                        if isinstance(seg, list) and len(seg) >= 2
                    ]

        elif p.suffix == ".csv":
            import csv as csvmod

            with open(p, newline="") as f:
                reader = csvmod.DictReader(f)
                for row in reader:
                    vid = row.get("video_id")
                    try:
                        start = int(row.get("start", -1))
                        end = int(row.get("end", -1))
                    except (ValueError, TypeError):
                        continue
                    if vid is not None and start >= 0 and end >= 0:
                        data.setdefault(vid, []).append((start, end))
    except (OSError, json.JSONDecodeError, csvmod.Error) as exc:
        import warnings

        warnings.warn(f"缺口标注文件读取失败 [{p}]: {exc}")

    return data


def compute_miss_metrics(
    frames_data: list[dict],
    absent_segments: list[tuple[int, int]] | None = None,
) -> dict:
    """
    漏检指标。

    absent_segments: [(start, end), ...] — 目标不在画面的帧区间（人工标注）。
        通常是 lead-in（视频开头人不在）、lead-out（视频结尾人离开）、
        或已知的目标离开画面区间。
        区间为 inclusive [start, end]。

    指标说明：
      VDR        = n_det / total_frames（所有帧）
      VDR_present = n_det / n_present（目标在画面帧）
      n_present   = total_frames - sum(absent_segment_lengths)
      Critical Gap = 漏检段中 ≥ THRESH_CRITICAL_GAP 帧的段（目标在画面时模型失败）
      JumpRate     = n_jumps / n_det（跳变帧 / 检测帧）
    """
    total = len(frames_data)
    n_det = sum(1 for f in frames_data if f["detected"])
    n_miss = total - n_det

    # 合并 absent_segments 到集合
    absent_set: set[int] = set()
    if absent_segments:
        for start, end in absent_segments:
            for i in range(max(0, start), min(total, end + 1)):
                absent_set.add(i)

    n_absent = len(absent_set)
    n_present = total - n_absent

    # 找所有连续漏检段（仅在 present 区间内）
    missed_segs: list[tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for i, f in enumerate(frames_data):
        if i in absent_set:
            if in_seg:
                missed_segs.append((seg_start, i - 1))
                in_seg = False
            continue
        if not f["detected"]:
            if not in_seg:
                seg_start = i
                in_seg = True
        else:
            if in_seg:
                missed_segs.append((seg_start, i - 1))
                in_seg = False
    if in_seg:
        missed_segs.append((seg_start, total - 1))

    gap_lengths = [seg[1] - seg[0] + 1 for seg in missed_segs]
    max_gap = max(gap_lengths) if gap_lengths else 0

    # Track Loss：连续漏检 ≥ 3 帧的段
    track_loss = sum(1 for g in gap_lengths if g >= THRESH_GAP)

    # Critical Gap：连续漏检 ≥ THRESH_CRITICAL_GAP 帧的段（目标在画面时的模型失败）
    critical_gaps = [
        seg for seg in missed_segs if (seg[1] - seg[0] + 1) >= THRESH_CRITICAL_GAP
    ]
    n_critical_gaps = len(critical_gaps)

    # VDR / VDR_present
    vdr = round(n_det / total, 6) if total else 0.0
    vdr_present = round(n_det / n_present, 6) if n_present else 0.0

    return {
        "total_frames": total,
        "n_detected": n_det,
        "n_missed": n_miss,
        "n_present": n_present,
        "n_absent": n_absent,
        "VDR": vdr,
        "VDR_present": vdr_present,
        "max_gap": max_gap,
        "track_loss": track_loss,
        "n_critical_gaps": n_critical_gaps,
        "critical_gaps": critical_gaps,
        "missed_segments": missed_segs,
        "absent_segments": absent_segments or [],
    }


# ─── Overlay 可视化 ─────────────────────────────────────────────────────────

COLORS = {
    "detected": (0, 255, 0),  # 绿
    "missed": (0, 0, 255),  # 红
    "jump": (0, 165, 255),  # 橙
    "jump_scale": (255, 0, 255),  # 紫（尺度跳变）
}


def draw_frame(
    frame: np.ndarray,
    detection: dict,
    jump_info: dict | None,
    frame_h: int,
) -> np.ndarray:
    """在一帧上画检测框和状态标签。"""
    canvas = frame.copy()
    h_frame, w_frame = canvas.shape[:2]

    detected = detection.get("detected", False)
    cands = detection.get("candidates")
    primary_idx = detection.get("primary_idx", -1)

    if detected and cands and primary_idx >= 0 and primary_idx < len(cands):
        try:
            primary = cands[primary_idx]
            cx = int(round(primary.get("cx", 0)))
            cy = int(round(primary.get("cy", 0)))
            bw = int(round(primary.get("w", 0)))
            bh = int(round(primary.get("h", 0)))
            conf = primary.get("conf", 0.0)
        except (ValueError, TypeError, KeyError):
            cx = cy = bw = bh = 0
            conf = 0.0

        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(w_frame, cx + bw // 2)
        y2 = min(h_frame, cy + bh // 2)

        jtype = (jump_info or {}).get("jump_type", "")
        if jtype == "POS":
            color = COLORS["jump"]
        elif jtype == "SCALE":
            color = COLORS["jump_scale"]
        elif jtype == "POS+SCALE":
            color = (255, 255, 0)
        else:
            color = COLORS["detected"]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"F{detection.get('frame_idx', '?')} conf={conf:.2f}"
        if jtype:
            label += f" [{jtype}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 2, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
    else:
        label = f"F{detection.get('frame_idx', '?')} [MISSED]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cx_default = w_frame // 2
        cy_default = h_frame // 2
        cv2.putText(
            canvas,
            label,
            (cx_default - tw // 2, cy_default),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLORS["missed"],
            1,
        )

    return canvas


def generate_overlay(
    video_path: str,
    frames_data: list[dict],
    jump_frames: list[dict],
    output_path: str | Path,
    max_frames: int = 60,
) -> None:
    """
    生成帧级 overlay 拼接图。
    每视频最多 max_frames 帧（每隔 N 帧取一帧），拼成网格图。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    jump_set = {j["frame_idx"] for j in jump_frames}
    jump_map = {j["frame_idx"]: j for j in jump_frames}

    total = len(frames_data)
    step = max(1, total // max_frames)
    sampled = list(range(0, total, step))[:max_frames]

    # 网格布局
    cols = min(8, len(sampled))
    rows = (len(sampled) + cols - 1) // cols

    thumb_w = 160
    thumb_h = 90
    canvas = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

    for grid_idx, frame_idx in enumerate(sampled):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        thumb = cv2.resize(frame, (thumb_w, thumb_h))
        det = frames_data[frame_idx]
        jinfo = jump_map.get(frame_idx)
        annotated = draw_frame(thumb, det, jinfo, thumb_h)

        r = grid_idx // cols
        c = grid_idx % cols
        canvas[r * thumb_h : (r + 1) * thumb_h, c * thumb_w : (c + 1) * thumb_w] = (
            annotated
        )

    cap.release()
    cv2.imwrite(str(output_path), canvas)


def save_missed_frames(
    video_path: str,
    frames_data: list[dict],
    missed_frame_indices: list[int],
    output_dir: str | Path,
) -> int:
    """保存漏检帧截图到目录。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    saved = 0
    for fi in missed_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(out_dir / f"frame_{fi:04d}.jpg"), frame)
            saved += 1

    cap.release()
    return saved


# ─── 主审计流程 ─────────────────────────────────────────────────────────────


def compute_jump_rate(jump_summary: dict, n_det: int) -> float:
    """跳变帧率 = 跳变帧数 / 检测帧数。"""
    if n_det <= 0:
        return 0.0
    total_jumps = jump_summary.get("total", 0)
    return round(total_jumps / n_det, 6)


def audit_video(
    video_path: str,
    model_path: str,
    output_dir: str | Path,
    conf_thresh: float = 0.05,
    iou_thresh: float = 0.45,
    top_k: int = 5,
    generate_overlays: bool = True,
    absent_segments: list[tuple[int, int]] | None = None,
) -> dict:
    """
    对单个视频执行 Phase 0 审计，返回详细结果字典。

    absent_segments: [(start, end), ...] — 目标不在画面的帧区间（人工标注）。
    """
    video_id = Path(video_path).name
    out_dir = Path(output_dir)

    # Step 1: 检测
    frames, meta = detect_video_phase0(
        video_path,
        model_path,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        top_k=top_k,
    )

    # Step 2: 写原始 JSON
    raw_json_path = out_dir / f"{video_id}_phase0_raw.json"
    write_phase0_json(
        raw_json_path,
        video_path,
        model_path,
        frames,
        meta,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        top_k=top_k,
    )

    frames_data = [f.to_dict() for f in frames]

    # Step 3: 指标计算
    miss = compute_miss_metrics(frames_data, absent_segments=absent_segments)
    jump_frames, jump_summary = compute_jump_frames(frames_data)
    conf_curve = compute_conf_curve(str(raw_json_path))
    jump_rate = compute_jump_rate(jump_summary, miss["n_detected"])

    # Phase 1 入口判据
    passes_vdr = miss["VDR_present"] >= 0.995
    passes_cgaps = miss["n_critical_gaps"] == 0
    passes_jumprate = jump_rate <= THRESH_JUMP_RATE
    passes_phase1 = passes_vdr and passes_cgaps and passes_jumprate

    result = {
        "video_id": video_id,
        "config": {
            "model_hash": model_hash(model_path),
            "conf_thresh": conf_thresh,
            "iou_thresh": iou_thresh,
            "top_k": top_k,
            "version": "phase0-v3",
        },
        "metadata": meta,
        **miss,
        "jump_summary": jump_summary,
        "jump_frames": jump_frames,
        "jump_rate": jump_rate,
        "conf_curve": conf_curve,
        "phase1_entry": {
            "pass_vdr": passes_vdr,
            "pass_critical_gaps": passes_cgaps,
            "pass_jump_rate": passes_jumprate,
            "pass_all": passes_phase1,
        },
    }

    # Step 4: Overlay
    if generate_overlays:
        overlay_path = out_dir / f"{video_id}_overlay.jpg"
        generate_overlay(video_path, frames_data, jump_frames, overlay_path)

        # 漏检帧截图
        missed_idx = [i for i, f in enumerate(frames_data) if not f["detected"]]
        if missed_idx:
            save_missed_frames(
                video_path, frames_data, missed_idx, out_dir / f"{video_id}_missed"
            )

    # Step 5: 写单视频报告
    coverage_path = out_dir / f"{video_id}_coverage.json"
    try:
        with open(coverage_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"[警告] 无法写入 {coverage_path.name}: {exc}")

    return result


# ─── CLI ────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Phase 0 纯检测基线 — 覆盖率审计")
    ap.add_argument(
        "--input-dir",
        default="validation/dataset_benchmark/raw_videos/",
        help="视频目录（默认: validation/dataset_benchmark/raw_videos/）",
    )
    ap.add_argument(
        "--output-dir",
        default="validation/reports/coverage_phase0/",
        help="输出目录（默认: validation/reports/coverage_phase0/）",
    )
    ap.add_argument(
        "--model",
        default="models/best.onnx",
        help="ONNX 模型路径（默认: models/best.onnx）",
    )
    ap.add_argument(
        "--dataset-index",
        default="validation/dataset_benchmark/dataset_index.json",
        help="数据集索引路径",
    )
    ap.add_argument(
        "--tag",
        default="phase0",
        help="报告后缀（默认: phase0）",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="只跑文件名包含子串的视频（逗号分隔）",
    )
    ap.add_argument(
        "--hold-out",
        action="store_true",
        help="只跑 dataset_index 中 hold_out=true 的视频",
    )
    ap.add_argument(
        "--no-overlay",
        action="store_true",
        help="跳过 overlay 生成（加快批量测试）",
    )
    ap.add_argument(
        "--absent-segments",
        default="validation/dataset_benchmark/absent_segments.json",
        help="目标不在画面的帧区间标注文件（默认: absent_segments.json）",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    input_dir = (
        (repo / args.input_dir)
        if not Path(args.input_dir).is_absolute()
        else Path(args.input_dir)
    )
    output_dir = (
        (repo / args.output_dir)
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    ds_path = (
        (repo / args.dataset_index)
        if not Path(args.dataset_index).is_absolute()
        else Path(args.dataset_index)
    )
    if ds_path.exists():
        try:
            with open(ds_path) as f:
                dataset = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[警告] 数据集索引读取失败: {exc}，扫描目录替代")
            dataset = [{"video_id": p.name} for p in input_dir.glob("*.mp4")]
    else:
        # fallback: 扫描 input_dir
        dataset = [{"video_id": p.name} for p in input_dir.glob("*.mp4")]
        print(f"[警告] 未找到 dataset_index，从目录扫描到 {len(dataset)} 个视频")

    if args.hold_out:
        dataset = [d for d in dataset if d.get("hold_out", False)]
        print(f"--hold-out → {len(dataset)} 个视频")

    if args.only:
        subs = [s.strip() for s in args.only.split(",") if s.strip()]
        dataset = [d for d in dataset if any(s in d["video_id"] for s in subs)]
        print(f"--only {subs} → {len(dataset)} 个视频")

    # 加载 absent_segments 标注
    absent_path = Path(args.absent_segments)
    if absent_path.exists():
        try:
            with open(absent_path, encoding="utf-8") as f:
                absent_raw = json.load(f)
            absent_map = {k: [tuple(seg) for seg in v] for k, v in absent_raw.items()}
            print(f"加载 absent_segments: {absent_path} ({len(absent_map)} 条已标注)")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[警告] absent_segments 读取失败: {exc}，跳过")
            absent_map = {}
    else:
        absent_map = {}
        print(f"[提示] absent_segments 文件不存在，跳过标注区间（{absent_path}）")

    print(f"\n{'=' * 60}")
    print(f"Phase 0 Coverage Audit — {args.tag}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"模型: {args.model}")
    print(f"视频数: {len(dataset)}")
    print(f"{'=' * 60}\n")

    t_start = time.time()
    results = []
    for k, item in enumerate(dataset):
        vid = item["video_id"]
        vpath = input_dir / vid
        if not vpath.exists():
            print(f"[{k + 1}/{len(dataset)}] 跳过（文件不存在）: {vid}")
            continue

        print(f"[{k + 1:>2}/{len(dataset)}] {vid}", end=" ... ", flush=True)
        try:
            absent_segs = absent_map.get(vid)
            result = audit_video(
                str(vpath),
                args.model,
                str(output_dir),
                generate_overlays=not args.no_overlay,
                absent_segments=absent_segs,
            )
            vdr = result["VDR"]
            vdr_p = result["VDR_present"]
            max_gap = result["max_gap"]
            track_loss = result["track_loss"]
            n_jumps = result["jump_summary"]["total"]
            print(
                f"VDR={vdr:.4f} VDRp={vdr_p:.4f} "
                f"MaxGap={max_gap} TrackLoss={track_loss} Jumps={n_jumps}"
            )
            results.append(result)
        except Exception as ex:
            print(f"[错误] {ex}")
            results.append({"video_id": vid, "error": str(ex)})

    total_min = (time.time() - t_start) / 60

    # ── 汇总 ────────────────────────────────────────────────────────────────
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if valid:
        n = len(valid)
        vdrs = [r["VDR"] for r in valid]
        vdrs_p = [r["VDR_present"] for r in valid]
        max_gaps = [r["max_gap"] for r in valid]
        track_losses = [r["track_loss"] for r in valid]
        total_jumps = sum(r["jump_summary"]["total"] for r in valid)
        all_detected = sum(1 for r in valid if r["VDR"] == 1.0)
        all_track_ok = sum(1 for r in valid if r["track_loss"] == 0)

        critical_gaps_list = [r.get("n_critical_gaps", 0) for r in valid]
        jump_rates = [r.get("jump_rate", 0.0) for r in valid]
        pass_phase1 = [r.get("phase1_entry", {}).get("pass_all", False) for r in valid]

        try:
            summary = {
                "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tag": args.tag,
                "model": args.model,
                "n_videos": n,
                "n_errors": len(errors),
                "total_minutes": round(total_min, 1),
                "VDR_mean": round(float(np.mean(vdrs)), 6),
                "VDR_min": round(float(np.min(vdrs)), 6),
                "VDR_present_mean": round(float(np.mean(vdrs_p)), 6),
                "VDR_present_min": round(float(np.min(vdrs_p)), 6),
                "MaxGap_max": int(max(max_gaps)),
                "MaxGap_mean": round(float(np.mean(max_gaps)), 2),
                "TrackLoss_total": int(sum(track_losses)),
                "TrackLoss_videos": sum(1 for t in track_losses if t > 0),
                "CriticalGaps_total": int(sum(critical_gaps_list)),
                "CriticalGaps_videos": sum(1 for cg in critical_gaps_list if cg > 0),
                "JumpFrames_total": int(total_jumps),
                "JumpRate_mean": round(float(np.mean(jump_rates)), 6),
                "JumpRate_max": round(float(np.max(jump_rates)), 6),
                "VDR_full_coverage": f"{all_detected}/{n}",
                "TrackLoss_zero": f"{all_track_ok}/{n}",
                "Phase1_pass_all": f"{sum(pass_phase1)}/{n}",
                "phase1_entry": {
                    "VDR_present_min": round(float(np.min(vdrs_p)), 6),
                    "VDR_present >= 99.5% AND CriticalGaps == 0 AND JumpRate <= 0.5%": (
                        sum(pass_phase1) == n
                    ),
                },
                "rows": valid,
            }
        except (ValueError, TypeError) as exc:
            print(f"[警告] 汇总统计计算失败: {exc}")
            summary = {"error": str(exc), "n_videos": 0}
    else:
        summary = {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tag": args.tag,
            "n_videos": 0,
            "error": "no valid results",
        }

    # 写汇总 JSON
    sum_path = output_dir / f"COVERAGE_PHASE0_{args.tag}.json"
    try:
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    except (OSError, TypeError) as exc:
        print(f"[错误] 无法写入汇总 JSON: {exc}")

    # ── Markdown 报告 ────────────────────────────────────────────────────────
    lines = [
        f"# Phase 0 Coverage Report — {args.tag}",
        "",
        f"- 生成: {summary['generated']}",
        f"- 模型: `{args.model}`",
        f"- 视频: {summary['n_videos']} | 错误: {summary['n_errors']}",
        f"- 总耗时: {summary['total_minutes']:.1f} 分钟",
        "",
        "## 入口条件（Phase 0 → Phase 1）",
        "",
        "```text",
        "VDR_present >= 99.5%  AND",
        "Track_Loss == 0",
        "```",
        "",
        f"- **VDR 严格全帧均值: {summary.get('VDR_mean', 'N/A')}**",
        f"- **VDR_present 均值: {summary.get('VDR_present_mean', 'N/A')}**",
        f"- **VDR_present 最小值: {summary.get('VDR_present_min', 'N/A')}** （红线 ≥ 99.5%）",
        f"- Critical Gaps 总计: {summary.get('CriticalGaps_total', 'N/A')} | 触发视频: {summary.get('CriticalGaps_videos', 'N/A')}/{len(valid) if valid else 0}",
        f"- Jump Rate 均值: {summary.get('JumpRate_mean', 'N/A')} | 最大: {summary.get('JumpRate_max', 'N/A')} （红线 ≤ 0.5%）",
        f"- 跳变帧总数: {summary.get('JumpFrames_total', 'N/A')}",
        f"- Phase 1 入口通过: {summary.get('Phase1_pass_all', 'N/A')}",
        "",
        "| 视频 | VDR_present | CriticalGaps | JumpRate | Phase1 | N_absent | N_miss | N_total |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(valid, key=lambda x: x["video_id"]):
        pe = r.get("phase1_entry", {})
        pass_str = "✅" if pe.get("pass_all") else "❌"
        n_abs = r.get("n_absent", 0)
        n_miss = r.get("n_missed", 0)
        n_tot = r.get("total_frames", 0)
        lines.append(
            f"| {r['video_id']} | {r['VDR_present']:.4f} "
            f"| {r.get('n_critical_gaps', 0)} "
            f"| {r.get('jump_rate', 0.0):.4f} "
            f"| {pass_str} "
            f"| {n_abs} | {n_miss} | {n_tot} |"
        )

    lines += [
        "",
        "## 跳变帧详情（人工过审重点）",
        "",
        "| 视频 | 跳变帧数 | POS跳 | SCALE跳 | POS+SCALE |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(valid, key=lambda x: x["video_id"]):
        js = r["jump_summary"]
        lines.append(
            f"| {r['video_id']} | {js['total']} | {js['POS']} | "
            f"{js['SCALE']} | {js['POS+SCALE']} |"
        )

    if errors:
        lines += ["", "## 错误", ""]
        for r in errors:
            lines.append(f"- **{r['video_id']}**: {r['error']}")

    lines += [
        "",
        "## 原始数据",
        "",
        f"详细结果: `{sum_path}`",
        "Overlay 图: `{video_id}_overlay.jpg`",
        "漏检帧截图: `{video_id}_missed/`",
    ]

    md_path = output_dir / f"COVERAGE_PHASE0_{args.tag}.md"
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except OSError as exc:
        print(f"[警告] 无法写入 {md_path}: {exc}")

    print(f"\n{'=' * 60}")
    print(
        f"审计完成 — {summary['n_videos']} 个视频 | {summary['total_minutes']:.1f} 分钟"
    )
    if valid:
        print(
            f"VDR 均值: {summary['VDR_mean']} | VDR_present 均值: {summary['VDR_present_mean']}"
        )
        print(f"VDR_present 最小值: {summary['VDR_present_min']} （红线: 99.5%）")
        print(
            f"Track Loss 总计: {summary['TrackLoss_total']} | 触发视频: {summary['TrackLoss_videos']}/{summary['n_videos']}"
        )
        print(f"跳变帧总数: {summary['JumpFrames_total']}")
        pe = summary.get("phase1_entry", {})
        cond_key = "VDR_present >= 99.5% AND CriticalGaps == 0 AND JumpRate <= 0.5%"
        cond_val = pe.get(cond_key, "N/A") if isinstance(pe, dict) else "N/A"
        print(f"入口条件满足: {cond_val}")
    print(f"报告: {md_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
