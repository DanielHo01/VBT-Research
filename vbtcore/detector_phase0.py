"""
vbtcore.detector_phase0 — 纯检测基线（Phase 0）
=================================================
铁律：零卡尔曼、零光流（LK）、零 mpp/真实尺度换算、零 Rep 估算、零按视频调参。

逐帧 YOLO 检测，低阈值全量输出 + top-k 候选，原始数据零丢失。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .detector import PlateDetector

# ─── 数据类 ─────────────────────────────────────────────────────────────────


@dataclass
class CandidateBox:
    """单个候选框（原始帧坐标）。"""

    cx: float
    cy: float
    w: float
    h: float
    conf: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameDetection:
    """
    逐帧检测结果。

    detected=True 表示有候选框（不等于"正确"——目标可能不在画面），
    需联合跳变帧审计与 overlay 人工过审判定。
    """

    frame_idx: int
    pts_s: float
    candidates: list[CandidateBox]  # top-k 候选，按 conf 降序
    primary_idx: int  # 主框索引（最大面积）
    detected: bool  # 有候选框（低阈值）

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "pts_s": round(self.pts_s, 3),
            "candidates": [c.to_dict() for c in self.candidates],
            "primary_idx": self.primary_idx,
            "detected": self.detected,
        }


# ─── 核心检测函数 ─────────────────────────────────────────────────────────────


def detect_video_phase0(
    video_path: str,
    model_path: str,
    conf_thresh: float = 0.05,
    iou_thresh: float = 0.45,
    top_k: int = 5,
) -> tuple[list[FrameDetection], dict]:
    """
    逐帧 YOLO 检测，零上层逻辑。

    参数
    ----
    video_path   : 视频路径
    model_path   : ONNX 模型路径
    conf_thresh  : 置信度阈值（默认 0.05，低阈值全量输出）
    iou_thresh   : NMS IoU 阈值（默认 0.45）
    top_k        : 每帧最多保留候选框数（按 conf 降序）

    返回
    ----
    (逐帧结果列表, 视频元数据字典)
    """
    detector = PlateDetector(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps if fps > 0 else 0.0

    frames: list[FrameDetection] = []
    prev_primary: tuple[float, float] | None = None  # 前帧 primary 中心 (cx, cy)
    t_start = time.time()

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        pts_s = frame_idx / fps if fps > 0 else 0.0

        # 全量检测（低阈值）
        raw_dets = detector.detect(frame, conf_thresh=conf_thresh)

        # NMS（按 conf 降序）
        if raw_dets:
            boxes = np.array([[d.cx, d.cy, d.w, d.h, d.conf] for d in raw_dets])
            scores = boxes[:, 4]
            sorted_idx = np.argsort(-scores)
            boxes = boxes[sorted_idx]

            # NMS
            keep = []
            while len(boxes) > 0:
                keep.append(0)
                if len(boxes) == 1:
                    break
                cx1, cy1, w1, h1 = boxes[0, :4]
                x1 = cx1 - w1 / 2
                y1 = cy1 - h1 / 2
                x1o = x1 + w1
                y1o = y1 + h1
                rest = boxes[1:]
                cx2 = rest[:, 0]
                cy2 = rest[:, 1]
                w2 = rest[:, 2]
                h2 = rest[:, 3]
                x2 = cx2 - w2 / 2
                y2 = cy2 - h2 / 2
                x2o = x2 + w2
                y2o = y2 + h2
                inter_x1 = np.maximum(x1, x2)
                inter_y1 = np.maximum(y1, y2)
                inter_x2 = np.minimum(x1o, x2o)
                inter_y2 = np.minimum(y1o, y2o)
                inter_w = np.maximum(0.0, inter_x2 - inter_x1)
                inter_h = np.maximum(0.0, inter_y2 - inter_y1)
                inter = inter_w * inter_h
                area1 = w1 * h1
                area2 = w2 * h2
                union = area1 + area2 - inter
                iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
                boxes = rest[iou <= iou_thresh]

            nms_idx = sorted_idx[keep]
            nms_dets = [raw_dets[i] for i in nms_idx]
        else:
            nms_dets = []

        # Top-k
        k = min(top_k, len(nms_dets))
        top_dets = nms_dets[:k]

        # Primary 选择策略：top-3 候选 + 前帧位置匹配
        # 若前一帧有 primary：选与前帧中心距离最小的候选
        # 若前一帧无检测（primary_prev=None）：fallback 为最大面积
        if top_dets:
            if prev_primary is not None:
                # 前帧有检测 → 选中心距离最小的候选
                px, py = prev_primary[0], prev_primary[1]
                dists = [(d.cx - px) ** 2 + (d.cy - py) ** 2 for d in top_dets]
                primary_idx = int(np.argmin(dists))
            else:
                # 前帧无检测 → fallback 为最大面积
                areas = np.array([d.w * d.h for d in top_dets])
                primary_idx = int(np.argmax(areas))
        else:
            primary_idx = -1

        # 记录本帧 primary 中心，供下一帧使用（仅当检测到时）
        if primary_idx >= 0:
            primary_box = top_dets[primary_idx]
            prev_primary = (primary_box.cx, primary_box.cy)
        else:
            prev_primary = None

        candidates = [
            CandidateBox(
                cx=float(d.cx),
                cy=float(d.cy),
                w=float(d.w),
                h=float(d.h),
                conf=float(d.conf),
            )
            for d in top_dets
        ]

        frames.append(
            FrameDetection(
                frame_idx=frame_idx,
                pts_s=pts_s,
                candidates=candidates,
                primary_idx=primary_idx,
                detected=len(candidates) > 0,
            )
        )

    cap.release()
    elapsed = time.time() - t_start

    metadata = {
        "total_frames": total_frames,
        "fps": round(fps, 3),
        "duration_s": round(duration_s, 3),
        "width": width,
        "height": height,
        "n_detected": sum(1 for f in frames if f.detected),
        "n_missed": sum(1 for f in frames if not f.detected),
        "elapsed_s": round(elapsed, 2),
        "fps_processing": round(total_frames / elapsed, 1) if elapsed > 0 else 0,
    }
    return frames, metadata


# ─── JSON 写入 ───────────────────────────────────────────────────────────────


def model_hash(model_path: str) -> str:
    """SHA-256 hash of the model file."""
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()[:16]}"


def write_phase0_json(
    output_path: str | Path,
    video_path: str,
    model_path: str,
    frames: list[FrameDetection],
    metadata: dict,
    version: str = "phase0-v2",
    conf_thresh: float = 0.05,
    iou_thresh: float = 0.45,
    top_k: int = 5,
    imgsz: int = 640,
) -> dict:
    """
    将 Phase 0 检测结果写入 JSON 文件。

    返回写入的完整 JSON 对象（供 coverage_audit.py 汇总使用）。
    """
    out = {
        "video_id": Path(video_path).name,
        "config": {
            "model_hash": model_hash(model_path),
            "model_path": str(model_path),
            "imgsz": imgsz,
            "letterbox": True,
            "conf_thresh": conf_thresh,
            "iou_thresh": iou_thresh,
            "top_k": top_k,
            "version": version,
        },
        "metadata": metadata,
        "frames": [f.to_dict() for f in frames],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out
