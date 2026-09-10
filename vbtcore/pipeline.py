"""
vbtcore.pipeline — 端到端视频分析入口
=====================================
analyze_video(video_path, ...) -> SetResult
  reps + 完整诊断（永不静默失败：一切失败都有状态码）。

状态码（Stage 1 验收口径）：
  OK                  正常出数
  NO_PLATE_DETECTED   锚定失败（含"只有杆"场景 —— 应被 App 引导加片）
  NO_CLEAN_SEGMENT    跟到了轨迹但没有可用干净段
  TOO_SHORT           干净段过短
  VIDEO_ERROR         视频打不开/帧数不足
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import PlateDetector
from .engine import DetectFitTracker, TrackDiagnostics
from .geometry import resolve_plate_diameter
from .segment import Rep, SegmentResult, segment_reps


class StatusCodes:
    OK = "OK"
    NO_PLATE_DETECTED = "NO_PLATE_DETECTED"
    NO_CLEAN_SEGMENT = "NO_CLEAN_SEGMENT"
    TOO_SHORT = "TOO_SHORT"
    VIDEO_ERROR = "VIDEO_ERROR"


@dataclass
class SetResult:
    video: str
    status: str
    reps: list[Rep] = field(default_factory=list)
    mcv: list[float] = field(default_factory=list)
    mcv_mid: list[float] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    fps: float = 30.0
    mpp: float | None = None


def analyze_video(video_path: str,
                  model_path: str,
                  redet_every: int = 15,
                  user_hint: tuple[float, float] | None = None,
                  plate_diameter_m: float = 0.45,
                  outer_plate: str | None = None,
                  regrind_enabled: bool = True,
                  detector: PlateDetector | None = None) -> SetResult:
    """单视频 → SetResult。detector 可复用以省模型加载时间。
    outer_plate（如 "20kg"）提供时按查表覆盖 plate_diameter_m（M1.5）。
    regrind_enabled=False 关闭底部重锚定（烧蚀实验用）。"""
    det = detector or PlateDetector(model_path)
    diameter = (resolve_plate_diameter(outer_plate) if outer_plate
                else plate_diameter_m)
    tracker = DetectFitTracker(
        det, redet_every=redet_every,
        plate_diameter_m=diameter,
        user_hint=user_hint,
        regrind_enabled=regrind_enabled,
    )

    cap = cv2.VideoCapture(video_path)
    n_probe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if not n_probe or n_probe < 30:
        return SetResult(video=video_path, status=StatusCodes.VIDEO_ERROR,
                         diagnostics={"reason": f"frames={n_probe}"})

    y_track, h_samples, diag, fps = tracker.process(video_path)

    result = SetResult(
        video=video_path,
        status=diag.status or StatusCodes.OK,
        fps=fps,
        diagnostics={
            "coverage": round(diag.coverage, 3),
            "ms_per_frame": round(diag.ms_per_frame, 1),
            "yolo_ratio": round(diag.yolo_ratio, 3),
            "n_frames": diag.n_frames,
            "src_counts": diag.src_counts,
            "ncc_mean": round(diag.ncc_mean, 3),
            "anchor_frame": diag.anchor_frame,
            "anchor_h_px": round(diag.anchor_h_px, 1),
            "elapsed_s": round(diag.elapsed_s, 1),
            "plate_diameter_m": diameter,
            "outer_plate": outer_plate,
            "notes": diag.notes,
            "n_regrind_snap": diag.n_regrind_snap,
            "n_regrind_micro": diag.n_regrind_micro,
            "n_regrind_reject": diag.n_regrind_reject,
        },
    )

    mpp = tracker.calibrate(h_samples)
    result.mpp = mpp
    if mpp is None:
        result.status = (result.status if result.status != "TRACKED"
                         else StatusCodes.NO_CLEAN_SEGMENT)
        result.diagnostics["reason"] = "标定失败（片高样本无效）"
        return result

    if y_track is None or not np.any(~np.isnan(y_track)):
        return result  # NO_PLATE_DETECTED 等，无轨迹

    seg: SegmentResult = segment_reps(y_track, fps, mpp)
    result.reps = seg.reps
    result.mcv = [r.mcv for r in seg.reps]
    result.mcv_mid = [r.mcv_mid for r in seg.reps]
    result.status = StatusCodes.OK if seg.reps else seg.status
    result.diagnostics["segment_note"] = seg.note
    result.diagnostics["n_clipped"] = sum(1 for r in seg.reps if r.clipped)
    return result
