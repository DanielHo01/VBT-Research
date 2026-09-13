"""
vbtcore.pipeline — 重构版端到端视频分析入口
============================================
新架构（Step 4 Refactor）：
  · StaticPlateCalibrator  → 尺度标定（CV 门禁）
  · DenseVisualTracker     → LK 光流 + YOLO 物理空间卡尔曼
  · BiomechanicalRepSegmenter → 速度 FSM 分段（深蹲/卧推 + 硬拉）
  · 统一 PTS 时间戳，消灭 px/frame 量纲混乱

analyze_video(video_path, ...) -> SetResult（接口与旧版完全兼容）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from .calibrator import StaticPlateCalibrator
from .detector import PlateDetector
from .segmenter import BiomechanicalRepSegmenter, Rep
from .tracker import DenseVisualTracker


class StatusCodes:
    OK = "OK"
    NO_PLATE_DETECTED = "NO_PLATE_DETECTED"
    NO_CLEAN_SEGMENT = "NO_CLEAN_SEGMENT"
    TOO_SHORT = "TOO_SHORT"
    VIDEO_ERROR = "VIDEO_ERROR"
    CALIBRATION_FAILED = "CALIBRATION_FAILED"


@dataclass
class SetResult:
    """与 benchmark 兼容的输出格式"""

    video: str
    status: str
    reps: list = field(default_factory=list)  # list[Rep] (segmenter.Rep)
    mcv: list[float] = field(default_factory=list)
    mcv_mid: list[float] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    fps: float = 30.0
    mpp: float | None = None


def analyze_video(
    video_path: str,
    model_path: str,
    redet_every: int = 15,
    user_hint: tuple[float, float] | None = None,
    plate_diameter_m: float = 0.45,
    outer_plate: str | None = None,
    regrind_enabled: bool = True,
    detector: PlateDetector | None = None,
    exercise_type: Literal["squat_bench", "deadlift"] = "squat_bench",
) -> SetResult:
    """
    重构版端到端流水线。

    参数
    ─────
    exercise_type : "squat_bench"（SSC，离心→向心）或 "deadlift"（直接向心）
    其余参数与旧版保持兼容。
    """
    t0 = time.perf_counter()

    det = detector or PlateDetector(model_path)
    cap = cv2.VideoCapture(video_path)
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    cap.release()

    if n_frames_total < 30:
        return SetResult(
            video=video_path,
            status=StatusCodes.VIDEO_ERROR,
            diagnostics={"reason": f"frames={n_frames_total}"},
        )

    # ── 初始化标定器 ──────────────────────────────────────────────
    calibrator = StaticPlateCalibrator(
        real_diameter_m=plate_diameter_m,
        min_static_frames=20,
        max_cv=0.015,
    )
    tracker: DenseVisualTracker | None = None
    mpp: float | None = None

    timestamps: list[float] = []
    positions: list[float] = []  # 向上为正（米）— 内部取反
    velocities: list[float] = []  # 向上为正（米/秒）— 内部取反

    status = StatusCodes.OK
    frame_idx = 0
    n_yolo_frames = 0
    ms_total = 0.0

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t_frame_start = time.perf_counter()

        # ── 真实 PTS 时间戳（毫秒 → 秒）───────────────────────────
        pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time_s = pts_ms / 1000.0 if pts_ms > 0 else (frame_idx / fps)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── 阶段一：标定（mpp 未锁死）───────────────────────────────
        if mpp is None:
            results = det.detect(frame, conf_thresh=0.45)
            if results is not None and len(results) > 0:
                # 取最大面积检测框
                best = max(results, key=lambda b: b.w * b.h)
                h_px = float(best.h)
                calibrator.add_sample(h_px)

                if calibrator.is_ready():
                    mpp = calibrator.lock_scale()
                    # 转换为 x1,y1,x2,y2 格式供 DenseVisualTracker 使用
                    bbox = (
                        best.cx - best.w / 2,
                        best.cy - best.h / 2,
                        best.cx + best.w / 2,
                        best.cy + best.h / 2,
                    )
                    tracker = DenseVisualTracker(
                        mpp=mpp,
                        initial_bbox=bbox,
                        initial_gray=gray,
                        initial_time_s=current_time_s,
                    )

            ms_total += (time.perf_counter() - t_frame_start) * 1000
            frame_idx += 1
            continue

        # ── 阶段二：密集跟踪（标定已锁死）─────────────────────────
        is_keyframe = frame_idx % redet_every == 0

        if is_keyframe:
            results = det.detect(frame, conf_thresh=0.40)
            if results is not None and len(results) > 0:
                best = max(results, key=lambda b: b.w * b.h)
                bbox = (
                    best.cx - best.w / 2,
                    best.cy - best.h / 2,
                    best.cx + best.w / 2,
                    best.cy + best.h / 2,
                )
                y_m, v_mps = tracker.step_keyframe(gray, bbox, current_time_s)
                n_yolo_frames += 1
            else:
                y_m, v_mps = tracker.step_interframe(gray, current_time_s)
        else:
            y_m, v_mps = tracker.step_interframe(gray, current_time_s)

        # 图像 y 向下为正 → 取反为向上（物理正方向）
        timestamps.append(current_time_s)
        positions.append(-y_m)
        velocities.append(-v_mps)

        ms_total += (time.perf_counter() - t_frame_start) * 1000
        frame_idx += 1

    cap.release()

    elapsed_s = time.perf_counter() - t0

    # ── 诊断信息 ──────────────────────────────────────────────────
    coverage = frame_idx / max(n_frames_total, 1)
    yolo_ratio = n_yolo_frames / max(frame_idx, 1)
    diag = {
        "coverage": round(coverage, 3),
        "ms_per_frame": round(ms_total / max(frame_idx, 1), 1),
        "yolo_ratio": round(yolo_ratio, 3),
        "n_frames": frame_idx,
        "n_yolo_frames": n_yolo_frames,
        "elapsed_s": round(elapsed_s, 1),
        "plate_diameter_m": plate_diameter_m,
        "outer_plate": outer_plate,
        "exercise_type": exercise_type,
        "tracker": "dense_visual_kalman",
        "calibrator": "static_cv_gate",
    }

    if mpp is None:
        return SetResult(
            video=video_path,
            status=StatusCodes.NO_PLATE_DETECTED,
            fps=fps,
            diagnostics=diag,
        )

    # ── 阶段三：Rep 分段 ─────────────────────────────────────────
    t_arr = np.array(timestamps, dtype=float)
    y_arr = np.array(positions, dtype=float)
    v_arr = np.array(velocities, dtype=float)

    segmenter = BiomechanicalRepSegmenter(exercise_type=exercise_type)
    raw_reps: list[Rep] = segmenter.segment(t_arr, y_arr, v_arr)

    mcv_vals = [r.mcv_mps for r in raw_reps]
    mcv_mid_vals = [r.pcv_mps for r in raw_reps]

    return SetResult(
        video=video_path,
        status=StatusCodes.OK if raw_reps else StatusCodes.NO_CLEAN_SEGMENT,
        reps=raw_reps,
        mcv=mcv_vals,
        mcv_mid=mcv_mid_vals,
        diagnostics=diag,
        fps=fps,
        mpp=mpp,
    )
