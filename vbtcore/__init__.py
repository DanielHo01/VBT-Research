"""
vbtcore — VBT Core Engine v5（重构版）
=========================================
Step 4 重构（2026-09-12）核心改动：
  1. calibrator.py  静态中位数锁死 + CV 变异系数门禁
  2. tracker.py     物理空间卡尔曼（米/秒）+ LK 光流密集跟踪
  3. segmenter.py    速度状态机分段（SSC + 硬拉两种拓扑）
  4. pipeline.py     PTS 时间戳 + 密集跟踪集成
Benchmark 结果：14/34 (41%) 计数通过（vs 旧版 3/34 (9%)）
"""

from .calibrator import StaticPlateCalibrator
from .detector import Detection, PlateDetector
from .engine import (
  BottomRegrind,
  DetectFitTracker,
  TrackDiagnostics,
  anchor_score,
  regrind_verdict,
  select_regrind_candidate,
)
from .geometry import (
  PLATE_DIAMETERS_M,
  compute_mpp,
  extract_plate_crop,
  fit_plate_ellipse,
  resolve_plate_diameter,
)
from .kalman import BarbellKalmanTracker
from .pipeline import SetResult, StatusCodes, analyze_video
from .segment import Rep, SegmentResult, segment_reps
from .segmenter import BiomechanicalRepSegmenter
from .tracker import DenseVisualTracker, KinematicKalmanTracker
from .tracker_ek import EKTrackDiagnostics, EllipseKalmanTracker

__all__ = [
  "PlateDetector",
  "Detection",
  "DetectFitTracker",
  "TrackDiagnostics",
  "anchor_score",
  "BottomRegrind",
  "select_regrind_candidate",
  "regrind_verdict",
  "resolve_plate_diameter",
  "PLATE_DIAMETERS_M",
  "extract_plate_crop",
  "fit_plate_ellipse",
  "compute_mpp",
  "BarbellKalmanTracker",
  "EllipseKalmanTracker",
  "EKTrackDiagnostics",
  "StaticPlateCalibrator",
  "KinematicKalmanTracker",
  "DenseVisualTracker",
  "BiomechanicalRepSegmenter",
  "Rep",
  "SegmentResult",
  "segment_reps",
  "analyze_video",
  "SetResult",
  "StatusCodes",
]
