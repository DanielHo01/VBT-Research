"""
vbtcore — VBT Core Engine v1（M0 基建版）
=========================================
修复的四个历史 bug（详见各模块 docstring）：
  1. geometry.py    竖屏旋转逆映射镜像错误（可视化实证）
  2. detector.py    置信度双重 sigmoid + 直 resize 挤压畸变
  3. engine.py      锚定最高置信≠工作片（物理先验+运动探针+点选兜底）
                    + 魔法系数 scale_factor=1.15 移除
  4. segment.py     MCV 定义统一（向心段全程平均，对齐 GymAware ACV）
M1.5（移植 TroyKaneshiro/barbell-velocity-tracker）：
  engine.py       rep 底部重锚定 regrind（UP/DOWN 双相，真正每 rep 一次）
  geometry.py     外层片直径查表（非 450mm 铁片不再硬假设）
架构：检测→拟合混合跟踪（每 N 帧 YOLO 重检测 + NCC 模板 + 匀速预测），
实测 ~12ms/帧（CPU），YOLO 调用率 7-14%。
"""
from .detector import PlateDetector, Detection
from .engine import (DetectFitTracker, TrackDiagnostics, anchor_score,
                     BottomRegrind, select_regrind_candidate,
                     regrind_verdict)
from .geometry import resolve_plate_diameter, PLATE_DIAMETERS_M
from .segment import Rep, SegmentResult, segment_reps
from .pipeline import analyze_video, SetResult, StatusCodes

__all__ = [
    "PlateDetector", "Detection",
    "DetectFitTracker", "TrackDiagnostics", "anchor_score",
    "BottomRegrind", "select_regrind_candidate", "regrind_verdict",
    "resolve_plate_diameter", "PLATE_DIAMETERS_M",
    "Rep", "SegmentResult", "segment_reps",
    "analyze_video", "SetResult", "StatusCodes",
]
