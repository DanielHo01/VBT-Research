"""
vbtcore — VBT Core Engine (Step 4 Refactor, v0.1.0-baseline)
============================================================
Phase 1 cleanup (2026-09-12)：
  - 公共 API 收敛到 9 个核心符号
  - engine.py / tracker_ek.py / detector_phase0.py 已归档
  - segment.py 改为 deprecation shim（兼容 1 个 release 周期）

新架构（Step 4 Refactor）：
  · StaticPlateCalibrator  → 尺度标定（CV 门禁）
  · DenseVisualTracker     → LK 光流 + YOLO 物理空间卡尔曼
  · BiomechanicalRepSegmenter → 速度 FSM 分段（深蹲/卧推 + 硬拉）
  · 统一 PTS 时间戳，消灭 px/frame 量纲混乱

成绩：不在此处抄写数字。全仓库唯一事实表见
      validation/reports/SCOREBOARD.md
"""

from __future__ import annotations

from .calibrator import StaticPlateCalibrator
from .detector import Detection, PlateDetector
from .pipeline import SetResult, StatusCodes, analyze_video
from .segmenter import BiomechanicalRepSegmenter, Rep
from .tracker import DenseVisualTracker, KinematicKalmanTracker

# 可从子模块访问（非核心 9）：
#   vbtcore.geometry.PLATE_DIAMETERS_M / compute_mpp / fit_plate_ellipse / resolve_plate_diameter
#   vbtcore.kalman.BarbellKalmanTracker

# 公共 API：9 个核心符号
__all__ = [
    # 检测
    "PlateDetector",
    "Detection",
    # 标定
    "StaticPlateCalibrator",
    # 跟踪
    "KinematicKalmanTracker",
    "DenseVisualTracker",
    # 分段
    "BiomechanicalRepSegmenter",
    "Rep",
    # 入口
    "SetResult",
    "StatusCodes",
    "analyze_video",
]

# ────────────────────────────────────────────────────────────────────────
# 兼容性 shim（v0.1.x 周期内保留，后续移除）
# ────────────────────────────────────────────────────────────────────────


def __getattr__(name: str):
    """
    向后兼容层：当用户访问已归档符号时，从 _archive/ 加载并发出
    DeprecationWarning。新代码应使用上述 9 个核心符号。
    """
    import importlib.util as _ilu
    import sys as _sys
    import warnings as _w
    from pathlib import Path as _P

    _HERE = _P(__file__).resolve().parent
    _ARCHIVE_DIR = _HERE.parent / "_archive" / "vbtcore_history"

    _ARCHIVED = {
        # engine.py → DetectFitTracker, TrackDiagnostics, anchor_score, regrind_*
        "DetectFitTracker": ("engine_v2_reverted.py", ["DetectFitTracker"]),
        "TrackDiagnostics": ("engine_v2_reverted.py", ["TrackDiagnostics"]),
        "anchor_score": ("engine_v2_reverted.py", ["anchor_score"]),
        "BottomRegrind": ("engine_v2_reverted.py", ["BottomRegrind"]),
        "select_regrind_candidate": (
            "engine_v2_reverted.py",
            ["select_regrind_candidate"],
        ),
        "regrind_verdict": ("engine_v2_reverted.py", ["regrind_verdict"]),
        # tracker_ek.py → EllipseKalmanTracker, EKTrackDiagnostics
        "EllipseKalmanTracker": ("tracker_ek_v4.py", ["EllipseKalmanTracker"]),
        "EKTrackDiagnostics": ("tracker_ek_v4.py", ["EKTrackDiagnostics"]),
        # detector_phase0.py → detect_video_phase0, write_phase0_json, model_hash
        "detect_video_phase0": (
            "detector_phase0_v0.py",
            ["detect_video_phase0"],
        ),
        "write_phase0_json": (
            "detector_phase0_v0.py",
            ["write_phase0_json"],
        ),
        "model_hash": ("detector_phase0_v0.py", ["model_hash"]),
    }

    if name not in _ARCHIVED:
        raise AttributeError(f"module 'vbtcore' has no attribute {name!r}")

    fname, attrs = _ARCHIVED[name]
    modname = f"vbtcore._archive_{fname[:-3]}"
    if modname not in _sys.modules:
        spec = _ilu.spec_from_file_location(modname, str(_ARCHIVE_DIR / fname))
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载归档模块 {fname}")
        mod = _ilu.module_from_spec(spec)
        _sys.modules[modname] = mod
        spec.loader.exec_module(mod)

    mod = _sys.modules[modname]
    for attr in attrs:
        if hasattr(mod, attr):
            setattr(mod, attr, getattr(mod, attr))  # noqa: F841 (idempotent)
    _w.warn(
        f"vbtcore.{name} 已废弃（v0.1.0-baseline 起归档至 "
        f"_archive/vbtcore_history/{fname}）。"
        f"新代码请使用 vbtcore 公共 API（见 vbtcore.__all__）。",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(mod, name)
