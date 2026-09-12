"""DEPRECATED vbtcore.tracker_ek — 已归档至 _archive/vbtcore_history/tracker_ek_v4.py

此模块保留仅为 1 个 release 周期兼容（v0.1.x），新代码请使用：
  - vbtcore.DenseVisualTracker + vbtcore.KinematicKalmanTracker (LK 光流)
  - vbtcore.BarbellKalmanTracker (1D 卡尔曼)

归档原因：椭圆状态卡尔曼（EllipseKalmanTracker）已被 LK 光流 +
1D 卡尔曼融合替代。椭圆拟合并入 vbtcore.geometry.fit_plate_ellipse，
仅作几何工具，不再参与状态估计。
"""
from __future__ import annotations

import importlib.util as _ilu
import sys as _sys
import warnings as _w
from pathlib import Path as _P

_HERE = _P(__file__).resolve().parent
_ARCHIVE_FILE = _HERE.parent / "_archive" / "vbtcore_history" / "tracker_ek_v4.py"

_w.warn(
    "vbtcore.tracker_ek 已废弃（v0.1.0-baseline 起归档至 "
    "_archive/vbtcore_history/tracker_ek_v4.py）。"
    "新代码请使用 vbtcore.DenseVisualTracker 或 vbtcore.BarbellKalmanTracker。",
    DeprecationWarning,
    stacklevel=2,
)

if _ARCHIVE_FILE.exists():
    spec = _ilu.spec_from_file_location(
        "vbtcore._archive_tracker_ek_v4", str(_ARCHIVE_FILE)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {_ARCHIVE_FILE} 加载归档模块")
    _mod = _ilu.module_from_spec(spec)
    _sys.modules["vbtcore._archive_tracker_ek_v4"] = _mod
    _sys.modules[__name__] = _mod
    spec.loader.exec_module(_mod)
else:
    raise ImportError(f"归档模块不存在：{_ARCHIVE_FILE}")
