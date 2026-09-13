"""DEPRECATED vbtcore.engine — 已归档至 _archive/vbtcore_history/engine_v2_reverted.py

此模块保留仅为 1 个 release 周期兼容（v0.1.x），新代码请使用：
  - vbtcore.pipeline.analyze_video() (主入口)
  - vbtcore.DenseVisualTracker + vbtcore.KinematicKalmanTracker

归档原因：v5 重构后 DetectFitTracker 已被 vbtcore.tracker.DenseVisualTracker +
vbtcore.pipeline.analyze_video 替代（直接 LK 光流 + 物理空间卡尔曼，
不再需要 DetectFitTracker 的稀疏 NCC 模板拟合）。
"""

from __future__ import annotations

import importlib.util as _ilu
import sys as _sys
import warnings as _w
from pathlib import Path as _P

_HERE = _P(__file__).resolve().parent
_ARCHIVE_FILE = _HERE.parent / "_archive" / "vbtcore_history" / "engine_v2_reverted.py"

_w.warn(
  "vbtcore.engine 已废弃（v0.1.0-baseline 起归档至 "
  "_archive/vbtcore_history/engine_v2_reverted.py）。"
  "新代码请使用 vbtcore.analyze_video() 或 vbtcore.DenseVisualTracker。",
  DeprecationWarning,
  stacklevel=2,
)

if _ARCHIVE_FILE.exists():
  spec = _ilu.spec_from_file_location("vbtcore._archive_engine_v2", str(_ARCHIVE_FILE))
  if spec is None or spec.loader is None:
    raise ImportError(f"无法从 {_ARCHIVE_FILE} 加载归档模块")
  _mod = _ilu.module_from_spec(spec)
  _sys.modules["vbtcore._archive_engine_v2"] = _mod
  _sys.modules[__name__] = _mod  # 关键：注册到当前模块名
  spec.loader.exec_module(_mod)
