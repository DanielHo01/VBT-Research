"""DEPRECATED vbtcore.detector_phase0 — 已归档至 _archive/vbtcore_history/detector_phase0_v0.py

此模块保留仅为 1 个 release 周期兼容（v0.1.x），新代码请使用：
  - vbtcore.PlateDetector (生产 ONNX 检测器)
  - 覆盖率审计脚本：scripts/coverage_audit.py 仍可引用此模块

归档原因：phase0 检测器是研究阶段的纯检测基线，已被 vbtcore.detector.PlateDetector
（带 letterbox + 置信度 sigmoid 修复）替代。覆盖率审计脚本（coverage_audit.py）
仍通过此 shim 调用，便于历史审计数据复现。
"""

from __future__ import annotations

import importlib.util as _ilu
import sys as _sys
import warnings as _w
from pathlib import Path as _P

_HERE = _P(__file__).resolve().parent
_ARCHIVE_FILE = _HERE.parent / "_archive" / "vbtcore_history" / "detector_phase0_v0.py"

_w.warn(
  "vbtcore.detector_phase0 已废弃（v0.1.0-baseline 起归档至 "
  "_archive/vbtcore_history/detector_phase0_v0.py）。"
  "生产代码请使用 vbtcore.PlateDetector；"
  "覆盖率审计脚本（scripts/coverage_audit.py）仍可引用此模块。",
  DeprecationWarning,
  stacklevel=2,
)

if _ARCHIVE_FILE.exists():
  spec = _ilu.spec_from_file_location(
    "vbtcore._archive_detector_phase0_v0", str(_ARCHIVE_FILE)
  )
  if spec is None or spec.loader is None:
    raise ImportError(f"无法从 {_ARCHIVE_FILE} 加载归档模块")
  _mod = _ilu.module_from_spec(spec)
  _sys.modules["vbtcore._archive_detector_phase0_v0"] = _mod
  _sys.modules[__name__] = _mod
  spec.loader.exec_module(_mod)
else:
  raise ImportError(f"归档模块不存在：{_ARCHIVE_FILE}")
