"""DEPRECATED vbtcore.segment — 已归档至 _archive/vbtcore_history/segment_v3_savgol.py

此模块保留仅为 1 个 release 周期兼容（v0.1.x），新代码请使用：
  - vbtcore.segmenter.BiomechanicalRepSegmenter (FSM 速度过零点分段)
  - vbtcore.segmenter.Rep

用法（DeprecationWarning 仍可见，便于迁移）：
  from vbtcore.segment import segment_reps   # ← 仍可用但会警告
  from vbtcore.segmenter import BiomechanicalRepSegmenter  # ← 新代码路径
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

# Locate the archived module file (3 parents up + sibling)
_HERE = Path(__file__).resolve().parent
_ARCHIVE_FILE = _HERE.parent / "_archive" / "vbtcore_history" / "segment_v3_savgol.py"

if _ARCHIVE_FILE.exists():
    # Use importlib spec to load the archived module under its old name
    _spec = importlib.util.spec_from_file_location(  # type: ignore[attr-defined]
        "vbtcore._archive_segment_v3", str(_ARCHIVE_FILE)
    )
    if _spec is not None and _spec.loader is not None:
        _archived = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
        sys.modules["vbtcore._archive_segment_v3"] = _archived
        sys.modules[__name__] = _archived  # 关键：注册到当前模块名
        _spec.loader.exec_module(_archived)
        # Re-export public names with DeprecationWarning
        warnings.warn(
            "vbtcore.segment 已废弃（v0.1.0-baseline 起归档至 "
            "_archive/vbtcore_history/segment_v3_savgol.py）。"
            "新代码请使用 vbtcore.segmenter.BiomechanicalRepSegmenter。",
            DeprecationWarning,
            stacklevel=2,
        )
else:
    raise ImportError(
        f"归档模块不存在：{_ARCHIVE_FILE}。"
        "若您是从旧仓库升级，请确认 _archive/ 目录完整。"
    )
