# vbtcore/calibrator.py
"""
尺度标定与门禁
==============
静态中位数锁死 + CV 变异系数晃动门禁。
在起始静止期采样检测框高度，完成门禁核验后永久冻结 mpp。
"""

from __future__ import annotations

import numpy as np


class StaticPlateCalibrator:
    def __init__(
        self,
        real_diameter_m: float = 0.45,
        min_static_frames: int = 20,
        max_cv: float = 0.015,
    ):
        """
        real_diameter_m : 杠铃片标准物理外径（默认 450 mm）
        min_static_frames : 计算标定所需的最小静止有效帧数
        max_cv : 变异系数门禁阈值；超过该值判定为机位晃动或严重遮挡
        """
        self.real_diameter_m = real_diameter_m
        self.min_static_frames = min_static_frames
        self.max_cv = max_cv
        self._samples: list[float] = []
        self._mpp_locked: float | None = None

    def add_sample(self, bbox_height: float) -> None:
        """仅在杠铃静止期输入每帧检测到的边界框高度（像素）"""
        if self._mpp_locked is None and bbox_height > 10.0:
            self._samples.append(bbox_height)

    def is_ready(self) -> bool:
        return len(self._samples) >= self.min_static_frames

    def lock_scale(self) -> float:
        """
        完成门禁核验并永久冻结 mpp。
        返回 : 米/像素（mpp）
        """
        if self._mpp_locked is not None:
            return self._mpp_locked

        if len(self._samples) < self.min_static_frames:
            raise ValueError(
                f"标定样本不足：当前 {len(self._samples)} 帧，"
                f"至少需要 {self.min_static_frames} 帧。"
            )

        heights = np.array(self._samples)

        # ── IQR 剔除离群点 ──────────────────────────────
        q25, q75 = np.percentile(heights, [25, 75])
        iqr = q75 - q25
        valid = heights[(heights >= q25 - 1.5 * iqr) & (heights <= q75 + 1.5 * iqr)]

        if valid.size == 0:
            raise RuntimeError("IQR 过滤后无有效样本，拒绝标定。")
        mean_h = float(np.mean(valid))  # noqa: pi-lens=unsafe-call
        std_h = float(np.std(valid))  # noqa: pi-lens=unsafe-call
        cv = std_h / mean_h

        # CV 超门限 → 记录警告，但使用中位数（降级运行，不直接报错）
        if cv > self.max_cv:
            import warnings

            warnings.warn(
                f"CV={cv:.4f} > {self.max_cv}（起始期有动作/晃动），"
                f"降级使用中位数标定。"
            )

        median_h = float(np.median(valid))  # noqa: pi-lens=unsafe-call
        self._mpp_locked = self.real_diameter_m / median_h
        return self._mpp_locked

    @property
    def mpp_locked(self) -> float | None:
        return self._mpp_locked
