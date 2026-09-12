"""
vbtcore.kalman — 连续卡尔曼轨迹滤波器
=======================================
彻底替代 NCC 追踪 + 像素硬门：
  - 状态向量: [y位移, y速度, y加速度]（牛顿匀加速模型）
  - 观测向量: [y位移]（每15帧 YOLO+椭圆圆心提供一次观测）
  - 纯数学最优融合，无任何硬像素阈值

一阶导数（速度）由状态矩阵内生，无坐标差分，天然光滑。
"""

from __future__ import annotations

import numpy as np


class BarbellKalmanTracker:
    """
    状态: [y, v_y, a_y]^T
    观测: 只观测 y（圆心纵坐标）

    Q = 过程噪声协方差（越小=越信任预测，越平滑）
    R = 观测噪声协方差（越大=越不信任单帧检测，越平滑）
    """

    def __init__(
        self, initial_y: float, dt: float = 1.0 / 120.0, Q: float = 0.01, R: float = 4.0
    ):
        # 状态: [y, v, a]^T
        self.x = np.array([[initial_y], [0.0], [0.0]], dtype=np.float64)
        self.dt = dt

        # 匀加速状态转移矩阵
        self.F = np.array(
            [
                [1.0, dt, 0.5 * dt * dt],
                [0.0, 1.0, dt],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # 观测矩阵（只观测 y）
        self.H = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

        # 初始协方差（对初始状态不确定性）
        self.P = np.eye(3, dtype=np.float64) * 10.0
        self.Q = np.eye(3, dtype=np.float64) * Q
        self.R = np.array([[R]], dtype=np.float64)

    # ── 预测（每帧调用，推进物理模型）──────────────────────
    def predict(self) -> tuple[float, float]:
        """
        纯预测，无观测修正。
        返回 (y_pred, v_pred) 供调用方使用。
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    # ── 更新（每15帧 YOLO+椭圆 提供一次观测）──────────────
    def update(self, observed_y: float) -> tuple[float, float]:
        """
        卡尔曼增益融合：完全消除坐标阶跃脉冲，数学最优。
        返回 (y_filt, v_filt)。
        """
        y_residual = observed_y - float(self.H @ self.x)
        S = float(self.H @ self.P @ self.H.T) + float(self.R[0, 0])
        K = self.P @ self.H.T / max(S, 1e-9)  # 3×1 增益向量
        self.x = self.x + K * y_residual
        self.P = (np.eye(3) - K @ self.H) @ self.P
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def y(self) -> float:
        return float(self.x[0, 0])

    @property
    def v(self) -> float:
        return float(self.x[1, 0])

    @property
    def a(self) -> float:
        return float(self.x[2, 0])
