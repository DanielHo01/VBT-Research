# vbtcore/tracker.py
"""
物理空间卡尔曼密集跟踪器
========================
状态向量直接运行在物理空间（米、秒），彻底消除 px/frame 量纲混乱。
- KinematicKalmanTracker : 恒加速度（CA）模型，卡尔曼滤波在物理空间运行
- DenseVisualTracker : 混合跟踪器
  · 普通帧 → Lucas-Kanade 局部光流提供密集速度证据
  · 每 15 帧 → YOLO 绝对坐标强力校正，消除光流漂移
"""

from __future__ import annotations

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 卡尔曼滤波器（物理空间）
# ─────────────────────────────────────────────────────────────────────────────


class KinematicKalmanTracker:
    """
    状态向量 x = [y(米), v(米/秒), a(米/秒²)]ᵀ
    恒加速度（CA）模型，严格牛顿运动学方程式。

    Q : 过程噪声协方差矩阵（连续白噪声离散化）
    R : 观测噪声协方差（YOLO 位置观测）
    """

    def __init__(self, initial_y_m: float, initial_time_s: float):
        self.x = np.array([[initial_y_m], [0.0], [0.0]], dtype=np.float64)
        self.last_time_s = initial_time_s

        # 初始协方差
        self.P = np.diag([0.001, 0.1, 1.0])

        # 过程噪声参数（允许最大约 5 m/s² 加速度冲击）
        self.q_a = 25.0

        # 观测矩阵：仅观测位置 y
        self.H_pos = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        self.R_pos = np.array([[0.0004]], dtype=np.float64)  # (2cm)²

    # ── 预测步（时间更新）────────────────────────────────────────
    def predict(self, current_time_s: float) -> tuple[float, float]:
        """
        返回预测后的 (y_m, v_mps)
        """
        dt = current_time_s - self.last_time_s
        if dt <= 0.0:
            dt = 1.0 / 60.0
        self.last_time_s = current_time_s

        # 状态转移矩阵 F（牛顿运动学）
        F = np.array(
            [
                [1.0, dt, 0.5 * dt * dt],
                [0.0, 1.0, dt],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # 过程噪声离散化 Q
        G = np.array([[0.5 * dt * dt], [dt], [1.0]], dtype=np.float64)
        Q = (G @ G.T) * self.q_a

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    # ── 位置测量更新（YOLO / 光流修正）───────────────────────────
    def update_position(self, observed_y_m: float) -> tuple[float, float]:
        """
        用绝对位置观测更新卡尔曼状态。
        返回更新后的 (y_m, v_mps)。
        """
        H = self.H_pos
        R = self.R_pos

        z = np.array([[observed_y_m]], dtype=np.float64)
        y_residual = z - H @ self.x

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_residual
        self.P = (np.eye(3, dtype=np.float64) - K @ H) @ self.P
        return float(self.x[0, 0]), float(self.x[1, 0])


# ─────────────────────────────────────────────────────────────────────────────
# 密集视觉跟踪器（LK 光流 + YOLO 校正）
# ─────────────────────────────────────────────────────────────────────────────


class DenseVisualTracker:
    """
    混合跟踪器：
    - 普通帧：稀疏 Lucas-Kanade 光流 + 卡尔曼预测
    - Keyframe（第 15 帧）：YOLO 检测绝对校正

    输出物理量（米、米/秒），图像坐标系（向下为正）已在内部取反。
    """

    def __init__(
        self,
        mpp: float,
        initial_bbox,
        initial_gray,
        initial_time_s: float,
    ):
        self.mpp = mpp

        # 初始检测框中心
        x1, y1, x2, y2 = map(int, initial_bbox)
        self.center_x_px = (x1 + x2) / 2.0
        self.center_y_px = (y1 + y2) / 2.0  # 像素坐标（向下为正）

        # 初始化物理空间卡尔曼（图像 y 取反 → 向上为正的物理坐标）
        init_y_m = self.center_y_px * self.mpp  # 转换到米
        self.kalman = KinematicKalmanTracker(init_y_m, initial_time_s)

        self.prev_gray = initial_gray
        self.template_points = self._extract_roi_points(initial_gray, initial_bbox)

    def _extract_roi_points(self, gray, bbox) -> np.ndarray | None:
        """
        在 bbox 区域内提取 Shi-Tomasi 特征点作为光流跟踪种子。
        仅在外侧 70%% 取点，避开内侧手腕/杠铃杆区域，防止人体干扰。
        """
        x1, y1, x2, y2 = map(int, bbox)
        bw = x2 - x1
        bh = y2 - y1
        mask = np.zeros_like(gray, dtype=np.uint8)
        # 外侧 70%（假设摄像机在侧面偏后，圆盘外侧完全可见）
        safe_x1 = x1
        safe_x2 = int(x1 + bw * 0.70)
        safe_y1 = int(y1 + bh * 0.05)
        safe_y2 = int(y2 - bh * 0.05)
        if safe_x2 > safe_x1 and safe_y2 > safe_y1:
            mask[safe_y1:safe_y2, safe_x1:safe_x2] = 255
        points = cv2.goodFeaturesToTrack(
            gray,
            mask=mask,
            maxCorners=25,
            qualityLevel=0.02,
            minDistance=6,
        )
        return points

    # ── 普通帧：光流 + 卡尔曼预测 ────────────────────────────────
    def step_interframe(self, gray, current_time_s: float) -> tuple[float, float]:
        """
        普通帧更新。
        返回 (y_m, v_mps)，图像 y 坐标已在内部取反（向上为正）。
        """
        y_pred, v_pred = self.kalman.predict(current_time_s)

        if self.template_points is not None and len(self.template_points) >= 5:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                self.template_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
            )
            good_new = p1[st == 1]
            good_old = self.template_points[st == 1]

            if len(good_new) >= 5:
                dy_px = float(np.median(good_new[:, 1] - good_old[:, 1]))
                self.center_y_px += dy_px
                self.kalman.update_position(self.center_y_px * self.mpp)
                self.template_points = good_new.reshape(-1, 1, 2)

        self.prev_gray = gray
        # 卡尔曼状态中 x[0] 已是米为单位（向上为正），直接返回
        return float(self.kalman.x[0, 0]), float(self.kalman.x[1, 0])

    # ── Keyframe：YOLO 绝对校正 ──────────────────────────────────
    def step_keyframe(
        self, gray, yolo_bbox, current_time_s: float
    ) -> tuple[float, float]:
        """
        Keyframe 更新（每 15 帧一次）。
        YOLO 绝对坐标强力校正，消除光流漂移。
        """
        self.kalman.predict(current_time_s)

        det_cy = ((yolo_bbox[1] + yolo_bbox[3]) / 2.0) * self.mpp
        det_cx = (yolo_bbox[0] + yolo_bbox[2]) / 2.0

        self.kalman.update_position(det_cy)
        self.center_y_px = (yolo_bbox[1] + yolo_bbox[3]) / 2.0
        self.center_x_px = det_cx

        self.prev_gray = gray
        self.template_points = self._extract_roi_points(gray, yolo_bbox)
        return float(self.kalman.x[0, 0]), float(self.kalman.x[1, 0])
