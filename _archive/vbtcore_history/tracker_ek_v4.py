"""
vbtcore.tracker_ek — KalmanTracker + bbox-static-mpp 混合跟踪器
==============================================================
Step 3 简化版（替代 v2 的 NCC + 像素门）：
  1. 静态标定期（前 ~5 帧杠铃静止）：
     - YOLO 粗定位 → 收集 bbox 高度 h_samples
     - mpp_locked = plate_diameter / median(h_samples)
     - 锁死后 mpp 不再改变（v2 防线一，成功消除 bias 正偏）
  2. 跟踪期：
     - 每帧：KalmanTracker.predict() → 输出 (y, v)
     - 每 15 帧：YOLO → 中心点 → KalmanTracker.update()
     - 一阶导数（速度）由 Kalman 状态内生，无需 SG 差分

与 DetectFitTracker 的差异：
  - 不需要 NCC 模板匹配（不抗运动模糊）
  - 不需要像素硬门（20px / 150px）
  - 不需要 IoU 启发式
  - 速度由 Kalman 状态直接给出，不再做 SG 差分

放弃的组件：
  - EllipseCalibrator（椭圆拟合在边视图上不可靠：最大轮廓不一定是片）
  - IoU/Re-anchor 启发式（Kalman 的协方差矩阵天然处理不确定性）
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import PlateDetector
from .engine import anchor_score, select_anchor_track
from .kalman import BarbellKalmanTracker

# ══════════════════════════════════════════════════════════
#  诊断信息
# ══════════════════════════════════════════════════════════


@dataclass
class EKTrackDiagnostics:
    status: str = ""
    n_frames: int = 0
    elapsed_s: float = 0.0
    ms_per_frame: float = 0.0
    n_yolo_calls: int = 0
    yolo_ratio: float = 0.0
    coverage: float = 0.0
    src_counts: dict = field(default_factory=dict)
    ncc_mean: float = 0.0
    anchor_frame: int = -1
    anchor_h_px: float = 0.0
    notes: list[str] = field(default_factory=list)
    # mpp 锁死专用
    mpp_lock_frame: int = -1
    mpp_lock_value: float | None = None
    n_static_samples: int = 0


# ══════════════════════════════════════════════════════════
#  主跟踪器：bbox-static-mpp + Kalman
# ══════════════════════════════════════════════════════════


class EllipseKalmanTracker:
    """
    主循环（每帧）：
      锚定期：扫帧至 YOLO 多次命中同一目标（运动探针确认）
      静态标定期：收集 bbox 高度样本 → 锁定 mpp
      跟踪期：Kalman predict 每帧，YOLO+update 每 15 帧
    """

    def __init__(
        self,
        detector: PlateDetector,
        redet_every: int = 15,
        yolo_conf: float = 0.30,
        bootstrap_conf: float = 0.30,
        boot_max_scan: int = 90,
        plate_diameter_m: float = 0.45,
        user_hint: tuple[float, float] | None = None,
        # 静态标定参数
        n_static_min: int = 3,
        static_window_frames: int = 30,
        # Kalman 参数
        kalman_Q: float = 0.01,
        kalman_R: float = 4.0,
        # Kalman update 时的最大跳变阈值（相对 plate_h）
        # 超过此值视为噪声，忽略该次 update（信任预测）
        update_max_jump_factor: float = 0.6,
    ):
        self.det = detector
        self.redet_every = redet_every
        self.yolo_conf = yolo_conf
        self.bootstrap_conf = bootstrap_conf
        self.boot_max_scan = boot_max_scan
        self.plate_diameter_m = plate_diameter_m
        self.user_hint = user_hint
        self.n_static_min = n_static_min
        self.static_window_frames = static_window_frames
        self.update_max_jump_factor = update_max_jump_factor
        self._kalman_Q = kalman_Q
        self._kalman_R = kalman_R

    # ── 主流程 ───────────────────────────────────────────
    def process(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        if fps <= 1 or fps > 240:
            fps = 30.0
        dt = 1.0 / fps

        t0 = time.time()
        ys: list[float] = []
        vs: list[float] = []
        srcs: list[str] = []
        h_samples: list[float] = []
        diag = EKTrackDiagnostics()

        # 锚定期 pending
        pend: list[dict] = []
        anchored = False
        # 椭圆标定期累积（这里用 bbox h）
        static_h: list[float] = []
        mpp_locked: float | None = None
        kalman: BarbellKalmanTracker | None = None
        plate_h: float = 0.0  # 当前估算的片高（用于 update 跳变门）

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            diag.n_frames += 1
            n_frames = diag.n_frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            H, W = gray.shape[:2]

            # ── 1) 锚定（YOLO + 物理先验 + 运动探针）────────
            if not anchored:
                if n_frames > self.boot_max_scan:
                    diag.status = "NO_PLATE_DETECTED"
                    cap.release()
                    return None, None, h_samples, diag, fps

                dets = self.det.detect(frame, self.yolo_conf)
                diag.n_yolo_calls += 1
                cands = [
                    d
                    for d in dets
                    if d.conf >= self.bootstrap_conf and anchor_score(d, H, W) > 0
                ]
                if self.user_hint is not None:
                    hx, hy = self.user_hint
                    cands = sorted(cands, key=lambda d: np.hypot(d.cx - hx, d.cy - hy))[
                        :1
                    ]

                # pending 40px 邻域匹配（与 engine.py 相同策略）
                used: set[int] = set()
                for tr in pend:
                    bi, bd = -1, 1e9
                    for i, d in enumerate(cands):
                        if i in used:
                            continue
                        dist = float(np.hypot(d.cx - tr["cx"], d.cy - tr["cy"]))
                        if dist < 40 and dist < bd:
                            bi, bd = i, dist
                    if bi >= 0:
                        used.add(bi)
                        d = cands[bi]
                        tr.update(
                            cx=d.cx,
                            cy=d.cy,
                            h=d.h,
                            conf=d.conf,
                            hits=tr["hits"] + 1,
                            miss=0,
                        )
                        tr["ys"].append(d.cy)
                    else:
                        tr["miss"] += 1
                for i, d in enumerate(cands):
                    if i not in used:
                        pend.append(
                            {
                                "cx": d.cx,
                                "cy": d.cy,
                                "h": d.h,
                                "conf": d.conf,
                                "hits": 1,
                                "miss": 0,
                                "ys": [d.cy],
                            }
                        )
                pend[:] = [t for t in pend if t["miss"] <= 5]

                best = select_anchor_track(pend, min_hits=5)
                if best is None:
                    ys.append(np.nan)
                    vs.append(np.nan)
                    srcs.append("none")
                    continue

                # 锚定成功 → 进入静态标定期
                anchored = True
                diag.anchor_frame = n_frames
                diag.anchor_h_px = best["h"]
                cx, cy = best["cx"], best["cy"]
                plate_h = best["h"]
                static_h.append(plate_h)
                diag.notes.append(f"f{n_frames}: 锚定 h={plate_h:.1f}px")
                ys.append(cy)
                vs.append(0.0)
                srcs.append("anchor")
                continue

            # ── 2) 静态标定期：累计 bbox h → mpp_locked ─────
            if mpp_locked is None:
                # 每帧 YOLO 跑（杠铃静止，YOLO 稳定命中）
                dets = self.det.detect(frame, self.yolo_conf)
                diag.n_yolo_calls += 1
                # 取最近的检测
                nearby = None
                for d in dets:
                    if d.conf < self.bootstrap_conf:
                        continue
                    dist = float(np.hypot(d.cx - cx, d.cy - cy))
                    if dist < 60:
                        if nearby is None or dist < nearby[0]:
                            nearby = (dist, d)
                if nearby is not None:
                    _, d = nearby
                    static_h.append(d.h)
                    plate_h = d.h
                    cy = d.cy
                    cx = d.cx

                # 检查是否该锁 mpp
                enough_samples = len(static_h) >= self.n_static_min
                window_exceeded = (
                    n_frames - diag.anchor_frame >= self.static_window_frames
                )
                if enough_samples and window_exceeded:
                    med_h = float(np.median(static_h))
                    if med_h > 2:
                        mpp_locked = self.plate_diameter_m / med_h
                        # 初始化 Kalman
                        kalman = BarbellKalmanTracker(
                            initial_y=cy,
                            dt=dt,
                            Q=self._kalman_Q,
                            R=self._kalman_R,
                        )
                        diag.mpp_lock_frame = n_frames
                        diag.mpp_lock_value = mpp_locked
                        diag.n_static_samples = len(static_h)
                        diag.notes.append(
                            f"mpp 锁死于帧{n_frames} "
                            f"(samples={len(static_h)}, h_med={med_h:.1f}px, "
                            f"mpp={mpp_locked:.5f}m/px)"
                        )
                        # 把静态样本也存进 h_samples 供 calibrate() 使用
                        h_samples.extend(static_h)
                        ys.append(cy)
                        vs.append(0.0)
                        srcs.append("cal_lock")
                        continue

                ys.append(cy)
                vs.append(0.0)
                srcs.append("cal_static")
                continue

            # ── 3) 跟踪期：Kalman predict/update ──────────────
            assert kalman is not None
            run_yolo = n_frames % self.redet_every == 0

            # Kalman 预测（每帧都做，保证连续性）
            kalman.predict()

            if run_yolo:
                dets = self.det.detect(frame, self.yolo_conf)
                diag.n_yolo_calls += 1
                obs_y = None
                if dets:
                    nearby = None
                    for d in dets:
                        if d.conf < self.bootstrap_conf:
                            continue
                        dist = float(np.hypot(d.cx - cx, d.cy - cy))
                        if dist < 120:
                            if nearby is None or dist < nearby[0]:
                                nearby = (dist, d)
                    if nearby is not None:
                        _, d = nearby
                        # 跳变门：观测值与 Kalman 预测偏差 > factor × plate_h
                        # → 视为噪声，忽略（信任预测）
                        pred_y = kalman.y
                        jump = abs(d.cy - pred_y)
                        max_jump = self.update_max_jump_factor * plate_h
                        if d.conf >= self.bootstrap_conf and jump <= max_jump:
                            obs_y = d.cy
                            plate_h = 0.7 * d.h + 0.3 * plate_h  # 平滑更新 plate_h
                            cx = d.cx
                            srcs.append("yolo+kalman")
                        else:
                            srcs.append("yolo_jump_reject")
                    else:
                        srcs.append("yolo_miss")
                else:
                    srcs.append("yolo_empty")

                if obs_y is not None:
                    kalman.update(obs_y)
                    cy = obs_y
            else:
                srcs.append("kalman")

            ys.append(kalman.y)
            vs.append(kalman.v)

        cap.release()
        elapsed = time.time() - t0
        diag.elapsed_s = elapsed
        diag.ms_per_frame = elapsed / max(1, diag.n_frames) * 1000
        diag.yolo_ratio = diag.n_yolo_calls / max(1, diag.n_frames)
        diag.src_counts = dict(Counter(srcs))
        diag.coverage = sum(
            1 for s in srcs if s in ("anchor", "cal_lock", "yolo+kalman", "kalman")
        ) / max(1, diag.n_frames)
        diag.ncc_mean = 0.0
        diag.status = "TRACKED"
        # Step 4: 返回 (y_track, v_track, h_samples, diag, fps)
        # 向心为正（y 向下增加，杠铃上升则 y 减少 → -v 表示上升 → v 反转后正表示向心）
        # 约定: vs[i] = 向心为正（与 segment_reps_from_velocity 期望一致）
        v_arr = np.array(vs, dtype=float)
        # tracker 内部 Kalman 状态 v 向量是 y 的时间导数；y 向下为正，
        # 所以 y 增加 = 杠铃下沉 = 离心；y 减少 = 杠铃上升 = 向心。
        # 因此 v_kalman（dy/dt）为负表示向心；我们要取反使向心为正。
        v_arr = -v_arr
        return np.array(ys, dtype=float), v_arr, h_samples, diag, fps

    # ── 标定（返回锁定 mpp 或基于 bbox h 结算）────────────
    def calibrate(self, h_samples: list[float]) -> float | None:
        """
        兼容接口：返回基于 bbox h 样本的 mpp。
        如果 h_samples 为空，返回 None（实际 mpp 在 diag.mpp_lock_value 中）。
        """
        if not h_samples:
            return None
        med = float(np.median(h_samples))
        if med < 2:
            return None
        return self.plate_diameter_m / med
