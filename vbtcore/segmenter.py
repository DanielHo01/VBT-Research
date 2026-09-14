# vbtcore/segmenter.py
"""
运动学分段器
=============
支持深蹲/卧推（SSC 模式）与硬拉（无 SSC 直接向心）。
速度过零点 + 滞回死区 + 物理门禁。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Rep:
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    duration_s: float
    rom_m: float  # 位移幅值（米）
    mcv_mps: float  # 向心段平均速度（主指标）
    pcv_mps: float  # 向心段峰值速度


class BiomechanicalRepSegmenter:
    """
    速度状态机分段。

    exercise_type
    ──────────────
    "squat_bench" : SSC 模式
      IDLE → ECCENTRIC → (底部换向) → CONCENTRIC → TOP → IDLE
    "deadlift" : 无 SSC，直接向心
      IDLE → CONCENTRIC → (速度归零) → IDLE

    速度约定
    ────────
    外部需传入：向上为正（+），向下为负（-）
    （即：图像 y 取反后的物理坐标）
    """

    def __init__(
        self,
        exercise_type: str = "squat_bench",
        min_rom_m: float = 0.12,
        min_dur_s: float = 0.20,
        v_thresh_start: float = 0.08,
        v_zero_band: float = 0.02,
        confirm_frames: int = 3,
    ):
        """
        confirm_frames : 状态跃迁的连续确认帧数（默认 3）
        ─────────────────────────────────────────────────
        【2026-09-14 修复】旧实现只看 (i, i+1) 两帧判定底部换向与向心结束，
        蹲底停顿期的单帧速度抖动即可触发「早产的 CONCENTRIC」：

          110kg_0.61_0.52_0.55_0.51_0.38.mp4 谷底 18.93s 实测
            18.67s v=-0.074   仍在下行
            18.77s v=+0.045   ← 单帧翻正，旧逻辑在此误判换向
            18.87s v=-0.062   又转负 → 该 rep 以 rom=0.0007m 收尾被门禁拒绝
            19.07s v=+0.248   真正的向心上行开始，但状态已回 IDLE，无人接管

        结果：轨迹里有全部 5 个往返，FSM 却只产出 3 个有效 rep，
        其余退化为 28 个碎片候选（rom 多在 0.001~0.04m 量级）。

        要求连续 confirm_frames 帧同向后才跃迁，可滤掉蹲底抖动。
        实测（保持其余参数不变）：
          该视频 3/5 → 5/5；confirm=1/2 仍为 3，confirm>=3 才修复。
        取 3（0.1s @30fps）：既跨过抖动，又远短于最短向心时长（~0.3s）。
        """
        self.exercise_type = exercise_type
        self.min_rom_m = min_rom_m
        self.min_dur_s = min_dur_s
        self.v_thresh = v_thresh_start  # 向心启动门限
        self.v_band = v_zero_band
        self.confirm_frames = max(1, int(confirm_frames))

    # ──────────────────────────────────────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────────────────────────────────────
    def segment(
        self,
        timestamps: np.ndarray,
        positions: np.ndarray,  # 向上为正（米）
        velocities: np.ndarray,  # 向上为正（米/秒）
    ) -> list[Rep]:
        """
        输入均为真实物理量（米、秒）：
        timestamps : 时间数组（秒，基于 PTS）
        positions  : 纵向位移（米），向上为正
        velocities : 纵向速度（米/秒），向上为正
        """
        n = len(timestamps)
        if n < 10:
            return []

        reps: list[Rep] = []
        state = "IDLE"
        rep_start_idx = 0

        cf = self.confirm_frames

        def _sustained(i: int, positive: bool) -> bool:
            """i+1 .. i+cf 连续 cf 帧是否稳定同向（滤蹲底/顶部单帧抖动）。"""
            for k in range(1, cf + 1):
                vk = float(velocities[i + k])  # noqa: PI-LENS=unsafe-call
                if positive:
                    if vk <= self.v_band:
                        return False
                elif vk >= -self.v_band:
                    return False
            return True

        for i in range(1, n - max(2, cf + 1)):
            v = float(velocities[i])  # noqa: PI-LENS=unsafe-call
            v_next = float(velocities[i + 1])  # noqa: PI-LENS=unsafe-call

            if self.exercise_type == "deadlift":
                # ── 硬拉拓扑（无 SSC，直接向心）────────────────────
                if state == "IDLE":
                    if v > self.v_thresh and v_next > self.v_thresh:
                        state = "CONCENTRIC"
                        rep_start_idx = i - 1

                elif state == "CONCENTRIC":
                    if v < self.v_band and v_next < self.v_band:
                        rep_end_idx = i
                        self._validate_and_append(
                            reps,
                            rep_start_idx,
                            rep_end_idx,
                            timestamps,
                            positions,
                            velocities,
                        )
                        state = "IDLE"

            else:
                # ── 深蹲/卧推拓扑（SSC：离心 → 换向 → 向心）────────
                if state == "IDLE":
                    if v < -self.v_thresh:
                        state = "ECCENTRIC"

                elif state == "ECCENTRIC":
                    # 底部换向：速度由负转正，且需连续 cf 帧确认（防蹲底抖动早产）
                    if v >= -self.v_band and _sustained(i, positive=True):
                        state = "CONCENTRIC"
                        rep_start_idx = i  # 向心起始 = 底部换向点

                elif state == "CONCENTRIC":
                    # 向心结束：速度归零（顶部停顿），同样需连续确认
                    if v <= self.v_band and _sustained(i, positive=False):
                        rep_end_idx = i
                        self._validate_and_append(
                            reps,
                            rep_start_idx,
                            rep_end_idx,
                            timestamps,
                            positions,
                            velocities,
                        )
                        state = "IDLE"

        return reps

    # ──────────────────────────────────────────────────────────────────────────
    # 物理门禁 + Rep 构造
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_and_append(
        self,
        reps: list[Rep],
        start_idx: int,
        end_idx: int,
        t_arr: np.ndarray,
        y_arr: np.ndarray,
        v_arr: np.ndarray,
    ) -> None:
        duration = float(t_arr[end_idx]) - float(t_arr[start_idx])  # noqa: PI-LENS=unsafe-call
        rom = abs(float(y_arr[end_idx]) - float(y_arr[start_idx]))  # noqa: PI-LENS=unsafe-call

        if duration < self.min_dur_s or rom < self.min_rom_m:
            return

        conc_v = v_arr[start_idx : end_idx + 1]
        pos_v = conc_v[conc_v > 0]

        reps.append(
            Rep(
                start_idx=int(start_idx),  # noqa: PI-LENS=unsafe-call
                end_idx=int(end_idx),  # noqa: PI-LENS=unsafe-call
                start_time=float(t_arr[start_idx]),  # noqa: PI-LENS=unsafe-call
                end_time=float(t_arr[end_idx]),  # noqa: PI-LENS=unsafe-call
                duration_s=round(duration, 3),
                rom_m=round(rom, 4),
                # 严格位移积分 MCV（对齐 GymAware 工业定义）：位移/时间，消除滤波正偏差
                mcv_mps=round(rom / duration if duration > 0 else 0.0, 3),
                pcv_mps=round(float(np.max(conc_v)) if len(conc_v) else 0.0, 3),  # noqa: PI-LENS=unsafe-call
            )
        )
