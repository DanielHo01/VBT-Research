"""
vbtcore.segment — rep 分段与指标结算
====================================
MCV 定义（本引擎统一，与 GymAware 对齐）：
  MCV = 向心段（bottom→top）平均正速度   ← 主指标
  mcv_mid = 向心段中点瞬时速度           ← 仅诊断用（旧 AnchorTemplateEngine 定义）
  PV    = 向心段速度峰值
  ROM   = bottom→top 位移（米）

历史不一致（已消除）：两套旧引擎分别用 mean 与 mid 定义，
标定系数一个乘 1.15 一个不乘——同一仓库两把尺子。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter, find_peaks


@dataclass
class Rep:
    start_frame: int
    end_frame: int
    mcv: float            # 向心段平均正速度 m/s（主指标）
    mcv_mid: float        # 中点瞬时速度 m/s（诊断）
    pv: float             # 峰值速度 m/s
    rom_m: float          # 米
    duration_s: float
    clipped: bool = False # 速度触 sanity 界（诊断）


@dataclass
class SegmentResult:
    reps: list[Rep]
    status: str
    note: str = ""
    y_used: np.ndarray | None = None
    v_used: np.ndarray | None = None


def longest_clean_run(y: np.ndarray, min_len: int = 30) -> tuple[int, int] | None:
    """最长无 NaN 连续段（>min_len 帧）。大 gap 不插值原则：跨 gap 的 rep 一律拒绝。"""
    best = None
    start = None
    for i in range(len(y)):
        if np.isnan(y[i]):
            if start is not None:
                seg = (start, i - 1)
                if seg[1] - seg[0] > min_len and (best is None or seg[1] - seg[0] > best[1] - best[0]):
                    best = seg
                start = None
        elif start is None:
            start = i
    if start is not None:
        seg = (start, len(y) - 1)
        if seg[1] - seg[0] > min_len and (best is None or seg[1] - seg[0] > best[1] - best[0]):
            best = seg
    return best


def small_gap_interp(y: np.ndarray, max_gap: int = 10) -> np.ndarray:
    """仅填充 ≤max_gap 帧的 NaN（线性），其余保持 NaN。"""
    out = y.copy()
    valid = np.where(~np.isnan(y))[0]
    if len(valid) < 2:
        return out
    i = 0
    while i < len(out):
        if np.isnan(out[i]):
            j = i
            while j < len(out) and np.isnan(out[j]):
                j += 1
            left = valid[valid < i]
            right = valid[valid >= j]
            if len(left) and len(right) and (j - i) <= max_gap:
                a, b = left[-1], right[0]
                for k in range(i, j):
                    out[k] = np.interp(k, [a, b], [y[a], y[b]])
            i = j
        else:
            i += 1
    return out


def segment_reps(y_track: np.ndarray, fps: float, mpp: float,
                 dur_range: tuple[float, float] = (0.25, 4.5),
                 rom_min_m: float = 0.012,
                 v_sanity: tuple[float, float] = (0.05, 2.5),
                 sg_window: int = 15) -> SegmentResult:
    """
    轨迹（像素，y 向下）→ rep 列表。
    流程：小 gap 插值 → 最长干净段 → SG(15,3) 平滑 → 速度（SG 一阶导）
         → bottom→top 峰谷配对 → 物理过滤。
    """
    y = small_gap_interp(y_track)
    run = longest_clean_run(y)
    if run is None:
        return SegmentResult(reps=[], status="NO_CLEAN_SEGMENT",
                             note="无 >30 帧连续轨迹段")
    a, b = run
    ys = y[a:b + 1]

    win = min(sg_window, len(ys) - 1)
    if win % 2 == 0:
        win -= 1
    if win < 5:
        return SegmentResult(reps=[], status="TOO_SHORT",
                             note=f"干净段仅 {len(ys)} 帧")
    y_s = savgol_filter(ys, win, 3)
    # y 向下为正 → 向心（向上）速度为负取反；单位 m/s
    v = -savgol_filter(ys, win, 3, deriv=1) * fps * mpp

    dur_min, dur_max = dur_range
    bottoms, _ = find_peaks(y_s, distance=int(fps * dur_min))
    tops, _ = find_peaks(-y_s, distance=int(fps * dur_min))
    events = sorted([(int(f), "b") for f in bottoms] + [(int(f), "t") for f in tops])

    reps: list[Rep] = []
    i = 0
    while i < len(events) - 1:
        f1, t1 = events[i]
        f2, t2 = events[i + 1]
        if t1 == "b" and t2 == "t":
            dur = (f2 - f1) / fps
            rom = abs(y_s[f2] - y_s[f1]) * mpp
            if dur_min <= dur <= dur_max and rom >= rom_min_m:
                seg_v = v[f1:f2 + 1]
                pos_v = seg_v[seg_v > 0]
                mid = int(np.clip((f1 + f2) // 2, 0, len(v) - 1))
                mcv = float(np.mean(pos_v)) if len(pos_v) else 0.0
                clipped = not (v_sanity[0] <= mcv <= v_sanity[1])
                reps.append(Rep(
                    start_frame=a + f1, end_frame=a + f2,
                    mcv=round(mcv, 3),
                    mcv_mid=round(float(np.clip(v[mid], *v_sanity)), 3),
                    pv=round(float(np.max(seg_v)) if len(seg_v) else 0.0, 3),
                    rom_m=round(rom, 4),
                    duration_s=round(dur, 2),
                    clipped=clipped,
                ))
            i += 2
        else:
            i += 1

    return SegmentResult(reps=reps,
                         status="OK" if reps else "NO_REPS",
                         y_used=y_s, v_used=v,
                         note=f"干净段 [{a},{b}] {b-a+1}帧")
