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
                 rom_keep_ratio: float = 0.45,
                 mcv_min: float = 0.10,
                 prominence_px: float = 6.0,
                 v_sanity: tuple[float, float] = (0.05, 2.5),
                 sg_window: int = 15) -> SegmentResult:
    """
    轨迹（像素，y 向下）→ rep 列表。
    流程：小 gap 插值 → 最长干净段 → SG(15,3) 平滑 → 速度（SG 一阶导）
         → bottom→top 峰谷配对 → 物理过滤 → 两遍法质量门。

    两遍法质量门（M1，依据 110kg/105kg 碎片化诊断）：
      同一组内真 rep 的 ROM 彼此相近（同一蹲深），而跟踪抖动产生的
      假 rep ROM ≤ 真值的 1/3（实测：真 54-63cm vs 假 1.4-20cm）。
      因此先收集全部候选，再保留 ROM ≥ rom_keep_ratio × 最大候选 ROM
      且 MCV ≥ mcv_min 者。绝对下限 rom_min_m 仍然生效。
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
    bottoms, _ = find_peaks(y_s, distance=int(fps * dur_min),
                            prominence=prominence_px)
    tops, _ = find_peaks(-y_s, distance=int(fps * dur_min),
                         prominence=prominence_px)

    # ── 近邻同侧峰合并（M1.3）──────────────────────────
    # SG 滤波在平台角部会产生成对过冲峰（实测相距 ~9 帧），
    # 不合并会让 bottom→top 配对错位。保留更深/更高的那个。
    def _merge_peaks(idx: np.ndarray, vals: np.ndarray) -> np.ndarray:
        if len(idx) == 0:
            return idx
        md = max(6, int(fps * 0.33))
        out = [(int(idx[0]), float(vals[0]))]   # (帧号, 峰值)
        for i, v in zip(idx[1:], vals[1:]):
            if i - out[-1][0] < md:
                if v > out[-1][1]:
                    out[-1] = (int(i), float(v))
            else:
                out.append((int(i), float(v)))
        return np.array([f for f, _ in out])

    bottoms = _merge_peaks(bottoms, y_s[bottoms])
    tops = _merge_peaks(tops, -y_s[tops])

    events = sorted([(int(f), "b") for f in bottoms] + [(int(f), "t") for f in tops])

    # ── 边界 top 合成（M1.3）──────────────────────────
    # 视频常在杠架回位（平台）处结束：真实最后一个 top 贴边界时
    # find_peaks 检不到（峰不能在边界）。若最后一个事件是 bottom
    # 且到末帧的时长/位移在合理范围，则用末帧合成 top。
    if events and events[-1][1] == "b":
        last_b = events[-1][0]
        dur_end = (len(y_s) - 1 - last_b) / fps
        rom_end = (y_s[last_b] - y_s[-1]) * mpp
        if dur_range[0] <= dur_end <= dur_range[1] and rom_end >= rom_min_m:
            events.append((len(y_s) - 1, "t"))
            events.sort()

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

    # ── 两遍法质量门 ─────────────────────────────────
    gate_note = ""
    if reps:
        max_rom = max(r.rom_m for r in reps)
        rom_keep = max(rom_min_m, rom_keep_ratio * max_rom)
        n_before = len(reps)
        reps = [r for r in reps
                if r.rom_m >= rom_keep and r.mcv >= mcv_min]
        gate_note = (f"; 两遍门: {n_before}->{len(reps)} "
                     f"(max_rom={max_rom*100:.0f}cm, 门={rom_keep*100:.1f}cm)")

    return SegmentResult(reps=reps,
                         status="OK" if reps else "NO_REPS",
                         y_used=y_s, v_used=v,
                         note=f"干净段 [{a},{b}] {b-a+1}帧{gate_note}")
