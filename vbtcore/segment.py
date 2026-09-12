"""
vbtcore.segment — rep 分段与指标结算
====================================
MCV 定义（本引擎统一，与 GymAware 对齐）：
  MCV = 向心段（bottom→top）全程平均速度（= ROM/duration 的伸缩求和）← 主指标
  mcv_pos = 向心段平均正速度            ← 诊断（M1.5 前的主指标，已降级）
  mcv_mid = 向心段中点瞬时速度           ← 仅诊断用（旧 AnchorTemplateEngine 定义）
  PV    = 向心段速度峰值
  ROM   = bottom→top 位移（米）

M1.5 口径修正依据：TroyKaneshiro/barbell-velocity-tracker METHODOLOGY.md
实证——GymAware ACV 是全程 plain mean；只平均正速度/最快窗口会系统性
高估 MCV。本仓库最差的两条视频（110kg_0.45_0.31 bias=+0.97、
110kg_0.57_0.55 bias=+0.55）均为 +bias，与该机制一致。

Step 4 重构（2026-09-11）：
  引入 segment_reps_from_velocity()，直通消费 Kalman 速度向量。
  废除 savgol_filter 二次差分与 prominence_px 像素硬阈值，
  改用「v 速度过零点」驱动的有限状态机（FSM）。
  物理意义：杠铃动作底部 = 向心起步瞬间 = v 由负转正过零点。
"""

from __future__ import annotations  # noqa: pi-lens=unsafe-call

from dataclasses import dataclass

import numpy as np


@dataclass
class Rep:
    start_frame: int
    end_frame: int
    mcv: float  # 向心段全程平均速度 m/s（主指标，M1.5 起）
    mcv_mid: float  # 中点瞬时速度 m/s（诊断）
    pv: float  # 峰值速度 m/s
    rom_m: float  # 米
    duration_s: float
    mcv_pos: float = 0.0  # 向心段平均正速度 m/s（诊断，M1.5 前的主指标）
    clipped: bool = False  # 速度触 sanity 界（诊断）


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
                if seg[1] - seg[0] > min_len and (
                    best is None or seg[1] - seg[0] > best[1] - best[0]
                ):
                    best = seg
                start = None
        elif start is None:
            start = i
    if start is not None:
        seg = (start, len(y) - 1)
        if seg[1] - seg[0] > min_len and (
            best is None or seg[1] - seg[0] > best[1] - best[0]
        ):
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


# ══════════════════════════════════════════════════════════
#  旧版 segment_reps（SG 差分 + 像素硬阈值）—— 保留作 fallback
# ══════════════════════════════════════════════════════════


def segment_reps(  # noqa: pi-lens=unsafe-call,pi-lens=unsafe-call
    y_track: np.ndarray,
    fps: float,
    mpp: float,
    dur_range: tuple[float, float] = (0.25, 4.5),
    rom_min_m: float = 0.012,
    rom_keep_ratio: float = 0.45,
    mcv_min: float = 0.10,
    prominence_px: float = 6.0,
    v_sanity: tuple[float, float] = (0.05, 2.5),
    sg_window: int = 15,
) -> SegmentResult:
    """
    轨迹（像素，y 向下）→ rep 列表。
    流程：小 gap 插值 → 最长干净段 → SG(15,3) 平滑 → 速度（SG 一阶导）
         → bottom→top 峰谷配对 → 物理过滤 → 两遍法质量门。

    Step 4 起**不建议**继续使用此函数（双重平滑 + 像素硬阈值）。
    优先用 segment_reps_from_velocity() 直通消费 Kalman v。
    """
    from scipy.signal import find_peaks, savgol_filter

    y = small_gap_interp(y_track)
    run = longest_clean_run(y)
    if run is None:
        return SegmentResult(
            reps=[], status="NO_CLEAN_SEGMENT", note="无 >30 帧连续轨迹段"
        )
    a, b = run
    ys = y[a : b + 1]

    win = min(sg_window, len(ys) - 1)
    if win % 2 == 0:
        win -= 1
    if win < 5:
        return SegmentResult(reps=[], status="TOO_SHORT", note=f"干净段仅 {len(ys)} 帧")
    y_s = savgol_filter(ys, win, 3)
    # y 向下为正 → 向心（向上）速度为负取反；单位 m/s
    v = -savgol_filter(ys, win, 3, deriv=1) * fps * mpp

    dur_min, dur_max = dur_range
    bottoms, _ = find_peaks(y_s, distance=int(fps * dur_min), prominence=prominence_px)
    tops, _ = find_peaks(-y_s, distance=int(fps * dur_min), prominence=prominence_px)

    # ── 近邻同侧峰合并（M1.3）──────────────────────────
    def _merge_peaks(idx: np.ndarray, vals: np.ndarray) -> np.ndarray:
        if len(idx) == 0:
            return idx
        md = max(6, int(fps * 0.33))
        out = [(int(idx[0]), float(vals[0]))]
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
                seg_v = v[f1 : f2 + 1]
                mid = int(np.clip((f1 + f2) // 2, 0, len(v) - 1))
                mcv = float(np.mean(seg_v)) if len(seg_v) else 0.0
                pos_v = seg_v[seg_v > 0]
                mcv_pos = float(np.mean(pos_v)) if len(pos_v) else 0.0
                clipped = not (v_sanity[0] <= mcv <= v_sanity[1])
                reps.append(
                    Rep(
                        start_frame=a + f1,
                        end_frame=a + f2,
                        mcv=round(mcv, 3),
                        mcv_pos=round(mcv_pos, 3),
                        mcv_mid=round(float(np.clip(v[mid], *v_sanity)), 3),
                        pv=round(float(np.max(seg_v)) if len(seg_v) else 0.0, 3),
                        rom_m=round(rom, 4),
                        duration_s=round(dur, 2),
                        clipped=clipped,
                    )
                )
            i += 2
        else:
            i += 1

    gate_note = ""
    if reps:
        max_rom = max(r.rom_m for r in reps)
        rom_keep = max(rom_min_m, rom_keep_ratio * max_rom)
        n_before = len(reps)
        reps = [r for r in reps if r.rom_m >= rom_keep and r.mcv >= mcv_min]
        gate_note = (
            f"; 两遍门: {n_before}->{len(reps)} "
            f"(max_rom={max_rom * 100:.0f}cm, 门={rom_keep * 100:.1f}cm)"
        )

    return SegmentResult(
        reps=reps,
        status="OK" if reps else "NO_REPS",
        y_used=y_s,
        v_used=v,
        note=f"干净段 [{a},{b}] {b - a + 1}帧{gate_note}",
    )


# ══════════════════════════════════════════════════════════
#  Step 4 新版：直通 Kalman 速度，FSM 过零点分段
# ══════════════════════════════════════════════════════════


def segment_reps_from_velocity(  # noqa: pi-lens=unsafe-call,pi-lens=unsafe-call
    v_track: np.ndarray,
    y_track: np.ndarray,
    fps: float,
    mpp: float,
    dur_range: tuple[float, float] = (0.25, 4.5),
    rom_min_m: float = 0.012,
    rom_keep_ratio: float = 0.45,
    mcv_min: float = 0.10,
    v_sanity: tuple[float, float] = (0.05, 2.5),
    # FSM 阈值（针对 Kalman 平滑后的 v 设定，默认偏保守以适配 over-smooth）
    stop_velocity_mps: float = 0.01,  # |v| < 此值视为杠铃停止
    min_eccentric_mps: float = 0.015,  # 离心期 |v| 必须超过此值才算真离心
    # find_peaks 补侦参数
    sg_window: int = 9,  # 位置平滑窗口（3 次多项式）
    prominence_m: float = 0.02,  # ROM 突起度下限（米）
    min_rep_gap_s: float = 0.3,  # 相邻同侧事件最小间隔（秒）
) -> SegmentResult:
    """
    直通消费 Kalman v_track（单位 px/frame），输出 m/s 速度并按
    「过零点 FSM」分段。

    物理状态机：
      IDLE    → ECCENTRIC：|v| > min_eccentric（开始下放）
      ECCENTRIC → BOTTOM：v 由负转正过零点（动作底部）
      BOTTOM → CONCENTRIC：v > stop_velocity（向心起动后持续上升）
      CONCENTRIC → TOP：v < stop_velocity 持续若干帧（杠到顶停止）
      TOP → IDLE → 等待下一次离心

    MCV = (concentric 段 v × mpp × fps) 全程平均（GymAware ACV 口径）。
    ROM = (y_top - y_bottom) × mpp（y 向下为正，故 ROM = (y_b - y_t) × mpp）。
    """
    if len(v_track) != len(y_track):
        return SegmentResult(
            reps=[],
            status="NO_CLEAN_SEGMENT",
            note=f"v/y 长度不一致 {len(v_track)} vs {len(y_track)}",
        )

    # 单位转换：v_track 是 Kalman 状态输出的像素/秒（dt=1/fps 已代入 F 矩阵）
    # 所以 v × mpp 直接就是 m/s，不需要再乘 fps
    v_mps = v_track * mpp  # 向心为正（tracker 中已反转符号）
    y_m = y_track * mpp  # 米（向下为正）

    # 找最长干净段（v 不为 NaN 的连续段）
    run = longest_clean_run(v_mps)
    if run is None:
        return SegmentResult(
            reps=[], status="NO_CLEAN_SEGMENT", note="无 >30 帧连续速度段"
        )
    a, b = run
    v_seg = v_mps[a : b + 1]
    y_seg = y_m[a : b + 1]

    dur_min, dur_max = dur_range

    # ── 位置平滑 + find_peaks 分段（Step 4 实战版）──────────
    # 使用 Savitzky-Golay 轻微平滑位置（窗口 sg_window，2 次多项式），
    # 用 find_peaks 找杠铃底部（y 最大值）和顶部（y 最小值），
    # 底→顶配对作为 rep 边界。该方法远比手写 FSM 鲁棒。
    from scipy.signal import find_peaks, savgol_filter

    win = min(sg_window, len(y_seg) - 1)
    if win % 2 == 0:
        win -= 1
    if win < 5:
        return SegmentResult(
            reps=[], status="TOO_SHORT", note=f"干净段仅 {len(y_seg)} 帧"
        )
    y_smooth = savgol_filter(y_seg, win, 2)
    dist_min = max(1, int(fps * min_rep_gap_s))
    # y_seg 已经是米（y_track * mpp），直接用 prominence_m（米）
    bottoms, _ = find_peaks(y_smooth, distance=dist_min, prominence=prominence_m)
    tops, _ = find_peaks(-y_smooth, distance=dist_min, prominence=prominence_m)
    events = sorted([(int(f), "b") for f in bottoms] + [(int(f), "t") for f in tops])
    # 边界 top 合成（视频末尾的 bottom 补一个 top）
    if events and events[-1][1] == "b":
        last_b = events[-1][0]
        dur_end = (len(y_seg) - 1 - last_b) / fps
        rom_end = y_smooth[last_b] - y_smooth[-1]
        if dur_range[0] <= dur_end <= dur_range[1] and rom_end >= rom_min_m:
            events.append((len(y_seg) - 1, "t"))
            events.sort()

    reps: list[Rep] = []
    i = 0
    while i < len(events) - 1:
        f1, t1 = events[i]
        f2, t2 = events[i + 1]
        if t1 == "b" and t2 == "t":
            dur = (f2 - f1) / fps
            rom = abs(y_smooth[f2] - y_smooth[f1])
            if dur_min <= dur <= dur_max and rom >= rom_min_m:
                seg_v = v_seg[f1 : f2 + 1]
                pos_v = seg_v[seg_v > 0]
                mcv = float(np.mean(seg_v)) if len(seg_v) else 0.0
                mcv_pos = float(np.mean(pos_v)) if len(pos_v) else 0.0
                mid = int(np.clip((f1 + f2) // 2, 0, len(v_seg) - 1))
                clipped = not (v_sanity[0] <= mcv <= v_sanity[1])
                reps.append(
                    Rep(
                        start_frame=a + f1,
                        end_frame=a + f2,
                        mcv=round(mcv, 3),
                        mcv_pos=round(mcv_pos, 3),
                        mcv_mid=round(float(np.clip(v_seg[mid], *v_sanity)), 3),
                        pv=round(float(np.max(seg_v)) if len(seg_v) else 0.0, 3),
                        rom_m=round(rom, 4),
                        duration_s=round(dur, 2),
                        clipped=clipped,
                    )
                )
            i += 2
        else:
            i += 1

    # ── 两遍法质量门（M1）────────────────────────────────
    gate_note = ""
    if reps:
        max_rom = max(r.rom_m for r in reps)
        rom_keep = max(rom_min_m, rom_keep_ratio * max_rom)
        n_before = len(reps)
        reps = [r for r in reps if r.rom_m >= rom_keep and r.mcv >= mcv_min]
        gate_note = (
            f"; 两遍门: {n_before}->{len(reps)} "
            f"(max_rom={max_rom * 100:.0f}cm, 门={rom_keep * 100:.1f}cm)"
        )

    return SegmentResult(
        reps=reps,
        status="OK" if reps else "NO_REPS",
        y_used=y_seg,
        v_used=v_seg,
        note=f"FSM 干净段 [{a},{b}] {b - a + 1}帧{gate_note}",
    )
