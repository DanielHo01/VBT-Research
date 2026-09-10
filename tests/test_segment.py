"""
tests/test_segment.py — rep 分段回归测试（纯合成信号，无需模型）
锁定决策：MCV = 向心段平均正速度（对齐 GymAware），mcv_mid 仅诊断。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vbtcore.segment import segment_reps, small_gap_interp, longest_clean_run


def _synthetic_set(n_reps=4, rom_px=120.0, fps=30.0, rep_dur_s=2.0, pause_frames=15):
    """
    合成深蹲轨迹（y 向下为正，单位 px）：
    每个 rep = 线性下蹲(half) + 线性站起(half) + 顶端停留(pause)。
    向心段（站起）为匀速 → MCV 理论值 = rom_px*mpp/(half/fps)。
    """
    half = int(rep_dur_s * fps / 2)
    parts = []
    for _ in range(n_reps):
        parts.append(np.linspace(0.0, rom_px, half, endpoint=False))          # 离心
        parts.append(np.linspace(rom_px, 0.0, half, endpoint=False))          # 向心
        parts.append(np.full(pause_frames, 0.0))                              # 停留
    return np.concatenate(parts)


def test_segment_finds_all_reps_with_mean_mcv():
    fps = 30.0
    rom_px = 120.0
    rep_dur = 2.0          # 向心段 30 帧，稀释 SG 端点效应
    n_reps = 4
    y = _synthetic_set(n_reps=n_reps, rom_px=rom_px, rep_dur_s=rep_dur, fps=fps)
    y = y[:-15]   # 去掉末尾部分停顿（模拟视频在回位后不久结束）
    mpp = 0.45 / 100.0   # 假设片高 100px
    res = segment_reps(y, fps, mpp, sg_window=11)
    assert res.status == "OK", res.note
    assert len(res.reps) == n_reps, f"期望 {n_reps} reps, 得到 {len(res.reps)}"
    v_theory = (rom_px * mpp) / (rep_dur / 2)
    for r in res.reps:
        assert abs(r.mcv - v_theory) / v_theory < 0.15, \
            f"MCV(mean)={r.mcv} vs 理论 {v_theory:.3f}"
        assert abs(r.mcv_mid - v_theory) / v_theory < 0.15
        # SG 在合成信号的尖锐转角/末段边界处削峰 ~3-7%（真实光滑信号影响更小）
        assert abs(r.rom_m - rom_px * mpp) < 0.05
        assert abs(r.duration_s - rep_dur / 2) < 0.15


def test_gap_rejection():
    """跨大 gap（>10 帧不插值）的轨迹：rep 永不跨越 NaN 帧。"""
    y = _synthetic_set(n_reps=6)
    y[120:200] = np.nan   # 80 帧 NaN 大 gap
    res = segment_reps(y, 30.0, 0.0045)
    for r in res.reps:
        seg = y[r.start_frame:r.end_frame + 1]
        assert not np.isnan(seg).any(), \
            f"rep [{r.start_frame},{r.end_frame}] 跨越了 NaN gap"
    assert len(res.reps) <= 3, f"碎片化未抑制: {len(res.reps)}"


def test_small_gap_interp():
    # 洞1: idx 3-4（2 帧）；洞2: idx 7-17（11 帧）；两端有效
    y = np.array([0., 1., 2., np.nan, np.nan, 5., 6.] + [np.nan] * 11 + [20.])
    assert len(y) == 19
    out = small_gap_interp(y, max_gap=10)
    assert not np.isnan(out[3]) and not np.isnan(out[4])   # 小洞 → 线性填充
    assert np.isnan(out[7:18]).all()                        # 11 帧大洞 → 保持 NaN
    out2 = small_gap_interp(y, max_gap=15)
    assert not np.isnan(out2[7:18]).any()                   # max_gap 放宽 → 全填


def test_longest_clean_run():
    y = np.array([0.] * 40 + [np.nan] + [1.] * 40 + [np.nan, np.nan] + [2.] * 45)
    a, b = longest_clean_run(y, min_len=30)
    assert (a, b) == (83, 127), f"得到 ({a},{b})"
    # 短段会被 min_len 过滤（跨度语义: b-a > min_len；3 帧段需 min_len<2）
    assert longest_clean_run(np.array([0., 1., np.nan, 2., 3., 4.]), min_len=1) == (3, 5)


def _add_jitter_bumps(y, positions, amp_px=8.0, width=5):
    """在轨迹的停留段注入抖动包（模拟 NCC 漂移产生的假 rep）。"""
    out = y.copy()
    for pos in positions:
        for k, w in enumerate(range(-width, width + 1)):
            idx = pos + w
            if 0 <= idx < len(out):
                out[idx] += amp_px * np.exp(-0.5 * (w / (width / 2)) ** 2)
    return out


def test_two_pass_rom_gate_kills_jitter_reps():
    """M1 核心：真 rep(ROM 120px) + 抖动假 rep(ROM 8px) → 只留真 rep。"""
    fps = 30.0
    mpp = 0.0045
    y = _synthetic_set(n_reps=3, rom_px=120.0, rep_dur_s=2.0, pause_frames=40)
    # 在每个停留段中间注入一个抖动包（会产生假峰谷对）
    pause_len = 40
    positions = []
    block = int(2.0 * fps) + pause_len      # 一个 rep 块的长度
    for i in range(3):
        positions.append(i * block + int(2.0 * fps) + pause_len // 2)
    y_jitter = _add_jitter_bumps(y, positions, amp_px=8.0, width=5)
    res = segment_reps(y_jitter, fps, mpp)
    assert len(res.reps) == 3, \
        f"两遍门失效: 得到 {len(res.reps)} reps（期望 3）: " \
        f"{[(r.start_frame, round(r.rom_m*100,1)) for r in res.reps]}"
    for r in res.reps:
        assert abs(r.rom_m - 120 * mpp) < 0.03   # SG 转角削峰 ~3%


def test_two_pass_gate_ratio_param():
    """rom_keep_ratio=0（关闭相对门）时假 rep 会保留 —— 参数确实起作用。"""
    fps = 30.0
    mpp = 0.0045
    y = _synthetic_set(n_reps=2, rom_px=120.0, rep_dur_s=2.0, pause_frames=40)
    block = int(2.0 * fps) + 40
    y_jitter = _add_jitter_bumps(y, [block // 2 + 20], amp_px=8.0, width=5)
    res_off = segment_reps(y_jitter, fps, mpp, rom_keep_ratio=0.0, mcv_min=0.0)
    assert len(res_off.reps) == 3, "相对门关闭时应有 3 个候选（2真+1假）"
    res_on = segment_reps(y_jitter, fps, mpp, rom_keep_ratio=0.45)
    assert len(res_on.reps) == 2, "相对门开启时应只剩 2 个真 rep"


def test_two_pass_gate_all_fake_input():
    """全是小抖动时（无真 rep），相对门不该硬留 —— 返回 NO_REPS 而非垃圾。"""
    fps = 30.0
    mpp = 0.0045
    y = np.zeros(300)
    y = _add_jitter_bumps(y, [50, 120, 200], amp_px=6.0, width=6)
    res = segment_reps(y, fps, mpp)
    assert len(res.reps) == 0, f"全抖动输入不应出 rep: {len(res.reps)}"
