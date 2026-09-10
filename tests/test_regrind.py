"""
tests/test_regrind.py — M1.5 回归测试（纯合成信号/纯函数，无需模型）
移植依据：TroyKaneshiro/barbell-velocity-tracker
  - Bottom-of-rep regrind 触发器（常数 = Troy 实测值）
  - MCV 全程平均口径（GymAware ACV；旧平均正速度降级为诊断）
  - 外层片直径查表
锁定修正：Troy 原实现触发后 ref=触发点 → 第二 rep 起永不武装
（实为每组一次）；本实现 UP/DOWN 双相做到真正每 rep 一次。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vbtcore.detector import Detection
from vbtcore.engine import (BottomRegrind, select_regrind_candidate,
                            regrind_verdict)
from vbtcore.geometry import resolve_plate_diameter
from vbtcore.segment import segment_reps


def _fires(ys, plate_r=100.0, **kw):
    rg = BottomRegrind(**kw)
    return [i for i, y in enumerate(ys) if rg.update(float(y), plate_r)]


def _valley(top=300.0, bottom=500.0, down=30, hold=6, up=30, pause=10):
    parts = [np.full(pause, top),
             np.linspace(top, bottom, down, endpoint=False),
             np.full(hold, bottom),
             np.linspace(bottom, top, up, endpoint=False),
             np.full(pause, top)]
    return np.concatenate(parts)


def test_fires_once_per_valley():
    """核心：连续两谷必须各触发一次（锁定 per-rep，防退化成 Troy 的每组一次）。"""
    one = _valley()
    y = np.concatenate([one, one])
    fires = _fires(y)
    assert len(fires) == 2, f"两谷应触发两次，实际 {fires}"
    # 触发点应在离底后不久（回落 0.3r≈30px 约 5 帧 + 确认 4 帧）
    vlen = len(one)
    for k, f in enumerate(fires):
        bottom_end = k * vlen + 10 + 30 + 6
        assert bottom_end + 3 <= f <= bottom_end + 14, \
            f"触发时机异常：{f}（谷底结束 {bottom_end}）"


def test_setup_jitter_no_fire():
    """setup 抖动（±15px « 1.5r 武装线）永不触发。"""
    rng = np.random.default_rng(7)
    y = 400.0 + rng.normal(0, 5, 300)
    assert _fires(y) == []


def test_shallow_dip_no_fire():
    """100px 浅蹲（< 1.5r=150）不武装 → 不触发。"""
    y = np.concatenate([np.full(30, 300.0),
                        np.linspace(300, 400, 20),
                        np.linspace(400, 300, 20),
                        np.full(30, 300.0)])
    assert _fires(y) == []


def test_paused_squat_fires_once():
    """蹲底停顿 40 帧：停顿时不触发，起身后触发一次。"""
    fires = _fires(_valley(hold=40))
    assert len(fires) == 1, f"暂停蹲应触发一次，实际 {fires}"


def test_reset_requires_full_rearm():
    """大跳变（夺回）重锚后：需重新走完 位移→反转，小回落不触发。"""
    rg = BottomRegrind()
    for y in np.linspace(300, 500, 30):   # 下蹲 200px → 已武装
        rg.update(float(y), 100.0)
    assert rg.armed
    rg.reset(800.0)
    assert not rg.armed and not rg.phase_up
    for y in np.full(10, 790.0):
        assert rg.update(float(y), 100.0) is False


def test_candidate_gate():
    """门内最近者选中；门外（>4r）即使高置信也拒绝。"""
    near = Detection(cx=100, cy=100, w=80, h=80, conf=0.3)
    far = Detection(cx=600, cy=600, w=80, h=80, conf=0.9)
    assert select_regrind_candidate([far, near], 110, 110, 100.0) is near
    assert select_regrind_candidate([far], 110, 110, 100.0) is None
    assert select_regrind_candidate([near], 110, 110, 0.0) is None


def test_regrind_verdict_tiers():
    """纠正分级：身份失败拒绝；微偏只刷新模板；真漂移全量纠正。"""
    assert regrind_verdict(100.0, 0.3, 0.3, 100.0) is None   # 身份失败
    assert regrind_verdict(4.0, 0.8, 0.6, 100.0) == "micro"   # 4px 微偏
    assert regrind_verdict(100.0, 0.8, 0.3, 100.0) == "snap"  # NCC 可信
    assert regrind_verdict(100.0, 0.2, 0.7, 100.0) == "snap"  # 高 conf 可信
    assert regrind_verdict(19.0, 0.9, 0.9, 100.0) == "micro"  # 边界 <0.2r
    assert regrind_verdict(21.0, 0.9, 0.9, 100.0) == "snap"   # 边界 >0.2r


def test_plate_diameter_lookup():
    assert resolve_plate_diameter(None) == 0.45
    assert resolve_plate_diameter("20kg") == 0.45
    assert resolve_plate_diameter(" 45LB ") == 0.45
    assert resolve_plate_diameter("15kg") == 0.38
    assert resolve_plate_diameter("10lb") == 0.28
    assert resolve_plate_diameter("35lb") == 0.42
    assert resolve_plate_diameter("未知规格") == 0.45   # 未知回退 bumper 默认


def _osc_set():
    """2 reps；向心腿叠加 8px/20帧正弦（制造段内负速度，区分两种 MCV 口径）。"""
    half = 30
    parts = []
    for _ in range(2):
        parts.append(np.linspace(0.0, 120.0, half, endpoint=False))
        t = np.arange(half)
        parts.append(np.linspace(120.0, 0.0, half, endpoint=False)
                     + 8.0 * np.sin(2 * np.pi * 1.5 * t / half))
        parts.append(np.full(40, 0.0))
    return np.concatenate(parts), 30.0


def test_mcv_is_whole_rom_mean():
    """MCV = 全程平均 = ROM/duration；段内负速度使平均正速度严格更大。"""
    y, fps = _osc_set()
    mpp = 0.0045
    res = segment_reps(y, fps, mpp, sg_window=11)
    assert res.status == "OK" and len(res.reps) == 2, \
        f"{res.note}: {[(r.start_frame, r.end_frame) for r in res.reps]}"
    for r in res.reps:
        ratio = abs(r.mcv - r.rom_m / r.duration_s) / (r.rom_m / r.duration_s)
        assert ratio < 0.10, f"mcv={r.mcv} vs ROM/dur={r.rom_m / r.duration_s:.3f}"
        assert r.mcv_pos > r.mcv + 0.005, f"mcv_pos={r.mcv_pos} mcv={r.mcv}"
        assert not r.clipped
