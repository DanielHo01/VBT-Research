"""
tests/test_anchor.py — 锚定物理先验 + 运动探针回归测试
锁定 bug：最高置信 ≠ 工作杠铃片（80kg 架下片堆 / 130kg 贴边片堆实验实证）。
架构决策：尺寸/贴边先验只过滤明显非法候选；片堆 vs 工作片的最终
区分靠运动探针（y 方差）—— 单一尺寸阈值不可分（80kg 片堆 0.141
vs 30kg 近距工作片 0.137）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vbtcore.detector import Detection
from vbtcore.engine import anchor_score, select_anchor_track

H, W = 1280, 720


def test_reject_offscreen_stack():
    """130kg 场景：贴左缘的部分出画片堆（cx=-136）必须被拒绝。"""
    d = Detection(cx=-136, cy=187, w=378, h=288, conf=0.73)
    assert anchor_score(d, H, W) == 0.0


def test_flat_rectangle_rejected():
    """侧视片堆呈扁矩形（ratio>1.6）→ 拒绝。"""
    d = Detection(cx=360, cy=640, w=300, h=100, conf=0.9)
    assert anchor_score(d, H, W) == 0.0


def test_size_domain():
    """太小（远处噪声）与太大（贴脸）都拒绝。"""
    tiny = Detection(cx=360, cy=640, w=20, h=25, conf=0.9)     # h/frame=0.02
    huge = Detection(cx=360, cy=640, w=500, h=400, conf=0.9)   # h/frame=0.31
    assert anchor_score(tiny, H, W) == 0.0
    assert anchor_score(huge, H, W) == 0.0


def test_working_plate_scores_positive():
    """正常工作片（居中、圆形、尺寸合理）得分 > 0。"""
    d = Detection(cx=380, cy=640, w=80, h=75, conf=0.55)
    s = anchor_score(d, H, W)
    assert s > 0.3, f"工作片得分过低: {s}"


def test_motion_probe_prefers_moving_plate():
    """80kg 场景：静态高置信片堆 vs 运动工作片 → 探针必须选工作片。"""
    stack = {"cx": 192, "cy": 779, "h": 181, "conf": 0.86,
             "hits": 40, "miss": 0, "ys": [779.0] * 40}          # 静止
    working = {"cx": 380, "cy": 640, "h": 77, "conf": 0.50,
               "hits": 30, "miss": 0, "ys": list(np.linspace(500, 780, 30))}  # 运动
    picked = select_anchor_track([stack, working])
    assert picked is working, "运动探针未选中工作片"


def test_motion_probe_requires_min_hits():
    """确认门槛：hits 不足的轨迹不能锚定（防误锁瞬时误检）。"""
    t = {"cx": 380, "cy": 640, "h": 77, "conf": 0.9,
         "hits": 2, "miss": 0, "ys": [640.0, 641.0]}
    assert select_anchor_track([t], min_hits=5) is None


import numpy as np  # noqa: E402  (置于顶部会与 docstring 测试顺序冲突，此处显式导入)
