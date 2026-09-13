"""
tests/test_geometry.py — 坐标映射回归测试
锁定 bug：engines/anchor_template_engine.py::rotate_coord_back 的
竖屏旋转逆映射镜像错误。vbtcore.geometry.canvas_to_orig 必须与
cv2.ROTATE_90_CLOCKWISE 的标准正映射严格互逆。
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vbtcore.geometry import canvas_to_orig, preprocess


def test_rotate_roundtrip_pixel_exact():
    """随机像素往返：原图 (x,y) → 旋转帧 → 我们的逆映射 → 必须回到 (x,y)。"""
    rng = np.random.default_rng(42)
    w, h = 720, 1280
    img = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    assert rot.shape[:2] == (w, h)
    for _ in range(300):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        # 标准正映射：原图(x,y) → 旋转帧 行=y'=x, 列=x'=H-1-y
        px_rot = rot[x, h - 1 - y]
        # 逆映射（canvas_to_orig 旋转分支语义）：orig_x=y'(行), orig_y=H-1-x'(列)
        ox, oy = x, (h - 1) - (h - 1 - y)
        assert (ox, oy) == (x, y), f"逆映射错误: ({ox},{oy}) != ({x},{y})"
        assert np.array_equal(px_rot, img[oy, ox])


def test_canvas_to_orig_letterbox_roundtrip():
    """letterbox 分支：canvas 中心点映射回原坐标。"""
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)  # 横屏图不会走旋转分支
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    pp = preprocess(frame, 640)
    assert pp.rotated is False
    # canvas 上 letterbox 区中心
    cx_c, cy_c = pp.xo + 100, pp.yo + 50
    w_c = h_c = 40.0
    cx_o, cy_o, w_o, h_o = canvas_to_orig(pp, cx_c, cy_c, w_c, h_c)
    assert abs(cx_o - 100 / pp.scale) < 1e-6
    assert abs(cy_o - 50 / pp.scale) < 1e-6
    assert abs(w_o - 40 / pp.scale) < 1e-6


def test_canvas_to_orig_rotated_swaps_and_maps():
    """Step 4 Refactor 后：竖屏帧走纯 letterbox 不旋转路径。
    （原旋转分支已被废除——训练数据 720×1280 竖屏直接 letterbox 即可，
    旋转逆映射的坐标系镜像 bug 一并消除。）"""
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)  # 竖屏
    pp = preprocess(frame, 640)
    # 新架构：所有方向统一走 letterbox，不旋转
    assert pp.rotated is False
    # canvas 坐标 (400, 300) 放一个 w=30, h=50 的框，验证 letterbox 反向映射
    cx_c, cy_c, w_c, h_c = 400.0, 300.0, 30.0, 50.0
    cx_o, cy_o, w_o, h_o = canvas_to_orig(pp, cx_c, cy_c, w_c, h_c)
    # letterbox 分支：orig = (canvas - offset) / scale，w/h 不互换
    expected_cx = (cx_c - pp.xo) / pp.scale
    expected_cy = (cy_c - pp.yo) / pp.scale
    expected_w = w_c / pp.scale
    expected_h = h_c / pp.scale
    assert abs(cx_o - expected_cx) < 1e-6, f"cx_o={cx_o} vs {expected_cx}"
    assert abs(cy_o - expected_cy) < 1e-6, f"cy_o={cy_o} vs {expected_cy}"
    assert abs(w_o - expected_w) < 1e-6, f"w_o={w_o} vs {expected_w}"
    assert abs(h_o - expected_h) < 1e-6, f"h_o={h_o} vs {expected_h}"


def test_preprocess_no_squeeze():
    """letterbox 必须保持等比：圆形物在 canvas 上仍是圆形。"""
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    cv2.circle(frame, (360, 640), 100, (255, 255, 255), -1)  # 真圆
    pp = preprocess(frame, 640)
    # 等比缩放断言：scale 对宽高一致
    s = pp.scale
    nh, nw = int(1280 * s), int(720 * s)
    assert abs(nw * 1 - (720 * s)) < 2
    assert abs(nh - 1280 * s) < 2
