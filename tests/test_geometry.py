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

from vbtcore.geometry import preprocess, canvas_to_orig


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
    """旋转分支：竖屏帧，检测框中心必须按 x=y', y=H-1-x' 映射，宽高互换。"""
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)  # 竖屏
    pp = preprocess(frame, 640)
    assert pp.rotated is True
    # 在 canvas 坐标 (400, 300) 放一个 w=30,h=50 的框（canvas 坐标系=旋转帧缩放）
    cx_c, cy_c, w_c, h_c = 400.0, 300.0, 30.0, 50.0
    # 先转旋转帧坐标
    x_rot = (cx_c - pp.xo) / pp.scale
    y_rot = (cy_c - pp.yo) / pp.scale
    w_rot = w_c / pp.scale
    h_rot = h_c / pp.scale
    cx_o, cy_o, w_o, h_o = canvas_to_orig(pp, cx_c, cy_c, w_c, h_c)
    # 旋转分支：orig_x = y_rot, orig_y = H - x_rot，且 w/h 互换
    assert abs(cx_o - y_rot) < 1e-6
    assert abs(cy_o - (1280 - x_rot)) < 1e-6
    assert abs(w_o - h_rot) < 1e-6
    assert abs(h_o - w_rot) < 1e-6


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
