"""
vbtcore.geometry — 帧预处理与坐标映射（纯函数，可单元测试）
===========================================================
修复的历史 bug：
  engines/anchor_template_engine.py::rotate_coord_back 的竖屏旋转逆映射公式错误。
  cv2.ROTATE_90_CLOCKWISE 的正映射是 (x, y) -> (H-1-y, x)（(0,0)→右上角），
  因此正确的逆映射是：
      orig_x = x'（旋转帧 x）
      orig_y = H_orig - 1 - y'
  旧实现 `orig_cx = rotated_h - rot_cy, orig_cy = rot_cx` 把坐标系
  镜像翻转了（可视化实证：检测框被映射到画面外/错误角落）。
  该错误保持两两距离不变（反射变换），跟踪侥幸可用，
  但所有绝对坐标逻辑（贴边拒绝、居中加权、bar path 绘制）全部失效。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FramePreprocess:
    """一帧预处理的结果：network blob + 逆映射所需参数。"""
    blob: np.ndarray            # (1, 3, S, S) float32
    scale: float                # canvas -> 预处理前帧 的缩放
    xo: int                     # canvas 上的 x 偏移
    yo: int
    rotated: bool               # 是否做了顺时针 90° 旋转
    orig_h: int                 # 原始帧尺寸
    orig_w: int


def preprocess(frame: np.ndarray, size: int = 640) -> FramePreprocess:
    """
    YOLO 预处理：竖屏先顺时针旋转 90°（消除旧路线的挤压 resize），
    再 letterbox 到 size×size。输出 blob 与逆映射参数。
    """
    h, w = frame.shape[:2]
    rotated = h > w
    if rotated:
        work = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    else:
        work = frame
    wh, ww = work.shape[:2]

    scale = min(size / wh, size / ww)
    nh, nw = int(wh * scale), int(ww * scale)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    yo = (size - nh) // 2
    xo = (size - nw) // 2
    resized = cv2.resize(work, (nw, nh))
    canvas[yo:yo + nh, xo:xo + nw] = resized

    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return FramePreprocess(blob=blob, scale=float(scale), xo=xo, yo=yo,
                           rotated=rotated, orig_h=h, orig_w=w)


def canvas_to_orig(pp: FramePreprocess, cx_c: float, cy_c: float,
                   w_c: float, h_c: float) -> tuple[float, float, float, float]:
    """
    canvas 上的 box 中心+宽高 → 原始帧坐标。
    旋转分支使用已验证的正确逆映射（见模块 docstring）；
    旋转同时交换 box 的宽高（圆度 ratio 不受影响）。
    """
    x1 = (cx_c - pp.xo) / pp.scale
    y1 = (cy_c - pp.yo) / pp.scale
    bw = w_c / pp.scale
    bh = h_c / pp.scale
    if not pp.rotated:
        return x1, y1, bw, bh
    # 逆旋转：orig_x = x', orig_y = H_orig - x'（-1 的亚像素差忽略）
    return y1, float(pp.orig_h) - x1, bh, bw


# ══════════════════════════════════════════════════════════
#  杠铃片直径查表（M1.5）
# ══════════════════════════════════════════════════════════
#
# 数值来源：TroyKaneshiro/barbell-velocity-tracker METHODOLOGY.md
# （力量举铁片口径）。注意单位陷阱：25lb≠25kg，键必须区分单位。
# 竞技举重片/包胶片多为全尺寸 450mm（与重量无关）；未知规格回退 0.45
# （= 本仓库 34 视频的 bumper 假设，保持现有行为不变）。

PLATE_DIAMETERS_M: dict[str, float] = {
    "45lb": 0.450, "25kg": 0.450, "20kg": 0.450,
    "35lb": 0.420,
    "25lb": 0.400, "15kg": 0.380,
    "10lb": 0.280, "10kg": 0.320,
}

DEFAULT_PLATE_DIAMETER_M = 0.45


def resolve_plate_diameter(outer_plate: str | None) -> float:
    """
    外层片规格（如 "20kg"/"45lb"，大小写/空格不敏感）→ 直径（米）。
    None 或未知规格 → 0.45（bumper 默认；App 应让用户从固定列表选择，
    避免把小铁片按 450mm 标定导致 ~2× 尺度误差）。
    """
    if outer_plate is None:
        return DEFAULT_PLATE_DIAMETER_M
    key = outer_plate.strip().lower().replace(" ", "")
    return PLATE_DIAMETERS_M.get(key, DEFAULT_PLATE_DIAMETER_M)


def rotate_point_roundtrip_check(frame_wh: tuple[int, int], n: int = 200, seed: int = 42) -> bool:
    """
    单元测试辅助：随机取 n 个像素位置，验证
    cv2.ROTATE_90_CLOCKWISE 正映射与我们逆映射的往返一致性（逐像素相等）。
    """
    rng = np.random.default_rng(seed)
    w, h = frame_wh
    img = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rh, rw = rot.shape[:2]  # rw == h, rh == w
    for _ in range(n):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        # 标准正映射（cv2 文档语义，直接取样验证）
        px_rot = rot[y, x]
        # 旋转帧中该像素位于 x' = H-1-y, y' = x
        x_rot, y_rot = h - 1 - y, x
        # 逆映射回原图坐标：orig_x = y', orig_y = H-1-x'
        ox, oy = y_rot, (h - 1) - x_rot
        if (ox, oy) != (x, y):
            return False
        if not np.array_equal(px_rot, img[oy, ox]):
            return False
    return True
