"""
vbtcore.detector — 杠铃片检测器（统一修复版）
=============================================
修复的历史 bug（algorithms/common.py::YoloPlateDetector）：
  1. 置信度双重 sigmoid：YOLO 导出的 row[4] 已是概率，旧代码再套
     1/(1+exp(-x))，任何非负置信度都 ≥0.5，conf 门控永久失效，
     标定防线被噪声击穿（20kg 视频 scale 虚大 3 倍的根因之一）。
  2. 直 resize 预处理：720×1280 → 416×416 非等比压扁，
     圆形片长宽比被系统性放大 ~1.78×，圆形度门控 1.4 必然误杀。
     现改用 letterbox（vbtcore.geometry.preprocess）。

模型兼容性：
  - yolo11_plate.onnx / plate_v1.onnx：输出 [1,5,N]，row[4]=概率 ✅
  - barbell_v4.onnx：输出 [1,6,3549]（4+2 类旧导出），本引擎不支持，
    需要时请用明确的解析器（避免再靠 argmax 侥幸工作）。
"""
from __future__ import annotations

from dataclasses import dataclass

import os

import cv2
import numpy as np
import onnxruntime as ort

from .geometry import FramePreprocess, preprocess, canvas_to_orig


@dataclass
class Detection:
    """原始帧坐标系下的一个检测。"""
    cx: float
    cy: float
    w: float
    h: float
    conf: float

    @property
    def ratio(self) -> float:
        """长宽比（≥1）。旋转交换宽高不影响该值。"""
        m = max(self.w, self.h)
        n = min(self.w, self.h)
        return m / n if n > 0 else 999.0


class PlateDetector:
    """ONNX YOLO 杠铃片检测器（letterbox + 正确置信度解析）。"""

    def __init__(self, model_path: str, input_size: int = 640,
                 providers: list[str] | None = None):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
        self.sess = ort.InferenceSession(
            model_path, sess_options=opts,
            providers=providers or ["CPUExecutionProvider"])
        self.inp_name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.size = int(shape[2]) if len(shape) >= 4 and shape[2] else input_size
        out_shape = self.sess.get_outputs()[0].shape
        self.n_channels = int(out_shape[1]) if out_shape and len(out_shape) >= 3 else 5
        if self.n_channels != 5:
            raise ValueError(
                f"模型输出 {out_shape} 不是 [1,5,N] 单类格式（可能是旧 barbell_v4 导出），"
                f"vbtcore 引擎不支持，请使用 yolo11_plate/plate_v1 或重新导出。")

    def detect(self, frame: np.ndarray, conf_thresh: float = 0.20) -> list[Detection]:
        """全帧检测 → 原始帧坐标 Detection 列表。"""
        pp = preprocess(frame, self.size)
        out = self.sess.run(None, {self.inp_name: pp.blob})[0]  # [1,5,N]
        row = out[0]
        confs = row[4]  # 已是概率，禁止再套 sigmoid
        keep = confs > conf_thresh
        dets: list[Detection] = []
        for i in np.where(keep)[0]:
            cx_o, cy_o, w_o, h_o = canvas_to_orig(
                pp, float(row[0, i]), float(row[1, i]),
                float(row[2, i]), float(row[3, i]))
            dets.append(Detection(cx=cx_o, cy=cy_o, w=w_o, h=h_o,
                                  conf=float(confs[i])))
        return dets
