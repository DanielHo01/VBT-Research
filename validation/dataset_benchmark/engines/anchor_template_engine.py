"""
AnchorTemplateEngine — yolo11_plate 检测器 + 首帧锚定 + NCC 模板追踪
"""

from __future__ import annotations
import os
import numpy as np
import cv2
import onnxruntime as ort
from typing import Optional
from scipy.signal import savgol_filter


# ─────────────────────────────────────────────────────────────────────────────
#  内部工具
# ─────────────────────────────────────────────────────────────────────────────

def letterbox(img: np.ndarray, size: int = 640):
    """Standard letterbox缩放到size×size，保持宽高比。返回blob+scale+yo+xo"""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    yo = (size - nh) // 2
    xo = (size - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return blob, float(scale), yo, xo


def canvas_to_orig(cx_c: float, cy_c: float,
                   scale: float, yo: int, xo: int) -> tuple[float, float]:
    """Letterbox 画布坐标 → 原始帧坐标"""
    return (cx_c - xo) / scale, (cy_c - yo) / scale


def prepare_frame_for_yolo(frame: np.ndarray, size: int = 640):
    """
    竖屏视频旋转90°再送入YOLO，坐标旋回原帧坐标系。
    返回 (blob, scale, yo, xo, rotated_h, rotated_w, was_rotated)
    """
    h, w = frame.shape[:2]
    if h > w:  # 竖屏 → 旋转90°变为横屏
        rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        rh, rw = rotated.shape[:2]
        scale = min(size / rh, size / rw)
        nh, nw = int(rh * scale), int(rw * scale)
        resized = cv2.resize(rotated, (nw, nh))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        yo = (size - nh) // 2
        xo = (size - nw) // 2
        canvas[yo:yo + nh, xo:xo + nw] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return blob, float(scale), yo, xo, rh, rw, True
    else:  # 横屏 → 直接letterbox
        scale = min(size / h, size / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        yo = (size - nh) // 2
        xo = (size - nw) // 2
        canvas[yo:yo + nh, xo:xo + nw] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return blob, float(scale), yo, xo, h, w, False


def rotate_coord_back(cx_c: float, cy_c: float, scale: float, yo: int, xo: int,
                      rotated_h: int, rotated_w: int) -> tuple[float, float]:
    """
    旋转帧的画布坐标 → 原始竖屏帧坐标。
    竖屏帧被 ROTATE_90_CLOCKWISE → 得到横屏帧 (rw_orig=h, rh_orig=w)。
    检测在横屏帧中坐标 (rot_cx, rot_cy)。
    旋回：orig_cx = rh_orig - rot_cy, orig_cy = rot_cx
    其中 rot_cx = (cx_c-xo)/scale, rot_cy = (cy_c-yo)/scale
    """
    rot_cx = (cx_c - xo) / scale
    rot_cy = (cy_c - yo) / scale
    # ROTATE_90_CLOCKWISE: (x,y)→(h-y, x) in the rotated space
    # rotated_h = original_w, rotated_w = original_h
    orig_cx = rotated_h - rot_cy
    orig_cy = rot_cx
    return orig_cx, orig_cy


def parse_yolo11(output: np.ndarray, conf_thresh: float = 0.20) -> list[dict]:
    """
    解析 yolo11_plate ONNX 输出 [1, 5, 8400]。
    row[4] 已经是 sigmoid 概率（不要重复 sigmoid）。
    """
    if output is None or output.size == 0:
        return []
    row = output[0]          # (5, 8400)
    confs = row[4]          # already sigmoid probability
    mask = confs > conf_thresh
    if not mask.any():
        return []
    idxs = np.where(mask)[0]
    return [
        {"cx": float(row[0, i]), "cy": float(row[1, i]),
         "w":  float(row[2, i]), "h":  float(row[3, i]),
         "conf": float(confs[i])}
        for i in idxs
    ]


def canvas_to_orig(cx_c: float, cy_c: float,
                   scale: float, yo: int, xo: int) -> tuple[float, float]:
    """Letterbox 画布坐标 → 原始帧坐标"""
    return (cx_c - xo) / scale, (cy_c - yo) / scale


# ─────────────────────────────────────────────────────────────────────────────
#  AnchorTemplateEngine
# ─────────────────────────────────────────────────────────────────────────────

class AnchorTemplateEngine:

    def __init__(self,
                 onnx_path: str,
                 plate_diameter_m: float = 0.45,
                 roi_expand: float = 2.0,
                 track_interval: int = 45,
                 track_score_thresh: float = 0.25,
                 conf_thresh: float = 0.20,
                 max_jump_px: float = 60.0,
                 sg_window: int = 15,
                 sg_poly: int = 3,
                 debug: bool = False,
                 yolo_size: int = 640,
                 detect_interval: int = 1):
        self.plate_diameter_m = plate_diameter_m
        self.roi_expand = roi_expand
        self.track_interval = track_interval
        self.track_score_thresh = track_score_thresh
        self.conf_thresh = conf_thresh
        self.max_jump_px = max_jump_px
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.debug = debug

        providers = ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.inp_name = self.sess.get_inputs()[0].name
        self.detect_interval = detect_interval

    # ── YOLO 检测 ──────────────────────────────────────────────────────────────

    def _detect_letterbox(self, frame: np.ndarray,
                       size: int = 640) -> tuple[list[dict], float, int, int, int, int, bool]:
        """
        全帧 YOLO（letterbox `size`）。
        竖屏视频自动旋转90°检测，坐标旋回原帧坐标系。
        返回: (dets, scale, yo, xo, frame_h, frame_w, was_rotated)
        """
        blob, scale, yo, xo, f_h, f_w, rotated = prepare_frame_for_yolo(frame, size)
        out = self.sess.run(None, {self.inp_name: blob})[0]
        dets = parse_yolo11(out, self.conf_thresh)
        return dets, scale, yo, xo, f_h, f_w, rotated

    def _detect_in_roi_orig(self, frame: np.ndarray,
                            roi_x: int, roi_y: int,
                            roi_w: int, roi_h: int,
                            size: int = 640) -> list[dict]:
        """
        在原始帧 ROI 内做 letterbox YOLO。
        roi_* 为原始帧坐标；crop → letterbox size → YOLO → 映射回原始坐标。
        """
        x1 = max(0, roi_x); y1 = max(0, roi_y)
        x2 = min(frame.shape[1], roi_x + roi_w)
        y2 = min(frame.shape[0], roi_y + roi_h)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return []
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        blob, scale, yo, xo = letterbox(crop, size)
        out = self.sess.run(None, {self.inp_name: blob})[0]
        dets_c = parse_yolo11(out, self.conf_thresh)

        results = []
        for d in dets_c:
            cx_o = (d["cx"] - xo) / scale + x1
            cy_o = (d["cy"] - yo) / scale + y1
            results.append({
                "cx": cx_o, "cy": cy_o,
                "w":  d["w"]  / scale,
                "h":  d["h"]  / scale,
                "conf": d["conf"],
            })
        return results

    # ── 首帧锚定 ──────────────────────────────────────────────────────────────

    def _bootstrap_anchor(self, cap: cv2.VideoCapture) -> Optional[dict]:
        """
        扫描前 30 帧，打分选最佳锚定检测。
        Returns: {cx, cy, w, h, scale, frame_idx} 原始帧坐标。
        """
        best_score = -1.0
        best = None
        h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_area = h_frame * w_frame

        for fi in range(30):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                break

            dets, scale, yo, xo = self._detect_letterbox(frame)

            for d in dets:
                # 映射到原始帧坐标
                cx_o, cy_o = canvas_to_orig(d["cx"], d["cy"], scale, yo, xo)
                w_o = d["w"] / scale
                h_o = d["h"] / scale

                # 过滤极端尺寸
                area = w_o * h_o
                if area < 0.001 * frame_area or area > 0.30 * frame_area:
                    continue
                aspect = h_o / max(w_o, 1e-6)
                if aspect < 0.2 or aspect > 3.5:
                    continue

                # 打分：直接用置信度（yolo11_plate 已足够准）
                score = d["conf"]

                if score > best_score:
                    best = {
                        "cx": cx_o, "cy": cy_o,
                        "w": w_o,   "h": h_o,
                        "conf": d["conf"],
                        "frame_idx": fi,
                    }
                    best_score = score

        return best

    # ── 模板锁定 ───────────────────────────────────────────────────────────────

    def _lock_template(self, frame: np.ndarray, anchor: dict) -> dict:
        """
        以 anchor 为中心裁剪小块模板，用于 NCC 匹配。
        模板尺寸：anchor 高×1.2，上限 120px，下限 30px。
        """
        h_f, w_f = frame.shape[:2]
        cx, cy = anchor["cx"], anchor["cy"]
        anchor_h = max(anchor["h"], 10.0)
        anchor_w = max(anchor["w"], 10.0)

        t_h = int(np.clip(anchor_h * 1.2, 30, 120))
        t_w = int(np.clip(anchor_w * 1.2, 30, 120))

        # 安全裁剪
        x1 = int(np.clip(cx - t_w / 2, 0, max(0, w_f - t_w)))
        y1 = int(np.clip(cy - t_h / 2, 0, max(0, h_f - t_h)))
        x2 = min(w_f, x1 + t_w)
        y2 = min(h_f, y1 + t_h)
        # 重新确保尺寸正确
        t_h = y2 - y1; t_w = x2 - x1

        template = frame[y1:y2, x1:x2].copy()
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        return {
            "template":       template,
            "template_gray": template_gray,
            "anchor":        anchor,
            "cx": float(cx), "cy": float(cy),
            "last_yolo_cx": float(cx),
            "last_yolo_cy": float(cy),
            "last_yolo_frame": 0,
            "frame_h": h_f,  "frame_w": w_f,
            "valid": True,
        }

    # ── 单帧检测（逐帧最近邻主路径）──────────────────────────────────────

    def _detect_frame(self, frame: np.ndarray,
                      ref_cx: float | None = None,
                      ref_cy: float | None = None) -> list[dict]:
        """
        全帧 YOLO 检测，解析为原始帧坐标。
        当 ref_cx/ref_cy 给定时，使用 "单峰选择策略"：
          - 如果最高 conf 检测在合理范围，优先用它
          - 否则降级到次优检测（尺寸 + 位置匹配 barbell 特征）
        """
        dets, scale, yo, xo, f_h, f_w, rotated = self._detect_letterbox(frame)
        if not dets:
            return []

        # 解析到原始坐标
        results = []
        for d in dets:
            if rotated:
                # 旋转帧：坐标旋回原竖屏帧
                cx_o, cy_o = rotate_coord_back(d["cx"], d["cy"],
                                                scale, yo, xo, f_h, f_w)
                h_o = d["h"] / scale
                w_o = d["w"] / scale
            else:
                cx_o = (d["cx"] - xo) / scale
                cy_o = (d["cy"] - yo) / scale
                h_o = d["h"] / scale
                w_o = d["w"] / scale
            results.append({
                "cx": cx_o, "cy": cy_o,
                "w":  w_o,
                "h":  h_o,
                "conf": d["conf"],
            })

        if ref_cx is not None and ref_cy is not None and len(results) > 1:
            # 单峰选择：综合 conf + 位置 + 尺寸，选最佳 barbell 候选
            frame_h = frame.shape[0]
            frame_w = frame.shape[1]
            for d in results:
                dist_ref = np.hypot(d["cx"] - ref_cx, d["cy"] - ref_cy)
                h = d["h"]; w = d["w"]
                # 尺寸打分：中等大小（80-350 px）优先
                size_ok = 80 < h < 350 and 50 < w < 400
                size_score = 1.0 if size_ok else 0.2
                # 中心度打分
                cx_n = abs(d["cx"] - frame_w / 2) / (frame_w / 2)
                cy_n = abs(d["cy"] - frame_h / 2) / (frame_h / 2)
                center_score = max(0, 1.0 - (cx_n + cy_n) / 2)
                d["_score"] = d["conf"] * (0.4 + 0.6 * size_score) * (0.5 + 0.5 * center_score)
            results.sort(key=lambda x: x["_score"], reverse=True)

            # 返回前两名，让最近邻追踪有选择余地
            return results[:2]

        return results

    # ── 主流程 ────────────────────────────────────────────────────────────────

    def process_video(self, video_path: str,
                     anchor_xy: tuple[float, float] | None = None) -> list[dict]:
        """
        经典 3 步法（新检测器）：
          1) 逐帧 YOLO + 最近邻追踪（60px 门禁）
          2) 中位数锁定标定
          3) SG(15,3) 平滑 + bottom→top rep 提取
        """
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

        # ── Pass 1: 逐帧检测 + 最近邻追踪 ────────────────────────────────
        raw_y: list[float] = []
        raw_x: list[float] = []
        plate_heights: list[float] = []

        last_cx: float | None = None
        last_cy: float | None = None
        max_jump = self.max_jump_px

        frame_idx = 0
        last_det_cx = last_cx
        last_det_cy = last_cy
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            #稀疏检测：每 detect_interval 帧跑一次 YOLO
            if frame_idx % self.detect_interval == 0:
                dets = self._detect_frame(frame, ref_cx=last_det_cx, ref_cy=last_det_cy)
            else:
                dets = []

            if not dets:
                # 稀疏跳帧 → 沿用上一帧
                cx = last_det_cx if last_det_cx is not None else 0.0
                cy = last_det_cy if last_det_cy is not None else 0.0
            elif last_det_cx is None:
                # 首帧：选置信度最高的检测
                best = max(dets, key=lambda d: d["conf"])
                cx, cy = best["cx"], best["cy"]
                last_det_cx, last_det_cy = cx, cy
                plate_heights.append(best["h"])
            else:
                # 最近邻追踪
                closest = min(dets,
                              key=lambda d: np.hypot(d["cx"] - last_det_cx, d["cy"] - last_det_cy))
                dist = np.hypot(closest["cx"] - last_det_cx, closest["cy"] - last_det_cy)
                if dist < max_jump:
                    cx, cy = closest["cx"], closest["cy"]
                    last_det_cx, last_det_cy = cx, cy
                    plate_heights.append(closest["h"])
                else:
                    cx, cy = last_det_cx, last_det_cy

            raw_x.append(cx); raw_y.append(cy)
            if cx != 0.0:
                last_cx = cx
            if cy != 0.0:
                last_cy = cy

        cap.release()

        if len(raw_y) < 30 or len(plate_heights) < 5:
            return []

        # ── Pass 2: 锁定标定 ───────────────────────────────────────────────────
        median_h = float(np.median(plate_heights))
        if median_h < 2.0:
            return []
        scale_m_per_px = self.plate_diameter_m / median_h

        # ── Pass 3: SG 平滑与微分 ──────────────────────────────────────────────
        from scipy.signal import savgol_filter as _sg
        from scipy.signal import find_peaks

        y_arr = np.array(raw_y, dtype=float)
        win = min(self.sg_window, len(y_arr) - 1)
        if win % 2 == 0:
            win -= 1
        if win < 5:
            return []

        y_smooth = _sg(y_arr, win, self.sg_poly)
        # Y-down: concentric(up)=decreasing → velocity positive
        v = -_sg(y_arr, win, self.sg_poly, deriv=1) * fps * scale_m_per_px

        # ── Pass 4: Rep 提取 (bottom=maxY → top=minY) ──────────────────────
        rom_min = 0.012   # 12 cm
        dur_min = 0.25   # s
        dur_max = 4.5    # s

        bottoms, _ = find_peaks(y_smooth, distance=int(fps * dur_min))
        tops, _    = find_peaks(-y_smooth, distance=int(fps * dur_min))

        events = sorted([(b, "bottom") for b in bottoms] + [(t, "top") for t in tops])

        reps = []
        i = 0
        while i < len(events) - 1:
            f1, t1 = events[i]
            f2, t2 = events[i + 1]
            if t1 == "bottom" and t2 == "top":
                dur = (f2 - f1) / fps
                if dur_min <= dur <= dur_max:
                    dy = abs(y_smooth[f2] - y_smooth[f1])
                    rom = dy * scale_m_per_px
                    if rom >= rom_min:
                        mid = (f1 + f2) // 2
                        mid = max(0, min(mid, len(v) - 1))
                        mcv = float(np.clip(v[mid], 0.10, 2.50))
                        reps.append({
                            "mcv":       round(mcv, 3),
                            "pv":        round(float(np.mean(v[max(0, f1):f2 + 1])), 3),
                            "rom":       round(rom * 100, 1),   # cm
                            "duration":  round(dur, 2),
                            "frames":    (int(f1), int(f2)),
                        })
                i += 2
            else:
                i += 1

        # ── Debug ─────────────────────────────────────────────────────────────
        if self.debug:
            self._save_debug(video_path, y_smooth, v, reps, raw_y)

        return reps

    def _save_debug(self, video_path: str,
                    y_smooth: np.ndarray, v: np.ndarray,
                    reps: list, raw_y: list):
        import os
        vid = os.path.basename(video_path)
        out_dir = "validation/dataset_benchmark/anchor_track_debug"
        os.makedirs(out_dir, exist_ok=True)

        fig_h = max(200, len(raw_y) // 3)
        fig = np.ones((fig_h, 800, 3), dtype=np.uint8) * 255

        yn = 100 + (y_smooth - np.nanmin(y_smooth)) / (
            max(1e-6, np.nanmax(y_smooth) - np.nanmin(y_smooth))
        ) * (fig_h - 120)

        for xi in range(min(len(yn) - 1, 799)):
            cv2.line(fig, (xi, int(yn[xi])), (xi + 1, int(yn[xi + 1])),
                     (200, 200, 200), 1)

        for rep in reps:
            f1, f2 = rep["frames"]
            cv2.line(fig, (f1, 0), (f1, fig_h), (0, 255, 0), 1)
            cv2.line(fig, (f2, 0), (f2, fig_h), (255, 0, 0), 1)
            cx = (f1 + f2) // 2
            cy = int(np.interp(cx, np.arange(len(yn)), yn))
            cv2.putText(fig, f'{rep["mcv"]}',
                        (cx, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imwrite(os.path.join(out_dir, vid.replace(".mp4", ".png")), fig)
