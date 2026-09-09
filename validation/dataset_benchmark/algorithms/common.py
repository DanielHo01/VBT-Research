"""
algorithms/common.py
Core detection, subpixel refinement, and rep-segmentation primitives.

检测器适配多型号 YOLO ONNX 导出：
  - 5 通道: [cx, cy, w, h, conf]            （plate_v1 / yolo11_plate）
  - 6 通道: [cx, cy, w, h, conf, ...]       （barbell_v4）
坐标单位为输入分辨率像素，输出时缩放回原图。

重要修复（2026-09）：ONNX 导出的 conf 通道已经是 [0,1] 概率，
旧代码又做了一次 sigmoid，导致空白帧也被当成 ~0.50 置信度的检测，
这是旧基准 RMSE=0.88 异常偏大的主因之一。
"""
import cv2
import numpy as np
import onnxruntime as ort

try:  # 作为包被导入
    from .. import config as _cfg
except ImportError:  # 直接以脚本目录运行
    import config as _cfg


# ── 检测器 ───────────────────────────────────────────────────

class YoloPlateDetector:
    """
    YOLO plate 检测器（5ch/6ch 通用）。

    输出格式：[cx, cy, w, h, conf]（5 channels）或
              [cx, cy, w, h, conf, cls]（6 channels）。
    conf 通道若已是 [0,1] 概率则直接使用，否则应用 sigmoid。
    """

    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = _cfg.default_model_path()
        self.model_path = str(model_path)

        providers = ['CPUExecutionProvider']
        try:
            available = [p for p in ort.get_available_providers()]
            for p in ['CUDAExecutionProvider', 'CPUExecutionProvider']:
                if p in available:
                    providers = [p] + [x for x in available if x != p]
                    break
        except Exception:
            pass

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        # 从 ONNX 输入 shape 推断输入尺寸（YOLOv8n=640, 旧barbell=416）
        input_shape = self.session.get_inputs()[0].shape  # e.g. [1,3,640,640] or [1,3,416,416]
        self.input_size = int(input_shape[2])

    # ── 推理 ──────────────────────────────────────────────

    def _run(self, frame: np.ndarray) -> np.ndarray:
        """返回 [C, N] 的原始输出（C=5 或 6）。"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]

        outputs = self.session.run(None, {self.input_name: img})[0]  # [1, C, N]
        row = outputs[0]
        if row.shape[0] < 5:
            raise RuntimeError(
                f"不支持的 ONNX 输出通道数 {row.shape[0]}（{self.model_path}）"
            )
        return row

    @staticmethod
    def _confidences(row: np.ndarray) -> np.ndarray:
        """conf 通道转概率：已在 [0,1] 则直接用，否则用 sigmoid。"""
        raw = row[4].astype(np.float64)
        if 0.0 <= raw.min() and raw.max() <= 1.0:
            return raw
        return 1.0 / (1.0 + np.exp(-raw))

    def detect_all(self, frame: np.ndarray,
                   conf_threshold: float = 0.25,
                   top_k: int = 50,
                   nms_iou: float = 0.45) -> list[dict]:
        """
        返回按置信度降序的检测列表（含简单 NMS 去重）。
        每个元素: {cx, cy, w, h, score}（原图像素坐标）。
        """
        h, w = frame.shape[:2]
        row = self._run(frame)
        confs = self._confidences(row)

        sx = w / self.input_size
        sy = h / self.input_size

        # 大于阈值者按置信度降序
        keep = np.where(confs >= conf_threshold)[0]
        if keep.size == 0:
            return []
        order = keep[np.argsort(confs[keep])[::-1]]

        boxes = []
        for i in order:
            boxes.append({
                'cx': float(row[0][i] * sx),
                'cy': float(row[1][i] * sy),
                'w':  float(row[2][i] * sx),
                'h':  float(row[3][i] * sy),
                'score': float(confs[i]),
            })

        # 简单 NMS（按面积 IoU）
        keep_boxes: list[dict] = []
        for b in boxes:
            x1, y1 = b['cx'] - b['w'] / 2, b['cy'] - b['h'] / 2
            x2, y2 = b['cx'] + b['w'] / 2, b['cy'] + b['h'] / 2
            area_b = max(b['w'], 0.0) * max(b['h'], 0.0)
            suppressed = False
            for k in keep_boxes:
                kx1, ky1 = k['cx'] - k['w'] / 2, k['cy'] - k['h'] / 2
                kx2, ky2 = k['cx'] + k['w'] / 2, k['cy'] + k['h'] / 2
                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                area_k = max(k['w'], 0.0) * max(k['h'], 0.0)
                union = area_b + area_k - inter
                if union > 0 and inter / union > nms_iou:
                    suppressed = True
                    break
            if not suppressed:
                keep_boxes.append(b)
            if len(keep_boxes) >= top_k:
                break
        return keep_boxes

    def detect(self, frame: np.ndarray,
               conf_threshold: float = 0.25) -> dict | None:
        """返回最高置信度检测（兼容旧接口）；无有效检测返回 None。"""
        dets = self.detect_all(frame, conf_threshold=conf_threshold, top_k=1)
        return dets[0] if dets else None


# ── 亚像素精修 ──────────────────────────────────────────────

def refine_subpixel_moment(frame: np.ndarray, cx: float, cy: float,
                          roi_radius: int = 40) -> tuple[float, float]:
    """基于 OpenCV 图像矩的亚像素重心精修。"""
    h, w = frame.shape[:2]
    x1 = max(0, int(cx - roi_radius))
    y1 = max(0, int(cy - roi_radius))
    x2 = min(w, int(cx + roi_radius))
    y2 = min(h, int(cy + roi_radius))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return cx, cy
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    M = cv2.moments(thresh)
    if M['m00'] > 0:
        return float(x1 + M['m10'] / M['m00']), float(y1 + M['m01'] / M['m00'])
    return cx, cy


# ── Rep 分段 ────────────────────────────────────────────────

def segment_reps(y_pos_px: np.ndarray, v_smooth: np.ndarray,
                 fps: float, scale_m_px: float,
                 min_gap_frames: int | None = None,
                 x_pos_px: np.ndarray | None = None,
                 h_pos_px: np.ndarray | None = None,
                 vx_smooth: np.ndarray | None = None,
                 vh_smooth: np.ndarray | None = None) -> list[dict]:
    """
    Peak-First 分段，支持 Y / X / Height 三信号自适应：

    策略：
      1. 选 Y/X/Height 中变化最大的信号
      2. find_peaks(sig, distance=fps*0.5, prominence=15)
      3. 每 peak 向后找首个深层 trough
      4. 计算 duration、ROM、MCV

    返回 list[dict]: {start_frame, end_frame, mcv, pv, rom_cm, duration_s}
    """
    from scipy.signal import find_peaks

    n = len(y_pos_px)
    if n < 30:
        return []

    min_gap = min_gap_frames or max(int(fps * 0.5), 20)


    # ── Step 1: 选最佳信号 ────────────────────────────────
    y_range = float(np.ptp(y_pos_px))
    x_range = float(np.ptp(x_pos_px)) if x_pos_px is not None else 0.0
    h_range = float(np.ptp(h_pos_px)) if h_pos_px is not None else 0.0

    if   y_range >= x_range and y_range >= h_range:
        sig = y_pos_px;  vel = v_smooth
    elif x_range >= h_range:
        sig = x_pos_px;  vel = vx_smooth if vx_smooth is not None else v_smooth
    else:
        sig = h_pos_px;  vel = vh_smooth if vh_smooth is not None else v_smooth

    # ── Step 2: peaks ──────────────────────────────────
    peaks, _ = find_peaks(sig, distance=min_gap, prominence=12.0)
    if len(peaks) < 1:
        return []

    # ── Step 3: troughs（用于 deep-filter）──────────────
    raw_troughs, _ = find_peaks(-sig, distance=max(min_gap // 2, 10))

    # ── Step 4: Peak → Trough 配对 ───────────────────
    y_60pct = float(np.percentile(sig, 60))
    reps = []
    for p in peaks:
        candidates = [t for t in raw_troughs
                     if t > p + int(fps * 0.15) and sig[t] < y_60pct]
        if not candidates:
            window_end = min(p + max(int(fps * 3.0), 60), n - 1)
            t = int(np.argmin(sig[p:window_end + 1])) + p
        else:
            t = candidates[0]

        duration = (t - p) / fps
        rom_px  = abs(sig[t] - sig[p])
        rom_cm  = rom_px * scale_m_px * 100.0

        if duration < 0.2 or duration > 7.5 or rom_cm < 8.0:
            continue

        cv_slice = vel[p:t]
        pos_v    = cv_slice[cv_slice > 0]
        mcv = float(np.mean(pos_v)) if len(pos_v) > 0 else 0.0
        pv  = float(np.max(cv_slice))          if len(cv_slice) > 0 else 0.0

        reps.append({
            'start_frame': int(p),
            'end_frame':   int(t),
            'mcv':         mcv,
            'pv':          pv,
            'rom_cm':      rom_cm,
            'duration_s':  duration,
        })

    return reps


# ── 标定 ────────────────────────────────────────────────────

def calibrate_scale(plate_heights_px: list[float],
                    plate_diameter_m: float = 0.45,
                    scale_factor: float = 1.0) -> tuple[float, float]:
    """
    由 plate 直径（已知物理尺寸）推算 米/像素。

    参数
    ----
    plate_diameter_m : 标准杠铃片直径（IWF 450mm=0.45m，IPF 431.8mm=0.4318m）。
    scale_factor     : 经验修正系数，默认 1.0。
                       旧代码默认 1.15 会给所有速度引入 +15% 系统偏差，
                       已移除；如需对比旧结果可显式传 1.15。

    返回 (scale_m_per_px, median_height_px)
    """
    median_h = float(np.median(plate_heights_px))
    if median_h <= 0:
        return 0.0, median_h
    scale = (plate_diameter_m * scale_factor) / median_h
    return scale, median_h
