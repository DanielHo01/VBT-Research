"""
algorithms/common.py
Core detection, subpixel refinement, and rep-segmentation primitives.
适配 plate_v1.onnx（YOLOv8n 单类 plate 检测器）。
ONNX 输出格式 [1, 5, 8400] → [cx, cy, w, h, conf]（单类，无 cls 维度）
"""
import cv2
import numpy as np
import onnxruntime as ort


# ── 检测器 ───────────────────────────────────────────────────

class YoloPlateDetector:
    """
    YOLOv8n plate 检测器（适配 plate_v1.onnx → [1, 5, 8400]）。
    输出格式：[cx, cy, w, h, conf]（5 channels，无 cls）
    """

    def __init__(self, model_path: str = r"D:\EasyVBT-Research\models\barbell_v4.onnx.backup"):
        providers = ['CPUExecutionProvider']
        try:
            available = [p for p in ort.get_available_providers()]
            for p in ['CUDAExecutionProvider', 'CPUExecutionProvider']:
                if p in available:
                    providers = [p] + [x for x in available if x != p]
                    break
        except Exception:
            pass

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        # 从 ONNX 输入 shape 推断输入尺寸（YOLOv8n=640, 旧barbell=416）
        input_shape = self.session.get_inputs()[0].shape  # e.g. [1,3,640,640] or [1,3,416,416]
        self.input_size = int(input_shape[2])

    def detect(self, frame: np.ndarray) -> dict | None:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]

        outputs = self.session.run(None, {self.input_name: img})[0]  # [1, 5, 8400]
        # YOLOv8 输出: [cx, cy, w, h, conf]
        row = outputs[0]  # [5, 8400]

        # Sigmoid on confidence
        confs = 1.0 / (1.0 + np.exp(-row[4].astype(float)))
        best_idx = int(np.argmax(confs))
        best_conf = float(confs[best_idx])

        if best_conf < 0.25:
            return None

        sx = w / self.input_size
        sy = h / self.input_size

        return {
            'cx':    float(row[0][best_idx] * sx),
            'cy':    float(row[1][best_idx] * sy),
            'w':     float(row[2][best_idx] * sx),
            'h':     float(row[3][best_idx] * sy),
            'score': best_conf,
        }


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
                    scale_factor: float = 1.15) -> tuple[float, float]:
    median_h = float(np.median(plate_heights_px))
    scale = (plate_diameter_m * scale_factor) / median_h
    return scale, median_h
