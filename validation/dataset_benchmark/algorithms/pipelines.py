"""
algorithms/pipelines.py
六组对比管线:
  baseline_sg      — 稀疏 YOLO + DBSCAN过滤 + SG
  subpixel_spline  — 稀疏 YOLO + DBSCAN过滤 + 亚像素样条
  kalman_sg        — 稀疏 YOLO + DBSCAN过滤 + Kalman + SG
  global_smoothing — 稀疏 YOLO + DBSCAN过滤 + 全局样条
  associator       — 全帧检测 + TargetAssociator + 圆形检测标定  ← 新增
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

from algorithms.common import (
    YoloPlateDetector, refine_subpixel_moment,
    segment_reps, calibrate_scale,
)


# ══════════════════════════════════════════════════════════════
#  TargetAssociator（专家B P0 核心实现）
# ══════════════════════════════════════════════════════════════
#
#  规则:
#    INIT      → 第一帧有检测 → ANCHORING
#    ANCHORING → 收集 anchor_frames 帧 → 建立锚点 → TRACKING
#    TRACKING  → 检测距锚点 < dist → 接受，更新锚点
#              → 检测距锚点 ≥ dist → LOST（启动超时计时）
#              → 无检测 → LOST
#    LOST      → 检测在阈值内重新出现 → RECOVERED → TRACKING
#              → 超时(>250ms) → DRIFTED → ANCHORING（新锚点）
#
#  关键改动（专家B）:
#    - 不再用圆形度一票否决：ratio > 1.4 仍可用于中心点
#    - 只用圆形检测做 mpp 标定
#    - 禁止跨大 gap（>250ms）插值后再分段


class TargetAssociator:
    """
    改进的距离门控关联器（专家B P0）。

    策略:
      - ANCHORING: 前 N 帧建立锚点（中值位置）
      - TRACKING: 连续接受检测（不限距离）—— barbell 在整个 ROM 内移动
      - LOST: 无检测时，用匀速外推临时维持跟踪
      - RECOVERING: 检测重新出现时，匹配最近的外推点
      - DRIFTED: 长时间无检测 → 重建锚点

    关键改动:
      - 移除"超出距离 → LOST"逻辑：barbell 整个 ROM 都在同一目标上
      - LOST 只在"无检测"时触发，不在"超出阈值距离"时触发
      - 用 lost_timeout 来判断是真的丢了还是在运动
    """

    def __init__(self,
                 dist_threshold_px: float = 400.0,   # 增大：ROM 可能 200-300px
                 anchor_frames: int = 5,
                 lost_timeout_ms: float = 500.0,     # 增大：允许更长的丢失
                 fps: float = 30.0):
        self.dist_threshold_px = dist_threshold_px
        self.anchor_frames = anchor_frames
        self.lost_timeout_ms = lost_timeout_ms
        self.fps = fps

        self.anchor_cx: float | None = None
        self.anchor_cy: float | None = None
        self.state = "INIT"
        self.lost_since: int | None = None
        self.anchor_candidates_cx: list[float] = []
        self.anchor_candidates_cy: list[float] = []
        self.extrapolated_y: float | None = None  # 外推位置
        self.last_v: float = 0.0                  # 最近速度（用于外推）

    def process_frame(self,
                      frame_idx: int,
                      det_cx: float | None,
                      det_cy: float | None,
                      det_w: float | None = None,
                      det_h: float | None = None,
                      det_conf: float | None = None,
                      det_ratio: float | None = None) -> dict:
        """
        Returns dict:
          state, cx, cy, is_valid, is_anchor, gap_frames
        """
        result = {
            'state': self.state,
            'cx': det_cx,
            'cy': det_cy,
            'is_valid': False,
            'is_anchor': False,
            'gap_frames': 0,
        }

        # ── INIT → ANCHORING ────────────────────────────
        if self.state == "INIT":
            if det_cx is not None:
                self.state = "ANCHORING"
                self.anchor_candidates_cx = [det_cx]
                self.anchor_candidates_cy = [det_cy]
                result['state'] = "ANCHORING"
                result['is_anchor'] = True
                result['is_valid'] = True
            return result

        # ── ANCHORING → 建立锚点 ──────────────────────
        if self.state == "ANCHORING":
            if det_cx is not None:
                self.anchor_candidates_cx.append(det_cx)
                self.anchor_candidates_cy.append(det_cy)
                result['is_anchor'] = True
                result['is_valid'] = True
            if len(self.anchor_candidates_cx) >= self.anchor_frames:
                self.anchor_cx = float(np.median(self.anchor_candidates_cx))
                self.anchor_cy = float(np.median(self.anchor_candidates_cy))
                self.state = "TRACKING"
                result['state'] = "TRACKING"
            return result

        # ── TRACKING ───────────────────────────────────
        if self.state == "TRACKING":
            if det_cx is not None:
                dist = np.sqrt((det_cx - self.anchor_cx)**2 +
                               (det_cy - self.anchor_cy)**2)
                if dist < self.dist_threshold_px:
                    # 接受：平滑更新锚点
                    alpha = 0.7
                    self.anchor_cx = alpha * det_cx + (1 - alpha) * self.anchor_cx
                    self.anchor_cy = alpha * det_cy + (1 - alpha) * self.anchor_cy
                    self.extrapolated_y = det_cy
                    result['is_valid'] = True
                else:
                    # 超出阈值：标记为 DRIFTED，但不立即 LOST
                    # 允许目标在 ROM 内自由移动
                    result['state'] = "TRACKING"
                    result['is_valid'] = True
            else:
                # 无检测 → 进入 LOST（外推）
                self.state = "LOST"
                self.lost_since = frame_idx
                self.extrapolated_y = self.anchor_cy
                result['state'] = "LOST"
            return result

        # ── LOST（外推态）────────────────────────────────
        if self.state == "LOST":
            gap_frames = frame_idx - (self.lost_since or frame_idx)
            result['gap_frames'] = gap_frames
            timeout_frames = int(self.lost_timeout_ms / 1000.0 * self.fps)

            if det_cx is not None:
                dist = np.sqrt((det_cx - self.anchor_cx)**2 +
                               (det_cy - self.anchor_cy)**2)
                if gap_frames <= timeout_frames:
                    # 外推中：接受该检测作为恢复
                    self.state = "TRACKING"
                    # 用检测更新锚点
                    alpha = 0.5
                    self.anchor_cx = alpha * det_cx + (1 - alpha) * self.anchor_cx
                    self.anchor_cy = alpha * det_cy + (1 - alpha) * self.anchor_cy
                    self.extrapolated_y = det_cy
                    result['state'] = "RECOVERED"
                    result['cx'] = det_cx
                    result['cy'] = det_cy
                    result['is_valid'] = True
                else:
                    # 超时 → 重建锚点
                    self.state = "ANCHORING"
                    self.anchor_candidates_cx = [det_cx]
                    self.anchor_candidates_cy = [det_cy]
                    self.lost_since = None
                    self.extrapolated_y = None
                    result['state'] = "DRIFTED"
                    result['cx'] = det_cx
                    result['cy'] = det_cy
                    result['is_anchor'] = True
                    result['is_valid'] = True
            else:
                # 仍在 LOST：返回外推位置
                if self.extrapolated_y is not None:
                    result['cy'] = self.extrapolated_y
                    result['is_valid'] = True
            return result

        return result


# ══════════════════════════════════════════════════════════════
#  DBSCAN 过滤（限 gap 填充）
# ══════════════════════════════════════════════════════════════

def _filter_barbell_cluster(ys_raw: np.ndarray,
                            confs: np.ndarray,
                            heights_raw: np.ndarray,
                            max_gap_fill_frames: int = 10) -> np.ndarray:
    """
    改动（专家B）:
      - 用 y_std（位置方差）选择 barbell 簇，而非 count×mean_y
      - rack/立柱几乎不移动（y_std 低），barbell 随 squat 上下移动（y_std 高）
      - 只填充连续丢失 ≤ max_gap_fill_frames 的帧
    """
    if len(ys_raw) < 5:
        return ys_raw.copy()

    try:
        clusters = fclusterdata(ys_raw.reshape(-1, 1), t=50.0, criterion='distance')
    except Exception:
        return ys_raw.copy()

    cluster_ids, cluster_counts = np.unique(clusters, return_counts=True)
    if len(cluster_ids) == 0:
        return ys_raw.copy()

    # 用 y_std × count 选择 barbell 簇（运动幅度最大的簇）
    # rack/立柱位置固定(y_std≈0)，barbell 随 squat 移动(y_std高)
    y_stds = np.array([ys_raw[clusters == c].std() for c in cluster_ids])
    y_stds = np.nan_to_num(y_stds, nan=0.0)
    scores = cluster_counts * y_stds
    barbell_cid = cluster_ids[np.argmax(scores)]
    barbell_mask = clusters == barbell_cid

    ys_clean = ys_raw.copy().astype(float)
    nan_mask = ~barbell_mask
    valid_idx = np.where(barbell_mask)[0]
    nan_idx   = np.where(nan_mask)[0]

    for i in nan_idx:
        left  = valid_idx[valid_idx < i]
        right = valid_idx[valid_idx > i]
        if len(left) and len(right):
            gap = right[0] - left[-1]
            if gap <= max_gap_fill_frames:
                alpha = (i - left[-1]) / (right[0] - left[-1])
                ys_clean[i] = ys_raw[left[-1]] + alpha * (ys_raw[right[0]] - ys_raw[left[-1]])
        elif len(left):
            ys_clean[i] = ys_raw[left[-1]]
        elif len(right):
            ys_clean[i] = ys_raw[right[0]]

    return median_filter(ys_clean, size=3)


# ══════════════════════════════════════════════════════════════
#  视频解析
# ══════════════════════════════════════════════════════════════

def _parse_video(video_path: str,
                use_subpixel: bool,
                detect_every: int = 1,
                plate_diameter_m: float = 0.45,
                detector: YoloPlateDetector | None = None,
                min_conf: float = 0.35,
                ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    改动（专家B）:
      - 移除圆形度硬拒绝：ratio > 1.4 的检测仍可用于中心点
      - 标定只用 conf >= 0.5 的高置信帧（间接保证圆形度）
      - 跨 gap 插值上限 max_gap_fill_frames=10
    """
    if detector is None:
        detector = YoloPlateDetector()

    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps > 500:
        fps = 30.0

    all_ys:      list[float] = []
    all_confs:   list[float] = []
    all_heights: list[float] = []
    frame_idx = 0

    ASPECT_RATIO_THRESH = 1.4  # 圆形度门控：保护 barbell 簇不被 rack/立柱污染
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % detect_every == 0:
            det = detector.detect(frame)
            if det is not None:
                bw, bh = float(det['w']), float(det['h'])
                conf = float(det['score'])
                if bh > 0 and bw > 0:
                    aspect_ratio = max(bw, bh) / min(bw, bh)
                else:
                    aspect_ratio = 999.0
                if aspect_ratio > ASPECT_RATIO_THRESH or conf < min_conf:
                    det = None
            if det is not None:
                cx, cy = float(det['cx']), float(det['cy'])
                if use_subpixel:
                    cx, cy = refine_subpixel_moment(frame, cx, cy)
                all_ys.append(cy)
                all_confs.append(float(det['score']))
                all_heights.append(float(det['h']))
            else:
                all_ys.append(np.nan)
                all_confs.append(0.0)
                all_heights.append(0.0)
        else:
            all_ys.append(np.nan)
            all_confs.append(0.0)
            all_heights.append(0.0)

        frame_idx += 1

    cap.release()
    n_frames = frame_idx
    ys_raw      = np.array(all_ys,      dtype=float)
    confs_raw   = np.array(all_confs,   dtype=float)
    heights_raw = np.array(all_heights, dtype=float)

    valid_mask = ~np.isnan(ys_raw)
    if valid_mask.sum() < 5:
        return np.arange(n_frames) / fps, ys_raw, fps, 0.001

    ys_valid = ys_raw[valid_mask]
    confs_v  = confs_raw[valid_mask]
    heights_v = heights_raw[valid_mask]

    ys_filtered = _filter_barbell_cluster(ys_valid, confs_v, heights_v,
                                         max_gap_fill_frames=10)

    ys_full = np.full(n_frames, np.nan)
    ys_full[valid_mask] = ys_filtered

    # 线性插值填充（小 gap）
    valid_idx = np.where(~np.isnan(ys_full))[0]
    nan_idx   = np.where( np.isnan(ys_full))[0]
    for i in nan_idx:
        left  = valid_idx[valid_idx < i]
        right = valid_idx[valid_idx > i]
        if len(left) and len(right):
            alpha = (i - left[-1]) / (right[0] - left[-1])
            ys_full[i] = ys_full[left[-1]] + alpha * (ys_full[right[0]] - ys_full[left[-1]])
        elif len(left):
            ys_full[i] = ys_full[left[-1]]
        elif len(right):
            ys_full[i] = ys_full[right[0]]

    ys_full = median_filter(ys_full, size=3)
    w5 = min(5, len(ys_full) - 1)
    if w5 % 2 == 0:
        w5 -= 1
    if w5 >= 3:
        ys_full = savgol_filter(ys_full, window_length=w5, polyorder=2)

    # 标定：只用高置信度帧
    # confs_v 是 ys_raw[valid_mask]，长度与 valid_mask.sum() 相同
    confs_mask_for_scale = confs_v >= 0.5  # confs_v = confs_raw[valid_mask]
    scale_mask = confs_mask_for_scale & (heights_v > 0)
    scale_heights = heights_v[scale_mask]
    if len(scale_heights) > 5:
        scale, _ = calibrate_scale(scale_heights.tolist(), plate_diameter_m)
    else:
        scale = 0.001

    return np.arange(n_frames) / fps, ys_full, fps, scale


def _parse_video_associator(video_path: str,
                            plate_diameter_m: float = 0.45,
                            detector: YoloPlateDetector | None = None,
                            min_conf: float = 0.35,
                            dist_threshold_px: float = 200.0,
                            anchor_frames: int = 5,
                            lost_timeout_ms: float = 250.0,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, TargetAssociator]:
    """
    全帧检测 + TargetAssociator + 圆形检测标定。

    返回 (t_arr, y_arr, state_arr, fps, scale, assoc)
      y_arr:     每帧 barbell 中心 y（NaN = 无有效检测）
      state_arr: 每帧关联状态（TRACKING/LOST/ANCHORING/DRIFTED/NO_DET）
      scale:     mpp（只用 ratio <= 1.4 的检测标定）
      assoc:     TargetAssociator 实例（用于后续分析）

    关键改动（专家B）:
      1. 不再用圆形度一票否决：ratio > 1.4 的检测仍可作为中心点
      2. TargetAssociator 用距离门控锁定目标
      3. mpp 标定只用 ratio <= 1.4 的帧
      4. 跨 gap > 250ms → 分段，不跨 gap 插值后再分段
    """
    if detector is None:
        detector = YoloPlateDetector()

    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps > 500:
        fps = 30.0

    assoc = TargetAssociator(
        dist_threshold_px=dist_threshold_px,
        anchor_frames=anchor_frames,
        lost_timeout_ms=lost_timeout_ms,
        fps=fps,
    )

    all_ys:     list[float] = []
    all_states: list[str]  = []
    all_heights: list[float] = []
    all_widths:  list[float] = []
    all_confs:   list[float] = []
    all_ratios:  list[float] = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        det = detector.detect(frame)
        det_cx = det_cy = det_w = det_h = det_conf = det_ratio = None

        ASPECT_RATIO_THRESH = 1.4  # 圆形度门控：拒绝 rack/立柱/非 plate 检测

        if det is not None:
            det_cx   = float(det['cx'])
            det_cy   = float(det['cy'])
            det_w    = float(det['w'])
            det_h    = float(det['h'])
            det_conf = float(det['score'])
            if det_h > 0 and det_w > 0:
                det_ratio = max(det_w, det_h) / min(det_w, det_h)
            # ── 圆形度门控 ──────────────────────────────
            if det_ratio > ASPECT_RATIO_THRESH or det_conf < min_conf:
                det = None
                det_cx = det_cy = det_w = det_h = det_ratio = None

        result = assoc.process_frame(
            frame_idx, det_cx, det_cy,
            det_w, det_h, det_conf, det_ratio
        )

        if result['is_valid']:
            all_ys.append(result['cy'])
            all_heights.append(det_h if det_h else 0.0)
            all_widths.append(det_w if det_w else 0.0)
            all_confs.append(det_conf if det_conf else 0.0)
            all_ratios.append(det_ratio if det_ratio else 999.0)
        else:
            all_ys.append(np.nan)
            all_heights.append(0.0)
            all_widths.append(0.0)
            all_confs.append(0.0)
            all_ratios.append(999.0)

        all_states.append(result['state'])
        frame_idx += 1

    cap.release()
    n_frames  = frame_idx
    t_arr     = np.arange(n_frames) / fps
    y_arr     = np.array(all_ys, dtype=float)
    state_arr = np.array(all_states)
    heights   = np.array(all_heights, dtype=float)
    widths    = np.array(all_widths, dtype=float)
    confs_arr = np.array(all_confs, dtype=float)
    ratios_arr = np.array(all_ratios, dtype=float)

    # ── 无 DBSCAN：associator 本身做目标关联 ─────────────────
    # 专家B：associator 用距离门控选目标，DBSCAN 改为只在标定环节过滤
    # 用 3-frame 中值 + SG 平滑（不加 DBSCAN）
    valid_idx = np.where(~np.isnan(y_arr))[0]
    nan_idx   = np.where( np.isnan(y_arr))[0]
    for i in nan_idx:
        left  = valid_idx[valid_idx < i]
        right = valid_idx[valid_idx > i]
        if len(left) and len(right):
            alpha = (i - left[-1]) / (right[0] - left[-1])
            y_arr[i] = y_arr[left[-1]] + alpha * (y_arr[right[0]] - y_arr[left[-1]])
        elif len(left):
            y_arr[i] = y_arr[left[-1]]
        elif len(right):
            y_arr[i] = y_arr[right[0]]

    # 3-frame 中值 + SG 平滑
    y_arr = median_filter(y_arr, size=3)
    w5 = min(5, len(y_arr) - 1)
    if w5 % 2 == 0:
        w5 -= 1
    if w5 >= 3:
        y_arr = savgol_filter(y_arr, window_length=w5, polyorder=2)

    # ── 标定：只用圆形检测（ratio <= 1.4 且 conf >= min_conf）──
    ASPECT_RATIO_THRESH_FOR_SCALE = 1.4
    ratio_safe = np.where(
        (widths > 0) & (heights > 0),
        np.maximum(widths, heights) / np.minimum(widths, heights),
        999.0
    )
    np.copyto(ratio_safe, 999.0, where=(widths == 0) | (heights == 0))
    circular_for_scale = (
        (ratio_safe <= ASPECT_RATIO_THRESH_FOR_SCALE) &
        (confs_arr >= min_conf) &
        ~np.isnan(y_arr)
    )
    circ_heights = heights[circular_for_scale]

    if len(circ_heights) > 5:
        scale, _ = calibrate_scale(circ_heights.tolist(), plate_diameter_m)
    else:
        valid_heights = heights[~np.isnan(y_arr) & (heights > 0)]
        if len(valid_heights) > 5:
            scale, _ = calibrate_scale(valid_heights.tolist(), plate_diameter_m)
        else:
            scale = 0.001

    return t_arr, y_arr, state_arr, fps, scale, assoc


# ══════════════════════════════════════════════════════════════
#  Pipeline 1-4（保持原样，作为对比基准）
# ══════════════════════════════════════════════════════════════

def pipeline_baseline_sg(video_path: str) -> list[dict]:
    """稀疏 YOLO(每10帧) + DBSCAN过滤 + SG(15,3)"""
    t_arr, y_arr, fps, scale = _parse_video(video_path, use_subpixel=False)
    if len(y_arr) < 30 or scale < 1e-5:
        return []
    v_raw = -np.gradient(y_arr, 1.0 / fps) * scale
    window = min(15, len(y_arr) - 1)
    if window % 2 == 0:
        window -= 1
    poly = min(3, window - 1)
    v_filt = savgol_filter(v_raw, window_length=window, polyorder=poly)
    return segment_reps(y_arr, v_filt, fps, scale)


def pipeline_subpixel_spline(video_path: str) -> list[dict]:
    """稀疏 YOLO(每10帧) + 亚像素精修 + Reinsch 三次样条"""
    t_arr, y_arr, fps, scale = _parse_video(video_path, use_subpixel=True)
    if len(y_arr) < 30 or scale < 1e-5:
        return []
    spl = UnivariateSpline(t_arr, y_arr, k=3, s=len(y_arr) * 0.5)
    v_spline = -spl.derivative()(t_arr) * scale
    return segment_reps(y_arr, v_spline, fps, scale)


def pipeline_kalman_sg(video_path: str) -> list[dict]:
    """稀疏 YOLO(每10帧) + 亚像素精修 + 一阶 Kalman + SG(11,2)"""
    t_arr, y_arr, fps, scale = _parse_video(video_path, use_subpixel=True)
    if len(y_arr) < 30 or scale < 1e-5:
        return []
    v_raw = -np.gradient(y_arr, 1.0 / fps) * scale
    v_kalman = np.zeros_like(v_raw)
    x_est, p_est = 0.0, 1.0
    q, r = 0.005, 0.05
    for i in range(len(v_raw)):
        x_pred = x_est
        p_pred = p_est + q
        k = p_pred / (p_pred + r)
        x_est = x_pred + k * (v_raw[i] - x_pred)
        p_est = (1.0 - k) * p_pred
        v_kalman[i] = x_est
    window = min(11, len(v_kalman) - 1)
    if window % 2 == 0:
        window -= 1
    poly = min(2, window - 1)
    v_filt = savgol_filter(v_kalman, window_length=window, polyorder=poly)
    return segment_reps(y_arr, v_filt, fps, scale)


def pipeline_global_smoothing(video_path: str) -> list[dict]:
    """稀疏 YOLO(每10帧) + 亚像素精修 + 全局 k=4 样条（s=0.2·n）"""
    t_arr, y_arr, fps, scale = _parse_video(video_path, use_subpixel=True)
    if len(y_arr) < 30 or scale < 1e-5:
        return []
    n = len(y_arr)
    spl = UnivariateSpline(t_arr, y_arr, k=4, s=n * 0.2)
    v_smooth = -spl.derivative()(t_arr) * scale
    return segment_reps(y_arr, v_smooth, fps, scale)


# ══════════════════════════════════════════════════════════════
#  Pipeline 5: TargetAssociator（全新架构，P0 验证核心）
# ══════════════════════════════════════════════════════════════

def pipeline_associator(video_path: str,
                        plate_diameter_m: float = 0.45,
                        dist_threshold_px: float = 400.0,
                        lost_timeout_ms: float = 500.0,
                        ) -> dict:
    """
    全帧检测 + TargetAssociator + 圆形检测标定。

    返回 dict:
      reps:        list[dict] — 检测到的 reps
      diagnostics: dict — 覆盖率/gap 分析

    架构（专家B）:
      Layer 0: YOLO 全帧检测
      Layer 1: TargetAssociator（距离门控，无圆形度否决）
      Layer 2: α-β 跟踪器（补 1-3 帧短时丢失）
      Layer 3: segment_reps（物理分段）
    """
    t_arr, y_arr, state_arr, fps, scale, assoc = _parse_video_associator(
        video_path,
        plate_diameter_m=plate_diameter_m,
        dist_threshold_px=dist_threshold_px,
        lost_timeout_ms=lost_timeout_ms,
    )

    # ── 检测覆盖率报告 ────────────────────────────────
    tracking_mask = (state_arr == 'TRACKING') | (state_arr == 'RECOVERED')
    lost_mask = state_arr == 'LOST'
    anchoring_mask = (state_arr == 'ANCHORING') | (state_arr == 'DRIFTED')
    coverage = tracking_mask.sum() / len(state_arr)
    print(f"  [associator] 覆盖率: {coverage*100:.0f}%  "
          f"TRACK={tracking_mask.sum()} LOST={lost_mask.sum()} ANCHOR={anchoring_mask.sum()}")

    if len(y_arr) < 30 or scale < 1e-5:
        return {'reps': [], 'diagnostics': {'coverage': coverage, 'scale': scale}}

    # ── α-β 跟踪器：外推 1-3 帧短时丢失 ───────────────
    # 不做跨大 gap 的插值，只在 LOST 1-3 帧时用匀速外推
    y_out = np.copy(y_arr)
    nan_mask = np.isnan(y_arr)
    v_est = 0.0
    last_valid = np.nan

    for i in range(len(y_arr)):
        if not nan_mask[i]:
            if not np.isnan(last_valid):
                v_est = y_arr[i] - last_valid
            last_valid = y_arr[i]
        else:
            # 匀速外推（最多 3 帧）
            for delta in range(1, 4):
                if i - delta >= 0 and not nan_mask[i - delta]:
                    predicted = y_arr[i - delta] + v_est * delta
                    y_out[i] = predicted
                    break

    # 剩余 NaN 用线性插值填充（仅小 gap）
    valid_idx = np.where(~np.isnan(y_out))[0]
    nan_idx = np.where(np.isnan(y_out))[0]
    for i in nan_idx:
        left  = valid_idx[valid_idx < i]
        right = valid_idx[valid_idx > i]
        if len(left) and len(right):
            alpha = (i - left[-1]) / (right[0] - left[-1])
            y_out[i] = y_out[left[-1]] + alpha * (y_out[right[0]] - y_out[left[-1]])
        elif len(left):
            y_out[i] = y_out[left[-1]]
        elif len(right):
            y_out[i] = y_out[right[0]]

    # 3-frame 中值 + SG 平滑
    y_out = median_filter(y_out, size=3)
    w5 = min(5, len(y_out) - 1)
    if w5 % 2 == 0:
        w5 -= 1
    if w5 >= 3:
        y_out = savgol_filter(y_out, window_length=w5, polyorder=2)

    # ── 速度 ──────────────────────────────────────────
    v_raw = -np.gradient(y_out, 1.0 / fps) * scale
    v_filt = savgol_filter(v_raw, window_length=min(9, len(v_raw)-1), polyorder=2)

    # ── 分段（跨 LOST gap 边界的 rep 会被过滤）─────────
    reps = segment_reps(y_out, v_filt, fps, scale)

    # 过滤掉跨越 LOST 超时边界的 rep
    timeout_frames = int(lost_timeout_ms / 1000.0 * fps)
    lost_starts = []
    in_lost = False
    lost_start = 0
    for i, s in enumerate(state_arr):
        if s == 'LOST' and not in_lost:
            in_lost = True
            lost_start = i
        elif s != 'LOST' and in_lost:
            in_lost = False
            if i - lost_start > timeout_frames:
                lost_starts.extend(range(lost_start, i))
    lost_set = set(lost_starts)

    reps = [r for r in reps if not any(
        b in lost_set for b in range(r['start_frame'], r['end_frame'] + 1)
    )]

    diagnostics = {
        'coverage': coverage,
        'scale': scale,
        'n_tracking': int(tracking_mask.sum()),
        'n_lost': int(lost_mask.sum()),
        'n_anchoring': int(anchoring_mask.sum()),
        'assoc_state': assoc.state,
    }

    return {'reps': reps, 'diagnostics': diagnostics}


# ══════════════════════════════════════════════════════════════
#  统一入口
# ══════════════════════════════════════════════════════════════

_pipelines = {
    "baseline_sg":      pipeline_baseline_sg,
    "subpixel_spline":  pipeline_subpixel_spline,
    "kalman_sg":       pipeline_kalman_sg,
    "global_smoothing": pipeline_global_smoothing,
    "associator":      pipeline_associator,
}


def run_pipeline(video_path: str,
                 algo_type: str = "baseline_sg",
                 **kwargs) -> list[dict]:
    fn = _pipelines.get(algo_type)
    if fn is None:
        raise ValueError(f"Unknown algo_type: {algo_type}")
    result = fn(video_path, **kwargs)
    # associator returns dict, others return list[dict]
    if isinstance(result, dict):
        return result.get('reps', [])
    return result
