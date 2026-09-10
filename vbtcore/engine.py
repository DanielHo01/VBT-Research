"""
vbtcore.engine — 锚定 + 「检测→拟合」混合跟踪 + 标定
==================================================
实测依据（scripts/exp_detect_then_track.py，2026-09-10）：
  - 每帧全图 YOLO：~96ms/帧（CPU）——移动端不可行
  - 每 15 帧重检测 + NCC 模板拟合 + 匀速预测：10-16ms/帧，r=0.956（30kg）
  - 纯拟合（只锚定一次）：漂移不可控，rep 检出 1/21 —— 必须低频重检测

锚定身份教训（实验实锤）：
  80kg 视频检测器最高置信度目标是架下存放片堆（h≈180px 静态）；
  130kg 视频工作片漏检、检出的全是背景片堆。
  → 最高置信 ≠ 工作杠铃片，必须加物理先验 + 运动探针 + 用户点选兜底。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import Detection, PlateDetector


# ══════════════════════════════════════════════════════════
#  锚定：物理先验打分
# ══════════════════════════════════════════════════════════

def anchor_score(d: Detection, frame_h: int, frame_w: int,
                 margin: float = 0.03,
                 h_range: tuple[float, float] = (0.03, 0.15),
                 max_ratio: float = 1.6) -> float:
    """
    锚定候选打分；不合法返回 0。
    规则（每条都有实验依据）：
      - 贴边/出画拒绝：杠铃片不会半截出画（130kg 背景片堆贴左缘）
      - 圆度 ≤1.6：片堆侧视呈扁矩形
      - 尺寸域：工作片表观高度约在帧高 3%~15%
        （注意：单一尺寸阈值无法区分 80kg 架下片堆 0.141 与
        30kg 近距工作片 0.137 —— 二者都在域内，最终靠运动探针区分）
      - 居中加权：构图上工作片接近画面中心
    """
    m = margin
    if not (frame_w * m < d.cx < frame_w * (1 - m)):
        return 0.0
    if not (frame_h * m < d.cy < frame_h * (1 - m)):
        return 0.0
    if d.ratio > max_ratio:
        return 0.0
    hr = d.h / frame_h
    if not (h_range[0] <= hr <= h_range[1]):
        return 0.0
    cxn = abs(d.cx - frame_w / 2) / (frame_w / 2)
    cyn = abs(d.cy - frame_h / 2) / (frame_h / 2)
    cent = max(0.0, 1.0 - (cxn + cyn) / 2)
    return d.conf * (0.4 + 0.6 * cent)


# ══════════════════════════════════════════════════════════
#  运动探针：区分「工作片」与「静态片堆」
# ══════════════════════════════════════════════════════════

@dataclass
class MotionProbe:
    """对锚定候选簇做短窗 y 方差统计：工作片必动，片堆不动。"""
    window_frames: int = 45          # 探针窗（~1.5s）
    min_candidates: int = 2          # 候选 ≥2 时才启用（单一候选没必要）

    def select(self, frames_cx: dict[int, list[float]],
               frames_cy: dict[int, list[float]],
               cand_keys: list[tuple[int, int]],
               fps: float) -> tuple[int, int]:
        """
        cand_keys: 每个候选的 (量化cx, 量化cy) 标识。
        返回按 (y方差×覆盖) 选出的最优候选 key；无统计数据时返回首个。
        """
        if len(cand_keys) == 1 or self.window_frames <= 0:
            return cand_keys[0]
        best_key, best_score = cand_keys[0], -1.0
        for key in cand_keys:
            ys: list[float] = []
            for f in sorted(frames_cy.keys())[: self.window_frames]:
                for cx_q, cy in zip(frames_cx[f], frames_cy[f]):
                    if (round(cx_q), round(cy)) == key:
                        ys.append(cy)
            if len(ys) >= 5:
                score = float(np.std(ys)) * min(1.0, len(ys) / 20.0)
                if score > best_score:
                    best_score, best_key = score, key
        return best_key


def select_anchor_track(pend: list[dict], min_hits: int = 5) -> dict | None:
    """
    从 pending 轨迹中选出锚定目标（运动探针）：
      1. 只考虑 hits≥min_hits 的轨迹（确认存在感）
      2. y 方差最大者优先 —— 工作片必动，存放片堆静止
         （80kg 实测：架下片堆 conf 0.86 静态 vs 工作片 conf 0.5 运动）
      3. 方差同分时取累计置信度高者
    返回选中的轨迹 dict；无合格者返回 None。
    """
    ready = [t for t in pend if t["hits"] >= min_hits]
    if not ready:
        return None

    def _motion(t):
        return float(np.std(t["ys"])) if len(t["ys"]) >= 2 else 0.0
    ready.sort(key=lambda t: (_motion(t), t["conf"]), reverse=True)
    return ready[0]


# ══════════════════════════════════════════════════════════
#  NCC 模板
# ══════════════════════════════════════════════════════════

@dataclass
class Template:
    patch: np.ndarray        # 半分辨率灰度模板
    size: int                # 原始模板边长（px）
    cx: float
    cy: float


def make_template(frame: np.ndarray, cx: float, cy: float, h: float,
                  ncc_scale: float = 0.5) -> Template:
    H, W = frame.shape[:2]
    t = int(np.clip(h * 1.2, 30, 200))
    x1 = int(np.clip(cx - t / 2, 0, max(0, W - t)))
    y1 = int(np.clip(cy - t / 2, 0, max(0, H - t)))
    x2, y2 = int(min(W, x1 + t)), int(min(H, y1 + t))
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    ts = max(8, int(t * ncc_scale))
    return Template(patch=cv2.resize(gray, (ts, ts)), size=t,
                    cx=float(cx), cy=float(cy))


# ══════════════════════════════════════════════════════════
#  混合跟踪器（检测→拟合）
# ══════════════════════════════════════════════════════════

@dataclass
class TrackDiagnostics:
    status: str = ""                # 见 pipeline.StatusCodes
    n_frames: int = 0
    elapsed_s: float = 0.0
    ms_per_frame: float = 0.0
    n_yolo_calls: int = 0
    yolo_ratio: float = 0.0
    coverage: float = 0.0           # 有效跟踪帧占比
    src_counts: dict = field(default_factory=dict)
    ncc_mean: float = 0.0
    anchor_frame: int = -1
    anchor_h_px: float = 0.0
    notes: list[str] = field(default_factory=list)


class DetectFitTracker:
    """
    主循环（每帧）：
      锚定期：扫帧至物理先验通过 + 置信度达标（上限 boot_max_scan 帧）
      跟踪期：
        每 redet_every 帧（或连续丢Targets>redet_on_miss 帧）→ YOLO 重检测
          · 2D 邻近门禁 + NCC 身份核验（防跳到画面里其它同款片）
          · 高置信时刷新模板（适应光照/外观漂移）
        其余帧 → NCC 拟合（搜索窗随速度自适应）+ 匀速预测
      任何时刻都不跨大 gap 插值（交由分段层拒绝）
    """

    def __init__(self, detector: PlateDetector,
                 redet_every: int = 15,
                 redet_on_miss: int = 8,
                 yolo_conf: float = 0.30,
                 bootstrap_conf: float = 0.30,
                 boot_max_scan: int = 90,
                 ncc_thresh: float = 0.45,
                 template_refresh_conf: float = 0.45,
                 accept_gate_px: float = 150.0,
                 ncc_scale: float = 0.5,
                 plate_diameter_m: float = 0.45,
                 user_hint: tuple[float, float] | None = None):
        self.det = detector
        self.redet_every = redet_every
        self.redet_on_miss = redet_on_miss
        self.yolo_conf = yolo_conf
        self.bootstrap_conf = bootstrap_conf
        self.boot_max_scan = boot_max_scan
        self.ncc_thresh = ncc_thresh
        self.template_refresh_conf = template_refresh_conf
        self.accept_gate_px = accept_gate_px
        self.ncc_scale = ncc_scale
        self.plate_diameter_m = plate_diameter_m
        self.user_hint = user_hint        # 用户点选 (cx, cy)：产品兜底钩子

    # ── NCC ──────────────────────────────────────────────
    def _ncc_fit(self, gray: np.ndarray, tpl: Template,
                 pred_cx: float, pred_cy: float, v_pred: float):
        H, W = gray.shape
        t = tpl.size
        R = 40 + int(min(120, abs(v_pred) * 4))
        x1 = int(np.clip(pred_cx - t / 2 - R, 0, W - 1))
        y1 = int(np.clip(pred_cy - t / 2 - R, 0, H - 1))
        x2 = int(np.clip(pred_cx + t / 2 + R, x1 + t, W))
        y2 = int(np.clip(pred_cy + t / 2 + R, y1 + t, H))
        win = gray[y1:y2, x1:x2]
        if win.shape[0] <= tpl.patch.shape[0] or win.shape[1] <= tpl.patch.shape[1]:
            return None
        wh, ww = win.shape
        win_s = cv2.resize(win, (max(8, int(ww * self.ncc_scale)),
                                 max(8, int(wh * self.ncc_scale))))
        if (win_s.shape[0] < tpl.patch.shape[0]
                or win_s.shape[1] < tpl.patch.shape[1]):
            return None
        res = cv2.matchTemplate(win_s, tpl.patch, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(res)
        sc = self.ncc_scale
        cx = x1 + (loc[0] + tpl.patch.shape[1] / 2) / sc
        cy = y1 + (loc[1] + tpl.patch.shape[0] / 2) / sc
        return float(cx), float(cy), float(peak)

    def _ncc_score_at(self, gray: np.ndarray, tpl: Template,
                      cx: float, cy: float) -> float:
        H, W = gray.shape
        t = tpl.size
        x1 = int(np.clip(cx - t / 2, 0, max(0, W - t)))
        y1 = int(np.clip(cy - t / 2, 0, max(0, H - t)))
        win = gray[y1:y1 + t, x1:x1 + t]
        ts = max(8, int(t * self.ncc_scale))
        win_s = cv2.resize(win, (ts, ts))
        res = cv2.matchTemplate(win_s, tpl.patch, cv2.TM_CCOEFF_NORMED)
        return float(res.max()) if res.size else 0.0

    # ── 主流程 ───────────────────────────────────────────
    def process(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        if fps <= 1 or fps > 240:
            fps = 30.0

        t0 = __import__("time").time()
        ys: list[float] = []
        srcs: list[str] = []
        ncc_scores: list[float] = []
        h_samples: list[float] = []
        gray_prev: np.ndarray | None = None

        tpl: Template | None = None
        cx = cy = None
        v_pred = 0.0
        missing_run = 0
        n_yolo = 0
        n_frames = 0
        diag = TrackDiagnostics()

        # 锚定期 pending 轨迹（连续性确认 + 运动探针）
        pend: list[dict] = []
        anchored = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            n_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            H, W = gray.shape[:2]

            # ── 1) 锚定：pending 轨迹确认 + 运动探针 ─────
            # （v1.1 修复回归：量化 bin 在片快速移动时永远攒不够
            #   3 次命中 → 30kg/50kg 误报 NO_PLATE。改为 40px
            #   邻域连续性跟踪；多候选时用 y 方差选会动的=工作片）
            if not anchored:
                if n_frames > self.boot_max_scan:
                    diag.status = "NO_PLATE_DETECTED"
                    diag.n_frames = n_frames
                    return None, h_samples, diag, fps
                dets = self.det.detect(frame, self.yolo_conf)
                n_yolo += 1
                fH, fW = gray.shape[:2]
                cands = [d for d in dets
                         if d.conf >= self.bootstrap_conf
                         and anchor_score(d, fH, fW) > 0]
                if self.user_hint is not None:
                    hx, hy = self.user_hint
                    cands = sorted(cands,
                                   key=lambda d: np.hypot(d.cx - hx, d.cy - hy))[:1]
                # 更新 pending 轨迹（40px 邻域贪心匹配）
                used: set[int] = set()
                for tr in pend:
                    bi, bd = -1, 1e9
                    for i, d in enumerate(cands):
                        if i in used:
                            continue
                        dist = float(np.hypot(d.cx - tr["cx"], d.cy - tr["cy"]))
                        if dist < 40 and dist < bd:
                            bi, bd = i, dist
                    if bi >= 0:
                        used.add(bi)
                        d = cands[bi]
                        tr.update(cx=d.cx, cy=d.cy, h=d.h, conf=d.conf,
                                  hits=tr["hits"] + 1, miss=0)
                        tr["ys"].append(d.cy)
                    else:
                        tr["miss"] += 1
                # 未匹配候选 → 新 pending 轨迹
                for i, d in enumerate(cands):
                    if i not in used:
                        pend.append({"cx": d.cx, "cy": d.cy, "h": d.h,
                                     "conf": d.conf, "hits": 1, "miss": 0,
                                     "ys": [d.cy]})
                pend[:] = [t for t in pend if t["miss"] <= 5]

                # 确认：运动探针选出会动的轨迹（工作片）
                best = select_anchor_track(pend, min_hits=5)
                if best is not None:
                    anchored = True
                    cx, cy = best["cx"], best["cy"]
                    h_est = best["h"]
                    h_samples.append(h_est)
                    tpl = make_template(frame, cx, cy, h_est, self.ncc_scale)
                    v_pred = 0.0
                    ys.append(cy); srcs.append("boot"); ncc_scores.append(1.0)
                    diag.anchor_frame = n_frames
                    diag.anchor_h_px = h_est
                    gray_prev = gray
                    continue
                ys.append(np.nan); srcs.append("none"); ncc_scores.append(0.0)
                gray_prev = gray
                continue

            # ── 2) 跟踪期 ───────────────────────────────
            run_yolo = (n_frames % self.redet_every == 0
                        or missing_run > self.redet_on_miss)
            predict = True
            if run_yolo:
                dets = self.det.detect(frame, self.yolo_conf)
                n_yolo += 1
                pred_y = (cy or 0) + v_pred
                cands = sorted(
                    (d for d in dets
                     if np.hypot(d.cx - (cx or 0), d.cy - pred_y) < self.accept_gate_px),
                    key=lambda d: np.hypot(d.cx - (cx or 0), d.cy - pred_y))
                picked = None
                for cand in cands[:2]:
                    if tpl is not None and missing_run <= 2:
                        s = self._ncc_score_at(gray, tpl, cand.cx, cand.cy)
                        if s >= max(self.ncc_thresh * 0.8, 0.5):
                            picked = cand
                            break
                    else:
                        picked = cand
                        break
                if picked is not None:
                    if missing_run == 0:
                        v_pred = 0.7 * (picked.cy - cy) + 0.3 * v_pred
                    cx, cy = picked.cx, picked.cy
                    h_samples.append(picked.h)
                    if picked.conf >= self.template_refresh_conf and tpl is not None:
                        tpl = make_template(frame, cx, cy, picked.h, self.ncc_scale)
                    ys.append(cy); srcs.append("yolo"); ncc_scores.append(1.0)
                    missing_run = 0
                    predict = False
                else:
                    missing_run += 1

            if predict and tpl is not None:
                pred_cy = (cy or 0) + v_pred
                fit = self._ncc_fit(gray, tpl, cx or 0, pred_cy, v_pred)
                if fit is not None and fit[2] >= self.ncc_thresh:
                    if missing_run == 0:
                        v_pred = 0.7 * (fit[1] - cy) + 0.3 * v_pred
                    cx, cy = fit[0], fit[1]
                    ys.append(cy); srcs.append("ncc"); ncc_scores.append(fit[2])
                    missing_run = 0
                else:
                    missing_run += 1
                    ys.append(np.nan); srcs.append("miss"); ncc_scores.append(0.0)
            elif predict:
                missing_run += 1
                ys.append(np.nan); srcs.append("miss"); ncc_scores.append(0.0)

        cap.release()
        elapsed = __import__("time").time() - t0

        diag.n_frames = n_frames
        diag.elapsed_s = elapsed
        diag.ms_per_frame = elapsed / max(1, n_frames) * 1000
        diag.n_yolo_calls = n_yolo
        diag.yolo_ratio = n_yolo / max(1, n_frames)
        diag.src_counts = dict(Counter(srcs))
        diag.coverage = sum(1 for s in srcs if s in ("ncc", "yolo", "boot")) / max(1, n_frames)
        valid_ncc = [s for s in ncc_scores if s > 0]
        diag.ncc_mean = float(np.mean(valid_ncc)) if valid_ncc else 0.0
        if not anchored:
            diag.status = "NO_PLATE_DETECTED"
        else:
            diag.status = "TRACKED"
        return np.array(ys, dtype=float), h_samples, diag, fps

    # ── 标定 ─────────────────────────────────────────────
    def calibrate(self, h_samples: list[float]) -> float | None:
        """
        mpp = 片直径 / 片高中位数。
        历史教训：algorithms/common.py::calibrate_scale 的 scale_factor=1.15
        魔法系数是在补偿直 resize 造成的畸变——畸变已在源头（letterbox）消除，
        系数移除，物理量纲恢复 1:1。
        """
        if not h_samples:
            return None
        med = float(np.median(h_samples))
        if med < 2:
            return None
        return self.plate_diameter_m / med
