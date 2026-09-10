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
#  rep 底部重锚定 BottomRegrind（M1.5）
# ══════════════════════════════════════════════════════════

@dataclass
class BottomRegrind:
    """
    rep 底部重锚定的因果触发器（M1.5）。
    移植自 TroyKaneshiro/barbell-velocity-tracker 的 "Bottom-of-rep regrind"：
    跟踪 rep 底部（y 局部最大）并在确认反转后重跑一次 YOLO，
    在漂移喂入向心段之前纠正。底部是重捕获的最佳时机：
    杠铃瞬时速度≈0 → 运动模糊最小 → 检测器最准、模板最干净。
    与 M1 运动观察哨互补：观察哨治"完全锁错背景"，
    regrind 治"跟踪漂移偏移"（真目标仍在数倍片半径内）。
    无漂移证据不碰轨道：纠正 <0.2r 时触发器只记录不干预——实测连模板
    刷新都会经重检测 NCC 门控蝴蝶效应翻转分段（110kg_0.45_0.48：
    RMSE 0.02→0.44）。常规重检测已承担模板刷新职责。
    常数默认 = Troy 的实测值（以片半径 plate_r 为单位）。

    相对 Troy 原实现的修正（代码推演）：Troy 触发后 ref=触发点高度，
    下一 rep 下蹲时 running-max 相对 ref 最多增长"触发点→底部"残余
    （≈数帧回升量 « 1.5r），武装条件永不满足——实为每组触发一次。
    本实现加 UP/DOWN 双相：触发后跟踪上升段，从底部回升 ≥arm_disp
    即确认到顶、重锚顶部，做到真正每 rep 一次（漂移按组累积的
    50kg/105kg_0.69 类失败必须每 rep 纠）。
    """

    arm_disp: float = 1.5        # 相对锚点位移多大才武装（滤 setup 抖动）
    reversal_floor: float = 0.3  # 相对 running-max 回落多大算一帧反转
    confirm: int = 4             # 连续反转帧数才触发
    max_correction: float = 4.0  # 接受纠正的最大幅度（×plate_r）

    top_y: float | None = None   # DOWN 相：本 rep 顶部参考
    watch_max_y: float = 0.0
    armed: bool = False
    reversal_run: int = 0
    phase_up: bool = False       # True = 上升段（等到顶重锚）
    bottom_y: float = 0.0        # UP 相：本 rep 底部（触发时的 running-max）
    watch_min_y: float = 0.0

    def reset(self, y: float) -> None:
        """重锚（锚定/夺回/大跳变后调用）。"""
        self.top_y = float(y)
        self.watch_max_y = float(y)
        self.armed = False
        self.reversal_run = 0
        self.phase_up = False
        self.bottom_y = 0.0
        self.watch_min_y = float(y)

    def update(self, cy: float, plate_r: float) -> bool:
        """
        输入一帧平滑跟踪位置（调用方保证物理合理；跳变帧请改调 reset）。
        返回 True = 确认过底部，应立即重跑 YOLO 纠漂。
        触发后进 UP 相（无论纠正接受/拒绝都不在同一次折返重复触发，
        对应 Troy 的"触发即重锚"语义）；到顶后自动切回 DOWN 相。
        """
        if plate_r <= 0:
            return False
        if self.top_y is None:
            self.reset(cy)
            return False
        if self.phase_up:
            if cy < self.watch_min_y:
                self.watch_min_y = cy
            if self.bottom_y - self.watch_min_y >= self.arm_disp * plate_r:
                self.top_y = self.watch_min_y
                self.watch_max_y = self.watch_min_y
                self.armed = False
                self.reversal_run = 0
                self.phase_up = False
            return False
        if cy > self.watch_max_y:
            self.watch_max_y = cy
            self.reversal_run = 0
            if (not self.armed
                    and self.watch_max_y - self.top_y >= self.arm_disp * plate_r):
                self.armed = True
        elif self.armed:
            if self.watch_max_y - cy >= self.reversal_floor * plate_r:
                self.reversal_run += 1
            else:
                self.reversal_run = 0
            if self.reversal_run >= self.confirm:
                self.bottom_y = self.watch_max_y
                self.watch_min_y = cy
                self.phase_up = True
                self.armed = False
                self.reversal_run = 0
                return True
        return False


def select_regrind_candidate(dets: list[Detection], cx: float, cy: float,
                             plate_r: float,
                             max_correction: float = 4.0) -> Detection | None:
    """
    regrind 候选选择（纯函数）：纠正门内（≤max_correction×plate_r）最近者。
    半径门是主要防线（Troy：4r 既纠多帧漂移，又防跳到远处片堆）；
    身份核验（NCC/conf）由调用方 _try_regrind 叠加。
    """
    if plate_r <= 0:
        return None
    gate = max_correction * plate_r
    best: Detection | None = None
    best_d = gate
    for d in dets:
        dist = float(np.hypot(d.cx - cx, d.cy - cy))
        if dist <= best_d:
            best, best_d = d, dist
    return best


def regrind_verdict(corr: float, ncc: float, conf: float, plate_r: float,
                    min_correction: float = 0.2) -> str | None:
    """
    regrind 纠正分级（纯函数）：
      - 身份核验失败（NCC<0.5 且 conf<0.5）→ None（拒绝）
      - 纠正 < min_correction×plate_r → "micro"（轨道已准，调用方零干预。
        依据 110kg_0.45_0.48 实测：4px"纠正"经分段蝴蝶效应把 RMSE 从
        0.02 打到 0.44——连模板刷新都不做才是零扰动）
      - 否则 → "snap"（真漂移，全量纠正）
    """
    if not (ncc >= 0.5 or conf >= 0.5):
        return None
    if corr < min_correction * plate_r:
        return "micro"
    return "snap"


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
    n_regrind_snap: int = 0    # M1.5：真漂移全量纠正次数
    n_regrind_micro: int = 0   # M1.5：微偏零干预次数
    n_regrind_reject: int = 0  # M1.5：触发但拒绝次数（门内无候选/身份失败）


class DetectFitTracker:
    """
    主循环（每帧）：
      锚定期：扫帧至物理先验通过 + 置信度达标（上限 boot_max_scan 帧）
      跟踪期：
        每 redet_every 帧（或连续丢Targets>redet_on_miss 帧）→ YOLO 重检测
          · 2D 邻近门禁 + NCC 身份核验（防跳到画面里其它同款片）
          · 高置信时刷新模板（适应光照/外观漂移）
        其余帧 → NCC 拟合（搜索窗随速度自适应）+ 匀速预测
        rep 底部确认反转 → regrind：重跑 YOLO 纠漂（M1.5，移植 Troy）
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
                 max_step_factor: float = 0.25,
                 min_step_px: float = 15.0,
                 hold_max_frames: int = 40,
                 user_hint: tuple[float, float] | None = None,
                 regrind_enabled: bool = True,
                 regrind_conf: float = 0.25,
                 regrind_arm_disp: float = 1.5,
                 regrind_reversal_floor: float = 0.3,
                 regrind_confirm: int = 4,
                 regrind_max_correction: float = 4.0,
                 regrind_min_correction: float = 0.2):
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
        # NCC 单帧位移上限（M1 防漂移）：真实杠铃速度 ≤2.5m/s 时
        # 单帧位移 ≈ 0.19×片高；取 0.25×片高留余量，超过即视为漂移
        self.max_step_factor = max_step_factor
        self.min_step_px = min_step_px
        # 丢失桥接上限：蹲底遮挡实测 19-42 帧；40 为双视频网格搜索最优
        self.hold_max_frames = hold_max_frames
        self.user_hint = user_hint        # 用户点选 (cx, cy)：产品兜底钩子
        # rep 底部重锚定（M1.5，移植 Troy）：触发器常数 = Troy 实测值；
        # regrind_conf 略低于 yolo_conf——底部模糊最小，低门槛换召回，
        # 安全性由 4r 纠正门 + 身份核验承担
        self.regrind_enabled = regrind_enabled
        self.regrind_conf = regrind_conf
        self.regrind_arm_disp = regrind_arm_disp
        self.regrind_reversal_floor = regrind_reversal_floor
        self.regrind_confirm = regrind_confirm
        self.regrind_max_correction = regrind_max_correction
        # 纠正分级下限（×plate_r）：小于此的"纠正"是噪声，零干预
        self.regrind_min_correction = regrind_min_correction

    # ── hold 桥接（M1.2）──────────────────────────────────
    def _hold_or_nan(self, cy: float | None, missing_run: int) -> float:
        """丢失 ≤hold_max 帧时输出保持位（底部停顿的物理近似），
        更久则 NaN（大 gap 不插值原则仍成立）。"""
        if cy is None:
            return np.nan
        return cy if missing_run <= self.hold_max_frames else np.nan

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

    # ── rep 底部重锚定（M1.5）────────────────────────────
    def _try_regrind(self, frame: np.ndarray, gray: np.ndarray,
                     cx: float, cy: float, h_est: float,
                     tpl: Template | None
                     ) -> tuple[str | None, Detection | None, float, float, str]:
        """
        触发器确认过底部后：重跑 YOLO，纠正跟踪漂移。
        接受条件（三重防线）：
          1. 候选在纠正门内（≤4r，Troy 值）——防跳到远处片堆
          2. NCC 身份分 ≥0.5（模板仍可信时）或 conf ≥0.5
             （底部最清晰，高置信检测本身可信）——防锁错同款片
          3. 纠正分级（regrind_verdict）：微偏零干预，只纠真漂移
        返回 (动作, 候选, 纠正px, NCC分, 说明)；动作 None = 拒绝。
        注：YOLO 调用计数由调用方累加（n_yolo 是 process 局部量）。
        """
        dets = self.det.detect(frame, self.regrind_conf)
        plate_r = max(10.0, h_est / 2)
        cand = select_regrind_candidate(dets, cx, cy, plate_r,
                                        self.regrind_max_correction)
        if cand is None:
            return None, None, 0.0, 0.0, "门内无候选"
        corr = float(np.hypot(cand.cx - cx, cand.cy - cy))
        ncc = (self._ncc_score_at(gray, tpl, cand.cx, cand.cy)
               if tpl is not None else 0.0)
        action = regrind_verdict(corr, ncc, cand.conf, plate_r,
                                 self.regrind_min_correction)
        if action is None:
            return (None, None, corr, ncc,
                    f"身份核验失败(ncc={ncc:.2f},conf={cand.conf:.2f})")
        if action == "micro":
            return action, cand, corr, ncc, f"微偏{corr:.0f}px无需纠正"
        return action, cand, corr, ncc, f"纠正{corr:.0f}px"

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
        watch: dict[int, dict] = {}   # 运动簇观察哨（key=id(dict)）
        anchored = False
        # M1.5 regrind 触发器（每次 process 调用独立状态）
        rg = BottomRegrind(arm_disp=self.regrind_arm_disp,
                           reversal_floor=self.regrind_reversal_floor,
                           confirm=self.regrind_confirm,
                           max_correction=self.regrind_max_correction)

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
                    rg.reset(cy)   # M1.5：锚点 = regrind 参考零点
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
            rg_pending = False   # M1.5：本帧是否有平滑新位置喂触发器
            if run_yolo:
                dets = self.det.detect(frame, self.yolo_conf)
                n_yolo += 1
                # 多帧预测：遮挡 missing_run 帧后，真目标应在
                # cy + (missing_run+1)*v_pred 附近（旧代码只预测 1 帧，
                # 导致快速段遮挡恢复必然超 150px 门被误拒）
                pred_y = (cy or 0) + v_pred * (missing_run + 1)
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
                if picked is None:
                    # M1.2 identity-first 远距夺回：门禁内无合格候选时，
                    # 对高置信候选做模板身份核验（NCC 分数优先于距离）。
                    # 依据 50kg 诊断：NCC 锁到背景后真目标在 150-200px 外，
                    # 纯距离门禁死锁整组。同款片 + 模板高分 = 身份可信。
                    for cand in sorted(dets, key=lambda d: -d.conf)[:3]:
                        if cand.conf < 0.35 or tpl is None:
                            continue
                        if self._ncc_score_at(gray, tpl, cand.cx, cand.cy) >= 0.55:
                            picked = cand
                            break

                # M1.4 运动观察哨：某检测簇在大幅运动（y std>40px）
                # 而锁定轨迹是平的（|cy-lock|>60px）→ 错锁背景签名，
                # 强制夺回到该簇。静止片堆 y std≈0 永不触发；
                # 正确跟踪时工作片就在锁定位附近（距离条件排除）。
                # 邻域聚类：60px 内的候选归入同一运动簇（2D bin 会把
                # 连续运动的片切碎，导致每个 bin 永远凑不够 4 次观测）
                for d in dets:
                    if d.conf < self.yolo_conf:
                        continue
                    hit = None
                    for cl in watch.values():
                        if np.hypot(d.cx - cl["cx"], d.cy - cl["cy"]) < 60:
                            hit = cl
                            break
                    if hit is None:
                        hit = {"cx": d.cx, "cy": d.cy, "obs": []}
                        watch[id(hit)] = hit
                    hit["cx"] = 0.7 * d.cx + 0.3 * hit["cx"]
                    hit["cy"] = 0.7 * d.cy + 0.3 * hit["cy"]
                    hit["obs"].append((n_frames, d.cy, d.h))
                cutoff = n_frames - 90
                for k in list(watch.keys()):
                    cl = watch[k]
                    cl["obs"] = [o for o in cl["obs"] if o[0] >= cutoff]
                    if not cl["obs"]:
                        del watch[k]
                for cl in list(watch.values()):
                    obs = cl["obs"]
                    if len(obs) >= 4:
                        ys_w = [o[1] for o in obs]
                        last_f, last_cy, last_h = obs[-1]
                        if (float(np.std(ys_w)) > 40
                                and abs(last_cy - (cy or 0)) > 60):
                            cx, cy = cl["cx"], cl["cy"]
                            h_samples.clear()
                            h_samples.append(last_h)
                            h_est = last_h
                            tpl = make_template(frame, cx, cy, last_h,
                                                self.ncc_scale)
                            v_pred = 0.0
                            missing_run = 0
                            watch.clear()
                            rg.reset(cy)   # M1.5：夺回=身份跳变，触发器重锚
                            ys.append(cy); srcs.append("adopt")
                            ncc_scores.append(1.0)
                            predict = False
                            diag.notes.append(f"f{n_frames}: 观察哨夺回")
                            break
                if picked is not None:
                    if missing_run == 0:
                        v_pred = 0.7 * (picked.cy - cy) + 0.3 * v_pred
                    cy_prev = cy
                    cx, cy = picked.cx, picked.cy
                    h_est = picked.h
                    h_samples.append(picked.h)
                    # 恢复模式（此前有丢失/拒绝）：模糊使 conf 降低，
                    # 刷新门槛放宽到 0.35，让模板跟上外观变化
                    refresh_conf = self.template_refresh_conf
                    prev_src = srcs[-1] if srcs else None
                    if missing_run > 0 or prev_src in ("reject", "hold"):
                        refresh_conf = min(refresh_conf, 0.35)
                    if picked.conf >= refresh_conf and tpl is not None:
                        tpl = make_template(frame, cx, cy, picked.h, self.ncc_scale)
                    ys.append(cy); srcs.append("yolo"); ncc_scores.append(1.0)
                    missing_run = 0
                    predict = False
                    # M1.5：平滑更新喂触发器；大跳变（夺回）则重锚
                    jump_gate = max(self.min_step_px,
                                    self.max_step_factor * picked.h)
                    if abs(cy - (cy_prev or 0)) > jump_gate:
                        rg.reset(cy)
                    else:
                        rg_pending = True
                else:
                    missing_run += 1

            if predict and tpl is not None:
                pred_cy = (cy or 0) + v_pred
                fit = self._ncc_fit(gray, tpl, cx or 0, pred_cy, v_pred)
                max_step = max(self.min_step_px,
                               self.max_step_factor * (h_est or 0))
                if (fit is not None and fit[2] >= self.ncc_thresh
                        and np.hypot(fit[0] - (cx or 0), fit[1] - (cy or 0)) <= max_step):
                    if missing_run == 0:
                        v_pred = 0.7 * (fit[1] - cy) + 0.3 * v_pred
                    cx, cy = fit[0], fit[1]
                    ys.append(cy); srcs.append("ncc"); ncc_scores.append(fit[2])
                    missing_run = 0
                    rg_pending = True   # M1.5（NCC 位移已过物理门，平滑）
                elif fit is not None and fit[2] >= self.ncc_thresh:
                    # NCC 峰值可信但位移超物理上限 → 判为漂移，拒绝
                    missing_run += 1
                    ys.append(self._hold_or_nan(cy, missing_run))
                    srcs.append("reject" if missing_run > self.hold_max_frames else "hold")
                    ncc_scores.append(fit[2])
                else:
                    missing_run += 1
                    v_pred *= 0.6   # hold 期间速度衰减（模拟减速到停）
                    ys.append(self._hold_or_nan(cy, missing_run))
                    srcs.append("miss" if missing_run > self.hold_max_frames else "hold")
                    ncc_scores.append(0.0)
            elif predict:
                missing_run += 1
                ys.append(self._hold_or_nan(cy, missing_run))
                srcs.append("miss" if missing_run > self.hold_max_frames else "hold")
                ncc_scores.append(0.0)

            # ── M1.5 regrind：底部触发器（单帧至多更新一次）──
            if rg_pending and self.regrind_enabled and h_est:
                if rg.update(cy, h_est / 2):
                    n_yolo += 1
                    action, cand, _, _, note = self._try_regrind(
                        frame, gray, cx, cy, h_est, tpl)
                    if action == "snap" and cand is not None:
                        cx, cy = cand.cx, cand.cy
                        h_est = cand.h
                        h_samples.append(cand.h)
                        tpl = make_template(frame, cx, cy, cand.h,
                                            self.ncc_scale)
                        # 触发点≈底部：速度≈0；旧 v 指向错误方向，必须清零
                        # （对应 Troy 的"跨纠正跳变无有效速度对"）
                        v_pred = 0.0
                        missing_run = 0
                        ys[-1] = cy
                        srcs[-1] = "regrind"
                        ncc_scores[-1] = 1.0
                        diag.n_regrind_snap += 1
                        diag.notes.append(f"f{n_frames}: regrind {note}")
                    elif action == "micro":
                        # 轨道已准：零干预（连模板刷新都不做，见 verdict 注释）
                        diag.n_regrind_micro += 1
                        diag.notes.append(f"f{n_frames}: regrind {note}")
                    else:
                        diag.n_regrind_reject += 1
                        diag.notes.append(f"f{n_frames}: regrind 拒绝({note})")

        cap.release()
        elapsed = __import__("time").time() - t0

        diag.n_frames = n_frames
        diag.elapsed_s = elapsed
        diag.ms_per_frame = elapsed / max(1, n_frames) * 1000
        diag.n_yolo_calls = n_yolo
        diag.yolo_ratio = n_yolo / max(1, n_frames)
        diag.src_counts = dict(Counter(srcs))
        diag.coverage = sum(1 for s in srcs
                            if s in ("ncc", "yolo", "boot", "regrind")) / max(1, n_frames)
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
