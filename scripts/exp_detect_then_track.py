"""
exp_detect_then_track.py — 技术路线对比实验
=============================================
验证「先检测杠铃片，之后靠拟合(NCC模板)跟踪」是否可以
在保持精度的前提下大幅节省计算。

三种模式（同一套 rep 提取逻辑，公平对比）:
  A  full_yolo   — 每帧全帧 YOLO + 最近邻选择（AnchorTemplateEngine 路线的修复版）
  B  detect_fit  — 每 N 帧全帧 YOLO 重检测 + 帧间 NCC 模板拟合 + 匀速预测  ← 待验证思路
  C  ncc_only    — 仅开头检测一次锚定，之后纯 NCC（算力下限基准）

两种 MCV 定义同时输出（用 GT 判定哪个定义更对）:
  mcv_mid   — 向心段中点瞬时速度（anchor_template_engine 现用, clip 0.05-2.5）
  mcv_mean  — 向心段平均速度（GymAware MCV 的标准定义）

用法:
  python3 scripts/exp_detect_then_track.py
"""
from __future__ import annotations

import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter, find_peaks

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "validation" / "dataset_benchmark"
sys.path.insert(0, str(BENCH))

from engines.anchor_template_engine import (  # noqa: E402
    prepare_frame_for_yolo, rotate_coord_back, canvas_to_orig, parse_yolo11,
)
from metrics_evaluator import MetricsEvaluator  # noqa: E402

MODEL = str(REPO / "models" / "yolo11_plate.onnx")

VIDEOS = [
    # (video_id, 说明)
    ("20kg_0.87_0.88_0.89_0.91.mp4",          "只有杆（对照组，预期无片）"),
    ("30kg_1.03_0.89_0.76_0.65.mp4",          "轻负荷快rep"),
    ("80kg_0.88_0.88_0.94_0.90.mp4",          "中负荷"),
    ("102.5kg_0.51_0.49_0.42_0.30.mp4",       "重负荷疲劳"),
    ("105kg_0.62_0.58_0.53_0.44.mp4",         "重负荷"),
    ("130kg_0.52_0.51_0.45_0.50_0.40.mp4",    "大重量"),
]


# ════════════════════════════════════════════════════════════
#  检测-拟合混合跟踪器
# ════════════════════════════════════════════════════════════

class DetectFitTracker:
    """YOLO 稀疏重检测 + NCC 模板拟合 + 匀速预测。"""

    def __init__(self, sess, inp_name,
                 redet_every: int = 15,          # 每 N 帧 YOLO 重检测（1=每帧, None=仅锚定）
                 yolo_conf: float = 0.30,
                 bootstrap_conf: float = 0.40,
                 ncc_thresh: float = 0.45,       # NCC 峰值低于此 → 视为丢失
                 template_refresh_conf: float = 0.45,  # 仅高置信 YOLO 才刷新模板
                 accept_gate_px: float = 150.0,  # 重检测接受门禁（自适应下限）
                 ncc_scale: float = 0.5,         # NCC 在半分辨率下做（提速）
                 plate_diameter_m: float = 0.45):
        self.sess = sess
        self.inp_name = inp_name
        self.redet_every = redet_every
        self.yolo_conf = yolo_conf
        self.bootstrap_conf = bootstrap_conf
        self.ncc_thresh = ncc_thresh
        self.template_refresh_conf = template_refresh_conf
        self.accept_gate_px = accept_gate_px
        self.ncc_scale = ncc_scale
        self.plate_diameter_m = plate_diameter_m

    # ── YOLO 全帧检测（返回原始帧坐标）─────────────────────
    def _yolo_detect(self, frame) -> list[dict]:
        blob, scale, yo, xo, f_h, f_w, rotated = prepare_frame_for_yolo(frame, 640)
        out = self.sess.run(None, {self.inp_name: blob})[0]
        dets = parse_yolo11(out, self.yolo_conf)
        results = []
        for d in dets:
            if rotated:
                cx_o, cy_o = rotate_coord_back(d["cx"], d["cy"], scale, yo, xo, f_h, f_w)
            else:
                cx_o, cy_o = canvas_to_orig(d["cx"], d["cy"], scale, yo, xo)
            results.append({"cx": cx_o, "cy": cy_o,
                            "w": d["w"] / scale, "h": d["h"] / scale,
                            "conf": d["conf"]})
        return results

    def _make_template(self, frame, cx, cy, h):
        t = int(np.clip(h * 1.2, 30, 200))
        H, W = frame.shape[:2]
        x1 = int(np.clip(cx - t / 2, 0, max(0, W - t)))
        y1 = int(np.clip(cy - t / 2, 0, max(0, H - t)))
        x2, y2 = int(min(W, x1 + t)), int(min(H, y1 + t))
        patch = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        t_s = max(8, int(t * self.ncc_scale))
        patch_s = cv2.resize(patch, (t_s, t_s))
        return {"patch": patch_s, "size": t, "cx": float(cx), "cy": float(cy)}

    # ── NCC 拟合：在预测位置附近搜索 ───────────────────────
    def _ncc_fit(self, frame, tpl, pred_cx, pred_cy):
        H, W = frame.shape[:2]
        t = tpl["size"]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 搜索窗：预测中心 ± R（R 随速度放大，下限 40px）
        R = 40 + int(min(120, abs(self._v_pred) * 4))
        x1 = int(np.clip(pred_cx - t / 2 - R, 0, W - 1))
        y1 = int(np.clip(pred_cy - t / 2 - R, 0, H - 1))
        x2 = int(np.clip(pred_cx + t / 2 + R, x1 + t, W))
        y2 = int(np.clip(pred_cy + t / 2 + R, y1 + t, H))
        win = gray[y1:y2, x1:x2]
        if win.shape[0] <= t or win.shape[1] <= t:
            return None
        wh, ww = win.shape
        win_s = cv2.resize(win, (max(8, int(ww * self.ncc_scale)),
                                 max(8, int(wh * self.ncc_scale))))
        if win_s.shape[0] < tpl["patch"].shape[0] or win_s.shape[1] < tpl["patch"].shape[1]:
            return None
        res = cv2.matchTemplate(win_s, tpl["patch"], cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(res)
        # 半分辨率坐标 → 原始帧坐标（窗口中心对齐）
        sc = self.ncc_scale
        cx_w = (loc[0] + tpl["patch"].shape[1] / 2) / sc
        cy_w = (loc[1] + tpl["patch"].shape[0] / 2) / sc
        cx = x1 + cx_w
        cy = y1 + cy_w
        return {"cx": float(cx), "cy": float(cy), "score": float(peak)}

    # ── NCC 身份核验：检查某位置与模板的匹配分 ─────────────
    def _ncc_score_at(self, frame, tpl, cx, cy):
        """在 (cx,cy) 处取与模板同尺寸的窗口，算 NCC 分（半分辨率，快速）。"""
        H, W = frame.shape[:2]
        t = tpl["size"]
        x1 = int(np.clip(cx - t / 2, 0, max(0, W - t)))
        y1 = int(np.clip(cy - t / 2, 0, max(0, H - t)))
        win = cv2.cvtColor(frame[y1:y1 + t, x1:x1 + t], cv2.COLOR_BGR2GRAY)
        t_s = max(8, int(t * self.ncc_scale))
        win_s = cv2.resize(win, (t_s, t_s))
        res = cv2.matchTemplate(win_s, tpl["patch"], cv2.TM_CCOEFF_NORMED)
        return float(res.max()) if res.size else 0.0

    # ── 锚定候选打分：物理先验（防锁到存放片堆/出画物体）──
    def _anchor_score(self, d, frame_h, frame_w):
        """返回 (score, ok)。规则：拒绝出画/贴边物体；尺寸在合理杠铃片域内；
        中心加权。最高置信 ≠ 杠铃片（可能是背景片堆）。"""
        # 贴边/出画拒绝（杠铃片不会半截在画面外）
        m = 0.03
        if not (frame_w * m < d["cx"] < frame_w * (1 - m)):
            return 0.0, False
        if not (frame_h * m < d["cy"] < frame_h * (1 - m)):
            return 0.0, False
        bw, bh = d["w"], d["h"]
        if bw <= 0 or bh <= 0:
            return 0.0, False
        ratio = max(bw, bh) / min(bw, bh)
        if ratio > 1.6:  # 片堆侧视/斜视会呈扁矩形
            return 0.0, False
        # 尺寸域：杠铃片表观高度通常在帧高的 3%~15%
        hr = bh / frame_h
        if not (0.03 <= hr <= 0.15):
            return 0.0, False
        # 中心加权（构图上杠铃片接近画面中心）
        cxn = abs(d["cx"] - frame_w / 2) / (frame_w / 2)
        cyn = abs(d["cy"] - frame_h / 2) / (frame_h / 2)
        cent = max(0.0, 1.0 - (cxn + cyn) / 2)
        return d["conf"] * (0.4 + 0.6 * cent), True

    # ── 主流程 ─────────────────────────────────────────────
    def process(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

        t_start = time.time()
        ys, srcs, ncc_scores = [], [], []
        tpl = None
        cx = cy = None
        h_est = None
        v_pred = 0.0            # px/frame（带符号, y 向下为正）
        missing_run = 0
        self._v_pred = 0.0
        n_yolo_calls = 0
        n_frames = 0
        boot_ok = False
        h_samples: list[float] = []
        BOOT_MAX_SCAN = 90  # 锚定最多扫前 90 帧（杆-only 视频快速判负）

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            n_frames += 1

            # ── 1) 锚定：扫帧直到高置信检测 ─────────────────
            if tpl is None:
                if n_frames > BOOT_MAX_SCAN:
                    return {"status": "NO_PLATE_DETECTED", "n_frames": n_frames,
                            "elapsed": time.time() - t_start, "boot_ok": False}
                dets = self._yolo_detect(frame)
                n_yolo_calls += 1
                fH, fW = frame.shape[:2]
                scored = []
                for d in dets:
                    s, ok = self._anchor_score(d, fH, fW)
                    if ok:
                        scored.append((s, d))
                if scored:
                    scored.sort(key=lambda t: t[0], reverse=True)
                    best_s, best = scored[0]
                    if best_s >= self.bootstrap_conf * 0.5:
                        cx, cy, h_est = best["cx"], best["cy"], best["h"]
                    h_samples.append(h_est)
                    tpl = self._make_template(frame, cx, cy, h_est)
                    v_pred = 0.0
                    boot_ok = True
                    ys.append(cy); srcs.append("boot"); ncc_scores.append(1.0)
                else:
                    ys.append(np.nan); srcs.append("none"); ncc_scores.append(0.0)
                continue

            predict = True
            run_yolo = (self.redet_every is not None
                        and n_frames % self.redet_every == 0) or missing_run > 8

            # ── 2) YOLO 重检测帧 ────────────────────────────
            if run_yolo:
                dets = self._yolo_detect(frame)
                n_yolo_calls += 1
                picked = None
                if dets:
                    pred_y = cy + v_pred
                    gate = self.accept_gate_px
                    # 2D 门禁：必须整体距离近（防止跳到画面里其它同款杠铃片）
                    cands = [d for d in dets
                             if np.hypot(d["cx"] - cx, d["cy"] - pred_y) < gate]
                    if cands:
                        cands.sort(key=lambda d: np.hypot(d["cx"] - cx, d["cy"] - pred_y))
                        # 身份核验：候选处 NCC 分数也要过关（模板存在且 NCC 之前未失效时）
                        for cand in cands[:2]:
                            if tpl is not None and missing_run <= 2:
                                s = self._ncc_score_at(frame, tpl, cand["cx"], cand["cy"])
                                if s >= self.ncc_thresh * 0.8 or s >= 0.5:
                                    picked = cand
                                    break
                            else:
                                picked = cand
                                break
                if picked is not None:
                    if missing_run == 0:
                        v_pred = 0.7 * (picked["cy"] - cy) + 0.3 * v_pred
                    cx, cy = picked["cx"], picked["cy"]
                    h_est = picked["h"]
                    h_samples.append(picked["h"])
                    if picked["conf"] >= self.template_refresh_conf:
                        tpl = self._make_template(frame, cx, cy, h_est)
                    ys.append(cy); srcs.append("yolo"); ncc_scores.append(1.0)
                    missing_run = 0
                    predict = False
                else:
                    missing_run += 1
                    # YOLO 没找到 → 落回 NCC 拟合
                    pass

            # ── 3) NCC 拟合帧 ───────────────────────────────
            if predict:
                self._v_pred = v_pred  # 供 _ncc_fit 计算搜索半径
                pred_cy = cy + v_pred
                fit = self._ncc_fit(frame, tpl, cx, pred_cy)
                if fit is not None and fit["score"] >= self.ncc_thresh:
                    if missing_run == 0:
                        v_pred = 0.7 * (fit["cy"] - cy) + 0.3 * v_pred
                    cx, cy = fit["cx"], fit["cy"]
                    ys.append(cy); srcs.append("ncc"); ncc_scores.append(fit["score"])
                    missing_run = 0
                else:
                    missing_run += 1
                    ys.append(np.nan); srcs.append("miss"); ncc_scores.append(0.0)

        cap.release()
        elapsed = time.time() - t_start

        y_arr = np.array(ys, dtype=float)
        # 小 gap 线性插值（≤10 帧）
        valid = np.where(~np.isnan(y_arr))[0]
        if len(valid) >= 5 and valid[-1] - valid[0] > 30:
            y_full = np.interp(np.arange(n_frames), valid, y_arr[valid])
            # 大 gap 置 NaN（>10 帧不插值，后面 rep 过滤会剔除跨大 gap 的）
            gaps = np.diff(valid)
            for i, g in enumerate(gaps):
                if g > 11:
                    y_full[valid[i] + 1: valid[i + 1]] = np.nan
        else:
            return {"status": "NO_TRACK", "n_frames": n_frames, "elapsed": elapsed,
                    "boot_ok": boot_ok}

        # 标定：所有 YOLO 帧片高中位数（比单点锚定稳）
        if not h_samples or float(np.median(h_samples)) < 2:
            return {"status": "NO_SCALE", "n_frames": n_frames, "elapsed": elapsed}
        scale = self.plate_diameter_m / float(np.median(h_samples))

        return self._extract_reps(y_full, fps, scale, elapsed, n_frames,
                                  n_yolo_calls, srcs, ncc_scores, boot_ok)

    # ── rep 提取（所有模式共用）────────────────────────────
    def _extract_reps(self, y, fps, scale, elapsed, n_frames,
                      n_yolo_calls, srcs, ncc_scores, boot_ok):
        # 大 gap 前后分段：只保留最长连续段
        segs, start = [], 0
        for i in range(1, len(y)):
            if np.isnan(y[i]) or np.isnan(y[i - 1]):
                segs.append((start, i - 1)); start = i + 1
        segs.append((start, len(y) - 1))
        segs = [(a, b) for a, b in segs if b - a > 30 and not np.isnan(y[a:b + 1]).any()]
        if not segs:
            return {"status": "NO_CLEAN_SEGMENT", "elapsed": elapsed, "n_frames": n_frames}
        a, b = max(segs, key=lambda s: s[1] - s[0])
        ys_ = y[a:b + 1]

        win = min(15, len(ys_) - 1)
        if win % 2 == 0:
            win -= 1
        if win < 5:
            return {"status": "TOO_SHORT", "elapsed": elapsed}
        y_s = savgol_filter(ys_, win, 3)
        v = -savgol_filter(ys_, win, 3, deriv=1) * fps * scale   # m/s，向上为正

        rom_min, dur_min, dur_max = 0.012, 0.25, 4.5
        bottoms, _ = find_peaks(y_s, distance=int(fps * dur_min))
        tops, _ = find_peaks(-y_s, distance=int(fps * dur_min))
        events = sorted([(f, "b") for f in bottoms] + [(f, "t") for f in tops])

        reps = []
        i = 0
        while i < len(events) - 1:
            f1, t1 = events[i]; f2, t2 = events[i + 1]
            if t1 == "b" and t2 == "t":
                dur = (f2 - f1) / fps
                rom = abs(y_s[f2] - y_s[f1]) * scale
                if dur_min <= dur <= dur_max and rom >= rom_min:
                    mid = int(np.clip((f1 + f2) // 2, 0, len(v) - 1))
                    reps.append({
                        "mcv_mid": round(float(np.clip(v[mid], 0.05, 2.50)), 3),
                        "mcv_mean": round(float(np.mean(v[f1:f2 + 1])), 3),
                    })
                i += 2
            else:
                i += 1

        from collections import Counter
        return {
            "status": "OK", "elapsed": elapsed, "n_frames": n_frames,
            "ms_per_frame": elapsed / n_frames * 1000,
            "yolo_calls": n_yolo_calls,
            "yolo_call_ratio": n_yolo_calls / n_frames,
            "src_counts": dict(Counter(srcs)),
            "mean_ncc": float(np.mean([s for s in ncc_scores if s > 0])) if any(ncc_scores) else 0,
            "boot_ok": boot_ok, "segment_frames": (a, b),
            "reps": reps,
        }


# ════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════

def main():
    import onnxruntime as ort
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    with open(BENCH / "dataset_index.json") as f:
        index = {d["video_id"]: d["gt_reps_mcv"] for d in json.load(f)}

    modes = {
        "A_full_yolo": dict(redet_every=1),
        "B_detect_fit": dict(redet_every=15),
        "C_ncc_only": dict(redet_every=None),
    }

    all_rows = []
    for vid, note in VIDEOS:
        gt = index.get(vid, [])
        vp = str(BENCH / "raw_videos" / vid)
        print(f"\n{'='*78}\n▶ {vid}  ({note})  GT={gt}")
        for mode, kw in modes.items():
            # 20kg 杆-only 对照组只跑 B（其它模式结论一致）
            if "只有杆" in note and mode != "B_detect_fit":
                continue
            tr = DetectFitTracker(sess, inp_name, **kw)
            r = tr.process(vp)
            row = {"video": vid, "mode": mode, "note": note,
                   "status": r["status"],
                   "ms_per_frame": round(r.get("ms_per_frame", float("nan")), 1),
                   "yolo_ratio": round(r.get("yolo_call_ratio", 0), 3)}
            if r["status"] == "OK" and gt:
                pred_mid = [x["mcv_mid"] for x in r["reps"]]
                pred_mean = [x["mcv_mean"] for x in r["reps"]]
                ev_mid = MetricsEvaluator.evaluate_video(vid, gt, [{"mcv": m} for m in pred_mid])
                ev_mean = MetricsEvaluator.evaluate_video(vid, gt, [{"mcv": m} for m in pred_mean])
                row.update({
                    "n_pred": len(pred_mid),
                    "rmse_mid": round(ev_mid.rmse, 3) if not np.isnan(ev_mid.rmse) else None,
                    "rmse_mean": round(ev_mean.rmse, 3) if not np.isnan(ev_mean.rmse) else None,
                    "r_mean": round(ev_mean.pearson_r, 3),
                    "bias_mean": round(ev_mean.bias, 3),
                    "pred_mid": pred_mid,
                    "pred_mean": pred_mean,
                })
                print(f"  [{mode:<12}] {r['ms_per_frame']:6.1f} ms/帧  YOLO调用率 {row['yolo_ratio']*100:5.1f}%  "
                      f"reps {row['n_pred']}/{len(gt)}  "
                      f"RMSE(mid)={row['rmse_mid']}  RMSE(mean)={row['rmse_mean']}  "
                      f"r={row['r_mean']}  bias={row['bias_mean']:+.3f}")
                print(f"                pred_mid ={pred_mid}")
                print(f"                pred_mean={pred_mean}")
            else:
                print(f"  [{mode:<12}] {r.get('ms_per_frame', 0):6.1f} ms/帧  status={r['status']}")
            all_rows.append(row)

    # ── 汇总 ──────────────────────────────────────────────
    print(f"\n{'='*78}\n汇总（排除 20kg 对照组）")
    for mode in modes:
        rows = [r for r in all_rows if r["mode"] == mode and "只有杆" not in r["note"]]
        spd = [r["ms_per_frame"] for r in rows if r.get("ms_per_frame")]
        yolo = [r["yolo_ratio"] for r in rows if r.get("yolo_ratio") is not None]
        rm_mid = [r["rmse_mid"] for r in rows if r.get("rmse_mid") is not None]
        rm_mean = [r["rmse_mean"] for r in rows if r.get("rmse_mean") is not None]
        npred = [r["n_pred"] for r in rows if "n_pred" in r]
        ngts = [len(index.get(r["video"], [])) for r in rows if "n_pred" in r]
        if spd:
            print(f"  [{mode:<12}] 平均 {np.mean(spd):6.1f} ms/帧 | YOLO调用率 {np.mean(yolo)*100:5.1f}% | "
                  f"RMSE mean定义均值 {np.mean(rm_mean) if rm_mean else float('nan'):.3f} | "
                  f"RMSE mid定义均值 {np.mean(rm_mid) if rm_mid else float('nan'):.3f} | "
                  f"rep 检出 {sum(npred)}/{sum(ngts)}")


if __name__ == "__main__":
    main()
