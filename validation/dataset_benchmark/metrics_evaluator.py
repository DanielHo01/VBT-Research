"""
metrics_evaluator.py
学术标准统计评估器：RMSE / MAE / Bias / Pearson R / ICC / Bland-Altman
"""
from __future__ import annotations

import numpy as np
import scipy.stats as stats
from dataclasses import dataclass, field


@dataclass
class RepMatchResult:
    """单个 Rep 的配对评估结果"""
    gt_mcv: float
    pred_mcv: float
    error: float           # pred - gt


@dataclass
class VideoEvalResult:
    """单个视频的评估汇总"""
    video_id: str
    n_gt: int
    n_pred: int
    n_paired: int
    paired: list[RepMatchResult]

    rmse: float
    mae: float
    bias: float
    pearson_r: float
    icc: float            # ICC(2,1) two-way mixed, absolute
    loa_upper: float      # Bland-Altman 95% 上限
    loa_lower: float      # Bland-Altman 95% 下限

    # 原始数据（方便绘图）
    gt_arr: np.ndarray = field(default_factory=lambda: np.array([]))
    pred_arr: np.ndarray = field(default_factory=lambda: np.array([]))


class MetricsEvaluator:
    """
    评估工具类：
      1. 成对 rep 对齐（顺序截断 + 贪婪最近邻）
      2. 计算全套学术指标
    """

    @staticmethod
    def align_reps(gt_mcvs: list[float],
                   pred_mcvs: list[float],
                   strategy: str = "truncate") -> tuple[list[float], list[float]]:
        """
        将预测 reps 与真值 reps 对齐。
        strategy:
          - "truncate": 取 min(n_gt, n_pred)，顺序截断/填充
          - "best_pair": 全局最优 1-to-1 配对（对距离矩阵贪心）
        """
        gt_arr    = np.array(gt_mcvs, dtype=np.float64)
        pred_arr  = np.array(pred_mcvs, dtype=np.float64)
        n_gt, n_pred = len(gt_arr), len(pred_arr)

        if n_gt == 0 or n_pred == 0:
            return [], []

        if strategy == "truncate":
            n = min(n_gt, n_pred)
            return gt_arr[:n].tolist(), pred_arr[:n].tolist()

        elif strategy == "best_pair":
            # 贪婪最近邻：每个 gt 匹配预测列表中最近的
            matched_gt, matched_pred = [], []
            used_pred = set()
            for g in gt_arr:
                diffs = [abs(float(g) - float(p)) for i, p in enumerate(pred_arr) if i not in used_pred]
                if not diffs:
                    break
                best_j = np.argmin(diffs)
                # 重建索引
                idx_map = [i for i in range(len(pred_arr)) if i not in used_pred]
                best_idx = idx_map[best_j]
                used_pred.add(best_idx)
                matched_gt.append(float(g))
                matched_pred.append(float(pred_arr[best_idx]))
            return matched_gt, matched_pred

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @staticmethod
    def compute_icc(gt_arr: np.ndarray,
                   pred_arr: np.ndarray) -> float:
        """
        ICC(2,1) — Two-Way Mixed, Absolute Agreement
        适用于单个测量设备对同一被试的重复测量评估。
        """
        n = len(gt_arr)
        if n < 3:
            return 0.0

        # 构建 2×n 矩阵 (2 observers/raters, n subjects)
        data = np.vstack([gt_arr, pred_arr])

        # 组间均值
        row_means = data.mean(axis=1, keepdims=True)    # (2,1)
        col_means = data.mean(axis=0, keepdims=True)   # (1,n)
        grand_mean = data.mean()

        # SSB (between subjects)
        ssb = np.sum(col_means ** 2) / 2 - (grand_mean ** 2) * n
        dfb = n - 1

        # SSW (within, total)
        ssw = np.sum((data - row_means) ** 2)
        dfw = n

        # SSM (model)
        ssm = ssb
        dfm = dfb

        # SS_total
        ss_total = np.sum((data - grand_mean) ** 2)

        # MSR, MSE
        msr = ssm / dfm if dfm > 0 else 0.0
        mse = ssw / dfw if dfw > 0 else 0.0

        if mse == 0:
            return 0.0

        # ICC(2,1)
        icc = (msr - mse) / (msr + (2 - 1) * mse)
        return float(icc)

    @classmethod
    def evaluate_video(cls,
                       video_id: str,
                       gt_mcvs: list[float],
                       pred_reps: list[dict],
                       strategy: str = "truncate") -> VideoEvalResult:
        """
        评估单个视频。
        pred_reps: run_pipeline() 返回的 list[dict]，每个含 'mcv' 键
        """
        pred_mcvs = [r['mcv'] for r in pred_reps]

        paired_gt, paired_pred = cls.align_reps(gt_mcvs, pred_mcvs, strategy=strategy)

        if not paired_gt:
            return VideoEvalResult(
                video_id=video_id, n_gt=len(gt_mcvs), n_pred=len(pred_mcvs),
                n_paired=0, paired=[], rmse=np.nan, mae=np.nan,
                bias=np.nan, pearson_r=0.0, icc=0.0,
                loa_upper=np.nan, loa_lower=np.nan,
            )

        gt_arr    = np.array(paired_gt, dtype=np.float64)
        pred_arr  = np.array(paired_pred, dtype=np.float64)
        errors    = pred_arr - gt_arr

        rmse    = float(np.sqrt(np.mean(errors ** 2)))
        mae     = float(np.mean(np.abs(errors)))
        bias    = float(np.mean(errors))

        if len(gt_arr) > 1:
            r, _ = stats.pearsonr(gt_arr, pred_arr)
        else:
            r = 0.0

        sd_diff  = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0
        loa_up   = bias + 1.96 * sd_diff
        loa_low  = bias - 1.96 * sd_diff

        icc_val  = cls.compute_icc(gt_arr, pred_arr)

        paired_objs = [
            RepMatchResult(gt_mcv=g, pred_mcv=p, error=float(p - g))
            for g, p in zip(paired_gt, paired_pred)
        ]

        return VideoEvalResult(
            video_id=video_id,
            n_gt=len(gt_mcvs), n_pred=len(pred_mcvs), n_paired=len(paired_gt),
            paired=paired_objs,
            rmse=rmse, mae=mae, bias=bias,
            pearson_r=float(r), icc=icc_val,
            loa_upper=loa_up, loa_lower=loa_low,
            gt_arr=gt_arr, pred_arr=pred_arr,
        )


# ── 聚合统计（跨视频）───────────────────────────────────────

def aggregate_results(video_results: list[VideoEvalResult]) -> dict:
    """
    将多个视频的 VideoEvalResult 聚合为全局统计。
    """
    all_gt    = np.concatenate([r.gt_arr    for r in video_results if len(r.gt_arr) > 0])
    all_pred  = np.concatenate([r.pred_arr  for r in video_results if len(r.pred_arr) > 0])

    if len(all_gt) == 0:
        return {}

    errors = all_pred - all_gt

    return {
        'total_videos':      len(video_results),
        'total_reps':        len(all_gt),
        'rmse':              float(np.sqrt(np.mean(errors ** 2))),
        'mae':               float(np.mean(np.abs(errors))),
        'bias':              float(np.mean(errors)),
        'std_error':         float(np.std(errors, ddof=1) / np.sqrt(len(errors))),
        'pearson_r':         float(np.corrcoef(all_gt, all_pred)[0, 1]),
        'loa_upper':         float(np.mean(errors) + 1.96 * np.std(errors, ddof=1)),
        'loa_lower':         float(np.mean(errors) - 1.96 * np.std(errors, ddof=1)),
        'all_gt':            all_gt.tolist(),
        'all_pred':          all_pred.tolist(),
    }
