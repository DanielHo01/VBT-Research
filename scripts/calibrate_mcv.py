"""calibrate_mcv.py — MCV 偏差回归校准（M2 第一步）
=====================================================
依据：
  - TECH_ROUTE 第十节：r≈0.956 说明可用线性校准吃掉大部分系统偏差
  - 2019 高速相机法论文：相机法 MV 存在系统性高估（低负荷更明显）
  - 34 视频基准 rep 级配对数据（BENCHMARK_v3.json 的 rep_pairs）

方法（防"校准本身过拟合"）：
  1. 全量 OLS：pred_gt = a·pred_raw + b
  2. 留一视频交叉验证（LOVO）：每次留出一个视频拟合，预测该视频，
     校准后 RMSE 才是对新视频的无偏估计
  3. bootstrap（按视频重采样）给出 a/b/RMSE 的 95% CI

诚实纪律（docs/HOLDOUT.md 第五节）：
  本脚本只用开发集。正式成绩必须双报 dev｜held-out；
  留出集拿到后应复跑：--benchmark validation/reports/BENCHMARK_holdout-v0.json。

用法:
    python3 scripts/calibrate_mcv.py [--benchmark validation/reports/BENCHMARK_v3.json]
                                      [--out validation/reports/CALIBRATION_mcv.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def load_pairs(json_path: Path) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)
    pairs = data.get("rep_pairs", [])
    if not pairs:
        raise SystemExit(f"[错误] {json_path} 中没有 rep_pairs——"
                         f"请用 v3+ 的 run_benchmark_v0.py 重新生成基准。")
    return pairs


def fit_ols(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """pred_gt = a*pred + b 的最小二乘解。"""
    A = np.vstack([pred, np.ones_like(pred)]).T
    (a, b), *_ = np.linalg.lstsq(A, gt, rcond=None)
    return float(a), float(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="validation/reports/BENCHMARK_v3.json",
                    help="含 rep_pairs 的基准 JSON")
    ap.add_argument("--out", default="validation/reports/CALIBRATION_mcv.json",
                    help="校准结果输出 JSON")
    ap.add_argument("--seed", type=int, default=42, help="bootstrap 随机种子")
    args = ap.parse_args()

    jp = Path(args.benchmark)
    if not jp.is_absolute():
        jp = REPO / jp
    pairs = load_pairs(jp)

    gt = np.array([p["gt"] for p in pairs], dtype=np.float64)
    pred = np.array([p["pred"] for p in pairs], dtype=np.float64)
    videos = np.array([p["video"] for p in pairs])
    uniq_videos = sorted(set(videos.tolist()))

    # ── 1. 全量 OLS ────────────────────────────────────
    a_full, b_full = fit_ols(gt, pred)
    resid_full = (a_full * pred + b_full) - gt
    rmse_full = float(np.sqrt(np.mean(resid_full ** 2)))
    bias_full = float(np.mean(resid_full))

    # ── 2. 留一视频交叉验证 ────────────────────────────
    lovo_err = []
    for v in uniq_videos:
        mask = videos != v
        if mask.sum() < 2 or (~mask).sum() == 0:
            continue
        a_v, b_v = fit_ols(gt[mask], pred[mask])
        lovo_err.extend(((a_v * pred[~mask] + b_v) - gt[~mask]).tolist())
    lovo_err = np.array(lovo_err)
    rmse_lovo = float(np.sqrt(np.mean(lovo_err ** 2)))
    bias_lovo = float(np.mean(lovo_err))

    # ── 3. bootstrap（按视频重采样，保留组内相关）────────
    rng = np.random.default_rng(args.seed)
    n_boot = 2000
    idx = np.arange(len(pairs))
    by_video = {v: idx[videos == v] for v in uniq_videos}
    boot_a, boot_b, boot_rmse = [], [], []
    for _ in range(n_boot):
        chosen = np.concatenate([
            by_video[v] for v in rng.choice(uniq_videos, size=len(uniq_videos), replace=True)
        ])
        a_b, b_b = fit_ols(gt[chosen], pred[chosen])
        boot_a.append(a_b)
        boot_b.append(b_b)
        boot_rmse.append(float(np.sqrt(np.mean(((a_b * pred[chosen] + b_b) - gt[chosen]) ** 2))))

    def ci(xs: list[float]) -> tuple[float, float]:
        return round(float(np.percentile(xs, 2.5)), 3), round(float(np.percentile(xs, 97.5)), 3)

    rmse_raw = float(np.sqrt(np.mean((pred - gt) ** 2)))
    bias_raw = float(np.mean(pred - gt))
    report = {
        "source_benchmark": str(jp),
        "n_pairs": int(len(pairs)),
        "n_videos": len(uniq_videos),
        "raw": {"rmse": round(rmse_raw, 4), "bias": round(bias_raw, 4)},
        "ols": {"a": round(a_full, 4), "b": round(b_full, 4),
                "rmse_fit": round(rmse_full, 4), "bias_fit": round(bias_full, 4)},
        "lovo_cv": {"rmse": round(rmse_lovo, 4), "bias": round(bias_lovo, 4)},
        "bootstrap_ci": {"a": ci(boot_a), "b": ci(boot_b),
                         "rmse_lovo_like": ci(boot_rmse)},
        "formula": f"mcv_cal = {a_full:.4f} * mcv_raw + ({b_full:+.4f})",
        "notes": [
            "开发集校准（M2 探索），正式成绩必须双报 dev|held-out（HOLDOUT.md 第五节）",
            "留出集到货后复跑本脚本并对比 LOVO RMSE，判断校准是否可泛化",
        ],
    }
    out_p = Path(args.out)
    if not out_p.is_absolute():
        out_p = REPO / out_p
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"rep 级配对: n={report['n_pairs']}（{report['n_videos']} 视频）")
    print(f"原始:       RMSE={report['raw']['rmse']}  bias={report['raw']['bias']}")
    print(f"OLS 全量:   a={a_full:.4f}  b={b_full:+.4f}  "
          f"拟合后 RMSE={report['ols']['rmse_fit']}  bias={report['ols']['bias_fit']}")
    print(f"LOVO CV:    校准后 RMSE={report['lovo_cv']['rmse']}  "
          f"bias={report['lovo_cv']['bias']}   ← 对新视频的无偏估计")
    print(f"bootstrap 95% CI: a {ci(boot_a)}  b {ci(boot_b)}  "
          f"RMSE(LOVO-like) {ci(boot_rmse)}")
    print(f"公式: {report['formula']}")
    print(f"结果: {out_p}")


if __name__ == "__main__":
    main()
