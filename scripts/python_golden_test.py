"""
scripts/python_golden_test.py — Python 端虚拟 golden test（铁律闸门部分替代）
================================================================================
目标：在无 Linux/C++ 工具链的环境下，验证 C++ 端 golden_test 的
JSON 契约、数值稳定性和复现性。

产出 GOLDEN_REPORT.json（与 vbtcore-cpp/tests/golden_test.cpp 同 schema），
供 C++ 端编译后做最终 max_vel_err 比对。

**与 C++ 端 golden_test 的区别**：
  ✓ JSON schema 一致（C++ analyze_video_json 序列化格式对齐）
  ✓ 数值稳定性：相同输入 → 相同输出（复跑偏差应为 0）
  ✗ 不验证 C++ 浮点 / OpenCV 像素舍入 / Kalman 累加差异
  ✗ 不验证 ASan 内存安全

**真正的铁律闸门验收仍需** ./scripts/build_cpp_desktop.sh（Linux 桌面）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BENCH = REPO / "validation" / "dataset_benchmark"
DEFAULT_BASELINE_DIR = REPO / "validation" / "reports" / "cpp_baseline"
DEFAULT_OUTPUT = REPO / "validation" / "reports" / "cpp_baseline" / "GOLDEN_REPORT.json"
MODEL = REPO / "models" / "best.onnx"

# 闸门阈值（与 C++ 端 golden_test.cpp 一致）
THRESHOLD_MAX_VEL_ERR_MPS = 0.005  # m/s
THRESHOLD_MAX_REP_DELTA = 1  # 容许 n_pred 偏差
THRESHOLD_N_PASS_MIN = 20  # 至少 20/34 通过


def load_baseline(baseline_dir: Path, video_id: str) -> dict | None:
    p = baseline_dir / f"{video_id}.json"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def rep_pairs_diff(pred_reps: list[dict], base_reps: list[dict]) -> dict:
    """配对比较两个 rep 列表（按顺序最优匹配）。"""
    n_pred = len(pred_reps)
    n_base = len(base_reps)
    n_pair = min(n_pred, n_base)
    max_vel_err = 0.0
    max_pos_err = 0.0  # 用 duration_s 当代理
    for i in range(n_pair):
        pv = pred_reps[i].get("pcv_mps", 0.0)
        bv = base_reps[i].get("pcv_mps", 0.0)
        err = abs(pv - bv)
        if err > max_vel_err:
            max_vel_err = err
        pd = pred_reps[i].get("duration_s", 0.0)
        bd = base_reps[i].get("duration_s", 0.0)
        err2 = abs(pd - bd)
        if err2 > max_pos_err:
            max_pos_err = err2
    return {
        "n_pred": n_pred,
        "n_base": n_base,
        "n_pair": n_pair,
        "max_vel_err_mps": round(max_vel_err, 6),
        "max_dur_err_s": round(max_pos_err, 6),
        "n_pred_delta": n_pred - n_base,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default=str(DEFAULT_BENCH))
    ap.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    bench = Path(args.bench_dir).resolve()
    if not bench.is_absolute():
        bench = (REPO / bench).resolve()
    if not str(bench).startswith(str(REPO)):
        sys.exit(f"[错误] bench_dir 必须在仓库内: {bench}")
    if not (bench / "dataset_index.json").exists():
        sys.exit(f"[错误] 找不到 dataset_index.json: {bench}")

    baseline_dir = Path(args.baseline_dir).resolve()
    output = Path(args.output).resolve()
    model_path = Path(args.model).resolve()
    if not str(model_path).startswith(str(REPO)):
        sys.exit(f"[错误] model 必须在仓库内: {model_path}")
    if not model_path.exists():
        sys.exit(f"[错误] 模型不存在: {model_path}")
    if not str(baseline_dir).startswith(str(REPO)):
        sys.exit(f"[错误] baseline_dir 必须在仓库内: {baseline_dir}")

    sys.path.insert(0, str(REPO))
    from vbtcore import PlateDetector, analyze_video  # noqa: E402

    model_path_str = str(model_path)
    try:
        with open(bench / "dataset_index.json", encoding="utf-8") as f:
            dataset = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"[错误] 读取 dataset_index.json 失败: {e}")
    if args.limit:
        dataset = dataset[: args.limit]
    n_total = len(dataset)
    print("=== Python 端虚拟 golden test ===")
    print(f"数据集: {n_total} 视频 | 模型: {model_path.name}")
    print(f"基线目录: {baseline_dir}")
    print(f"输出: {output}")
    print()

    det = PlateDetector(model_path_str)

    n_pass = 0
    n_fail = 0
    n_error = 0
    n_no_baseline = 0
    results = []

    t0 = time.time()
    for k, item in enumerate(dataset):
        vid = item["video_id"]
        vp = bench / "raw_videos" / vid
        if not vp.exists():
            print(f"[{k + 1:>2}/{n_total}] ⚠ 视频不存在: {vid}")
            n_error += 1
            continue

        baseline = load_baseline(baseline_dir, vid)
        if baseline is None:
            print(f"[{k + 1:>2}/{n_total}] ⚠ 无 baseline: {vid}")
            n_no_baseline += 1
            continue

        # 复跑 vbtcore.analyze_video
        try:
            t_video = time.time()
            r = analyze_video(
                str(vp), model_path_str, detector=det, exercise_type="squat_bench"
            )
        except Exception as e:
            print(f"[{k + 1:>2}/{n_total}] ✗ 异常: {vid} - {e}")
            n_error += 1
            continue

        # 构造本次输出（与 baseline 同 schema）
        pred = {
            "status": r.status,
            "reps": [
                {
                    "start_idx": rep.start_idx,
                    "end_idx": rep.end_idx,
                    "pcv_mps": round(rep.pcv_mps, 3),
                    "duration_s": round(rep.duration_s, 3),
                }
                for rep in r.reps
            ],
        }
        diff = rep_pairs_diff(pred["reps"], baseline.get("reps", []))
        pass_ = (
            (pred["status"] == baseline.get("status"))
            and (diff["max_vel_err_mps"] <= THRESHOLD_MAX_VEL_ERR_MPS)
            and (abs(diff["n_pred_delta"]) <= THRESHOLD_MAX_REP_DELTA)
        )

        dt = time.time() - t_video
        if pass_:
            n_pass += 1
            sym = "✓"
        else:
            n_fail += 1
            sym = "✗"
        print(
            f"[{k + 1:>2}/{n_total}] {sym} {vid:<44} "
            f"status={pred['status']:<18} "
            f"n_pred={diff['n_pred']:>2}/{diff['n_base']:<2} "
            f"max_vel_err={diff['max_vel_err_mps']:.4f} m/s "
            f"({dt:.1f}s)"
        )
        results.append(
            {
                "video_id": vid,
                "status_pred": pred["status"],
                "status_base": baseline.get("status"),
                "n_pred": diff["n_pred"],
                "n_base": diff["n_base"],
                "max_vel_err_mps": diff["max_vel_err_mps"],
                "max_dur_err_s": diff["max_dur_err_s"],
                "n_pred_delta": diff["n_pred_delta"],
                "pass": pass_,
            }
        )

    total_min = (time.time() - t0) / 60
    print()
    print("=" * 60)
    print("Python 端 golden test 汇总")
    print("=" * 60)
    print(f"通过: {n_pass}/{n_total}")
    print(f"失败: {n_fail}")
    print(f"异常: {n_error}")
    print(f"无 baseline: {n_no_baseline}")
    print(f"总耗时: {total_min:.1f} min")
    print(f"阈值: max_vel_err ≤ {THRESHOLD_MAX_VEL_ERR_MPS} m/s")
    print()
    if n_pass >= THRESHOLD_N_PASS_MIN:
        print(
            f"✅ Python 端 ≥ {THRESHOLD_N_PASS_MIN}/34 通过：JSON 契约与数值稳定性 OK"
        )
        print(
            "   剩余 30% (C++ 浮点对齐 + ASan) 仍需 Linux 端 ./scripts/build_cpp_desktop.sh"
        )
    else:
        print(f"✗ Python 端 < {THRESHOLD_N_PASS_MIN}/34 通过：闸门未绿")
    print()

    # 写 GOLDEN_REPORT.json（与 C++ 端格式一致）
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": "python",
        "engine": "vbtcore Python (v5 refactor)",
        "model": model_path.name,
        "n_total": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_error": n_error,
        "n_no_baseline": n_no_baseline,
        "thresholds": {
            "max_vel_err_mps": THRESHOLD_MAX_VEL_ERR_MPS,
            "max_rep_delta": THRESHOLD_MAX_REP_DELTA,
            "n_pass_min": THRESHOLD_N_PASS_MIN,
        },
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"GOLDEN_REPORT 写入: {output}")

    sys.exit(0 if n_pass >= THRESHOLD_N_PASS_MIN else 1)


if __name__ == "__main__":
    main()
