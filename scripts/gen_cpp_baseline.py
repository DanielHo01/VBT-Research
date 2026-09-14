"""
scripts/gen_cpp_baseline.py — 为 C++ golden_test 生成 per-video baseline JSON
================================================================================
对开发集视频逐个跑 vbtcore.analyze_video()（stride=1 全量检测），输出
validation/reports/cpp_baseline/<video_id>.json 供 C++ golden_test 对比。

第一阶段（Make it Right）规范：
  · stride=1 逐帧全量 YOLO 检测（redet_every=1），不使用任何稀疏近似
  · 正确 letterbox 逆变换（vbtcore.geometry.canvas_to_orig）
  · 起始静止期中位数标定 + 全局冻结 mpp（严禁动作中动态刷新）
  · 真实 PTS 时间戳（严禁写死 dt=1/fps）
  · 生理速度区间核验：MCV ∈ [0.3, 1.5] m/s，越界视频显式告警

JSON 字段（与 vbtcore-cpp/src/analyze.cpp::analyze_video_json() 一致）：
  - status: OK | NO_PLATE_DETECTED | NO_CLEAN_SEGMENT | ...
  - fps: float
  - mpp: float
  - reps: [{start_idx, end_idx, start_time, end_time, duration_s, rom_m, mcv_mps, pcv_mps}]
  - diagnostics: {n_frames, n_yolo, elapsed_s, ...}

用法：
    # 5 视频快速冒烟（研发调试，先验证逻辑）
    python scripts/gen_cpp_baseline.py --smoke

    # 全量 34 视频，4 进程并发
    python scripts/gen_cpp_baseline.py --workers 4

    # 指定子集
    python scripts/gen_cpp_baseline.py --only 30kg,50kg

    # 跨盘 I/O 提速（WSL2：先把视频复制到 Linux 原生盘再跑）
    python scripts/gen_cpp_baseline.py --copy-to-tmp --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BENCH = REPO / "validation" / "dataset_benchmark"
BASELINE_DIR = REPO / "validation" / "reports" / "cpp_baseline"
MODEL = REPO / "models" / "best.onnx"

# ── 生理速度区间（深蹲向心平均速度的合理物理范围）────────────────
MCV_MIN_PHYSIOLOGICAL = 0.30
MCV_MAX_PHYSIOLOGICAL = 1.50

# ── 5 视频冒烟集：覆盖空杆 / 轻片 / 中片 / 多片 / 极限负荷 ────────
SMOKE_SET = [
    "20kg_0.87_0.88_0.89_0.91",  # 空杆 → 应为 NO_PLATE_DETECTED（真阴性）
    "30kg_1.03_0.89_0.76_0.65",  # 轻片高速
    "80kg_0.88_0.88_0.94_0.90",  # 中片
    "110kg_0.71_0.73",  # 多片堆叠（mpp 冻结的关键回归用例）
    "140kg_0.41",  # 极限负荷低速
]

# 每个 worker 进程复用的检测器（避免每个视频重建 ONNX session）
_WORKER_DETECTOR = None


def _get_detector(model_path: str):
    """每进程惰性初始化并复用 PlateDetector。"""
    global _WORKER_DETECTOR
    if _WORKER_DETECTOR is None:
        from vbtcore import PlateDetector

        _WORKER_DETECTOR = PlateDetector(model_path)
    return _WORKER_DETECTOR


def _serialize(vid: str, r) -> dict:
    """SetResult → C++ golden_test 期望的 JSON 结构。"""
    return {
        "video": vid,
        "status": r.status,
        "fps": round(r.fps, 2),
        "mpp": round(r.mpp, 6) if r.mpp is not None else None,
        "reps": [
            {
                "start_idx": rep.start_idx,
                "end_idx": rep.end_idx,
                "start_time": round(rep.start_time, 3),
                "end_time": round(rep.end_time, 3),
                "duration_s": round(rep.duration_s, 3),
                "rom_m": round(rep.rom_m, 4),
                "mcv_mps": round(rep.mcv_mps, 3),
                "pcv_mps": round(rep.pcv_mps, 3),
            }
            for rep in r.reps
        ],
        "diagnostics": {
            "n_frames": r.diagnostics.get("n_frames", 0),
            "n_yolo": r.diagnostics.get("n_yolo_frames", 0),
            "elapsed_s": round(r.diagnostics.get("elapsed_s", 0.0), 2),
            "coverage": round(r.diagnostics.get("coverage", 0.0), 3),
            "ms_per_frame": round(r.diagnostics.get("ms_per_frame", 0.0), 1),
            "yolo_ratio": round(r.diagnostics.get("yolo_ratio", 0.0), 3),
            "plate_diameter_m": r.diagnostics.get("plate_diameter_m", 0.45),
            "exercise_type": r.diagnostics.get("exercise_type", "squat_bench"),
            "stride": r.diagnostics.get("stride", 1),
        },
    }


def _check_physiological(out: dict, gt_reps: list) -> list[str]:
    """核验 MCV 是否落在生理速度区间，返回告警列表。"""
    warns: list[str] = []
    mcvs = [rep["mcv_mps"] for rep in out["reps"]]
    for i, v in enumerate(mcvs):
        if not (MCV_MIN_PHYSIOLOGICAL <= v <= MCV_MAX_PHYSIOLOGICAL):
            warns.append(f"rep{i + 1} MCV={v:.3f} 越界 [{MCV_MIN_PHYSIOLOGICAL}, {MCV_MAX_PHYSIOLOGICAL}]")
    if out["status"] == "OK" and gt_reps and len(mcvs) != len(gt_reps):
        warns.append(f"计数 {len(mcvs)} ≠ 真值 {len(gt_reps)}")
    return warns


def _run_one(task: tuple) -> dict:
    """worker 入口：分析单个视频并写出 baseline JSON。"""
    vid, video_path, model_path, stride, out_dir, gt_reps = task
    sys.path.insert(0, str(REPO))
    from vbtcore import analyze_video

    t0 = time.time()
    try:
        det = _get_detector(model_path)
        r = analyze_video(
            video_path,
            model_path,
            redet_every=stride,
            detector=det,
            exercise_type="squat_bench",
        )
    except Exception as e:  # noqa: BLE001 — worker 需把异常回传主进程
        return {"video": vid, "ok": False, "error": f"{type(e).__name__}: {e}"}

    out = _serialize(vid, r)
    out["diagnostics"]["stride"] = stride
    Path(out_dir, f"{vid}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "video": vid,
        "ok": True,
        "status": out["status"],
        "mpp": out["mpp"],
        "n_reps": len(out["reps"]),
        "n_gt": len(gt_reps),
        "mcvs": [rep["mcv_mps"] for rep in out["reps"]],
        "warns": _check_physiological(out, gt_reps),
        "elapsed": round(time.time() - t0, 1),
    }


def _resolve_inside_repo(raw: str, label: str) -> Path:
    p = Path(raw)
    p = p.resolve() if p.is_absolute() else (REPO / p).resolve()
    if not str(p).startswith(str(REPO)):
        sys.exit(f"[错误] {label} 必须位于仓库内: {p}")
    return p


def main():
    ap = argparse.ArgumentParser(description="生成 C++ golden_test 的 Python baseline")
    ap.add_argument("--bench-dir", default=str(DEFAULT_BENCH))
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--out-dir", default=str(BASELINE_DIR))
    ap.add_argument("--only", default=None, help="只跑文件名包含任一子串的视频（逗号分隔）")
    ap.add_argument("--smoke", action="store_true", help="只跑 5 视频冒烟集（30 秒级）")
    ap.add_argument("--limit", type=int, default=None, help="最多跑 N 个视频")
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="YOLO 检测步长；第一阶段必须为 1（逐帧全量检测）",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="并发进程数（默认 min(4, CPU 核数)）",
    )
    ap.add_argument(
        "--copy-to-tmp",
        action="store_true",
        help="先把视频复制到本地 /tmp 再分析（消除 WSL2 9P 跨盘 I/O 瓶颈）",
    )
    args = ap.parse_args()

    bench = _resolve_inside_repo(args.bench_dir, "基准目录")
    if not (bench / "dataset_index.json").exists():
        sys.exit(f"[错误] 找不到基准索引: {bench / 'dataset_index.json'}")
    model_path = _resolve_inside_repo(args.model, "模型路径")
    if not model_path.exists():
        sys.exit(f"[错误] 模型不存在: {model_path}")
    out_dir = _resolve_inside_repo(args.out_dir, "输出目录")

    if args.stride != 1:
        print(
            f"⚠ 警告：stride={args.stride} ≠ 1。第一阶段（Make it Right）要求逐帧全量检测，"
            "稀疏检测属于第二阶段提速手段，其结果不得作为真值 baseline。"
        )

    try:
        with open(bench / "dataset_index.json", encoding="utf-8") as f:
            dataset = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"[错误] 无法读取基准索引: {e}")

    if args.smoke:
        dataset = [d for d in dataset if any(s in d["video_id"] for s in SMOKE_SET)]
        print(f"--smoke → {len(dataset)} 个视频（空杆/轻片/中片/多片/极限）")
    if args.only:
        subs = [s.strip() for s in args.only.split(",") if s.strip()]
        dataset = [d for d in dataset if any(s in d["video_id"] for s in subs)]
        print(f"--only {subs} → {len(dataset)} 个视频")
    if args.limit:
        dataset = dataset[: args.limit]
        print(f"--limit {args.limit} → {len(dataset)} 个视频")

    if not dataset:
        sys.exit("[错误] 筛选后没有任何视频")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 可选：复制到本地 tmp 盘，消除跨盘 I/O ────────────────────
    tmp_dir = None
    src_root = bench / "raw_videos"
    if args.copy_to_tmp:
        tmp_dir = Path(tempfile.mkdtemp(prefix="vbt_videos_"))
        print(f"复制视频到本地盘 {tmp_dir} …")
        for item in dataset:
            src = src_root / item["video_id"]
            if src.exists():
                shutil.copy2(src, tmp_dir / item["video_id"])
        src_root = tmp_dir

    tasks = []
    for item in dataset:
        vid = item["video_id"]
        vp = src_root / vid
        if not vp.exists():
            print(f"⚠ 视频不存在，跳过: {vid}")
            continue
        tasks.append(
            (vid, str(vp), str(model_path), args.stride, str(out_dir), item.get("gt_reps_mcv", []))
        )

    workers = max(1, min(args.workers, len(tasks)))
    print(
        f"共 {len(tasks)} 个视频 | 模型 {model_path.name} | stride={args.stride} | "
        f"{workers} 进程并发 | 输出 {out_dir}"
    )
    print("-" * 100)

    t0 = time.time()
    results: list[dict] = []
    try:
        if workers == 1:
            for t in tasks:
                results.append(_run_one(t))
                _print_row(results[-1], len(results), len(tasks))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_run_one, t): t[0] for t in tasks}
                for k, fut in enumerate(as_completed(futures), 1):
                    res = fut.result()
                    results.append(res)
                    _print_row(res, k, len(tasks))
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    total = time.time() - t0
    _summary(results, out_dir, total)


def _print_row(res: dict, k: int, n: int) -> None:
    if not res["ok"]:
        print(f"[{k:>2}/{n}] ✗ {res['video']:<48} {res['error']}")
        return
    mcv_str = ", ".join(f"{v:.3f}" for v in res["mcvs"]) or "-"
    flag = "⚠" if res["warns"] else "✓"
    print(
        f"[{k:>2}/{n}] {flag} {res['video']:<48} {res['status']:<18} "
        f"reps={res['n_reps']}/{res['n_gt']:<2} mpp={res['mpp']} "
        f"mcv=[{mcv_str}] {res['elapsed']}s"
    )
    for w in res["warns"]:
        print(f"          ↳ {w}")


def _summary(results: list[dict], out_dir: Path, total: float) -> None:
    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    statuses: dict[str, int] = {}
    for r in ok:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1

    all_mcvs = [v for r in ok for v in r["mcvs"]]
    out_of_range = [v for v in all_mcvs if not (MCV_MIN_PHYSIOLOGICAL <= v <= MCV_MAX_PHYSIOLOGICAL)]
    count_pass = sum(1 for r in ok if abs(r["n_reps"] - r["n_gt"]) <= 1 or (r["n_gt"] == 0))
    # 空杆真阴性：无片视频返回 NO_PLATE_DETECTED 记为正确拒绝
    correct_reject = sum(
        1 for r in ok if r["status"] == "NO_PLATE_DETECTED" and "20kg" in r["video"]
    )

    print("-" * 100)
    print(f"✓ 完成 {len(ok)}/{len(results)} 个视频 → {out_dir}")
    print(f"  总耗时: {total:.1f}s ({total / 60:.1f} min)")
    print(f"  状态分布: {statuses}")
    print(f"  计数通过(±1): {count_pass}/{len(ok)} ｜ 空杆正确拒绝: {correct_reject}")
    print(
        f"  rep 总数: {len(all_mcvs)} ｜ 生理区间外 MCV: {len(out_of_range)} "
        f"（区间 [{MCV_MIN_PHYSIOLOGICAL}, {MCV_MAX_PHYSIOLOGICAL}] m/s）"
    )
    if out_of_range:
        print(f"    越界值: {[round(v, 3) for v in sorted(out_of_range)]}")
    if failed:
        print(f"  ✗ 失败 {len(failed)} 个:")
        for r in failed:
            print(f"    - {r['video']}: {r['error']}")
    print("\n下一步：./scripts/build_cpp_desktop.sh 编译 C++，用 golden_test 比对本 baseline。")


if __name__ == "__main__":
    main()
