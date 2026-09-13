"""
scripts/gen_cpp_baseline.py — 为 C++ golden_test 生成 per-video baseline JSON
================================================================================
对 34 个开发集视频，逐个跑 vbtcore.analyze_video()，输出
validation/reports/cpp_baseline/<video_id>.json 供 C++ golden_test 对比。

JSON 字段（与 vbtcore-cpp/src/analyze.cpp::analyze_video_json() 一致）：
  - status: OK | NO_PLATE_DETECTED | NO_CLEAN_SEGMENT | ...
  - fps: float
  - mpp: float
  - reps: [{start_idx, end_idx, start_time, end_time, duration_s, rom_m, mcv_mps, pcv_mps}]
  - diagnostics: {n_frames, n_yolo, elapsed_s, ...}

用法：
    python scripts/gen_cpp_baseline.py
    # 或指定视频子集：
    python scripts/gen_cpp_baseline.py --only 30kg,50kg
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BENCH = REPO / "validation" / "dataset_benchmark"
BASELINE_DIR = REPO / "validation" / "reports" / "cpp_baseline"
MODEL = REPO / "models" / "best.onnx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default=str(DEFAULT_BENCH))
    ap.add_argument("--only", default=None,
                    help="只跑文件名包含任一子串的视频（逗号分隔）")
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--limit", type=int, default=None,
                    help="最多跑 N 个视频（用于快速验证）")
    args = ap.parse_args()

    bench = Path(args.bench_dir).resolve()
    if not bench.is_absolute():
        bench = (REPO / bench).resolve()
    # 路径遍历防护：限定在仓库根目录内
    if not str(bench).startswith(str(REPO)):
        sys.exit(f"[错误] 基准目录必须位于仓库内: {bench}")
    if not (bench / "dataset_index.json").exists():
        sys.exit(f"[错误] 找不到基准索引: {bench / 'dataset_index.json'}")

    model_path = Path(args.model).resolve()
    if not model_path.is_absolute():
        model_path = (REPO / model_path).resolve()
    if not str(model_path).startswith(str(REPO)):
        sys.exit(f"[错误] 模型路径必须位于仓库内: {model_path}")
    if not Path(model_path).exists():
        sys.exit(f"[错误] 模型不存在: {model_path}")

    sys.path.insert(0, str(REPO))
    from vbtcore import PlateDetector, analyze_video  # noqa: E402

    model_path_str = str(model_path)
    try:
        with open(bench / "dataset_index.json", encoding="utf-8") as f:
            dataset = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"[错误] 无法读取基准索引 {bench / 'dataset_index.json'}: {e}")
    if args.only:
        subs = [s.strip() for s in args.only.split(",") if s.strip()]
        dataset = [d for d in dataset if any(s in d["video_id"] for s in subs)]
        print(f"--only {subs} → {len(dataset)} 个视频")
    if args.limit:
        dataset = dataset[:args.limit]
        print(f"--limit {args.limit} → {len(dataset)} 个视频")
    print(f"共 {len(dataset)} 个视频 | 模型: {Path(model_path).name}")

    det = PlateDetector(model_path_str)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for k, item in enumerate(dataset):
        vid = item["video_id"]
        vp = bench / "raw_videos" / vid
        if not vp.exists():
            print(f"[{k+1:>2}/{len(dataset)}] ⚠ 视频不存在，跳过: {vid}")
            continue
        t_video = time.time()
        try:
            r = analyze_video(str(vp), model_path_str, detector=det,
                              exercise_type="squat_bench")
        except Exception as e:
            print(f"[{k+1:>2}/{len(dataset)}] ✗ 异常: {vid} - {e}")
            continue

        # 序列化为 C++ 端 golden_test 期望的 JSON 格式
        out = {
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
            },
        }
        out_path = BASELINE_DIR / f"{vid}.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        dt = time.time() - t_video
        print(f"[{k+1:>2}/{len(dataset)}] {vid:<44} status={r.status:<18} "
              f"reps={len(r.reps):>2}/{len(item.get('gt_reps_mcv', [])):<2} "
              f"elapsed={dt:.1f}s → {out_path.name}")

    total = time.time() - t0
    print(f"\n✓ 全部 {len(dataset)} 个视频已写入 {BASELINE_DIR}")
    print(f"  总耗时: {total:.1f}s ({total/60:.1f} min)")
    print("\n下一步：把 validation/reports/cpp_baseline/*.json 拷贝到 Linux 桌面端，")
    print("       与 C++ golden_test 输出的 JSON 做 max_vel_err 偏差比对。")


if __name__ == "__main__":
    main()
