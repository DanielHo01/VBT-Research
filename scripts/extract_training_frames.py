#!/usr/bin/env python3
"""
scripts/extract_training_frames.py — 从 22 条非 hold-out 视频均匀抽帧（训练扩充）

策略：按时长加权均匀采样，不依赖启发式分类（gym 视频条件单一）。
增强阶段再模拟暗光/遮挡/多片场景。

用法:
    python scripts/extract_training_frames.py [--n-frames 450]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC_VIDEOS = REPO / "validation/dataset_benchmark/raw_videos"
INDEX_PATH = REPO / "validation/dataset_benchmark/dataset_index.json"
OUT_DIR = REPO / "datasets/barbell_dataset/phase0_expansion"


def sample_video_uniform(
    video_path: str,
    n_samples: int,
) -> list[tuple[int, np.ndarray, float, float]]:
    """
    在视频中均匀采样 n_samples 帧，返回 (frame_idx, frame, brightness, variance)。
    等距分段，每段首帧（避免相邻帧冗余）。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = max(n_frames, 1)

    step = n_frames / n_samples if n_samples < n_frames else 1.0
    samples = []
    pos = 0.0
    while len(samples) < n_samples and int(pos) < n_frames:
        fi = int(pos)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            variance = float(gray.var())
            samples.append((fi, frame.copy(), brightness, variance))
        pos += step
        pos = round(pos)
    cap.release()
    return samples


def main():
    ap = argparse.ArgumentParser(description="从非 hold-out 视频均匀抽帧供训练扩充")
    ap.add_argument("--n-frames", type=int, default=450, help="目标总帧数（默认: 450）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（默认: 42）")
    ap.add_argument("--dry-run", action="store_true", help="仅统计，不写入文件")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 加载非 hold-out 视频
    with open(INDEX_PATH) as f:
        dataset = json.load(f)
    non_ho = [
        e
        for e in dataset
        if not e.get("hold_out", False)
        and e.get("video_id") != "20kg_0.87_0.88_0.89_0.91.mp4"
    ]
    print(f"非 hold-out 视频: {len(non_ho)} 条")

    # 统计每视频时长
    video_info = []
    for e in non_ho:
        vid = e["video_id"]
        path = SRC_VIDEOS / vid
        if not path.exists():
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            continue
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        video_info.append(
            {
                "video_id": vid,
                "n_frames": n_frames,
                "fps": fps,
                "duration_s": n_frames / fps if fps > 0 else 0,
            }
        )

    total_frames = sum(v["n_frames"] for v in video_info)
    total_duration = sum(v["duration_s"] for v in video_info)
    print(f"总帧数: {total_frames} ({total_duration:.0f}s)")

    # 按时长加权分配
    n_target = args.n_frames
    per_video = {}
    for v in video_info:
        share = v["n_frames"] / total_frames
        per_video[v["video_id"]] = max(1, int(round(share * n_target)))

    # 微调使总和 ≈ 目标
    while sum(per_video.values()) < n_target - 5:
        biggest = max(per_video, key=per_video.get)
        per_video[biggest] += 1
    while sum(per_video.values()) > n_target + 5:
        smallest = min(per_video, key=per_video.get)
        if per_video[smallest] > 1:
            per_video[smallest] -= 1

    print(f"目标: {n_target} 帧，实际分配: {sum(per_video.values())}")

    # 采样
    all_selected = []
    if not args.dry_run:
        (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    for v in video_info:
        vid = v["video_id"]
        path = SRC_VIDEOS / vid
        n_alloc = per_video.get(vid, 1)
        samples = sample_video_uniform(str(path), n_alloc)

        for fi, frame, brightness, variance in samples:
            all_selected.append(
                {
                    "video_id": vid,
                    "frame_idx": fi,
                    "category": "uniform",
                    "image_path": str(
                        OUT_DIR / f"images/{vid.replace('.mp4', '')}_f{fi:05d}.jpg"
                    ),
                    "brightness": brightness,
                    "variance": variance,
                    "n_frames_total": v["n_frames"],
                }
            )
            if not args.dry_run:
                out_name = f"{vid.replace('.mp4', '')}_f{fi:05d}.jpg"
                cv2.imwrite(str(OUT_DIR / "images" / out_name), frame)

    print(f"\n总抽取: {len(all_selected)} 帧")

    # 亮度分布
    brights = [s["brightness"] for s in all_selected]
    variances = [s["variance"] for s in all_selected]
    print(
        f"亮度: min={min(brights):.0f}  p10={np.percentile(brights, 10):.0f}  "
        f"median={np.median(brights):.0f}  p90={np.percentile(brights, 90):.0f}  max={max(brights):.0f}"
    )
    print(
        f"方差: min={min(variances):.0f}  p10={np.percentile(variances, 10):.0f}  "
        f"median={np.median(variances):.0f}  p90={np.percentile(variances, 90):.0f}  max={max(variances):.0f}"
    )

    # 写入清单
    if not args.dry_run:
        manifest = {
            "generated": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            "n_frames": len(all_selected),
            "seed": args.seed,
            "frames": all_selected,
        }
        manifest_path = OUT_DIR / "extracted_frames.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # YAML split
        yaml_path = OUT_DIR / "extraction_split.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("# Uniform frame extraction for Phase 0 expansion\n")
            f.write(f"# Total frames: {len(all_selected)}\n\n")
            f.write("train:\n")
            for item in all_selected:
                f.write(f"  - {item['image_path']}\n")
        print(f"帧清单: {manifest_path}")
        print(f"YAML: {yaml_path}")


if __name__ == "__main__":
    main()
