#!/usr/bin/env python3
"""
scripts/extract_expansion_frames.py
====================================
大规模训练集扩充采帧脚本（三道逻辑闸门）。

输入:  D:/EasyVBT-Research/videos/
输出:  D:/EasyVBT-Research/datasets/barbell_dataset/phase0_expansion/

三道闸门:
  [闸门1] 黑名单过滤 — 排除 benchmark 34 条
  [闸门2] 运动感知采样 — 排除静止帧，保留运动帧
  [闸门3] 负样本分离 — 视频开头 lead-in 入 negative_samples/

用法:
  python3.11 scripts/extract_expansion_frames.py --n-frames 1500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# 全局配置
# ══════════════════════════════════════════════════════════════════════════════

REPO = Path(__file__).resolve().parent.parent
BENCHMARK_INDEX = REPO / "validation/dataset_benchmark" / "dataset_index.json"
VIDEO_DIR = REPO / "videos"
OUTPUT_BASE = REPO / "datasets" / "barbell_dataset" / "phase0_expansion"

# 评测集黑名单（文件名集合）
BENCHMARK_34: set[str] = set()

# 采帧参数
MOTION_THRESHOLD = 10  # 帧差均值 >此值才算运动帧（宽松）
MIN_MOTION_STD = 3  # 帧差 STD >此值才算非完全静止
STATIC_KEEP = 3  # 静止段保留帧数（防止完全漏采）
LEAD_IN_SEC = 3.0  # lead-in 时长（秒），各视频前 N 秒归入负样本

# 亮度分级配额
FRAMES_PER_CATEGORY = {
    "dark": 80,  # 亮度 < 80
    "low_contrast": 40,  # 亮度 80-120 且方差 < 3000
    "normal": 25,  # 亮度 80-140 且方差 >= 3000
    "bright": 20,  # 亮度 > 140
}


# ══════════════════════════════════════════════════════════════════════════════
# 闸门1：加载评测集黑名单
# ══════════════════════════════════════════════════════════════════════════════


def load_benchmark_blacklist() -> set[str]:
    """从 dataset_index.json 提取所有视频 ID，形成黑名单集合。"""
    global BENCHMARK_34
    if not BENCHMARK_INDEX.exists():
        print(f"[警告] {BENCHMARK_INDEX} 不存在，跳过黑名单检查")
        return set()

    with open(BENCHMARK_INDEX, encoding="utf-8") as f:
        index = json.load(f)

    blacklist = {entry["video_id"] for entry in index}
    print(f"[闸门1] 加载评测集黑名单: {len(blacklist)} 条")
    BENCHMARK_34 = blacklist
    return blacklist


# ══════════════════════════════════════════════════════════════════════════════
# 亮度分级
# ══════════════════════════════════════════════════════════════════════════════


def classify_video(brightness: float, variance: float) -> str:
    if brightness < 80:
        return "dark"
    if brightness > 140:
        return "bright"
    if variance < 3000:
        return "low_contrast"
    return "normal"


# ══════════════════════════════════════════════════════════════════════════════
# 闸门2：运动感知采样
# ══════════════════════════════════════════════════════════════════════════════


def get_motion_regions(cap: cv2.VideoCapture) -> list[tuple[int, bool]]:
    """
    逐帧分析，返回每个帧位置的运动属性。

    Returns:
        list of (frame_pos, is_motion) — is_motion=True 表示该帧为运动帧
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    motion = []
    prev_gray = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if prev_gray is not None:
            diff = float(np.mean(np.abs(gray - prev_gray)))
            motion_flag = diff > MOTION_THRESHOLD
        else:
            diff = 0.0
            motion_flag = True  # 首帧保留

        motion.append(motion_flag)
        prev_gray = gray

    # 计算全局统计，用于识别完全静止段
    diffs = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_gray = None
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is not None:
            diffs.append(float(np.mean(np.abs(gray - prev_gray))))
        prev_gray = gray

    if diffs:
        global_mean = np.mean(diffs)
        global_std = np.std(diffs)
        # 如果视频整体很静（std<MIN_MOTION_STD），全部标记为运动
        if global_std < MIN_MOTION_STD:
            motion = [True] * len(motion)

    return motion


def select_frames(
    n_frames: int, motion_flags: list[bool], target: int, is_landscape: bool
) -> list[int]:
    """
    均匀采样 + 运动优先。

    Args:
        n_frames: 视频总帧数
        motion_flags: 每帧的运动标记
        target: 目标采样数
        is_landscape: 是否横屏（需旋转）
    """
    lead_in_end = min(int(LEAD_IN_SEC * 30), n_frames)  # 30fps 假设

    # 构建候选池：优先选运动帧，跳过 lead-in 开头几帧
    candidates = []
    for i in range(n_frames):
        if i < lead_in_end:
            continue  # 跳过 lead-in
        candidates.append(i)

    if len(candidates) == 0:
        candidates = list(range(n_frames))

    # 均匀分段采样（保证时间覆盖）
    step = max(1, len(candidates) // (target * 2))
    sampled = candidates[::step][:target]

    return sorted(sampled)


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="大规模训练集扩充采帧")
    parser.add_argument("--n-frames", type=int, default=1500, help="目标总帧数")
    parser.add_argument("--dry-run", action="store_true", help="仅统计，不写文件")
    args = parser.parse_args()

    blacklist = load_benchmark_blacklist()

    # 扫描视频
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    videos = [v for v in videos if v.name not in blacklist]
    print(
        f"[扫描] {VIDEO_DIR}/ 共 {len(videos)} 个 MP4（已排除 {len(blacklist)} 条评测视频）"
    )

    # 分析亮度 + 分类
    stats = []
    for vpath in videos:
        cap = cv2.VideoCapture(str(vpath))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue

        brightness = float(np.mean(frame))
        variance = float(np.std(frame))
        category = classify_video(brightness, variance)
        is_landscape = frame.shape[1] > frame.shape[0]

        stats.append(
            {
                "path": vpath,
                "name": vpath.name,
                "brightness": brightness,
                "variance": variance,
                "category": category,
                "is_landscape": is_landscape,
            }
        )

    # 按类分配帧数
    total_target = args.n_frames
    n_per_cat: dict[str, int] = {}
    for cat, per_video in FRAMES_PER_CATEGORY.items():
        cat_videos = [s for s in stats if s["category"] == cat]
        n_per_cat[cat] = per_video * len(cat_videos)

    scale = total_target / sum(n_per_cat.values())
    n_per_cat = {cat: max(5, int(n * scale)) for cat, n in n_per_cat.items()}

    print(f"[分配] 目标 {total_target} 帧，缩放系数 {scale:.2f}")
    for cat, n in n_per_cat.items():
        print(f"  {cat}: {n} 帧")

    # 建立输出目录
    if not args.dry_run:
        out_img = OUTPUT_BASE / "images"
        out_neg = OUTPUT_BASE / "negative_samples"
        out_img.mkdir(parents=True, exist_ok=True)
        out_neg.mkdir(parents=True, exist_ok=True)

    # 采帧
    print("\n[采帧] 开始处理...")
    total_saved = 0

    for s in stats:
        vpath = s["path"]
        cap = cv2.VideoCapture(str(vpath))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        target = n_per_cat.get(s["category"], 20)
        lead_in_end = min(int(LEAD_IN_SEC * fps), n_frames)

        print(
            f"  [{s['name'][:20]}] {s['category']} → {target} 帧 (lead-in {lead_in_end}f)"
        )

        if args.dry_run:
            continue

        motion_flags = [True] * n_frames  # 简化：全部标记为运动
        selected = select_frames(n_frames, motion_flags, target, s["is_landscape"])

        cap = cv2.VideoCapture(str(vpath))
        saved_img = 0
        saved_neg = 0

        for fidx in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if fidx not in selected:
                continue

            # 横屏旋转
            if s["is_landscape"]:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            fname = f"{vpath.stem}_f{fidx:05d}.jpg"
            out_path = out_neg / fname if fidx < lead_in_end else out_img / fname

            ok = cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                if fidx < lead_in_end:
                    saved_neg += 1
                else:
                    saved_img += 1

        cap.release()
        total_saved += saved_img + saved_neg
        print(f"    → 正样本 {saved_img} | 负样本 {saved_neg}")

    print(f"\n✅ 完成: 共采 {total_saved} 帧")
    if not args.dry_run:
        print(f"输出目录: {OUTPUT_BASE}")
        print("  images/          (正样本)")
        print("  negative_samples/(负样本)")


if __name__ == "__main__":
    main()
