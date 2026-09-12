"""
视频帧提取脚本 — 适配 EzYOLO 标注流程

功能:
  1. 均匀抽帧 (每 N 帧)
  2. 自动检测蹲底换向帧 (帧间差异最小点 = 停顿换向)
  3. 输出 EzYOLO 兼容的图片 + 可选 YOLO 标签格式

用法:
  python scripts/extract_frames_for_labeling.py
  python scripts/extract_frames_for_labeling.py --video-dir D:/path/to/videos --step 15 --motion-window 15
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── 配置 ────────────────────────────────────────────────────────────────────

DEFAULT_VIDEO_DIR = Path(r"D:\EasyVBT-Research\videos")
DEFAULT_OUTPUT_DIR = Path(r"D:\EasyVBT-Research\datasets\barbell_dataset\images")
DEFAULT_LABEL_DIR = Path(r"D:\EasyVBT-Research\datasets\barbell_dataset\labels")
DEFAULT_STEP = 15  # 每 N 帧抽 1 张
DEFAULT_MOTION_WINDOW = 15  # 运动分析窗口大小 (帧)
DEFAULT_MIN_MOTION = 500  # 帧间差异阈值 (低于此值视为"静止/换向")
DEFAULT_BOTTOM_FRAMES_PER_VIDEO = 5  # 每视频额外保留的蹲底帧数


# ─── 核心函数 ────────────────────────────────────────────────────────────────


def compute_frame_motion(video_path: str, step: int = 15) -> np.ndarray:
    """
    计算每 step 帧处的帧间差异 (运动能量)。
    低运动能量 = 杠铃在换向/停顿 = 蹲底候选位置。

    Returns:
        motion_scores: shape (N,) 每 step 处的运动能量
        frame_indices: shape (N,) 对应的原始帧编号
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    prev_gray = None
    motion_scores = []
    frame_indices = []

    frame_idx = 0
    sampled_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 缩放加速计算
        gray_small = cv2.resize(gray, (w // 4, h // 4))

        if prev_gray is not None:
            diff = cv2.absdiff(gray_small, prev_gray)
            score = float(diff.mean())  # 0-255

            # 只在 step 整数倍处记录
            if frame_idx % step == 0:
                motion_scores.append(score)
                frame_indices.append(frame_idx)

        prev_gray = gray_small
        frame_idx += 1

    cap.release()

    if not motion_scores:
        return np.array([]), np.array([])

    return np.array(motion_scores), np.array(frame_indices)


def find_squat_bottom_frames(
    motion_scores: np.ndarray,
    frame_indices: np.ndarray,
    fps: float,
    n_frames: int = DEFAULT_BOTTOM_FRAMES_PER_VIDEO,
    min_gap_s: float = 0.8,
) -> np.ndarray:
    """
    从运动能量序列中找到蹲底换向帧 (能量局部最小点)。

    策略:
      - 取每段低能量区间的谷底
      - 相邻谷底之间至少间隔 min_gap_s 秒

    Returns:
        原始帧编号数组
    """
    if len(motion_scores) < 5:
        return np.array([])

    scores = motion_scores.copy()
    # 中值归一化，找相对低点
    median_score = float(np.median(scores))
    if median_score == 0:
        return np.array([])

    # 归一化到 0-1
    norm_scores = scores / median_score

    # 用简单的谷检测: 比左右邻居都低
    bottoms = []
    window = max(3, int(min_gap_s * fps / DEFAULT_STEP))

    for i in range(window, len(norm_scores) - window):
        left_min = norm_scores[i - window : i].min()
        right_min = norm_scores[i + 1 : i + window + 1].min()
        if norm_scores[i] <= left_min and norm_scores[i] <= right_min:
            # 必须是相对低点 (低于中值的 60%)
            if norm_scores[i] < 0.6:
                # 和上一个已选帧至少隔 window 帧
                if not bottoms or (i - bottoms[-1]) >= window:
                    bottoms.append(i)

    if len(bottoms) == 0:
        return np.array([])

    # 如果找到太多，取能量最低的 n_frames 个
    if len(bottoms) > n_frames * 2:
        bottom_scores = [(i, norm_scores[i]) for i in bottoms]
        bottom_scores.sort(key=lambda x: x[1])
        selected = [bottoms.index(b) for _, b in bottom_scores[:n_frames]]
        bottoms = sorted(selected)

    return frame_indices[np.array(bottoms)]


def extract_regular_frames(total_frames: int, step: int) -> np.ndarray:
    """均匀抽帧: 0, step, 2*step, ..."""
    return np.arange(0, total_frames, step)


def extract_and_save_frames(
    video_path: str,
    output_dir: Path,
    step: int = DEFAULT_STEP,
    motion_window: int = DEFAULT_MOTION_WINDOW,
    extra_bottom_frames: int = DEFAULT_BOTTOM_FRAMES_PER_VIDEO,
    save_all: bool = True,
) -> dict:
    """
    抽帧主逻辑:
      1. 均匀抽帧 (每 step 帧)
      2. 蹲底帧检测
      3. 合并 + 去重 + 保存

    Returns:
        stats dict
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    video_name = Path(video_path).stem
    out_dir = output_dir / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 均匀抽帧
    regular_indices = extract_regular_frames(total_frames, step)
    log.info(f"  [{video_name}] 均匀抽帧: {len(regular_indices)} 张 (step={step})")

    # Step 2: 蹲底帧检测
    motion_scores, sampled_indices = compute_frame_motion(video_path, step=step)
    if len(motion_scores) > 0:
        bottom_indices = find_squat_bottom_frames(
            motion_scores, sampled_indices, fps, n_frames=extra_bottom_frames
        )
    else:
        bottom_indices = np.array([])

    log.info(f"  [{video_name}] 蹲底帧: {len(bottom_indices)} 张")
    log.debug(f"  蹲底帧位置: {bottom_indices.tolist()}")

    # Step 3: 合并 + 去重
    all_indices = np.unique(np.concatenate([regular_indices, bottom_indices]))
    all_indices.sort()

    # 过滤首尾各留 5 帧 (架上准备 + 卸片)
    all_indices = all_indices[all_indices >= 5]
    all_indices = all_indices[all_indices <= total_frames - 5]

    # Step 4: 提取并保存
    cap = cv2.VideoCapture(video_path)
    saved_count = 0
    for seq_idx, frame_idx in enumerate(all_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # 文件名格式: {video_name}_{frame:06d}_r{regular_flag}_b{bottom_flag}.jpg
        is_regular = frame_idx in regular_indices
        is_bottom = frame_idx in bottom_indices
        tag = ""
        if not is_regular and is_bottom:
            tag = "_BOTTOM"  # 仅蹲底帧
        elif is_regular and is_bottom:
            tag = "_RB"  # 既是均匀帧也是蹲底帧

        filename = f"{video_name}_{int(frame_idx):06d}{tag}.jpg"
        save_path = out_dir / filename

        # 降噪压缩保存 (质量 85%, 节省空间)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        cv2.imwrite(str(save_path), frame, encode_params)
        saved_count += 1

    cap.release()

    stats = {
        "video": video_name,
        "total_frames": total_frames,
        "fps": fps,
        "regular_frames": len(regular_indices),
        "bottom_frames": len(bottom_indices),
        "saved_frames": saved_count,
        "output_dir": str(out_dir),
    }
    log.info(f"  [{video_name}] 已保存 {saved_count} 张 → {out_dir}")
    return stats


def generate_report(all_stats: list[dict], report_path: Path):
    """生成抽帧报告 (CSV 格式，方便审核)"""
    import csv

    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "fps",
                "total_frames",
                "regular_frames",
                "bottom_frames",
                "saved_frames",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(all_stats)

    total_saved = sum(s["saved_frames"] for s in all_stats)
    total_frames = sum(s["total_frames"] for s in all_stats)
    log.info(
        f"\n抽帧完成: {len(all_stats)} 条视频, {total_saved} 张图片 (总帧数 {total_frames})"
    )
    log.info(f"报告: {report_path}")


def create_yolo_dataset_structure(
    output_dir: Path, label_dir: Path, train_ratio: float = 0.85
):
    """
    创建 YOLO 数据集标准目录结构:
      datasets/barbell_dataset/
        images/train/
        images/val/
        labels/train/
        labels/val/
        data.yaml
    """
    for split in ["train", "val"]:
        (output_dir.parent / "images" / split).mkdir(parents=True, exist_ok=True)
        (label_dir.parent / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 移动图片到 train/val (按比例)
    all_videos = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    np.random.seed(42)
    indices = np.random.permutation(len(all_videos))
    split_idx = int(len(all_videos) * train_ratio)
    train_videos = [all_videos[i] for i in indices[:split_idx]]
    val_videos = [all_videos[i] for i in indices[split_idx:]]

    import shutil

    for video_dir in train_videos:
        for img in video_dir.glob("*.jpg"):
            shutil.move(str(img), output_dir.parent / "images" / "train" / img.name)
    for video_dir in val_videos:
        for img in video_dir.glob("*.jpg"):
            shutil.move(str(img), output_dir.parent / "images" / "val" / img.name)

    # 生成 data.yaml
    dataset_yaml = output_dir.parent / "data.yaml"
    yaml_content = f"""\
path: {output_dir.parent.as_posix()}
train: images/train
val: images/val

nc: 1
names:
  0: plate

# 标注规则: plate (视野中所有可见铃片，不区分内外)
# 生成时间: 自动抽帧
"""
    with open(dataset_yaml, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    log.info(
        f"数据集结构:\n"
        f"  train: {len(train_videos)} 视频, "
        f"val: {len(val_videos)} 视频\n"
        f"  data.yaml: {dataset_yaml}"
    )


# ─── 主入口 ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="视频帧提取 — 适配 EzYOLO 标注")
    parser.add_argument(
        "--video-dir", type=Path, default=DEFAULT_VIDEO_DIR, help="视频目录"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="图片输出目录"
    )
    parser.add_argument(
        "--step", type=int, default=DEFAULT_STEP, help="每 N 帧抽 1 张 [默认 15]"
    )
    parser.add_argument(
        "--bottom-frames",
        type=int,
        default=DEFAULT_BOTTOM_FRAMES_PER_VIDEO,
        help="每视频额外保留的蹲底帧数 [默认 5]",
    )
    parser.add_argument(
        "--motion-window",
        type=int,
        default=DEFAULT_MOTION_WINDOW,
        help="运动分析窗口大小 [默认 15]",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅扫描不保存")
    args = parser.parse_args()

    video_files = sorted(args.video_dir.glob("*.mp4"))
    if not video_files:
        log.error(f"未找到 .mp4 文件: {args.video_dir}")
        sys.exit(1)

    log.info(f"找到 {len(video_files)} 条视频:")
    for v in video_files:
        log.info(f"  {v.name}")
    log.info("")

    all_stats = []
    for video_path in tqdm(video_files, desc="抽帧进度"):
        if args.dry_run:
            cap = cv2.VideoCapture(str(video_path))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            saved = n // args.step + args.bottom_frames
            all_stats.append(
                {
                    "video": Path(video_path).stem,
                    "fps": 30,
                    "total_frames": n,
                    "regular_frames": n // args.step,
                    "bottom_frames": args.bottom_frames,
                    "saved_frames": saved,
                    "output_dir": str(args.output_dir / Path(video_path).stem),
                }
            )
            log.info(f"  [DRY RUN] {Path(video_path).stem}: ~{saved} 张")
        else:
            stats = extract_and_save_frames(
                str(video_path),
                args.output_dir,
                step=args.step,
                motion_window=args.motion_window,
                extra_bottom_frames=args.bottom_frames,
            )
            all_stats.append(stats)

    # 生成报告
    if not args.dry_run:
        report_path = args.output_dir.parent / "extraction_report.csv"
        generate_report(all_stats, report_path)

        log.info("\n是否创建 YOLO 数据集划分? (85% train / 15% val)")
        log.info("按回车跳过（仅抽帧），或输入 'y' 创建:")

        user_input = input("> ").strip().lower()
        if user_input == "y":
            create_yolo_dataset_structure(args.output_dir, DEFAULT_LABEL_DIR)


if __name__ == "__main__":
    main()
