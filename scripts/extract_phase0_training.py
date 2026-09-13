#!/usr/bin/env python3
"""
scripts/extract_phase0_training.py — Phase 0 扩充训练数据抽帧

从 22 条非 hold-out 视频抽取 400-500 帧，用于重训 YOLOv11n。
配额（分层采样）:
  - 底位 (eccentric bottom): ≥ 80 帧
  - 暗光 (low brightness):   ≥ 40 帧
  - 遮挡 (partial occlusion): ≥ 40 帧
  - 多片 (multi-plate):     ≥ 60 帧
  - 正常帧: 余量填满到 ~400 帧

策略:
  1. 读 coverage JSON 获取每帧检测结果 (primary box 位置/conf)
  2. 用 Phase 0 detector 重新跑（或复用已有 JSON）
  3. 按配额采样

用法:
  python scripts/extract_phase0_training.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ─── 配额常量 ────────────────────────────────────────────────────────────────
QUOTA = {
    "bottom": 80,
    "low_light": 40,
    "occlusion": 40,
    "multi_plate": 60,
}
TOTAL_TARGET = 420  # 目标总帧数

# ─── 阈值常量 ────────────────────────────────────────────────────────────────
BOTTOM_Y_THRESHOLD_RATIO = 0.65  # 主框 cy > h * ratio → 底位
BRIGHTNESS_DARK_THRESHOLD = 80  # 帧平均亮度 < threshold → 暗光
BRIGHTNESS_NORM_THRESHOLD = 160  # 帧平均亮度 > threshold → 正常光
OCCLUSION_CONF_THRESHOLD = 0.15  # conf < threshold → 可能是遮挡
MULTI_PLATE_COUNT = 2  # ≥ 2 个候选框 → 多片

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── 核心函数 ────────────────────────────────────────────────────────────────


def classify_frame(
    frame: np.ndarray,
    candidates: list[dict],
    primary_idx: int,
) -> tuple[str, float]:
    """
    判定帧的类型。
    返回 (category, score)：
      - bottom:       主框在画面下方（cy > h * threshold）
      - low_light:    帧亮度低
      - occlusion:    主框 conf 低（遮挡特征）
      - multi_plate:  有多个候选框
      - normal:       其他
      - no_detection: 无检测
    """
    h, w = frame.shape[:2]

    # no detection
    if not candidates:
        return "no_detection", 0.0

    # occlusion: conf 极低
    if primary_idx >= 0:
        conf = candidates[primary_idx].get("conf", 1.0)
        if conf < OCCLUSION_CONF_THRESHOLD:
            return "occlusion", conf
    else:
        conf = candidates[0].get("conf", 1.0)
        if conf < OCCLUSION_CONF_THRESHOLD:
            return "occlusion", conf

    # multi_plate: 多个候选框
    if len(candidates) >= MULTI_PLATE_COUNT:
        return "multi_plate", len(candidates)

    # bottom: 主框在画面下方
    if primary_idx >= 0:
        cy = candidates[primary_idx].get("cy", h / 2)
        if cy > h * BOTTOM_Y_THRESHOLD_RATIO:
            return "bottom", cy / h

    # low_light: 帧亮度低
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    if brightness < BRIGHTNESS_DARK_THRESHOLD:
        return "low_light", brightness

    return "normal", brightness


def sample_frames_from_raw_json(
    raw_json_path: str,
    video_path: str,
    quota_per_video: dict[str, int] | None = None,
) -> list[dict]:
    """
    从 raw JSON 读取检测结果 + OpenCV 读帧，按配额分类采样。

    Returns:
        list of {frame_idx, pts_s, category, category_score, candidates, primary_idx, bbox}
    """
    with open(raw_json_path, encoding="utf-8") as f:
        data = json.load(f)

    frames_data = data.get("frames", [])
    fps = data.get("metadata", {}).get("fps", 30.0)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        log.warning(f"无法打开视频: {video_path}")
        return []

    samples = []
    per_cat = dict.fromkeys(
        ["bottom", "low_light", "occlusion", "multi_plate", "normal"], 0
    )

    for frame_entry in tqdm(
        frames_data, desc=Path(raw_json_path).stem[:30], leave=False
    ):
        frame_idx = frame_entry["frame_idx"]
        pts_s = frame_entry["pts_s"]
        candidates = frame_entry.get("candidates", [])
        primary_idx = frame_entry.get("primary_idx", -1)

        # 读帧（只读，不做额外检测）
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        category, score = classify_frame(frame, candidates, primary_idx)

        # 只采样有检测的帧
        if category == "no_detection":
            continue

        # 配额控制（每视频）
        max_per_video = {
            "bottom": 8,
            "low_light": 4,
            "occlusion": 4,
            "multi_plate": 6,
            "normal": 10,
        }

        if per_cat.get(category, 0) >= max_per_video.get(category, 10):
            continue

        per_cat[category] += 1
        samples.append(
            {
                "frame_idx": frame_idx,
                "pts_s": round(pts_s, 3),
                "category": category,
                "category_score": round(score, 3),
                "candidates": candidates,
                "primary_idx": primary_idx,
                "video_id": data.get("video_id", ""),
            }
        )

    cap.release()
    return samples


def extract_frames(
    video_path: str,
    sample_list: list[dict],
    output_dir: Path,
    save_overlay: bool = True,
) -> list[dict]:
    """
    从视频中提取指定帧，保存为图片 + overlay。
    返回保存记录。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning(f"无法打开视频: {video_path}")
        return []

    records = []
    colors = {
        "bottom": (0, 200, 255),  # 橙色
        "low_light": (128, 128, 255),  # 浅红
        "occlusion": (0, 100, 255),  # 蓝
        "multi_plate": (0, 255, 0),  # 绿
        "normal": (200, 200, 200),  # 灰
    }

    for sample in tqdm(sample_list, desc="extract", leave=False):
        frame_idx = sample["frame_idx"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        category = sample["category"]
        color = colors.get(category, (200, 200, 200))

        # 画所有候选框
        candidates = sample.get("candidates", [])
        primary_idx = sample.get("primary_idx", -1)
        for i, cand in enumerate(candidates):
            cx = cand["cx"]
            cy = cand["cy"]
            bw = cand["w"]
            bh = cand["h"]
            conf = cand.get("conf", 0.0)
            x1 = max(0, int(cx - bw / 2))
            y1 = max(0, int(cy - bh / 2))
            x2 = min(w, int(cx + bw / 2))
            y2 = min(h, int(cy + bh / 2))
            is_primary = i == primary_idx
            thick = 3 if is_primary else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            label = f"{conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # 类别标签
        cv2.putText(
            frame, category.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

        # 保存
        video_id = sample.get("video_id", "unknown")
        safe_id = "".join(
            c for c in video_id.replace(".mp4", "") if c.isalnum() or c in "-_"
        )
        fname = f"{safe_id}_f{frame_idx:05d}_{category}.jpg"
        out_path = output_dir / fname
        cv2.imwrite(str(out_path), frame)

        records.append(
            {
                "file": str(out_path),
                "frame_idx": frame_idx,
                "pts_s": sample["pts_s"],
                "category": category,
                "category_score": sample["category_score"],
                "video_id": video_id,
                "n_candidates": len(candidates),
                "primary_idx": primary_idx,
                "primary_conf": candidates[primary_idx]["conf"]
                if (primary_idx >= 0 and candidates)
                else 0.0,
            }
        )

    cap.release()
    return records


# ─── 主函数 ────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Phase 0 扩充训练数据抽帧")
    ap.add_argument(
        "--input-dir",
        default="validation/dataset_benchmark/raw_videos/",
        help="视频目录",
    )
    ap.add_argument(
        "--coverage-dir",
        default="validation/reports/coverage_phase0/",
        help="coverage JSON 目录",
    )
    ap.add_argument(
        "--output-dir",
        default="datasets/phase0_extra/images",
        help="输出图片目录",
    )
    ap.add_argument(
        "--target",
        type=int,
        default=TOTAL_TARGET,
        help=f"目标总帧数（默认 {TOTAL_TARGET}）",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    coverage_dir = Path(args.coverage_dir)
    output_dir = Path(args.output_dir)

    # 加载 dataset_index
    index_path = Path("validation/dataset_benchmark/dataset_index.json")
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    else:
        log.error(f"未找到 dataset_index: {index_path}")
        sys.exit(1)

    # 筛选 22 条非 hold-out 视频
    non_ho = [d for d in index if d.get("hold_out", False) is False]
    log.info(f"非 hold-out 视频: {len(non_ho)} 条")
    for d in non_ho:
        log.info(f"  {d['video_id']}")

    # 从 coverage JSON 采样
    all_samples: list[dict] = []
    for entry in tqdm(non_ho, desc="采样"):
        vid = entry["video_id"]
        video_path = input_dir / vid
        coverage_path = coverage_dir / f"{vid}_coverage.json"

        if not video_path.exists():
            log.warning(f"视频不存在: {video_path}")
            continue

        if not coverage_path.exists():
            log.warning(f"Coverage JSON 不存在: {coverage_path}")
            continue

        # 读取 raw JSON（逐帧检测结果）
        raw_path = coverage_dir / f"{vid}_phase0_raw.json"
        if raw_path.exists():
            samples = sample_frames_from_raw_json(str(raw_path), str(video_path))
        else:
            log.warning(f"Raw JSON 不存在: {raw_path}，跳过")
            continue

        all_samples.extend(samples)

    log.info(f"\n总采样帧数: {len(all_samples)}")
    cat_counts = {}
    for s in all_samples:
        cat_counts[s["category"]] = cat_counts.get(s["category"], 0) + 1
    for cat, cnt in sorted(cat_counts.items()):
        quota = QUOTA.get(cat, 0)
        status = "✓" if cnt >= quota else "✗"
        log.info(f"  {cat}: {cnt} / {quota} {status}")

    # 提取帧
    log.info(f"\n提取 {len(all_samples)} 帧到 {output_dir} ...")
    records = []
    for entry in tqdm(non_ho, desc="提取"):
        vid = entry["video_id"]
        video_path = input_dir / vid
        raw_path = coverage_dir / f"{vid}_phase0_raw.json"
        if not raw_path.exists():
            continue
        samples = sample_frames_from_raw_json(str(raw_path), str(video_path))
        recs = extract_frames(str(video_path), samples, output_dir)
        records.extend(recs)

    # 保存记录
    records_path = output_dir.parent / "extraction_records.json"
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    log.info(f"记录已保存: {records_path}")
    log.info(f"总帧数: {len(records)}")

    # 配额总结
    cat_final = {}
    for r in records:
        cat_final[r["category"]] = cat_final.get(r["category"], 0) + 1
    log.info("\n最终配额:")
    for cat, cnt in sorted(cat_final.items()):
        quota = QUOTA.get(cat, 0)
        status = "✓" if cnt >= quota else "✗"
        log.info(f"  {cat}: {cnt} / {quota} {status}")


if __name__ == "__main__":
    main()
