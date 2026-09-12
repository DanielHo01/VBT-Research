#!/usr/bin/env python3
"""
scripts/annotate_absent.py — 标注目标不在画面的帧区间（lead-in / lead-out）

用法:
    # 查看视频总帧数
    python scripts/annotate_absent.py list

    # 标注 absent 区间
    python scripts/annotate_absent.py edit 102.5kg_0.53_0.38.mp4
    python scripts/annotate_absent.py edit 110kg_0.45_0.48_0.33.mp4
    ...

    # 导出 absent_segments.json（供 coverage_audit.py 使用）
    python scripts/annotate_absent.py export

输出:
    validation/dataset_benchmark/absent_segments.json
    格式: {"video_id": [[start, end], ...], ...}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
INDEX_PATH = REPO / "validation/dataset_benchmark" / "dataset_index.json"
ANNOTATION_PATH = REPO / "validation/dataset_benchmark" / "absent_segments.json"
VIDEO_DIR = REPO / "validation/dataset_benchmark" / "raw_videos"


def load_annotations() -> dict[str, list[tuple[int, int]]]:
    if ANNOTATION_PATH.exists():
        with open(ANNOTATION_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        return {k: [tuple(seg) for seg in v] for k, v in raw.items()}
    return {}


def save_annotations(data: dict[str, list[tuple[int, int]]]) -> None:
    out = {k: [list(seg) for seg in v] for k, v in data.items()}
    ANNOTATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATION_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"已保存: {ANNOTATION_PATH}")


def list_videos() -> None:
    with open(INDEX_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    annotations = load_annotations()
    print(f"\n{'视频ID':<55} {'absent区间数':>8} {'总帧':>6}")
    print("-" * 75)
    for item in dataset:
        vid = item["video_id"]
        path = VIDEO_DIR / vid
        if not path.exists():
            continue
        cap = cv2.VideoCapture(str(path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        segs = annotations.get(vid, [])
        print(f"{vid:<55} {len(segs):>8} {n_frames:>6}")
    print(f"\n共 {len(dataset)} 条视频，{len(annotations)} 条已标注")


def edit_video(vid: str) -> None:
    """交互式标注 absent 区间（lead-in / lead-out）。"""
    path = VIDEO_DIR / vid
    if not path.exists():
        print(f"[错误] 视频不存在: {path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    data = load_annotations()
    current = data.get(vid, [])
    print(f"\n{'=' * 60}")
    print(f"视频: {vid}")
    print(f"总帧: {n_frames}  ({n_frames / fps:.1f}s @ {fps:.0f}fps)")
    print(f"当前标注: {current}")
    print(f"{'=' * 60}")
    print("\n操作:")
    print("  [s,start]  设置区间起点")
    print("  [e,end]    设置区间终点（需先设起点）")
    print("  [a]        确认添加当前区间 [start, end]")
    print("  [d,N]      删除第N个区间")
    print("  [l,N]      跳到第N帧预览")
    print("  [p]        播放/暂停（逐帧浏览）")
    print("  [q]        保存并退出")

    annotations = list(current)
    seg_start = None
    playing = False

    cap = cv2.VideoCapture(str(path))
    frame_idx = 0
    current_frame = None

    while True:
        # 更新帧显示
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            frame_idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                print("[错误] 无法读取视频")
                break

        # 标注区间标签
        label_parts = []
        for i, (s, e) in enumerate(annotations):
            if s <= frame_idx <= e:
                label_parts.append(f"IN_ABSENT[{i}]({s}–{e})")
        in_absent = "; ".join(label_parts) if label_parts else ""

        canvas = frame.copy()
        H, W = canvas.shape[:2]

        # 显示信息
        info = f"F{frame_idx}/{n_frames - 1}  ({frame_idx / fps:.2f}s)  {in_absent}"
        if seg_start is not None:
            info += f"  [PENDING: {seg_start}–__]"
        cv2.putText(
            canvas, info[:80], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )

        # 如果在 absent 区间内，整帧红框
        if any(s <= frame_idx <= e for s, e in annotations):
            cv2.rectangle(canvas, (5, 5), (W - 5, H - 5), (0, 0, 255), 3)
            cv2.putText(
                canvas,
                "ABSENT",
                (W // 2 - 60, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )

        cv2.imshow(f"annotate: {vid}", canvas)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key in (ord("s"), ord("1")):
            seg_start = frame_idx
            print(f"  起点 → frame {frame_idx}")
        elif key in (ord("e"), ord("2")) and seg_start is not None:
            seg_end = frame_idx
            if seg_end < seg_start:
                seg_start, seg_end = seg_end, seg_start
            print(f"  终点 → frame {seg_end}  区间 [{seg_start}, {seg_end}]")
        elif key in (ord("a"), ord("3")) and seg_start is not None:
            seg_end = frame_idx
            if seg_end < seg_start:
                seg_start, seg_end = seg_end, seg_start
            annotations.append((seg_start, seg_end))
            print(f"  ✓ 添加区间 [{seg_start}, {seg_end}]  (共{len(annotations)}个)")
            seg_start = None
        elif key == ord("d"):
            print(f"  当前区间: {list(enumerate(annotations))}")
            # 找下一个数字输入
        elif chr(key) in "0123456789" and key != 255:
            # 简单：d+数字组合
            pass
        elif key in (ord("l"), ord("j")):
            # 下一步让用户输入帧号
            pass
        elif key == 83:  # 右箭头
            frame_idx = min(frame_idx + 1, n_frames - 1)
        elif key == 81:  # 左箭头
            frame_idx = max(frame_idx - 1, 0)
        elif key == 83 and playing:
            frame_idx = min(frame_idx + 1, n_frames - 1)
        elif playing:
            frame_idx = min(frame_idx + 1, n_frames - 1)
            if frame_idx == n_frames - 1:
                playing = False
                frame_idx = 0

    cap.release()
    cv2.destroyAllWindows()

    data[vid] = annotations
    save_annotations(data)
    print(f"\n已保存 {vid}: {annotations}")


def export_json() -> None:
    data = load_annotations()
    if not data:
        print("[警告] 没有标注数据，导出空文件")
    save_annotations(data)
    print("\n供 coverage_audit.py 使用:")
    print(f"  python scripts/coverage_audit.py --absent-segments {ANNOTATION_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description="标注视频中目标不在画面的区间")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出所有视频及其标注状态")
    sub.add_parser("export", help="导出 absent_segments.json")

    edit = sub.add_parser("edit", help="编辑单条视频的 absent 区间")
    edit.add_argument("video_id", help="视频文件名，如 102.5kg_0.53_0.38.mp4")

    args = ap.parse_args()

    if args.cmd == "list":
        list_videos()
    elif args.cmd == "edit":
        edit_video(args.video_id)
    elif args.cmd == "export":
        export_json()


if __name__ == "__main__":
    main()
