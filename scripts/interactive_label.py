"""interactive_label.py — M3 全帧标注工具 v3
===============================================
相对 v2（Phase 1.5）的关键修正：
  - 存全帧 + 全框 YOLO 标签（v2 存的是单目标 crop，训不了检测器）；
  - 零硬编码路径（v2 写死 D:\\EasyVBT-Research）；
  - proposer 懒加载（v2 import 即载 OWL-ViT，无权重直接崩）；
  - 新增关键帧插值（--interpolate）与进度（--status）模式。

标注语义（docs/M3_TRAINING_PLAN.md）：所有可见片全标（含背景片堆），
可见 <50% 的半截片不标，框贴边。

用法（Windows）：
    # 队列模式（推荐）：按挖掘队列逐帧标，断点续标
    python scripts/interactive_label.py --mine datasets/mining/queue_r0.json

    # 单视频模式：显式帧号 / 等步采样
    python scripts/interactive_label.py --video 110kg_0.45_0.48_0.33.mp4 --frames 0,150,300
    python scripts/interactive_label.py --video 50kg_0.89_1.09_1.12_1.11.mp4 --stride 100 --max-frames 12

    # 关键帧插值（无 GUI）：两人工关键帧之间自动填框，标 auto 待复核
    python scripts/interactive_label.py --interpolate --max-gap 90

    # 进度（无 GUI）
    python scripts/interactive_label.py --status --mine datasets/mining/queue_r0.json

GUI 操作：
    拖拽        画新框（自动 accepted）
    单击框      选中 + pending→accepted（一键确认）
    A / R       全部接受 / 删除全部 pending
    D           删除选中框
    N           本帧 accepted 框复制到下一帧（pending，静止片堆/微调工作片）
    S / Space   保存并停留 / 保存并下一帧（B 上一帧，均自动保存）
    Q / ESC     保存并退出
proposer：--proposer none（默认）/ yolo（现模型低阈值提候选）/ owlvit。
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from label_common import (clip_box_pixels, interpolate_keyframes,  # noqa: E402
                          load_manifest, load_mining_queue, pixels_to_yolo,
                          save_manifest, yolo_to_pixels)


# ── 帧队列 ───────────────────────────────────────────────────────

def build_queue(args) -> list[dict]:
    """返回 [{"video","frame","reason"}]。"""
    if args.mine:
        q = load_mining_queue(args.mine)
        items = q["items"][args.start:args.start + args.count
                           if args.count else None]
        return [{"video": it["video"], "frame": it["frame"],
                 "reason": ",".join(it["reasons"])} for it in items]
    if not args.video:
        raise SystemExit("需 --mine 或 --video 其一")
    if args.frames:
        frames = [int(x) for x in args.frames.split(",") if x.strip() != ""]
    else:
        cap = cv2.VideoCapture(str(Path(args.bench_dir) / "raw_videos" / args.video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()
        frames = list(range(0, n, args.stride))[:args.max_frames]
    return [{"video": args.video, "frame": f, "reason": "manual"} for f in frames]


def img_stem(video: str, frame: int) -> str:
    return f"{Path(video).stem}_f{frame:05d}"


def read_frame(bench_dir: str, video: str, frame: int):
    cap = cv2.VideoCapture(str(Path(bench_dir) / "raw_videos" / video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    cap.release()
    if not ret:
        raise SystemExit(f"读帧失败: {video} f{frame}")
    return img


# ── proposer（懒加载） ───────────────────────────────────────────

class Proposer:
    def __init__(self, kind: str, args):
        self.kind = kind
        self.args = args
        self._yolo = None
        self._owl = None

    def propose(self, img: np.ndarray) -> list[dict]:
        if self.kind == "none":
            return []
        if self.kind == "yolo":
            return self._propose_yolo(img)
        if self.kind == "owlvit":
            return self._propose_owlvit(img)
        raise SystemExit(f"未知 proposer: {self.kind}")

    def _propose_yolo(self, img):
        if self._yolo is None:
            from vbtcore.detector import PlateDetector  # noqa: PLC0415
            self._yolo = PlateDetector(self.args.model)
        dets = self._yolo.detect(img, self.args.proposer_conf)
        dets.sort(key=lambda d: -d.conf)
        return [{"cx": d.cx, "cy": d.cy, "w": d.w, "h": d.h,
                 "state": "pending", "auto": False}
                for d in dets[:self.args.proposer_max]]

    def _propose_owlvit(self, img):
        if self._owl is None:
            import os as _os  # noqa: PLC0415
            if self.args.offline:
                _os.environ["HF_HUB_OFFLINE"] = "1"
                _os.environ["TRANSFORMERS_OFFLINE"] = "1"
            import torch  # noqa: PLC0415
            from transformers import (AutoImageProcessor, AutoTokenizer,  # noqa: PLC0415
                                      OwlViTForObjectDetection)
            print("加载 OWL-ViT（离线）..." if self.args.offline else "加载 OWL-ViT...",
                  flush=True)
            self._TEXTS = ["a weight plate", "barbell plate", "gym weight",
                           "a red plate", "a blue plate", "a yellow plate",
                           "a green plate", "a white plate"]
            self._owl = (
                OwlViTForObjectDetection.from_pretrained(
                    "google/owlvit-base-patch32",
                    local_files_only=self.args.offline),
                AutoImageProcessor.from_pretrained(
                    "google/owlvit-base-patch32",
                    local_files_only=self.args.offline),
                AutoTokenizer.from_pretrained(
                    "google/owlvit-base-patch32",
                    local_files_only=self.args.offline),
                torch,
            )
            print("OWL-ViT 就绪", flush=True)
        model, processor, tokenizer, torch = self._owl
        model.eval()
        H, W = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = processor(images=rgb, return_tensors="pt")
        txt_in = tokenizer(self._TEXTS, padding=True, return_tensors="pt")
        with torch.no_grad():
            out = model(pixel_values=img_in["pixel_values"],
                        input_ids=txt_in["input_ids"],
                        attention_mask=txt_in["attention_mask"])
        res = processor.post_process_object_detection(
            out, target_sizes=torch.tensor([[H, W]]),
            threshold=self.args.proposer_conf)[0]
        boxes = []
        for i in range(len(res["scores"])):
            box = res["boxes"][i].numpy()
            bw, bh = float(box[2] - box[0]), float(box[3] - box[1])
            ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 999
            if ratio <= 1.5:
                boxes.append({"cx": float(box[0] + bw / 2),
                              "cy": float(box[1] + bh / 2),
                              "w": bw, "h": bh,
                              "state": "pending", "auto": False})
        boxes.sort(key=lambda b: 0)  # 保持检出顺序
        return boxes[:self.args.proposer_max]


# ── 存取 ─────────────────────────────────────────────────────────

def save_frame(out_dir: Path, video: str, frame: int, img: np.ndarray,
               boxes: list[dict], manifest: dict, proposer: str,
               auto: bool = False) -> int:
    """存全帧 jpg + YOLO txt + 更新 manifest。返回 accepted 框数。"""
    img_dir, lbl_dir = out_dir / "images", out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    stem = img_stem(video, frame)
    H, W = img.shape[:2]
    accepted = [b for b in boxes if b["state"] == "accepted"]
    cv2.imwrite(str(img_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lines = []
    for b in accepted:
        clip = clip_box_pixels(b["cx"], b["cy"], b["w"], b["h"], W, H)
        if clip is None:
            continue
        lines.append(pixels_to_yolo(0, *clip, W, H))
    (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                         encoding="utf-8")
    manifest["images"][f"{stem}.jpg"] = {
        "video": video, "frame": frame, "n_boxes": len(lines),
        "auto": auto, "proposer": proposer,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_manifest(out_dir / "manifest.json", manifest)
    return len(lines)


def load_frame_boxes(out_dir: Path, video: str, frame: int) -> list[dict] | None:
    """已标帧 → 框列表（None = 未标过）。"""
    stem = img_stem(video, frame)
    img_p, lbl_p = out_dir / "images" / f"{stem}.jpg", out_dir / "labels" / f"{stem}.txt"
    if not lbl_p.exists():
        return None
    img = cv2.imread(str(img_p)) if img_p.exists() else None
    if img is None:  # 图像缺失则以视频帧尺寸为准（调用方保证同视频同尺寸）
        return []
    H, W = img.shape[:2]
    boxes = []
    for line in lbl_p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            _, cx, cy, w, h = yolo_to_pixels(line, W, H)
            boxes.append({"cx": cx, "cy": cy, "w": w, "h": h,
                          "state": "accepted", "auto": False})
    return boxes


# ── GUI ──────────────────────────────────────────────────────────

COLORS = {"pending": (0, 200, 255), "accepted": (0, 255, 0)}
AUTO_COLOR = (255, 0, 255)


class GuiState:
    def __init__(self):
        self.boxes: list[dict] = []
        self.selected: int = -1
        self.drawing: tuple | None = None
        self.scale = 1.0
        self.dirty = True


def draw_gui(img, st: GuiState, info: str):
    H, W = img.shape[:2]
    disp = cv2.resize(img, (int(W * st.scale), int(H * st.scale)))
    for i, b in enumerate(st.boxes):
        x1 = int((b["cx"] - b["w"] / 2) * st.scale)
        y1 = int((b["cy"] - b["h"] / 2) * st.scale)
        x2 = int((b["cx"] + b["w"] / 2) * st.scale)
        y2 = int((b["cy"] + b["h"] / 2) * st.scale)
        color = AUTO_COLOR if b.get("auto") else COLORS[b["state"]]
        thick = 4 if i == st.selected else 2
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, thick)
        cv2.putText(disp, f"#{i + 1}", (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if st.drawing:
        (x0, y0), (x1, y1) = st.drawing
        cv2.rectangle(disp, (x0, y0), (x1, y1), (255, 255, 255), 2)
    bar_h = 56
    cv2.rectangle(disp, (0, 0), (disp.shape[1], bar_h), (20, 20, 20), -1)
    cv2.putText(disp, info[:150], (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(disp, "drag=draw click=accept A=all R=del-pend D=del-sel N=copy-next S/Space=save B=back Q=quit"[:150],
                (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    return disp


def on_mouse(event, x, y, flags, param: GuiState):
    st = param
    if event == cv2.EVENT_LBUTTONDOWN:
        st.drawing = ((x, y), (x, y))
    elif event == cv2.EVENT_MOUSEMOVE and st.drawing and (flags & cv2.EVENT_FLAG_LBUTTON):
        st.drawing = (st.drawing[0], (x, y))
        st.dirty = True
    elif event == cv2.EVENT_LBUTTONUP and st.drawing:
        (x0, y0), _ = st.drawing
        st.drawing = None
        if abs(x - x0) < 5 and abs(y - y0) < 5:
            # 单击：选中最近框并一键接受
            best, bd = -1, 1e9
            for i, b in enumerate(st.boxes):
                cx, cy = b["cx"] * st.scale, b["cy"] * st.scale
                d = abs(x - cx) + abs(y - cy)
                if d < bd:
                    best, bd = i, d
            if best >= 0:
                b = st.boxes[best]
                if (abs(x - b["cx"] * st.scale) < b["w"] * st.scale / 2 + 12
                        and abs(y - b["cy"] * st.scale) < b["h"] * st.scale / 2 + 12):
                    st.selected = best
                    b["state"] = "accepted"
                    b["auto"] = False
                else:
                    st.selected = -1
            else:
                st.selected = -1
        else:
            # 拖拽：新框（≥8px），直接 accepted
            x1, x2 = sorted([x0, x])
            y1, y2 = sorted([y0, y])
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                s = st.scale
                st.boxes.append({"cx": (x1 + x2) / 2 / s, "cy": (y1 + y2) / 2 / s,
                                 "w": (x2 - x1) / s, "h": (y2 - y1) / s,
                                 "state": "accepted", "auto": False})
                st.selected = len(st.boxes) - 1
        st.dirty = True


def gui_label(args, queue: list[dict]) -> int:
    out_dir = Path(args.out)
    manifest = load_manifest(out_dir / "manifest.json")
    proposer = Proposer(args.proposer, args)
    # 断点续标：跳过已标（--redo 则重访）
    labeled = set(manifest["images"])
    todo = []
    for q in queue:
        if not args.redo and f"{img_stem(q['video'], q['frame'])}.jpg" in labeled:
            continue
        todo.append(q)
    print(f"队列 {len(queue)} 帧，已标跳过 {len(queue) - len(todo)}，待标 {len(todo)}")
    if not todo:
        print("无待标帧（加 --redo 重访已标）。")
        return 0

    win = "M3 Labeler (全帧YOLO)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    frame_cache: dict[tuple[str, int], np.ndarray] = {}

    def get_frame(video, frame):
        key = (video, frame)
        if key not in frame_cache:
            frame_cache[key] = read_frame(args.bench_dir, video, frame)
            if len(frame_cache) > 8:
                frame_cache.pop(next(iter(frame_cache)))
        return frame_cache[key]

    # 预载已有框/候选
    box_cache: dict[tuple[str, int], list[dict]] = {}

    def get_boxes(video, frame):
        key = (video, frame)
        if key not in box_cache:
            old = load_frame_boxes(out_dir, video, frame)
            if old is not None:
                for b in old:
                    b["auto"] = False
                box_cache[key] = old
            else:
                box_cache[key] = proposer.propose(get_frame(video, frame))
        return box_cache[key]

    st = GuiState()
    cv2.setMouseCallback(win, on_mouse, st)
    idx = 0
    total_saved = 0
    t0 = time.time()
    while 0 <= idx < len(todo):
        q = todo[idx]
        img = get_frame(q["video"], q["frame"])
        if st.dirty or getattr(st, "_key", None) != (q["video"], q["frame"]):
            st.boxes = get_boxes(q["video"], q["frame"])
            st._key = (q["video"], q["frame"])
            st.selected = -1
            H, W = img.shape[:2]
            st.scale = min(1280 / W, 720 / H)
        n_a = sum(1 for b in st.boxes if b["state"] == "accepted")
        n_p = len(st.boxes) - n_a
        info = (f"[{idx + 1}/{len(todo)}] {q['video']} f{q['frame']} "
                f"reason={q['reason']} | acc={n_a} pend={n_p} | "
                f"saved={total_saved} {(time.time() - t0) / 60:.1f}min")
        cv2.imshow(win, draw_gui(img, st, info))
        st.dirty = False
        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            continue
        if key in (ord("q"), 27):
            total_saved += save_frame(out_dir, q["video"], q["frame"], img,
                                      st.boxes, manifest, args.proposer)
            break
        elif key == ord("a"):
            for b in st.boxes:
                b["state"] = "accepted"
                b["auto"] = False
            st.dirty = True
        elif key == ord("r"):
            st.boxes = [b for b in st.boxes if b["state"] != "pending"]
            box_cache[(q["video"], q["frame"])] = st.boxes
            st.selected = -1
            st.dirty = True
        elif key == ord("d") and 0 <= st.selected < len(st.boxes):
            st.boxes.pop(st.selected)
            st.selected = -1
            st.dirty = True
        elif key == ord("n") and idx + 1 < len(todo):
            nxt = todo[idx + 1]
            copied = [{"cx": b["cx"], "cy": b["cy"], "w": b["w"], "h": b["h"],
                       "state": "pending", "auto": False}
                      for b in st.boxes if b["state"] == "accepted"]
            if copied:
                box_cache[(nxt["video"], nxt["frame"])] = (
                    box_cache.get((nxt["video"], nxt["frame"]), []) + copied)
                print(f"  复制 {len(copied)} 框 → 下一帧")
        elif key == ord("s"):
            n = save_frame(out_dir, q["video"], q["frame"], img,
                           st.boxes, manifest, args.proposer)
            total_saved += 1
            print(f"  保存 {img_stem(q['video'], q['frame'])} ({n} 框)")
        elif key == ord(" "):
            save_frame(out_dir, q["video"], q["frame"], img,
                       st.boxes, manifest, args.proposer)
            total_saved += 1
            idx += 1
            st.dirty = True
        elif key == ord("b"):
            save_frame(out_dir, q["video"], q["frame"], img,
                       st.boxes, manifest, args.proposer)
            idx = max(0, idx - 1)
            st.dirty = True
    cv2.destroyAllWindows()
    el = (time.time() - t0) / 60
    print(f"本轮保存 {total_saved} 帧，用时 {el:.1f} 分钟 "
          f"（{el / max(1, total_saved):.1f} min/帧）")
    return 0


# ── 插值 / 进度（无 GUI） ────────────────────────────────────────

def cmd_interpolate(args) -> int:
    out_dir = Path(args.out)
    manifest = load_manifest(out_dir / "manifest.json")
    # 按视频收集人工关键帧
    by_video: dict[str, list[tuple[int, list]]] = {}
    for img_name, meta in manifest["images"].items():
        if meta.get("auto"):
            continue
        boxes = load_frame_boxes(out_dir, meta["video"], meta["frame"])
        if boxes is None:
            continue
        px = [(b["cx"], b["cy"], b["w"], b["h"]) for b in boxes
              if b["state"] == "accepted"]
        by_video.setdefault(meta["video"], []).append((meta["frame"], px))
    n_filled = 0
    for video, keys in sorted(by_video.items()):
        keys.sort()
        if len(keys) < 2:
            continue
        for (k0, b0), (k1, b1) in zip(keys, keys[1:]):
            if k1 - k0 - 1 <= 0 or k1 - k0 - 1 > args.max_gap:
                continue
            filled = interpolate_keyframes(k0, b0, k1, b1)
            for f, px_boxes in filled.items():
                stem = img_stem(video, f)
                if f"{stem}.jpg" in manifest["images"]:
                    continue  # 永不覆盖人工
                img = read_frame(args.bench_dir, video, f)
                boxes = [{"cx": x, "cy": y, "w": w, "h": h,
                          "state": "accepted", "auto": True}
                         for x, y, w, h in px_boxes]
                save_frame(out_dir, video, f, img, boxes, manifest,
                           proposer="interpolate", auto=True)
                n_filled += 1
        print(f"  {video}: {len(keys)} 关键帧")
    print(f"插值填充 {n_filled} 帧（auto=true，请用 GUI 复核：--redo 重访）")
    return 0


def cmd_status(args) -> int:
    out_dir = Path(args.out)
    manifest = load_manifest(out_dir / "manifest.json")
    n_man = sum(1 for m in manifest["images"].values() if not m.get("auto"))
    n_auto = sum(1 for m in manifest["images"].values() if m.get("auto"))
    n_box = sum(m.get("n_boxes", 0) for m in manifest["images"].values())
    print(f"已标：人工 {n_man} 帧 / 自动 {n_auto} 帧 / 共 {n_box} 框")
    if args.mine:
        q = load_mining_queue(args.mine)
        queued = {f"{img_stem(it['video'], it['frame'])}.jpg" for it in q["items"]}
        done = queued & set(manifest["images"])
        print(f"队列覆盖：{len(done)}/{len(queued)}（{args.mine}）")
    by_video: dict[str, int] = {}
    for m in manifest["images"].values():
        by_video[m["video"]] = by_video.get(m["video"], 0) + 1
    for v, c in sorted(by_video.items()):
        print(f"  {v}: {c} 帧")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="M3 全帧标注工具 v3")
    ap.add_argument("--bench-dir", default="validation/dataset_benchmark")
    ap.add_argument("--mine", default=None, help="挖掘队列 json")
    ap.add_argument("--video", default=None)
    ap.add_argument("--frames", default=None, help="逗号帧号")
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0, help="0=全部")
    ap.add_argument("--redo", action="store_true", help="重访已标帧")
    ap.add_argument("--out", default="datasets/interactive_labels")
    ap.add_argument("--proposer", default="none", choices=["none", "yolo", "owlvit"])
    ap.add_argument("--model", default="models/yolo11_plate.onnx")
    ap.add_argument("--proposer-conf", type=float, default=0.15)
    ap.add_argument("--proposer-max", type=int, default=20)
    ap.add_argument("--offline", action="store_true", default=True,
                    help="OWL-ViT 离线（默认开）")
    ap.add_argument("--online", action="store_false", dest="offline")
    ap.add_argument("--interpolate", action="store_true")
    ap.add_argument("--max-gap", type=int, default=90)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        return cmd_status(args)
    if args.interpolate:
        return cmd_interpolate(args)
    queue = build_queue(args)
    print(f"待标队列 {len(queue)} 帧，proposer={args.proposer}，输出 {args.out}")
    return gui_label(args, queue)


if __name__ == "__main__":
    sys.exit(main())
