"""label_common — M3 标注管线纯函数（stdlib only）
====================================================
设计约束：
  - 只用标准库，无第三方依赖 → tests/ 可直接单测，CI 零新增依赖。
  - 坐标约定：YOLO 文本行为归一化 (cls cx cy w h)；
    像素框为 (cx, cy, w, h) 四元组，中心点 + 宽高。

标注语义（docs/M3_TRAINING_PLAN.md 第二节）：
  - 所有可见片全标（含背景片堆），身份由引擎运动探针区分；
  - 可见面积 <50% 的半截片不标；
  - 框必须贴边（片高直接进 mpp 标定，松紧影响绝对速度）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path


# ── YOLO 文本行编解码 ─────────────────────────────────────────────

def pixels_to_yolo(cls: int, cx: float, cy: float, w: float, h: float,
                   W: int, H: int) -> str:
    """像素框 → YOLO 行（归一化，6 位小数）。退化框抛 ValueError。"""
    if W <= 0 or H <= 0:
        raise ValueError(f"图像尺寸非法: {W}x{H}")
    if w <= 0 or h <= 0:
        raise ValueError(f"退化框: w={w} h={h}")
    x = min(1.0, max(0.0, cx / W))
    y = min(1.0, max(0.0, cy / H))
    bw = min(1.0, max(0.0, w / W))
    bh = min(1.0, max(0.0, h / H))
    if bw <= 0 or bh <= 0:
        raise ValueError("框完全在画面外")
    return f"{int(cls)} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}"


def yolo_to_pixels(line: str, W: int, H: int) -> tuple[int, float, float, float, float]:
    """YOLO 行 → (cls, cx, cy, w, h) 像素。格式错误抛 ValueError。"""
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"YOLO 行必须 5 列: {line!r}")
    try:
        cls = int(float(parts[0]))
        cx, cy, bw, bh = (float(p) for p in parts[1:])
    except ValueError:
        raise ValueError(f"YOLO 行数值非法: {line!r}")
    return cls, cx * W, cy * H, bw * W, bh * H


def parse_label_text(text: str) -> list[tuple[int, float, float, float, float]]:
    """解析整个 .txt（归一化坐标），跳过空行。"""
    boxes = []
    for line in text.splitlines():
        if line.strip():
            parts = line.strip().split()
            if len(parts) != 5:
                raise ValueError(f"YOLO 行必须 5 列: {line!r}")
            cls = int(float(parts[0]))
            boxes.append((cls, float(parts[1]), float(parts[2]),
                          float(parts[3]), float(parts[4])))
    return boxes


def validate_box_norm(box: tuple, *, single_cls: bool = True,
                      max_ratio: float = 3.0,
                      min_side: float = 0.005) -> list[str]:
    """检查一个归一化框，返回问题列表（空 = 通过）。"""
    issues = []
    cls, cx, cy, bw, bh = box
    if single_cls and cls != 0:
        issues.append(f"class={cls} 非单类 plate(0)")
    for name, v in (("cx", cx), ("cy", cy), ("w", bw), ("h", bh)):
        if not (0.0 <= v <= 1.0):
            issues.append(f"{name}={v:.4f} 越界[0,1]")
        if math.isnan(v) or math.isinf(v):
            issues.append(f"{name} 非有限值")
    if bw < min_side or bh < min_side:
        issues.append(f"边过小 w={bw:.4f} h={bh:.4f}")
    if bw > 0 and bh > 0:
        ratio = max(bw, bh) / min(bw, bh)
        if ratio > max_ratio:
            issues.append(f"长宽比 {ratio:.2f} 超 {max_ratio}（片应近圆）")
    return issues


# ── 像素框运算 ───────────────────────────────────────────────────

def clip_box_pixels(cx: float, cy: float, w: float, h: float,
                    W: int, H: int) -> tuple[float, float, float, float] | None:
    """像素框裁到画面内；完全在外返回 None。"""
    x1 = min(W, max(0.0, cx - w / 2))
    y1 = min(H, max(0.0, cy - h / 2))
    x2 = min(W, max(0.0, cx + w / 2))
    y2 = min(H, max(0.0, cy + h / 2))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def box_iou(a: tuple, b: tuple) -> float:
    """两像素框 IoU（cx,cy,w,h）。"""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match_boxes(prev: list[tuple], cur: list[tuple],
                gate: float = 0.5, size_gate: float | None = None,
                ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """贪心最近中心匹配。

    门限 = gate × max(两框对角线)，默认 0.5（相邻帧的片位移远小于此；
    背景片堆静止，必匹配上）。
    size_gate：可选的尺寸一致性门（宽/高任一方向变化倍数上限），
    用于关键帧插值时防止大小片错配（固定机位下片表观尺寸不变）。
    返回 (配对[(i,j)], 未匹配prev下标, 未匹配cur下标)。
    """
    cands = []
    for i, a in enumerate(prev):
        for j, b in enumerate(cur):
            dist = math.hypot(a[0] - b[0], a[1] - b[1])
            diag = max(math.hypot(a[2], a[3]), math.hypot(b[2], b[3]))
            if dist > gate * diag:
                continue
            if size_gate is not None:
                if min(a[2], a[3], b[2], b[3]) <= 0:
                    continue
                ratios = (a[2] / b[2], b[2] / a[2], a[3] / b[3], b[3] / a[3])
                if max(ratios) > size_gate:
                    continue
            cands.append((dist, i, j))
    cands.sort()
    pairs, used_i, used_j = [], set(), set()
    for _, i, j in cands:
        if i not in used_i and j not in used_j:
            pairs.append((i, j))
            used_i.add(i)
            used_j.add(j)
    return (pairs,
            [i for i in range(len(prev)) if i not in used_i],
            [j for j in range(len(cur)) if j not in used_j])


def interpolate_keyframes(k0: int, boxes0: list[tuple],
                          k1: int, boxes1: list[tuple],
                          gate: float = 2.0, size_gate: float = 1.5,
                          ) -> dict[int, list[tuple]]:
    """两关键帧之间线性插值（不含端点）。

    关键帧是人工标的、杠铃运动平滑且同杠片锁步运动 → 中心门放宽到
    2.0×对角线（30 帧间隔、1m/s 向心段约 250px 位移仍可匹配），
    另加尺寸一致性门（固定机位片表观尺寸不变，防大小片错配）。
    保守策略：只有跨帧匹配上的框才插值；出现/消失的框不传播
    （宁可漏标待人工补，不造 ghost 框污染训练）。
    """
    if k1 <= k0:
        raise ValueError(f"关键帧顺序非法: {k0} -> {k1}")
    pairs, _, _ = match_boxes(boxes0, boxes1, gate, size_gate)
    out: dict[int, list[tuple]] = {}
    for f in range(k0 + 1, k1):
        t = (f - k0) / (k1 - k0)
        frame_boxes = []
        for i, j in pairs:
            a, b = boxes0[i], boxes1[j]
            frame_boxes.append(tuple(ai + (bi - ai) * t for ai, bi in zip(a, b)))
        out[f] = frame_boxes
    return out


# ── 数据集划分与混合 ─────────────────────────────────────────────

def split_by_video(records: list[dict], val_videos: list[str]
                   ) -> tuple[list[dict], list[dict]]:
    """按视频划分（防同视频帧泄漏）。records 须有 'video' 键；按 basename 归一。"""
    val_set = {Path(v).name for v in val_videos}
    train = [r for r in records if Path(r["video"]).name not in val_set]
    val = [r for r in records if Path(r["video"]).name in val_set]
    return train, val


def suggest_val_split(videos: list[str], n: int = 4, seed: int = 7) -> list[str]:
    """确定性抽 val 视频（默认 4 条）。调用方仍需目检轻/重/易/难覆盖。"""
    import random
    rng = random.Random(seed)
    names = sorted({Path(v).name for v in videos})
    if n >= len(names):
        raise ValueError(f"视频仅 {len(names)} 条，val 要 {n} 条太多")
    return sorted(rng.sample(names, n))


def plan_repeat_factor(n_public: int, n_self: int,
                       target_self_ratio: float = 0.25) -> int:
    """自标数据重复倍数 K，使 K*n_self 占混合训练集 target 比例。

    例：public 5600 + self 200 + 25% → K=10（自标有效 2000，占比 26%）。
    无公开数据时返回 1（不重复）。
    """
    if n_self <= 0 or n_public <= 0:
        return 1
    if not (0.0 < target_self_ratio < 1.0):
        raise ValueError("target_self_ratio 须在 (0,1)")
    k = target_self_ratio * n_public / ((1 - target_self_ratio) * n_self)
    return max(1, math.ceil(k))


# ── 挖掘队列 / manifest IO ───────────────────────────────────────

REQUIRED_QUEUE_KEYS = {"video", "frame", "score", "reasons"}


def save_mining_queue(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def load_mining_queue(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "items" not in payload or not isinstance(payload["items"], list):
        raise ValueError("挖掘队列缺 'items' 列表")
    for it in payload["items"]:
        missing = REQUIRED_QUEUE_KEYS - set(it)
        if missing:
            raise ValueError(f"队列条目缺键 {missing}: {it}")
    return payload


def load_manifest(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"images": {}}
    data = json.loads(p.read_text(encoding="utf-8"))
    if "images" not in data:
        raise ValueError("manifest 缺 'images' 键")
    return data


def save_manifest(path: str | Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False),
                          encoding="utf-8")
