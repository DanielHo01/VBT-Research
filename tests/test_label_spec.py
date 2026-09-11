"""test_label_spec — label_common 纯函数单测（M3 标注管线）。

约束：只能测 stdlib 模块 label_common，不导入 cv2/torch/ultralytics，
保证 CI 零新增依赖可跑。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import label_common as lc  # noqa: E402
import public_registry as pr  # noqa: E402


def test_yolo_roundtrip():
    s = lc.pixels_to_yolo(0, 360.0, 640.0, 90.0, 95.0, 720, 1280)
    cls, cx, cy, w, h = lc.yolo_to_pixels(s, 720, 1280)
    assert cls == 0
    assert abs(cx - 360.0) < 0.01 and abs(cy - 640.0) < 0.01
    assert abs(w - 90.0) < 0.01 and abs(h - 95.0) < 0.01


def test_yolo_rejects_degenerate():
    for bad in [(0, 10, 10, 0, 5), (0, 10, 10, 5, -1)]:
        try:
            lc.pixels_to_yolo(*bad, 100, 100)
        except ValueError:
            pass
        else:
            raise AssertionError(f"退化框未拒绝: {bad}")
    try:
        lc.yolo_to_pixels("0 0.5 0.5", 100, 100)
    except ValueError:
        pass
    else:
        raise AssertionError("3 列行未拒绝")


def test_validate_box_norm_ok_and_bad():
    assert lc.validate_box_norm((0, 0.5, 0.5, 0.1, 0.1)) == []
    bad_cls = lc.validate_box_norm((3, 0.5, 0.5, 0.1, 0.1))
    assert any("class" in i for i in bad_cls)
    oob = lc.validate_box_norm((0, 1.5, 0.5, 0.1, 0.1))
    assert any("越界" in i for i in oob)
    skinny = lc.validate_box_norm((0, 0.5, 0.5, 0.4, 0.05))
    assert any("长宽比" in i for i in skinny)
    tiny = lc.validate_box_norm((0, 0.5, 0.5, 0.001, 0.001))
    assert any("过小" in i for i in tiny)


def test_box_iou_identity_and_disjoint():
    a = (50.0, 50.0, 20.0, 20.0)
    assert abs(lc.box_iou(a, a) - 1.0) < 1e-9
    b = (500.0, 500.0, 20.0, 20.0)
    assert lc.box_iou(a, b) == 0.0


def test_match_boxes_static_and_moved():
    prev = [(100.0, 100.0, 40.0, 40.0),   # 静止片堆
            (300.0, 300.0, 40.0, 40.0)]   # 工作片
    cur = [(101.0, 99.0, 40.0, 40.0),     # 片堆微抖
           (305.0, 320.0, 40.0, 40.0)]    # 工作片移动 20px（<0.5×对角线≈28）
    pairs, um_p, um_c = lc.match_boxes(prev, cur)
    assert sorted(pairs) == [(0, 0), (1, 1)]
    assert um_p == [] and um_c == []
    far = [(700.0, 700.0, 40.0, 40.0)]
    pairs2, um_p2, um_c2 = lc.match_boxes(prev, far)
    assert pairs2 == [] and um_p2 == [0, 1] and um_c2 == [0]


def test_interpolate_linear_midpoint():
    b0 = [(100.0, 100.0, 40.0, 40.0)]
    b1 = [(100.0, 200.0, 40.0, 40.0)]
    out = lc.interpolate_keyframes(0, b0, 10, b1)
    assert sorted(out.keys()) == list(range(1, 10))
    mid = out[5][0]
    assert abs(mid[0] - 100.0) < 1e-9 and abs(mid[1] - 150.0) < 1e-9
    assert abs(mid[2] - 40.0) < 1e-9


def test_match_size_gate_blocks_big_small_mix():
    near = [(100.0, 100.0, 120.0, 120.0)]   # 近端大片
    far = [(110.0, 105.0, 40.0, 40.0)]      # 中心接近但尺寸差 3×
    pairs, _, _ = lc.match_boxes(near, far, gate=2.0, size_gate=1.5)
    assert pairs == []
    pairs2, _, _ = lc.match_boxes(near, far, gate=2.0, size_gate=None)
    assert pairs2 == [(0, 0)]


def test_interpolate_drops_unmatched_no_ghost():
    b0 = [(100.0, 100.0, 40.0, 40.0)]   # 关键帧0 只有片堆
    b1 = [(100.0, 100.0, 40.0, 40.0),   # 关键帧1 多出一个新框
          (500.0, 500.0, 40.0, 40.0)]
    out = lc.interpolate_keyframes(0, b0, 4, b1)
    for f, boxes in out.items():
        assert len(boxes) == 1, f"f{f} 出现 ghost 框"
    try:
        lc.interpolate_keyframes(5, b0, 5, b1)
    except ValueError:
        pass
    else:
        raise AssertionError("同帧插值未拒绝")


def test_split_by_video_no_leak():
    recs = [{"video": "a.mp4", "i": 1}, {"video": "a.mp4", "i": 2},
            {"video": "b.mp4", "i": 3}, {"video": "dir/c.mp4", "i": 4}]
    train, val = lc.split_by_video(recs, ["a.mp4"])
    assert [r["i"] for r in val] == [1, 2]
    assert [r["i"] for r in train] == [3, 4]
    # basename 归一：传全路径也应命中
    _, val2 = lc.split_by_video(recs, ["some/dir/c.mp4"])
    assert [r["i"] for r in val2] == [4]


def test_repeat_factor_math():
    assert lc.plan_repeat_factor(5600, 200, 0.25) == 10
    assert lc.plan_repeat_factor(0, 200) == 1
    assert lc.plan_repeat_factor(5600, 0) == 1
    # 50% 目标：K*200/(5600+K*200)=0.5 → K=28
    assert lc.plan_repeat_factor(5600, 200, 0.5) == 28


def test_registry_keep_first_and_unlisted_default():
    # keep 优先：'barbell-end' 命中 keep 的 end，不被 drop 的 barbell 误杀
    assert pr.match_class("barbell-end", ["end"], ["barbell"]) == "keep"
    assert pr.match_class("Barbell", ["end"], ["barbell"]) == "drop"
    assert pr.match_class("plate_25_red", ["plate"], ["barbell"]) == "keep"
    assert pr.match_class("weird_action", ["plate"], ["barbell"]) == "unlisted"
    assert pr.match_class("zacisk", ["kg", "plate"], ["zacisk"]) == "drop"
    assert pr.match_class("25kg", ["kg", "plate"], ["zacisk"]) == "keep"


def test_queue_manifest_io_roundtrip(tmp_path=None):
    import tempfile, os
    d = tempfile.mkdtemp()
    try:
        q = {"items": [{"video": "a.mp4", "frame": 3, "score": 2.0,
                        "reasons": ["no_pass"]}]}
        qp = os.path.join(d, "q.json")
        lc.save_mining_queue(qp, q)
        assert lc.load_mining_queue(qp) == q
        try:
            bad = os.path.join(d, "bad.json")
            lc.save_mining_queue(bad, {"items": [{"video": "a"}]})
            lc.load_mining_queue(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("缺键队列未拒绝")
        mp = os.path.join(d, "m.json")
        assert lc.load_manifest(mp) == {"images": {}}
        lc.save_manifest(mp, {"images": {"x.jpg": {"n": 1}}})
        assert lc.load_manifest(mp)["images"]["x.jpg"]["n"] == 1
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
