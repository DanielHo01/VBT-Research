"""
self_test.py — 无需视频的冒烟自检（在提交/发布前运行）
========================================================

验证 4 件事：
  1. 三个 ONNX 模型都能加载，且空白/噪声帧不会产生“错误检测”
     （回归测试：旧代码把已 sigmoid 的 conf 再 sigmoid 一次，
      空白帧置信度 ~0.50 被误判为检测 —— 这是 RMSE 异常的主因）
  2. calibrate_scale：450mm 杠铃片在 100px 高时，scale 应为 0.0045 m/px
  3. 检测器对 5ch / 6ch 两种输出都能解析并正确缩放坐标
  4. MetricsEvaluator 对齐与 RMSE 计算正确

用法:
    python3 scripts/self_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'validation' / 'dataset_benchmark'))

import config as cfg  # noqa: E402
from algorithms.common import YoloPlateDetector, calibrate_scale  # noqa: E402
from metrics_evaluator import MetricsEvaluator  # noqa: E402

PASSED = 0
FAILED = 0


def check(name: str, ok: bool, detail: str = ""):
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAILED += 1
        print(f"  [FAIL] {name}  {detail}")


def test_models_blank_frame():
    print("\n1) 模型加载 + 空白帧不应产生检测")
    import cv2
    for name in ['barbell_v4', 'plate_v1', 'yolo11_plate']:
        mp = cfg.REPO_ROOT / 'models' / f'{name}.onnx'
        if not mp.exists():
            check(f'{name}: 模型存在', False, f'缺少 {mp}')
            continue
        try:
            det = YoloPlateDetector(str(mp))
        except Exception as e:
            check(f'{name}: 模型可加载', False, str(e))
            continue
        check(f'{name}: 模型可加载', True)

        blank = np.zeros((720, 1280, 3), np.uint8)
        noise = np.random.randint(0, 255, (720, 1280, 3), np.uint8)
        b_det = det.detect(blank)
        n_det = det.detect(noise)

        # 旧 bug：空白帧 conf ≈ 0.50 会被当作检测
        check(f'{name}: 空白帧无检测 (conf<0.25)', b_det is None,
              f"got conf={b_det['score']:.3f}" if b_det else "")
        check(f'{name}: 噪声帧无检测 (conf<0.25)', n_det is None,
              f"got conf={n_det['score']:.3f}" if n_det else "")

        # 解析 5ch/6ch + 坐标缩放：手工注入一张模拟 YOLO 输出
        det._run = lambda f: np.array([[640.0], [360.0], [50.0], [50.0], [0.99]],
                                      dtype=np.float32)  # 5ch
        det.input_size = 640  # 原图 1280x720
        out = det.detect_all(blank, conf_threshold=0.25)
        check(f'{name}: 5ch 解析 + 坐标缩放', len(out) == 1 and abs(out[0]['cx'] - 1280.0) < 1e-3,
              f"got {out}")


def test_calibrate_scale():
    print("\n2) calibrate_scale")
    heights = [100.0] * 9 + [102.0, 98.0]  # 噪声但稳健
    scale, med = calibrate_scale(heights, plate_diameter_m=0.45, scale_factor=1.0)
    check('450mm @ 100px -> 0.0045 m/px', abs(scale - 0.0045) < 1e-9,
          f"scale={scale}")
    check('median 高度正确', med == 100.0)
    scale15, _ = calibrate_scale(heights, plate_diameter_m=0.45, scale_factor=1.15)
    check('scale_factor=1.15 仍可显式开启（对比旧结果）',
          abs(scale15 - 0.0045 * 1.15) < 1e-9)
    scale0, _ = calibrate_scale([0.0] * 5)
    check('全部高度为 0 返回 0（不再静默用 0.001）', scale0 == 0.0)


def test_evaluator():
    print("\n3) MetricsEvaluator")
    gt = [1.0, 0.8, 0.6]
    pred = [1.05, 0.75, 0.55]
    ev = MetricsEvaluator.evaluate_video('t', gt, [{'mcv': p} for p in pred])
    exp = float(np.sqrt(np.mean([0.05 ** 2, 0.05 ** 2, 0.05 ** 2])))
    check('RMSE 计算', abs(ev.rmse - exp) < 1e-9, f"rmse={ev.rmse} 期待={exp}")
    check('偏差计算', abs(ev.bias - (0.05 - 0.05 - 0.05) / 3) < 1e-9, f"bias={ev.bias}")

    # 顺序截断 vs 最优配对：pred 多一个“幻影 rep”（unrack 场景）
    pred2 = [1.05, 3.0, 0.75, 0.55]
    ev_t = MetricsEvaluator.evaluate_video('t', gt, [{'mcv': p} for p in pred2],
                                           strategy='truncate')
    ev_b = MetricsEvaluator.evaluate_video('t', gt, [{'mcv': p} for p in pred2],
                                           strategy='best_pair')
    check('best_pair 优于 truncate（幻影 rep 情形）',
          ev_b.rmse < ev_t.rmse, f"trunc={ev_t.rmse:.3f} best={ev_b.rmse:.3f}")


def main() -> int:
    print("VBT 冒烟自检（无需视频）")
    print("=" * 60)
    test_models_blank_frame()
    test_calibrate_scale()
    test_evaluator()
    print("=" * 60)
    print(f"  通过 {PASSED} / {PASSED + FAILED}")
    if FAILED:
        print("  有失败项，请勿提交推送！")
        return 1
    print("  全部通过 ✅")
    return 0


if __name__ == '__main__':
    sys.exit(main())
