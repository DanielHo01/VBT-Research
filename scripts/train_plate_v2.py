"""train_plate_v2.py — M3 检测器训练 + 导出 + 门控
====================================================
路线（docs/M3_TRAINING_PLAN.md）：COCO 预训练 YOLO11n + 全新单类检测头，
在 build_dataset.py 组装的 plate_v2 上训练（公开作底 + 自标加权 ~25%）。

跑哪里：Colab T4 / Kaggle P100（见 notebooks/colab_train_plate_v2.ipynb）。
本地 1650 4GB 可试（--batch 8），但慢且易 OOM，仅备选。

训练后自动三道门：
  Gate A 格式：ONNX 输出必须是 [1,5,N]（vbtcore 硬性要求），输入 [1,3,imgsz,imgsz]。
  Gate B 烟雾：--smoke-img 给一帧开发集图，vbtcore 检测器能跑通并打印检出数。
  Gate C 基准：（手动）在 34 视频 + holdout 上跑 run_benchmark_v0.py，
               假拒绝 6→≤2 且无回归才算过（--benchmark-cmd 打印命令）。

用法：
    python scripts/train_plate_v2.py --data datasets/plate_v2/data.yaml --name plate_v2a
    python scripts/train_plate_v2.py --data ... --epochs 150 --batch 32 --device 0
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fix_data_yaml(data_yaml: Path, run_dir: Path) -> Path:
    """data.yaml 可移植修正：path 重写为 yaml 所在目录（build 机与训练机路径不同）。

    ultralytics 训练时 train/val 相对 path 解析；build_dataset 写的是 build 机
    绝对路径，跨机必错。此处生成 run_dir/data_fixed.yaml 并返回。
    """
    import yaml  # noqa: PLC0415 — ultralytics 依赖，保证存在

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    cfg["path"] = str(data_yaml.resolve().parent)
    fixed = run_dir / "data_fixed.yaml"
    run_dir.mkdir(parents=True, exist_ok=True)
    fixed.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return fixed


def validate_onnx(onnx_path: Path, imgsz: int) -> tuple[int, ...]:
    """Gate A：零输入推理，断言 [1,3,imgsz,imgsz]→[1,5,N]。返回输出 shape。"""
    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: dummy})[0]
    shape = tuple(out.shape)
    if len(shape) != 3 or shape[0] != 1 or shape[1] != 5:
        raise SystemExit(f"Gate A 失败：ONNX 输出 {shape} 不是 [1,5,N] 单类格式")
    print(f"Gate A 通过：ONNX 输出 {shape}")
    return shape


def smoke_detect(onnx_path: Path, img_path: str, conf: float = 0.20) -> int:
    """Gate B：vbtcore 检测器在给定帧上跑通。返回检出数。"""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import cv2  # noqa: PLC0415
    from vbtcore.detector import PlateDetector  # noqa: PLC0415

    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"smoke 图读失败: {img_path}")
    dets = PlateDetector(str(onnx_path)).detect(img, conf)
    print(f"Gate B 通过：{img_path} 检出 {len(dets)} 框（conf≥{conf}）")
    for d in sorted(dets, key=lambda d: -d.conf)[:5]:
        print(f"  conf={d.conf:.2f} cx={d.cx:.0f} cy={d.cy:.0f} h={d.h:.0f}")
    return len(dets)


def main() -> int:
    ap = argparse.ArgumentParser(description="M3 检测器训练 + 导出 + 门控")
    ap.add_argument("--data", required=True, help="datasets/plate_v2/data.yaml")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="基座（默认官方 COCO 预训练；传 '' 则随机初始化=烧蚀）")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="plate_v2a")
    ap.add_argument("--project", default="runs/plate_v2")
    ap.add_argument("--device", default="", help="''=自动；0=首块 GPU")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--flipud", type=float, default=0.0,
                    help="垂直翻转（默认 0！竖屏视频禁翻）")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="烧蚀：随机初始化（期望更差，仅实验）")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--smoke-img", default=None, help="Gate B 烟雾帧")
    ap.add_argument("--smoke-conf", type=float, default=0.20)
    ap.add_argument("--copy-models", action="store_true",
                    help="ONNX 拷贝到 models/<name>.onnx（默认只放 runs，手动拷）")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO  # noqa: PLC0415
        import ultralytics  # noqa: PLC0415
    except ImportError:
        print("缺 ultralytics：pip install ultralytics（训练机上装）")
        return 1

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"data.yaml 不存在: {data_yaml}（先跑 build_dataset.py）")
        return 1
    run_dir = Path(args.project) / args.name
    fixed_yaml = fix_data_yaml(data_yaml, run_dir)
    print(f"data: {fixed_yaml}（path 已修正为 {data_yaml.resolve().parent}）")

    base = args.model if not args.no_pretrained else "yolo11n.yaml"
    print(f"基座: {base} ｜ epochs={args.epochs} batch={args.batch} "
          f"imgsz={args.imgsz} device='{args.device or 'auto'}' flipud={args.flipud}")
    model = YOLO(base)
    model.train(data=str(fixed_yaml), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device or None, seed=args.seed,
                patience=args.patience, workers=args.workers,
                single_cls=True, flipud=args.flipud,
                project=args.project, name=args.name, exist_ok=True,
                verbose=True, plots=True, val=True, deterministic=True)
    print(f"训练完成: {run_dir}")

    metrics = {}
    try:
        val_res = model.val()
        metrics = {"mAP50": round(float(val_res.box.map50), 4),
                   "mAP50_95": round(float(val_res.box.map), 4),
                   "precision": round(float(val_res.box.mp), 4),
                   "recall": round(float(val_res.box.mr), 4)}
        print(f"val: {metrics}（仅看趋势，真门是 34 视频基准）")
    except Exception as e:  # noqa: BLE001 — val 失败不挡导出
        print(f"val 指标读取失败（不挡导出）: {e}")

    card = {"name": args.name, "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ultralytics": ultralytics.__version__,
            "params": {"model": base, "imgsz": args.imgsz, "epochs": args.epochs,
                       "batch": args.batch, "seed": args.seed, "patience": args.patience,
                       "flipud": args.flipud, "single_cls": True},
            "data_yaml": str(data_yaml.resolve()), "metrics": metrics}

    if not args.no_export:
        best_pt = run_dir / "weights" / "best.pt"
        onnx_path = Path(model.export(format="onnx", opset=args.opset))
        print(f"导出: {onnx_path}")
        shape = validate_onnx(onnx_path, args.imgsz)
        card["onnx"] = str(onnx_path.resolve())
        card["onnx_shape"] = list(shape)
        if args.copy_models:
            dest = ROOT / "models" / f"{args.name}.onnx"
            dest.write_bytes(onnx_path.read_bytes())
            print(f"已拷贝: {dest}")
        else:
            print(f"手动入库：cp {onnx_path} models/{args.name}.onnx")
    if args.smoke_img and not args.no_export:
        card["smoke_n"] = smoke_detect(Path(card["onnx"]), args.smoke_img,
                                       args.smoke_conf)
    (run_dir / "run_card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"run_card: {run_dir / 'run_card.json'}")
    print(f"Gate C（手动）: python scripts/run_benchmark_v0.py --tag {args.name} "
          f"--model models/{args.name}.onnx --engine \"vbtcore v1.5 + {args.name}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
