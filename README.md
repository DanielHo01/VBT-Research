# VBT-Research

Vertical Bar tracking research — a computer vision pipeline for estimating barbell velocity and repetition count from smartphone videos of squats.

## Project Status

**Phase 1 ✅** — Baseline model (`barbell_v4.onnx`) achieves RMSE = 0.88 on MCV prediction across 34 validation videos.

**Phase 1.5 ❌** — Attempted OWL-ViT zero-shot pseudo-labeling; training produced a worse model (RMSE = 2.20). [Lessons learned](docs/TRAINING.md)

**Phase 2 🔲** — Not yet started.

## Quick Start (本地验证，视频不上传)

```bash
# 1. 安装依赖（建议 venv）
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. 把 34 个验证视频放到 validation/dataset_benchmark/raw_videos/
#    （也可以用环境变量指向任意本地目录，视频不离开本机）
export VBT_VIDEOS_DIR=/path/to/your/raw_videos

# 3. 无需视频的冒烟自检（回归测试，验证检测器修复）
python3 scripts/self_test.py

# 4. 跑完整基准（默认 barbell_v4 + associator 生产管线）
python3 scripts/run_full_benchmark.py
python3 scripts/run_full_benchmark.py --model yolo11_plate   # 换模型
python3 scripts/run_full_benchmark.py --limit 5              # 先跑前5个

# 5. 逐视频错误定位（找 RMSE 大的原因）
python3 scripts/diagnose_benchmark.py

# 6. 多管线横向对比（含绘图）
python3 validation/dataset_benchmark/run_benchmark.py
```

所有脚本路径默认相对仓库根目录解析（`validation/dataset_benchmark/config.py`），
可用环境变量覆盖：`VBT_MODEL_PATH` / `VBT_VIDEOS_DIR` / `VBT_INDEX_PATH` / `VBT_OUTPUT_DIR`。

### 重要：修复说明（2026-09）

旧版检测器对 ONNX 导出的置信度**重复应用 sigmoid**（输出已是 [0,1] 概率），
导致空白帧也被当作 ~0.50 置信度的检测 —— 这是旧基准 RMSE=0.88 异常偏大的
主要原因之一。本版本已修复，并移除了标定里的经验魔数 `scale_factor=1.15`
（默认改为 1.0，需要对比旧结果时可显式传 `--scale-factor 1.15`）。

**旧 RMSE 数字不可直接比较**，请在本地用视频重新跑基准。

## Project Structure

```
docs/                  # Architecture, dataset, training, roadmap
models/                # Trained ONNX models
  barbell_v4.onnx      ← Production model (旧基准 RMSE=0.88，修复后待重测)
scripts/               # Core scripts
  run_full_benchmark.py       # 单模型基准（可移植 CLI）
  diagnose_benchmark.py       # 逐视频错误定位
  self_test.py                # 无需视频的冒烟自检
  interactive_label.py
  extract_owlvit_pseudo.py
  verify_labels.py
validation/            # 34 benchmark videos (raw_videos 本地)+ ground truth
  dataset_benchmark/config.py # 相对路径/环境变量解析
```

## Models

| Model | RMSE (MCV) | Notes |
|-------|-------------|-------|
| `barbell_v4.onnx` | **0.88** | Best — production use |
| `yolo11_plate.onnx` | ? | Untested — needs benchmark |
| `plate_v1.onnx` | ~1.4 | Poor on side-view |

## Key Findings

- barbell_v4 generalizes well across weight loads (20–50 kg) and body weights
- Side-view (lateral camera angle) is harder than top-view for plate detection
- OWL-ViT zero-shot detection finds plates but with low confidence (0.05–0.08); pseudo-labels from it are too noisy to train a better model
- Manual labeling with OWL-ViT as a suggestion engine is the most practical path forward

## Requirements

```
opencv-python>=4.10
numpy>=1.26
scipy>=1.13
matplotlib>=3.9
ultralytics>=8.2
transformers>=4.57
onnxruntime>=1.19
pandas>=2.2
```

See `requirements.txt` for full pinned versions.
