# VBT-Research

Vertical Bar tracking research — a computer vision pipeline for estimating barbell velocity and repetition count from smartphone videos of squats.

## Project Status

**Phase 1 ✅** — Baseline model (`barbell_v4.onnx`) achieves RMSE = 0.88 on MCV prediction across 34 validation videos.

**Phase 1.5 ❌** — Attempted OWL-ViT zero-shot pseudo-labeling; training produced a worse model (RMSE = 2.20). [Lessons learned](docs/TRAINING.md)

**Phase 2 🔲** — Not yet started.

## vbtcore — 新引擎（M0）

> 2026-09-10 起的规范化引擎，修复了旧脚本的四类历史 bug（详见各模块 docstring）：
> 竖屏旋转逆映射镜像错误、置信度双重 sigmoid、直 resize 挤压畸变、
> 魔法标定系数 scale_factor=1.15。

```bash
# 单元测试（13+ 用例，锁定已修 bug）
PYTHONPATH=<deps> python3 -c "import sys; sys.path[:0]=['tests']; import test_geometry, test_segment, test_anchor; ..."

# 34 视频基准报告（validation/reports/BENCHMARK_v0.md）
python3 scripts/run_benchmark_v0.py
```

- 架构：检测→拟合混合跟踪（每 15 帧 YOLO 重检测 + NCC 模板 + 匀速预测），CPU ~15ms/帧
- M0 基线：计数通过(±1或正确拒绝) 19/34；配对视频 RMSE 均值 0.278；
  20kg 杆-only 被正确拒绝（NO_PLATE_DETECTED）
- 已知短板（M1 目标）：遮挡期假 rep（碎片化）、重负荷工作片检出率（检测域差，需数据闭环）
- `algorithms/pipelines.py` 等旧脚本保留作为对照基线，不再维护

## Quick Start

```bash
# Run full benchmark (34 videos)
python3 scripts/run_full_benchmark.py

# Run interactive labeling tool (OWL-ViT suggests, you click to accept/reject)
python3 scripts/interactive_label.py <video_file.mp4>

# Extract pseudo-labels via OWL-ViT
python3 scripts/extract_owlvit_pseudo.py
```

## Project Structure

```text
docs/                  # Architecture, dataset, training, roadmap
models/                # Trained ONNX models
  barbell_v4.onnx      ← Production model (RMSE=0.88)
scripts/               # Core scripts
  run_full_benchmark.py
  interactive_label.py
  extract_owlvit_pseudo.py
  verify_labels.py
validation/            # 34 benchmark videos + ground truth (videos in git)
```

## Models

| Model | RMSE (MCV) | Notes |
| ------- | ------------- | ------- |
| `barbell_v4.onnx` | **0.88** | Best — production use |
| `yolo11_plate.onnx` | ? | Untested — needs benchmark |
| `plate_v1.onnx` | ~1.4 | Poor on side-view |

## Key Findings

- barbell_v4 generalizes well across weight loads (20–50 kg) and body weights
- Side-view (lateral camera angle) is harder than top-view for plate detection
- OWL-ViT zero-shot detection finds plates but with low confidence (0.05–0.08); pseudo-labels from it are too noisy to train a better model
- Manual labeling with OWL-ViT as a suggestion engine is the most practical path forward

## Requirements

```text
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
