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
- M1 基线（诚实口径，假拒绝计为失败）：计数通过 25/34；配对视频 RMSE 均值 0.202；
  ~13 ms/帧；假拒绝 6 条（检测器域差，M3 数据闭环目标）
- 20kg 杆-only 被正确拒绝（NO_PLATE_DETECTED，确认用户判断）
- M1 修复：NCC 位移物理上限（防漂移）、identity-first 远距夺回、运动观察哨
  （邻域聚类检测错锁背景）、hold 桥接（蹲底遮挡 19-42 帧）、两遍法 ROM 质量门
  （清除抖动假 rep）、近邻峰合并 + 边界 top 合成
- M1.5（移植 TroyKaneshiro/barbell-velocity-tracker）：rep 底部重锚定 regrind
  （UP/DOWN 双相真每 rep 一次 + 纠正分级 snap/micro/拒绝）+ MCV 全程平均口径
  （GymAware ACV，旧平均正速度降级为诊断）+ 外层片直径查表；26 单元测试全过；
  基准 24/34、RMSE 0.155（同环境 M1 基线 24/34、0.152：中性零回归；
  44 次触发 0 snap——底部检测盲区 gating，增益待 M3 检测器）；见 BENCHMARK_v1
- 已知短板（M2/M3 目标）：50kg/105kg 部分组仍少计（错锁恢复不全）、
  速度校准（部分视频 |bias|>0.2）、重负荷工作片检出率（需数据闭环）
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
