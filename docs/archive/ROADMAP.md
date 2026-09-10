# 项目状态看板

> **已归档（2026-09-10，原 `docs/ROADMAP.md`）**：本看板停留在 2025-09-08 的
> Phase 1/1.5/2 叙事，当前唯一活路线图是 `../TECH_ROUTE.md`（M0–M5 体系）。
> 本文件仅作历史快照保留，请勿更新。

> 最近更新: 2025-09-08

## 当前最优模型

**barbell_v4.onnx** — RMSE=0.88, 91% pass rate (31/34 视频)

---

## 阶段进度

```
[Phase 1]   ████████████ 完成  barbell_v4 基线, RMSE=0.88
[Phase 1.5] ░░░░░░░░░░░░ 失败  OWL-ViT伪标签 → 模型更差 (已归档)
[Phase 2]   ░░░░░░░░░░░░ 未开始
```

---

## 核心文件

### 模型
```
models/
  barbell_v4.onnx              ← 生产用
  barbell_v4.onnx.backup       ← 原始备份
  yolo11_plate.onnx            ← 待验证
  plate_v1.onnx                ← 差,勿用
```

### 核心脚本
```
scripts/
  run_full_benchmark.py        ← 基准测试
  interactive_label.py         ← 交互式标注工具
  extract_owlvit_pseudo.py     ← OWL-ViT伪标签提取
  owlvit_visualize.py          ← OWL-ViT可视化
  verify_labels.py             ← YOLO标签验证
```

### 数据集
```
datasets/
  owlvit_pseudo/               ← 58张人工筛选伪标签
  interactive_labels/          ← 人工标注工作区
  weightlifting-plates/        ← 原始数据集
```

### 文档
```
docs/
  ARCHITECTURE.md              ← 项目架构
  DATASET.md                   ← 数据集说明
  ROADMAP.md                   ← 本文件
  TRAINING.md                  ← 训练记录和教训
```

---

## 下一步选项

1. **用 interactive_label.py 人工标注** — OWL-ViT提候选，你点选接受，积累干净数据
2. **验证 yolo11_plate.onnx** — 可能是预训练的通用检测器，跑一下基准测试
3. **轻量再训练** — 从 plate_v1.pt 出发，用 owlvit_pseudo 的58张干净数据，YOLOv8n+320px+batch=16

---

## 已归档（_archive/）

所有废弃文件移至 `_archive/` 目录，不会影响日常使用：

- `_archive/phase1.5_failed/` — Phase 1.5 所有失败产物
- `_archive/old_datasets/` — 废弃数据集
- `_archive/old_scripts/` — 废弃脚本
