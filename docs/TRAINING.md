# 训练记录

## 模型清单（当前状态）

| 模型 | 路径 | 质量 | 说明 |
|------|------|------|------|
| `barbell_v4.onnx` | `models/` | ✅ **生产最优** | RMSE=0.88, 91% pass |
| `barbell_v4.onnx.backup` | `models/` | ✅ 备份 | 同上，原始备份 |
| `yolo11_plate.onnx` | `models/` | ❓ 未验证 | 预训练模型，待测试 |
| `plate_v1.onnx` | `models/` | ❌ 差 | RMSE≈1.4, side-view效果差 |

---

## Phase 1 — 完成 ✅

**目标**: 建立基线，验证端到端流程
**结果**: barbell_v4 达到 RMSE=0.88，91%视频通过预测

---

## Phase 1.5 — 失败 ❌（已归档）

**目标**: 用 OWL-ViT 零样本检测做伪标签，重新训练 YOLOv8n
**结果**: 训练失败 + 产出的模型更差

### 失败原因

1. **伪标签本身有噪声**: barbell_v4 是顶视角模型，对侧视角视频的检测本身就差，用它的输出做伪标签会传播错误
2. **训练环境问题**: Windows + CPU训练极慢（30-83分钟/epoch），第一个epoch缓存构建卡住数小时
3. **combined_v2 模型质量下降**: 用有噪声的伪标签训练，产出的模型 RMSE=2.20，比 barbell_v4 差 2.5×

### 归档内容

所有 Phase 1.5 的失败产物已移至 `_archive/phase1.5_failed/`:
- `combined_v2_run/` — 训练 checkpoint（3 epochs后停止）
- `owlvit_combined/` — 5136张训练数据
- `owlvit_finetune_run/` — 未完成的微调尝试
- `owlvit_vis/` — 临时可视化输出
- `combined_v2_rmse2.20.onnx` — 比barbell_v4更差的模型

### OWL-ViT 检测质量分析

OWL-ViT 零样本在侧视角深蹲视频上：
- ✅ ratio 正确（1.0-1.4 = 圆形plate）
- ✅ 位置基本准确
- ❌ 置信度极低（0.05-0.08，专用模型高10倍）
- ❌ 误检背景物体
- ❌ 帧间不稳定（1→7→1 检测数量跳动）

**结论**: 低置信度检测混入背景干扰，伪标签本身就是错的

### 保留的有价值产出

- `datasets/owlvit_pseudo/` — 58张人工筛选后的高质量伪标签
- `scripts/interactive_label.py` — 交互式标注工具（OWL-ViT提候选+人工点选）
- `datasets/interactive_labels/` — 人工标注工作区

---

## Phase 2 — 规划中 🔲

待定。可能方向：
- 人工标注更多数据（用 interactive_label.py）
- 尝试更轻量的训练配置（YOLOv8n, 320px, batch=16）
- 直接在 pipeline 中使用 OWL-ViT（慢但准确）

---

## 训练教训

1. **伪标签需要高置信度过滤**: 仅用 score>0.10 的检测做伪标签
2. **domain mismatch 是致命问题**: 顶视角训练的模型不能直接用于侧视角
3. **Windows 训练环境**: 确认 GPU 真的被使用（`nvidia-smi --query-compute-apps`）
4. **多行 YOLO 标签是正常的**: 每行代表一个物体，不要按行数过滤

---

## 如何复现 barbell_v4 基线

```bash
# 运行完整基准测试
python3 scripts/run_full_benchmark.py

# 预期结果: RMSE=0.88, 91% pass (31/34 视频)
```
