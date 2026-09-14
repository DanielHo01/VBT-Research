# 历史文档（已归档）

> **归档日期：2026-09-14**
> 本目录是特定日期的**历史快照**，记录当时的调查结论与决策依据。
> **其中的成绩数字、待办清单、技术判断均可能已过期，请勿作为当前依据引用。**

## 当前有效文档

| 主题 | 文档 |
| --- | --- |
| 成绩（唯一事实表） | [`../../validation/reports/SCOREBOARD.md`](../../validation/reports/SCOREBOARD.md) |
| 技术路线 | [`../TECH_ROUTE.md`](../TECH_ROUTE.md) |
| 架构 | [`../ARCHITECTURE.md`](../ARCHITECTURE.md) |
| 公共 API | [`../VBTCORE_PUBLIC_API.md`](../VBTCORE_PUBLIC_API.md) |
| 环境配置 | [`../ENVIRONMENT.md`](../ENVIRONMENT.md) |
| 留出集规范 | [`../HOLDOUT.md`](../HOLDOUT.md) |
| 数据集 | [`../DATASET.md`](../DATASET.md) |
| 训练记录 | [`../TRAINING.md`](../TRAINING.md) |
| 分支策略 | [`../BRANCH_POLICY.md`](../BRANCH_POLICY.md) |
| Demo 构建 | [`../DEMO_GUIDE.md`](../DEMO_GUIDE.md) |

## 归档清单

| 文档 | 日期 | 内容与过期点 |
| --- | --- | --- |
| `ROADMAP.md` | 2025-09-08 | 旧 Phase 1/1.5/2 叙事。已被 `TECH_ROUTE.md` 的 M0–M5 体系取代。 |
| `DEEP_DIVE_REPORT.md` | 2026-09-10 | 首次全量代码走读与竞品调研。其中「工程质量」一节所列问题（硬编码 Windows 路径、无单测、无 CI）**均已修复**。 |
| `PLATE_LAYER_REVIEW.md` | 2026-09-10 | 杠铃片层路线复核。数字为旧逐帧检测口径；第四节「不做：先重训检测器」**已过期**（后续确实重训了 best.onnx）。 |
| `PUBLIC_DATASET_AUDIT.md` | 2026-09-10 | 公开数据集桌面审计。规模/类列表未经下载核实。 |
| `SESSION_HANDOFF_20260911.md` | 2026-09-11 | 会话交接。其 P0.1（4 条假拒绝）**已解决**；P0.2（亚像素椭圆+卡尔曼）经 v3 实验证伪后改走现行方案。 |
| `STEP2_ARCHITECTURE.md` | 2026-09-11 | YOLO 降维为 ROI 粗提取器的架构设计。 |
| `DATA_DD_20260912.md` | 2026-09-12 | 数据与文档尽职调查，含 datasets/ 散落情况盘点。 |

## 为什么保留

这些文档记录了**决策的来龙去脉**——尤其是哪些路走不通、为什么。
删掉它们会让后人重蹈覆辙（参见 README 的「负面约束清单」，
其证据链大量来自这批文档与 `validation/reports/archive/`）。
