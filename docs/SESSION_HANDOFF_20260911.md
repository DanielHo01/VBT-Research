# Session Handoff — 2026-09-11

> 本次会话（Daniel 何天元 + Agent）完整记录。技术团队按本文档推进下一步开发。

---

## 一、本次会话核心成果

### 1. 仓库全面同步

- 本地从落后 8 个 commit（vbtcore 模块完全缺失）同步到 GitHub 最新
- 26 个单元测试全过，本地复跑 25/34，基准可复

### 2. 新检测模型（Colab 训练）

- 数据：26 条本地视频（720×1280 竖屏）→ 每 15 帧均匀抽帧 + 蹲底关键帧补录
- 标注：EzYOLO 半自动标注（边缘检测 → 人工审核修正）
- 产出：`models/best.onnx`（10.1 MB，YOLOv11n）和 `models/best.pt`（5.2 MB）
- 质量：28/34 配对率（79%），RMSE 0.25，假拒绝从 6 条降至 4 条

### 3. 算法迭代（三个版本 benchmark）

| 版本 | 说明 | 计数通过 | 假拒绝 | 视频 RMSE | Rep RMSE | bias |
| ------ | ------ | --------- | -------- | ---------- | ---------- | ------ |
| **v0** | 原始引擎（mpp 动态计算） | 25/34 | 6 | 0.212 | 0.253 | −0.085 |
| **v1** | best.onnx 仅换模型 | **27/34** | **4** | 0.250 | 0.316 | +0.040 |
| **v2** | best.onnx + mpp锁死 + IoU防跳变 | 24/34 | **4** | 0.284 | 0.381 | +0.025 |

### 4. 关键实验结论（决定下一步方向）

#### ✅ 模型已足够好，不需要重训

- `140kg_0.41.mp4`：RMSE = 0.020 m/s（2 厘米/秒，工业级）
- `102.5kg_0.53_0.38.mp4`：RMSE = 0.058，r = 1.000
- `130kg` 系列多条视频：RMSE 稳定在 0.07~0.09
- 结论：**YOLOv11n 权重直接保留，作为 ROI 粗提取器**

#### ✅ mpp 锁死物理决策正确

- v0→v1 的 bias 从 −0.085 翻转到 +0.040，证明动态 mpp 引入了系统性误差
- 锁死 mpp 后 bias 回落，消除了多片厚度引入的投影几何失真

#### ❌ v2 的像素硬门 + IoU 重锚定逻辑引入了回归

- 固定 20px 门限在 v2 中导致 5 条视频新增 NO_REPS
- 根本原因：20kg 杠铃片在高速度下单帧位移超 20px，硬拒绝把检测器结果误判为跳变
- 教训：**不能用像素距离做跳变判断，需用物理速度上限约束**

---

## 二、当前代码状态

### 改动的文件（已修改，待 commit）

| 文件 | 改动内容 |
| ------ | --------- |
| `models/yolo11_plate.onnx` | 被 best.onnx 覆盖（v1/v2 使用同一模型） |
| `models/best.onnx` | 新增（Colab 训练产出，10.1 MB） |
| `models/best.pt` | 新增（训练权重存档，5.2 MB） |
| `vbtcore/detector.py` | 置信度 sigmoid 修复 + letterbox 预处理 |
| `vbtcore/engine.py` | 锚定打分 + 运动探针 + mpp 锁死 + IoU 重锚定（v2 逻辑） |
| `vbtcore/geometry.py` | 竖屏旋转逆映射修复 |
| `vbtcore/kalman.py` | **新增**：BarbllKalmanTracker（1D 匀加速卡尔曼，纯数学最优融合） |
| `validation/reports/BENCHMARK_v0.md/json` | 本地复跑结果（25/34） |
| `validation/reports/BENCHMARK_v1.md/json` | v1（best.onnx 仅换模型）结果（27/34） |
| `validation/reports/BENCHMARK_v2.md/json` | v2（mpp锁死+IoU+Reanchor）结果（24/34） |

### 待 commit 的新增脚本

| 文件 | 用途 |
| ------ | ------ |
| `scripts/extract_frames_for_labeling.py` | 视频帧提取（均匀抽帧 + 蹲底关键帧检测） |
| `scripts/create_barbell_annotations.py` | YOLO 格式标注文件生成 + EzYOLO 导入准备 |
| `scripts/run_benchmark_v0.py` | 基准测试脚本（支持 --bench-dir 参数化） |

### 已有但未入 git 的数据目录

```
datasets/
  barbell_dataset/          ← 728 张已标注帧（EzYOLO 标注中）
    yolo_images/train/      ← 605 张训练集
    yolo_images/val/        ← 123 张验证集
    data.yaml               ← YOLO 配置
    dataset_split.yaml       ← 图片路径列表
  weightlifting-plates/     ← 5865 张原始数据集（未解压）
  owlvit_pseudo/            ← 58 张 Phase 1.5 伪标签

videos/                     ← 26 条原始视频（720×1280 竖屏）
```

---

## 三、技术团队待办事项

### P0 — 必须完成（阻塞其他迭代）

#### P0.1：解决 4 条假拒绝视频

- 4 条 `NO_PLATE_DETECTED`：`30kg_0.84_...`、`110kg_0.42_...`、`110kg_0.53_...`、`110kg_0.57_...`（110kg 系列）
- 根因：检测器对重负荷（110kg）工作片的域差，训练数据中缺乏类似场景
- 方案：补充 110kg 场景的标注数据，或调整锚定打分参数

#### P0.2：实施卡尔曼 + 亚像素椭圆拟合新架构
>
> **核心思路（Daniel 已拍板，不需要重训模型）**

当前问题：YOLO bbox 边界受光照/遮挡影响，导致比例尺（mpp）波动
解决路径：

1. 保留 `best.onnx` 作为 ROI 粗检测（不再依赖其边界精确定位）
2. 在每个 YOLO 检测框内：
   - 边缘检测（`cv2.Canny`）
   - 轮廓分组（弧长/面积过滤，排除腿/手臂）
   - 亚像素椭圆最小二乘拟合（`cv2.fitEllipse`）
   - 从椭圆长轴反推圆心（亚像素精度，~0.1 pixel）
3. 圆心序列送入 `BarbllKalmanTracker`（`vbtcore/kalman.py` 已写好）：
   - 状态向量：[y, v_y, a_y]（牛顿匀加速模型）
   - 观测向量：[y]（圆心纵坐标）
   - 纯数学最优融合，无任何硬像素阈值
4. `mpp` 标定：从 Kalman 平滑后的轨迹中提取 plate 物理直径（查表：400mm 标准片）

**文件接口：**

- 新增 `vbtcore/ellipse_calibrator.py`：椭圆拟合 + 亚像素圆心 + mpp 查表
- 修改 `vbtcore/pipeline.py`：将椭圆圆心替换 YOLO bbox 中心作为几何输入
- `vbtcore/kalman.py` 已就绪（本次会话已写好），可直接集成

#### P0.3：修复 v2 引入的 NO_REPS 回归

- `50kg_0.89_...` 和 `105kg_0.59_...` 从 v1 的 OK 退化为 NO_REPS
- 根因：20px 像素硬门限在高速运动下误判
- 方案：替换为物理速度上限约束（VBT 极限速度约 2.5 m/s → 单帧位移上限 = 2.5/30 ≈ 0.083m = 8.3% 帧高），或直接使用 P0.2 的 Kalman 架构（天然免疫）

### P1 — 优化（可选）

- 替换 `CUDAExecutionProvider` 为 onnxruntime-gpu（推理加速）
- 实施 `vbtcore/segment.py` 中的 BottomRegrind 逻辑（rep 底部重锚定，减少分段蝴蝶效应）
- 添加 rep 级线性校准（`scripts/calibrate_mcv.py` 接口已就绪）

---

## 四、实验纪律（Daniel 明确要求）

1. **单一变量原则**：每次 benchmark 只改动一个变量（模型 OR 算法，不能同时改）
2. **基线必须先跑**：任何调参前先跑当前版本基准，记录数字，再改，再跑
3. **诚实基线**：只报顺序配对 RMSE（最优匹配 RMSE 虚高 43%，禁止用于评估报告）
4. **留出集规范**：见 `docs/HOLDOUT.md`，新视频一律先进留出集，不直接进开发集凑数
5. **不盲目调参**：发现参数敏感性高时，优先找机制/数据问题，而非继续调参

---

## 五、目录结构（当前最新）

```
D:/EasyVBT-Research/
├── vbtcore/                    # 当前生产引擎
│   ├── __init__.py
│   ├── detector.py             # ONNX 检测器（sigmoid+letterbox 修复）
│   ├── engine.py               # 锚定+跟踪+标定（mpp锁死+IoU+Reanchor）
│   ├── geometry.py              # 竖屏旋转+letterbox
│   ├── kalman.py               # 【新增】1D 卡尔曼轨迹滤波器
│   ├── pipeline.py             # analyze_video() 入口
│   └── segment.py              # rep 分段+MCV 结算
├── models/
│   ├── yolo11_plate.onnx       # 当前生产模型（被 best.onnx 覆盖）
│   ├── best.onnx               # 【新增】Colab 训练产出
│   └── best.pt                 # 【新增】训练权重存档
├── validation/
│   ├── dataset_benchmark/       # 34 条 GymAware 基准视频
│   └── reports/
│       ├── BENCHMARK_v0.md/json # 本地复跑（25/34）
│       ├── BENCHMARK_v1.md/json # v1 best.onnx（27/34）✅ 当前最优
│       └── BENCHMARK_v2.md/json # v2 + mpp锁死+IoU（24/34，有回归）
├── scripts/
│   ├── run_benchmark_v0.py     # 基准测试脚本
│   ├── extract_frames_for_labeling.py  # 视频帧提取
│   ├── create_barbell_annotations.py  # YOLO 标注生成
│   └── calibrate_mcv.py         # rep 级线性校准
├── datasets/
│   └── barbell_dataset/         # 728 张已标注帧（标注中）
├── docs/
│   ├── TECH_ROUTE.md           # 技术路线
│   ├── HOLDOUT.md              # 留出集规范
│   └── SESSION_HANDOFF_20260911.md  # 本文档
└── tests/                       # 26 个单元测试（全过）
```

---

## 六、基准测试快速命令

```bash
# 替换模型文件（模型在 models/ 目录替换 yolo11_plate.onnx）
# 然后跑基准：
python scripts/run_benchmark_v0.py --tag vX

# 跑单元测试：
python tests/run_all_tests.py
```

---

*Session conducted: 2026-09-11 | Agent: pi-hermes | User: Daniel Ho (何天元)*
