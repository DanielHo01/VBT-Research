# ARCHITECTURE

四层架构设计，从原始视频到最终速度输出。

---

## 系统总览

```
   ┌─────────────┐
   │  视频输入    │  手机竖屏侧视深蹲
   └──────┬──────┘
          ↓
   ┌─────────────┐
   │  Layer 0    │  YOLO 检测  →  plate bbox
   │  检测       │
   └──────┬──────┘
          ↓  bbox 序列 (frame_idx, cx, cy, w, h, conf)
   ┌─────────────┐
   │  Layer 1    │  TargetAssociator + DBSCAN
   │  关联       │  →  锁定同一片，过滤误检
   └──────┬──────┘
          ↓  单一目标的稳定轨迹
   ┌─────────────┐
   │  Layer 2    │  α-β Tracker + Savitzky-Golay
   │  平滑       │  →  帧间平滑 + 外推
   └──────┬──────┘
          ↓  干净的 (t, y) 序列
   ┌─────────────┐
   │  Layer 3    │  BatchAnalyzer
   │  结算       │  →  MV / PV / ROM / mCV
   └──────┬──────┘
          ↓
   ┌─────────────┐
   │  输出       │  mCV, rep count, 速度曲线
   └─────────────┘
```

---

## Layer 0：检测（YOLO）

**职责**：每帧（或跳帧）输出 plate 的 bbox。

**当前模型**：`barbell_v4.onnx`（YOLOv8s，11.6 MB）
**问题**：竖屏侧视深蹲底部，plate 被腿遮挡 → 框的长宽比骤增 → 被圆形度门控拒绝 → 轨迹断点
**方案**：替换为新训练的 `plate_v1.onnx`（单类微调，竖屏训练数据）

**输入**：单帧 RGB 图像 (resize 到 416×416)
**输出**：dict {cx, cy, w, h, score}
- cx, cy: bbox 中心
- w, h: bbox 宽高
- score: 置信度

**文件**：`validation/dataset_benchmark/algorithms/common.py::YoloPlateDetector`

---

## Layer 1：关联（Tracking）

**职责**：将每帧的 bbox 关联到同一个目标（plate），输出稳定的轨迹。

**两个组件**：
1. **TargetAssociator**（状态机）
   - 状态：INIT → ANCHORING → TRACKING → LOST → DRIFTED
   - 用距离门控（dist_threshold=400px）锁定目标
   - 处理短期遮挡（lost_timeout=500ms）

2. **DBSCAN**（聚类）
   - 在所有检测结果上做密度聚类
   - 选最大簇 = barbell 簇
   - 过滤误检

**文件**：`validation/dataset_benchmark/algorithms/pipelines.py`

**当前 Pipeline**：
- `pipeline_kalman_sg` — Kalman + SG（基准）
- `pipeline_associator` — Associator + α-β + SG（新）
- `pipeline_baseline_sg` — 简单 SG（最弱）
- `pipeline_kcf_sg` — KCF 跟踪（弃用，会跳目标）

---

## Layer 2：平滑（Filtering）

**职责**：在 Layer 1 输出基础上做帧间平滑和短期外推。

**两个方法**：
1. **Savitzky-Golay**（后处理）
   - 窗口=5, polyorder=2
   - 保留峰值特征
   - 适合速度曲线（rep 分割）

2. **α-β Tracker**（实时）
   - α=0.6, β=0.4
   - 用于实时外推被遮挡帧

**不做什么**：不做 Kalman 全局轨迹估计（专家 A/B 一致认为会让 BatchAnalyzer 输入污染）

---

## Layer 3：结算（BatchAnalyzer）

**职责**：从 (t, y) 序列算出 mCV、rep count、ROM。

**核心算法**（专家 A 路径）：
1. 计算速度 dy/dt（数值微分）
2. 用 speed threshold 检测 rep 边界
3. 对每个 rep：
   - 找 concentric 段（向上加速）
   - 求平均速度 = mCV

**输入**：完整的 (t, y) 序列（来自 Layer 2）
**输出**：dict {reps: [...], mcv: [...]}

**文件**：第三方或自实现（待补充）

---

## 数据流时间线

```
t=0     t=33ms   t=66ms   t=100ms ...
 │       │        │        │
 │       │        │        │
YOLO    YOLO     YOLO     YOLO     (30 fps，每帧检测)
 │       │        │        │
bbox    bbox     bbox     bbox
 │       │        │        │
 └───────┴────────┴────────┘
        ↓
  TargetAssociator (锁定 plate，丢弃误检)
        ↓
   稳定轨迹 (frame_idx, cx, cy)
        ↓
   α-β Tracker (实时平滑)
        ↓
   SG 后处理 (去除噪声，保留峰值)
        ↓
   BatchAnalyzer
        ↓
   mCV per rep
```

---

## 关键约束（设计原则）

1. **BatchAnalyzer 输入必须是 Layer 2 输出的原始/平滑轨迹**，不能再回去重新估计（避免循环依赖）
2. **不允许跨大 gap 插值**（gap > 10 帧保留 NaN），避免 BatchAnalyzer 误判为完整 rep
3. **关联器优先级 > DBSCAN**：先锁定目标，再做聚类
4. **圆形度门控是软门控**：用于标定（mpp 计算），不直接拒绝关联

---

## 当前痛点 → 决策

| 痛点 | 决策 |
|---|---|
| barbell_v4 在 110kg 上检测失败 | 重训模型（单类 plate）|
| RMS 跨视频不稳定 | 改用 ellipse short axis 算 mpp（竞品做法）|
| KCF 跳目标 | 改用 TargetAssociator（距离门控）|
| 缺失自标注数据 | 用公开数据集（weightlifting-plates-v11）|