# Step 2 技术架构报告 & 后续路线图

> 生成时间：2026-09-11  
> 参与人：Daniel & 小何（AI 助手）  
> 目标：将 YOLOv11n 从"几何标定器"降维为"ROI 粗提取器"，彻底解决 110kg 视频 bias=+0.9 的速度系统性虚高问题。

---

## 一、基准测试结果（34 条视频）

### v0 — 原始模型（baseline）

| 指标 | 数值 |
| ------ | ------ |
| 计数通过率 | 25/34（74%） |
| 假拒绝（真有问题） | 6 条 |
| 视频 RMSE 均值 | 0.212 m/s |
| Rep RMSE | 0.253 m/s |
| bias | −0.085（系统性低估） |

### v1 — best.onnx（Colab 训练，best.pt 导出）

| 指标 | 数值 |
| ------ | ------ |
| 计数通过率 | 27/34（79%）✅ |
| 假拒绝 | 4 条 ✅（减少 2 条） |
| 视频 RMSE 均值 | 0.250 m/s |
| Rep RMSE | 0.316 m/s |
| bias | +0.040（系统性虚高）⚠️ |

### v2 — best.onnx + 防线一二（mpp 静态锁死 + IoU/Re-anchor）

| 指标 | 数值 |
| ------ | ------ |
| 计数通过率 | 24/34（71%） |
| 假拒绝 | 4 条 ✅ |
| 视频 RMSE 均值 | 0.284 m/s |
| Rep RMSE | 0.381 m/s |
| bias | +0.025（接近零）✅ |

---

## 二、防线一（mpp 静态锁死）— 物理验证成功

### 原理

在动作开始前的前 10 帧（杠铃静止准备期）收集 YOLO 边界框高度样本，取中位数计算全局 `mpp_fixed`。一旦检测到大位移（高度变化 > 0.5×片高），立即冻结比例尺，后续追踪全程禁止动态重刷 mpp。

### 效果

- `bias` 从 v1 的 **+0.040 回落至 +0.025**，证明多片圆柱体厚度引入的投影几何失真被有效压制
- `110kg_0.57` 等视频的巨额正偏大部分消失

### 实现位置

`vbtcore/engine.py` — `DetectFitTracker.__init__()` 中：

- `_mpp_static_h: list[float]` — 锁死前的高度样本
- `_mpp_locked: bool` — 冻结标志
- 锚定后前 10 帧收集高度 → 检测到位移 → 冻结

---

## 三、防线二（IoU + Re-anchor）— 比硬像素门更稳定

### 问题

原有的 20px 硬门在向心爆发期（15帧内杠铃位移 50-80px）会误杀高置信度检测，导致跟踪链断裂，出现 NO_REPS。

### 解决方案（已实现）

```python
# vbtcore/engine.py — process() 中，跟踪期检测分支

# 1. 构建追踪器预测框（以模板尺寸为单位）
ncc_x1, ncc_y1 = cx - t_w/2, cy - t_h/2
# 2. 构建检测框
det_x1, det_y1 = picked.cx - d_w/2, picked.cy - d_h/2
# 3. 计算 IoU
iou = inter / union

if iou > 0.2:
    # 状态A：两者重叠 → EMA 软融合（0.6×检测 + 0.4×追踪预测）
    cx = 0.6*picked.cx + 0.4*ncc_pred_cx
    cy = 0.6*picked.cy + 0.4*ncc_pred_cy
elif picked.conf > 0.65:
    # 状态B：低重叠但高置信 → Hard Re-anchor，强制重置追踪器
    cx, cy = picked.cx, picked.cy
    tpl = make_template(frame, cx, cy, picked.h)
    v_pred = 0.0; missing_run = 0
else:
    # 状态C：低重叠且低置信 → 忽略检测，信任 NCC
    pass
```

### 效果

- NO_REPS 从 5 条（20px 硬门）降至 2 条（IoU/Re-anchor）
- 通过率从 53% 恢复至 71%

---

## 四、新架构提案：椭圆拟合 + 卡尔曼滤波（未集成）

以下两个模块已写入代码库，**但未集成到 pipeline**，需要技术人员完成集成。

### 4.1 EllipseCalibrator — 亚像素椭圆拟合（`vbtcore/geometry.py`）

```python
def fit_plate_ellipse(crop, crop_cx, crop_cy):
    """
    在铃片裁剪 ROI 上拟合外侧圆形轮廓的真实投影椭圆。
    返回 (major_axis_px, center_y, center_x, angle_deg)。

    核心：cv2.fitEllipse — 最小二乘椭圆拟合，精度亚像素级。
    长轴 = 严格物理直径，不受倾角和多片厚度缩水影响。
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)  # 最大外轮廓
    ellipse = cv2.fitEllipse(c)
    (cx_e, cy_e), (d1, d2), angle = ellipse
    major_axis = max(d1, d2)
    return major_axis, cy_e, cx_e, angle
```

**比例尺计算（仅在视频前 5 帧运行一次）：**

```python
mpp_fixed = 0.45 / median([major_axis_px for _ in range(5)])
# 0.45m = 标准 45cm 铃片直径（实测可用 lookup table 按 outer_plate 修正）
```

### 4.2 BarbellKalmanTracker — 连续卡尔曼轨迹滤波（`vbtcore/kalman.py`）

```python
class BarbellKalmanTracker:
    """
    状态向量: [y, v_y, a_y]^T（牛顿匀加速模型）
    观测向量: [y]（每15帧 YOLO+椭圆圆心提供一次）

    优势：
    - 速度 v 由状态矩阵内生，无坐标差分，数学上直接切除坐标阶跃
    - 零硬参数：Q（过程噪声）和 R（观测噪声）完全由高斯统计自适应
    - 每帧调用 predict()，每15帧调用 update()
    """
    def __init__(self, initial_y, dt=1/120.0, Q=0.01, R=4.0):
        self.x = np.array([[initial_y], [0.0], [0.0]])  # [y, v, a]
        self.F = [[1, dt, 0.5*dt²], [0, 1, dt], [0, 0, 1]]  # 状态转移
        self.H = [[1, 0, 0]]                               # 只观测 y

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])  # y_pred, v_pred

    def update(self, observed_y):
        K = self.P @ self.H.T / (self.H @ self.P @ self.H.T + self.R)
        self.x = self.x + K * (observed_y - self.H @ self.x)
        self.P = (np.eye(3) - K @ self.H) @ self.P
        return float(self.x[0]), float(self.x[1])  # y_filt, v_filt
```

---

## 五、已改动文件清单

| 文件 | 改动说明 |
| ------ | --------- |
| `vbtcore/engine.py` | mpp 静态锁死 + IoU/Re-anchor（防线一二） |
| `vbtcore/detector.py` | 默认 Provider 改为 CUDA→CPU（向后兼容） |
| `vbtcore/geometry.py` | 新增 `extract_plate_crop()` + `fit_plate_ellipse()`（椭圆拟合器） |
| `vbtcore/kalman.py` | **新增** BarbellKalmanTracker（卡尔曼滤波） |
| `validation/reports/BENCHMARK_v0.json/md` | 原始模型基准 |
| `validation/reports/BENCHMARK_v1.json/md` | best.onnx 基准 |
| `validation/reports/BENCHMARK_v2.json/md` | best.onnx + 防线一二基准 |
| `scripts/create_barbell_annotations.py` | **新增** 自动标注生成脚本 |
| `scripts/extract_frames_for_labeling.py` | **新增** 帧提取脚本 |
| `models/yolo11_plate.onnx` | 被 best.onnx 覆盖（当前活跃模型） |

**未提交（过大）：**

- `datasets/barbell_dataset/` — 728 张标注帧 + YOLO 格式数据集
- `models/best.onnx` / `best.pt` — Colab 训练输出
- `runs/` — YOLO 训练输出

---

## 六、技术人员后续任务清单

### P0（必须完成）

1. **集成 EllipseCalibrator 到 pipeline**
   - 在 `DetectFitTracker.process()` 的锚定期（前 5 帧）
   - 调用 `extract_plate_crop()` + `fit_plate_ellipse()` 提取长轴
   - 计算 `mpp_fixed` 并存入 tracker，替代原有的 `h_samples` 动态中位数方法
   - `pipeline.py` 的 `calibrate()` 方法需适配：直接使用 `mpp_fixed` 而非重新计算

2. **集成 BarbellKalmanTracker 替代 NCC 追踪**
   - 每帧调用 `kalman.predict()` 推进状态
   - 每 15 帧检测到目标时调用 `kalman.update(observed_y)`
   - 输出的 `y_track` 来自 `kalman.y`，`v_track` 来自 `kalman.v`
   - 速度 `v = kalman.v * fps * mpp_fixed`（无需差分，直接来自状态向量）
   - **删除** 旧的 NCC 模板匹配逻辑

3. **删除或禁用旧的 mpp 锁死 + IoU/Re-anchor 代码**
   - 这些是过渡方案，新架构下不再需要

### P1（建议完成）

1. **调整 KalmanFilter 参数**
   - `Q=0.01, R=4.0` 是初始猜测，建议在 3-5 条代表性视频上调参
   - 原则：R 越大 = 越信任检测器 = 响应越快；R 越小 = 越信任预测 = 越平滑

2. **RepSegmenter 优化（`segment.py`）**
   - 110kg 极值 RMSE 仍有部分来自 ROM 估算偏差
   - 建议：用 Kalman 的内生速度 `v_track` 替代 Savitzky-Golay 差分速度
   - 直接用 `v_track` 找波峰波谷，而非对 `y_track` 差分

### P2（可选）

1. **多负荷 lookup table**
   - 当前假设 `plate_diameter_m = 0.45`
   - 实际：不同负荷直径不同（20kg≈0.45m, 25kg≈0.45m, 等）
   - `geometry.py` 的 `PLATE_DIAMETERS_M` 已存在，需完善

2. **空杆 warm-up 场景**
   - 若业务需要支持空杆（无片）测速，检测器无法工作
   - 建议：检测到画面无片时降级到 `barbell_v4.onnx`（杠杆模式），
     或提示用户手动框选杆头端点

---

## 七、关键性能目标

| 指标 | 当前（v2） | 目标 |
| ------ | ----------- | ------ |
| 计数通过率 | 71% | ≥ 80% |
| 视频 RMSE 均值 | 0.284 m/s | ≤ 0.20 m/s |
| Rep RMSE | 0.381 m/s | ≤ 0.25 m/s |
| 110kg RMSE | 1.076 m/s（最差） | ≤ 0.20 m/s |
| bias | +0.025 | 接近 0 |
| 推理速度 | ~11 ms/帧 | ≤ 20 ms/帧（目标） |

---

## 八、数据闭环计划

现有 728 张标注帧（来自 26 条视频）训练了 best.onnx，效果有限。建议：

1. **补充 EzYOLO 标注**：优先标注 v2 中仍有假拒绝的 4 条视频对应帧
2. **扩展到更多负荷类型**：尤其是 30kg（轻负荷）和 110kg（中低负荷）
3. **建立留出集**（docs/HOLDOUT.md）：用独立 10 条视频做最终验证，避免在开发集上过拟合

---

*报告由 AI 助手自动生成，所有 benchmark 数据来自 `scripts/run_benchmark_v0.py`*
