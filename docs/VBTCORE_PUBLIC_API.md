# vbtcore 公共 API 文档（v0.1.0-baseline）

> 生效日期：v0.1.0-baseline 起
> 公共 API：9 个核心符号
> 兼容性策略：已归档符号保留 1 个 release 周期（v0.1.x）

---

## 公共 API（9 个核心符号）

```python
from vbtcore import (
    PlateDetector,          # 检测器（生产 ONNX）
    Detection,              # 检测结果 dataclass
    StaticPlateCalibrator,  # 尺度标定（CV 门禁 + 静态中位数锁死）
    KinematicKalmanTracker, # 物理空间卡尔曼（米/秒）
    DenseVisualTracker,     # LK 光流 + 卡尔曼融合
    BiomechanicalRepSegmenter,  # 速度 FSM 分段（深蹲/卧推 + 硬拉）
    Rep,                    # rep 结果 dataclass
    SetResult,              # 端到端分析结果 dataclass
    StatusCodes,            # 状态码常量
    analyze_video,          # 主入口函数
)
```

### `PlateDetector`

**路径**：`vbtcore.detector.PlateDetector`

```python
det = PlateDetector(model_path="models/best.onnx", providers=["CPUExecutionProvider"])
results = det.detect(frame_bgr, conf_thresh=0.40)
# 返回 list[Detection]
```

### `Detection`

```python
@dataclass
class Detection:
    cx: float        # 中心 x（像素）
    cy: float        # 中心 y（像素）
    w: float         # 宽（像素）
    h: float         # 高（像素）
    conf: float      # 置信度
    ratio: float     # w/h 圆度比
```

### `StaticPlateCalibrator`

**路径**：`vbtcore.calibrator.StaticPlateCalibrator`

```python
cal = StaticPlateCalibrator(real_diameter_m=0.45, min_static_frames=20, max_cv=0.015)
cal.add_sample(bbox_height_px)
if cal.is_ready():
    mpp = cal.lock_scale()  # 米/像素
```

### `KinematicKalmanTracker`

**路径**：`vbtcore.tracker.KinematicKalmanTracker`

物理空间 1D 卡尔曼（牛顿匀加速模型），状态向量 `[y, v_y, a_y]`，观测 `y`。
适用于帧间 `predict()` + 稀疏检测 `update()`。

### `DenseVisualTracker`

**路径**：`vbtcore.tracker.DenseVisualTracker`

LK 光流 + YOLO 物理空间卡尔曼融合，跟踪器入口。

```python
tracker = DenseVisualTracker(mpp=mpp, initial_bbox=(x1, y1, x2, y2),
                              initial_gray=gray, initial_time_s=0.0)
y_m, v_mps = tracker.step_interframe(gray, current_time_s)
```

### `BiomechanicalRepSegmenter`

**路径**：`vbtcore.segmenter.BiomechanicalRepSegmenter`

```python
seg = BiomechanicalRepSegmenter(exercise_type="squat_bench")
reps = seg.segment(timestamps, positions_m, velocities_mps)
# 返回 list[Rep]
```

支持的 `exercise_type`：

- `"squat_bench"`：SSC 模式（深蹲、卧推）
- `"deadlift"`：无 SSC（直接向心）

### `Rep`

```python
@dataclass
class Rep:
    start_idx: int           # 起始帧（闭区间）
    end_idx: int             # 结束帧
    start_time: float        # 起始时间（秒）
    end_time: float          # 结束时间（秒）
    duration_s: float        # 向心段时间
    rom_m: float             # 位移幅值（米）
    mcv_mps: float           # 向心段平均速度（主指标 = ROM / duration）
    pcv_mps: float           # 向心段峰值速度
```

### `analyze_video`

**路径**：`vbtcore.pipeline.analyze_video`

```python
from vbtcore import analyze_video, SetResult

result: SetResult = analyze_video(
    video_path="input.mp4",
    model_path="models/best.onnx",
    redet_every=15,
    plate_diameter_m=0.45,
    outer_plate="20kg",
    exercise_type="squat_bench",
)

if result.status == StatusCodes.OK:
    for rep in result.reps:
        print(f"MCV={rep.mcv_mps:.3f} m/s, ROM={rep.rom_m:.3f}m")
```

### `StatusCodes`

```python
class StatusCodes:
    OK = "OK"
    NO_PLATE_DETECTED = "NO_PLATE_DETECTED"
    NO_CLEAN_SEGMENT = "NO_CLEAN_SEGMENT"
    TOO_SHORT = "TOO_SHORT"
    VIDEO_ERROR = "VIDEO_ERROR"
    CALIBRATION_FAILED = "CALIBRATION_FAILED"
```

---

## 辅助 API（按需从子模块导入）

```python
from vbtcore.geometry import (
    PLATE_DIAMETERS_M,         # 杠铃片直径查表
    compute_mpp,               # 像素/米比例尺
    fit_plate_ellipse,         # 亚像素椭圆拟合
    resolve_plate_diameter,    # 外层片规格→直径
    preprocess,                # YOLO letterbox 预处理
    canvas_to_orig,            # canvas→原图坐标逆映射
)
from vbtcore.kalman import BarbellKalmanTracker  # 1D 卡尔曼（不带 LK 光流）
```

---

## 已归档符号（DEPRECATED，v0.1.x 内仍可访问）

通过 `vbtcore.engine`、`vbtcore.tracker_ek`、`vbtcore.detector_phase0`、
`vbtcore.segment` 模块访问已归档实现，**访问时会触发 `DeprecationWarning`**。

| 已归档模块 | 已归档符号 | 替代 API |
| --- | --- | --- |
| `vbtcore.engine` | `DetectFitTracker`, `TrackDiagnostics`, `anchor_score`, `BottomRegrind`, `select_regrind_candidate`, `regrind_verdict`, `make_template`, `MotionProbe`, `select_anchor_track` | `analyze_video()` / `DenseVisualTracker` |
| `vbtcore.tracker_ek` | `EllipseKalmanTracker`, `EKTrackDiagnostics` | `DenseVisualTracker` / `BarbellKalmanTracker` |
| `vbtcore.detector_phase0` | `detect_video_phase0`, `write_phase0_json`, `model_hash` | `PlateDetector` |
| `vbtcore.segment` | `Rep`, `SegmentResult`, `segment_reps`, `segment_reps_from_velocity`, `small_gap_interp`, `longest_clean_run` | `BiomechanicalRepSegmenter` / `Rep` (segmenter) |

迁移示例：

```python
# ❌ 旧（已废弃）
from vbtcore.engine import DetectFitTracker
tracker = DetectFitTracker(detector)
ys = tracker.process(video_path)

# ✅ 新（推荐）
from vbtcore import analyze_video
result = analyze_video(video_path, model_path)
```

---

## 归档文件位置

```
_archive/vbtcore_history/
├── engine_v2_reverted.py         # 914 行 DetectFitTracker 实现
├── tracker_ek_v4.py              # EllipseKalmanTracker 实现
├── detector_phase0_v0.py         # 纯检测基线（被 PlateDetector 替代）
└── segment_v3_savgol.py          # savgol + 两遍门 rep 分段（被 FSM 替代）
```

---

*本文档随 vbtcore 公共 API 同步更新。修改公共 API 时必须同步更新本文档。*
