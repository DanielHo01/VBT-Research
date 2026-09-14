# VBT-Research

手机单目视觉的杠铃速度测量（Velocity-Based Training）研究仓库。
输入竖屏侧视深蹲视频 → 输出每个 rep 的平均向心速度（MCV）与计次。

> **成绩引用规则**：全仓库唯一的成绩事实表是
> [`validation/reports/SCOREBOARD.md`](validation/reports/SCOREBOARD.md)。
> 本 README 与任何 docstring 都不再各自抄写数字。

---

## 当前状态（2026-09-14）

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| **第一阶段 · Make it Right** | ✅ **已达成** | stride=1 全量检测，开发集 rep RMSE **0.0664**、r **0.9423**、计数 **31/34** |
| **第二阶段 · Make it Fast** | ⏸ 未启动 | 需先通过 Iron Gate（C++ 对齐 ≥20/34），当前 17/34 |
| **第三阶段 · Make it Robust** | 🔲 未开始 | 陀螺仪水平引导、遮挡自愈、热降频 |

**当前卡点**：Iron Gate 17/34（详见
[`IRON_GATE_20260913.md`](validation/reports/IRON_GATE_20260913.md)）。
闸门未绿 → 按铁律**未启动 Android NDK 交叉编译**。

**重要口径**：以上均为 **dev（开发集）成绩**。留出集 `validation/holdout/`
尚未采集，按 [`docs/HOLDOUT.md`](docs/HOLDOUT.md) 的双报纪律，
最终成绩必须以 `dev ｜ held-out` 双报形式给出。

---

## 快速开始

```bash
# 依赖（单测与基准只需这四个，无需 ultralytics 等重型包）
pip install opencv-python-headless numpy scipy onnxruntime

# 单元测试（26 个，锁定已修复的历史 bug）
python tests/run_all_tests.py

# 5 视频冒烟（~74 秒：空杆/轻片/中片/多片/极限）
python scripts/gen_cpp_baseline.py --smoke --workers 2

# 全量 34 视频重建基线
python scripts/gen_cpp_baseline.py --workers 4 --copy-to-tmp
```

### 最小调用示例

```python
from vbtcore import analyze_video

r = analyze_video(
    "validation/dataset_benchmark/raw_videos/110kg_0.71_0.73.mp4",
    "models/best.onnx",
    redet_every=1,          # 第一阶段口径：逐帧全量检测
)
print(r.status, r.mpp)                     # OK 0.002471
print([rep.mcv_mps for rep in r.reps])     # [0.705, 0.762]  真值 [0.71, 0.73]
```

---

## 架构

```text
视频帧
  ├─ Layer 0 检测     PlateDetector（YOLOv11n ONNX，letterbox 预处理）
  ├─ Layer 1 标定     StaticPlateCalibrator（起始静止期中位数 → mpp 全局冻结）
  ├─ Layer 2 跟踪     DenseVisualTracker（物理空间 CA 卡尔曼 + LK 光流）
  └─ Layer 3 分段     BiomechanicalRepSegmenter（速度 FSM → MCV / ROM / 计次）
```

时间基准统一使用真实 PTS 时间戳（`CAP_PROP_POS_MSEC`），
全链路运行在物理量纲（米、秒），无 px/frame 混用。

详见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) 与
[`docs/VBTCORE_PUBLIC_API.md`](docs/VBTCORE_PUBLIC_API.md)。

---

## 目录结构

```text
vbtcore/                  Python 引擎（公共 API 9 个符号）
vbtcore-cpp/              C++ 移植（7 源文件 + golden_test）
android/                  Android Demo（Compose UI + JNI 桥接）
models/
  best.onnx               生产检测模型（YOLOv11n，mAP@0.5=0.994）
  best.pt                 训练权重存档
scripts/
  gen_cpp_baseline.py     基线生成（--smoke / --workers / --copy-to-tmp）
  build_cpp_desktop.sh    C++ 编译 + Iron Gate
tests/                    26 个单元测试
validation/
  dataset_benchmark/      开发集 34 视频 + GymAware 真值
  holdout/                留出集（待采集，见 docs/HOLDOUT.md）
  reports/
    SCOREBOARD.md         ← 成绩唯一事实表
    IRON_GATE_20260913.md   C++ 对齐报告
    cpp_baseline/         34 个 per-video baseline JSON
    archive/              历史过程报告（勿引用）
docs/                     架构、技术路线、数据、留出集规范
```

---

## 模型

| 模型 | 指标 | 状态 |
| --- | --- | --- |
| `models/best.onnx` | mAP@0.5 = 0.994 ／ mAP@0.5:0.95 = 0.932 | ✅ 生产使用 |

基于 2058 张全场景图片（含暗光、低对比度、大运动模糊、25 张纯背景负样本）
微调的 YOLOv11n，输出维度 `[1, 5, 8400]`。

> **历史说明**：早期 README 曾宣称 `barbell_v4.onnx` 的「RMSE = 0.88」为最佳成绩。
> 该模型**不在本仓库中**，其数字不可复现，且口径与当前 MCV 定义不同，已作废。
> 当前唯一生产模型是 `best.onnx`。

---

## 负面约束清单（严禁倒退）

以下路径均已被实测证伪，详细证据见
[`validation/reports/archive/`](validation/reports/archive/)：

1. **动作中动态刷新 mpp** — 多片堆叠厚度投影使框高虚夸，速度翻倍（RMSE 飙至 0.897）。
   铁律：起始静止前 30 帧取中位数后**全局冻结**。
2. **卡尔曼位置上叠长窗滤波**（如 15 阶 SG）— 磨平向心爆发峰，计数跌至 18%、bias −0.396。
3. **固定像素硬剔除门限**（如 20px）— 爆发位移达 50–80px，硬门限误拒有效检测 → 脱轨。
4. **时间量纲混乱**（写死 `dt=1/fps` 或漏乘帧率）— 手机是可变帧率（VFR）。
   铁律：严格读取真实 PTS。
5. **把 20kg 空杆列为漏检** — 空杆物理上无片，`NO_PLATE_DETECTED` 是正确的真阴性。
6. **C++ letterbox 反变换坐标错误** — 曾致框高放大 4.26×、mpp=0.000571。已修复。
7. **C++ 置信度双重 sigmoid** — `row[4]` 已是概率，再套 sigmoid 会让
   8400/8400 个 anchor 全部过阈值（正确应为 ~20 个），mpp 虚小 4.37×。已修复。

---

## 开发约定

- 分支策略见 [`docs/BRANCH_POLICY.md`](docs/BRANCH_POLICY.md)
- 环境配置见 [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)
- CI：26 单测 + 2 视频 smoke 基准（`.github/workflows/ci.yml`）
- `validation/dataset_benchmark/algorithms/` 等旧管线保留作对照基线，不再维护

## License

Apache 2.0
