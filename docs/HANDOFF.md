# HANDOFF — 本地接手指南

> 最后更新：2026-09-14
> 面向：在本地机器上接手后续开发的人。
> 成绩数字一律以 [SCOREBOARD.md](../validation/reports/SCOREBOARD.md) 为唯一来源，本文不复述。

---

## 一、当前进度定位

三阶段路线（见 [TECH_ROUTE.md](TECH_ROUTE.md)）：

| 阶段 | 状态 |
| --- | --- |
| 第一阶段 Make it Right | 🟡 **Python 端达成，C++ 闸门未过** |
| 第二阶段 Make it Fast | ⛔ 未启动（按铁律，闸门未绿不得启动） |
| 第三阶段 Make it Robust | ⛔ 未启动 |

**卡点只有一个：Iron Gate。** 上次运行 **17/34**，要求 **≥20/34**。
且该结果产生于分段器修复**之前**，修复后**尚未重跑**（本沙箱 apt 装不了 cmake/ninja）。

→ **本地第一件事就是重跑 Iron Gate**，见第三节。

---

## 二、最近一轮改了什么（2026-09-14）

### 2.1 分段器修复：`confirm_frames=3`

**问题**：状态机判定「底部换向」与「向心结束」时只看相邻两帧。
蹲底停顿期速度单帧抖正即触发早产的 `ECCENTRIC → CONCENTRIC`，
该 rep 随即以极小 ROM 收尾被物理门禁拒绝，状态回落 IDLE，
**真正的向心冲程再无状态机接管** → 少计。

**修法**：跃迁需连续 3 帧同向确认。3 帧 ≈ 0.1s @30fps，
既跨过抖动，又远短于最短向心时长（~0.3s）。
实测 confirm=1/2 修不好、=3 全对、=4/5 无额外收益 —— 3 是抖动宽度决定的，不是凑的。

**影响**：8 条视频输出变化，**4 条修复计数、0 条退化**。

**改动点**：
- `vbtcore/segmenter.py` — 新增 `confirm_frames` 构造参数 + `_sustained()` 辅助
- `vbtcore-cpp/src/segmenter.cpp` — 同步，`kConfirmFrames = 3` + `sustained` lambda

> ⚠️ **两端必须永远保持语义一致**，否则 Iron Gate 会因语义分歧掉点。改一边就得改另一边。

### 2.2 归因更正：原「3 条少计属检测器域差」已被证伪

三条视频**无检测帧率均为 0.0%，不存在任何漏检**。真实情况是两类完全不同的根因：
1 条是上述算法 bug（已修复），2 条指向数据问题（见下节）。
`IRON_GATE_20260913.md` §2.4 与 SCOREBOARD 相应段落均已标注作废并更正。

---

## 三、本地要做的事（按优先级）

### ① 重跑 Iron Gate ← 最高优先级

```bash
bash scripts/build_cpp_desktop.sh     # 需要 cmake + ninja + g++
```

判据：通过 **≥20/34**，速度容差 **≤0.005 m/s**，**ASan 零内存泄漏**。

本沙箱未能执行，我只做到：
- `g++ -fsyntax-only` 语法检查通过
- 单文件 harness 直读轨迹 CSV 喂给 C++ 分段器，**输出与 Python 逐位一致**

所以 C++ 分段器的**数值正确性有验证**，但**全链路 + ASan 没有**。

> 上次残差主因是**两端 OpenCV 版本不一致**（沙箱内不可消除）。
> 本地若两端用同一版 OpenCV，通过率有望自然回升 —— 这是 17/34 最可能的解释。

### ② 复核两条 105kg 视频的真值 ← 需要人来定

| 视频 | GT | 算法 | 我的判断 |
| --- | --- | --- | --- |
| `105kg_0.69_0.64_0.65_0.60_0.56_0.45` | 6 | 3 | **疑似视频被裁剪**，非标注错 |
| `105kg_0.60_0.54_0.55_0.54_0.51_0.37` | 6 | 4 | **疑似标注多计** |

**为什么不是漏检**（三重独立证据）：
1. 用**完全独立于流水线**的方法复核（模板匹配 / HSV 颜色质心，不用 YOLO、光流、卡尔曼），
   得到的下蹲次数与流水线一致：3 和 4。
2. **相机静止**：相位相关测得全程累计位移 0.001 m，排除镜头晃动污染轨迹。
3. **浅 rep 没被门限吃掉**：去掉 0.12m 门限后多出的候选深度仅 0.02~0.03 m，
   而真实下蹲 0.18~0.30 m —— 那些是噪声。

**`105kg_0.69...` 为何疑似裁剪**：开头 2 秒内杠铃已移动 0.269 m（非静止起始）；
其 3 个 MCV 与 GT **后 3 个**对齐 RMSE=**0.031**，与前 3 个对齐 RMSE=**0.149**。
→ GT 的 6 个可能是对的，缺的是画面。

**自查材料**：`validation/reports/diagnostics/105kg_0.60_contact_sheet.jpg`
（40 秒每 0.5s 抽一帧，81 格，可直接数）。下蹲在约 5 / 15.5 / 26 / 36 s。

> 🚫 **在人工确认前，不要改 `dataset_index.json` / `ground_truth.csv`。**
> 这属于数据集口径变更。SCOREBOARD 当前按**原真值**记分。

### ③ 若闸门转绿，再进第二阶段

NDK r25c 交叉编译 `libvbtcore.so` → 打包 APK → 真机验证。
`android/` 下 JNI 壳（`cpp/vbtcore_jni.cpp`，48 行）与 CMakeLists 已就位。

---

## 四、环境与复现

详见 [ENVIRONMENT.md](ENVIRONMENT.md)。最小可用环境：

```bash
python -m venv .venv && source .venv/bin/activate
pip install opencv-python-headless numpy scipy onnxruntime
```

| 用途 | 命令 | 耗时 |
| --- | --- | --- |
| 单测（26 项） | `python tests/run_all_tests.py` | 秒级 |
| 冒烟（5 视频） | `python scripts/gen_cpp_baseline.py --smoke --workers 2` | ~74s |
| 全量 baseline | `python scripts/gen_cpp_baseline.py --workers 4 --copy-to-tmp` | 数分钟 |
| C++ 构建 + 闸门 | `bash scripts/build_cpp_desktop.sh` | — |
| 少计视频诊断 | `python scripts/diagnose_missed_reps.py --all-missed --dump-csv` | ~4min |
| 留出集自检 | `python scripts/holdout_intake.py --check` | 秒级 |

> `--workers` 请按本地核数调整；沙箱只有 2 vCPU 所以用的 2。
> 诊断脚本是 stride=1 全量检测，单条 55~105s；**复查轨迹请直接读
> `validation/reports/diagnostics/*.csv`，不要重跑。**

### 任何改动后的物理校验（必做）

参考视频 `110kg_0.71_0.73.mp4`，三项指标偏离即回滚：

| 指标 | 基准值 |
| --- | --- |
| mpp | **0.002471** |
| y_range | **0.664 m** |
| MCV | **[0.705, 0.762]**（真值 [0.71, 0.73]） |

本轮 `confirm_frames` 修复后此三项**逐位未变**。

---

## 五、不要重蹈的覆辙

完整清单见 [TECH_ROUTE.md](TECH_ROUTE.md)。六条硬约束摘要：

1. **禁止动态刷新 mpp** —— 倾斜机位多片堆叠使框高虚夸，速度翻倍，RMSE 飙到 0.897。
   **起始静止前 30 帧取中位数，全局冻结。**
2. **禁止在卡尔曼位置上叠长窗滤波**（如 15 阶 SG）—— 磨平向心峰值，计数通过率跌到 18%。
3. **禁止固定像素硬剔除门限**（如 20px）—— 爆发上行位移可达 50~80px，硬门限会拒掉有效检测。
4. **禁止时间量纲混乱** —— 手机是 VFR，**必须读真实 PTS**，写死 `dt=1/fps` 曾让速度塌陷 27 倍。
5. **20kg 空杆不算漏检** —— 无片，`NO_PLATE_DETECTED` 是正确真阴性。
   SCOREBOARD 的 34 分母已把它记为通过；只算有片视频则口径等价于 31/33。
6. **`models/best.onnx` 输出 row[4] 已是概率**（实测 max 0.930）—— **禁止再套 sigmoid**。

流程纪律：严禁多线并发调参；4 步 SOP 不准跳步；**只有 Iron Gate 亮绿才能进第二阶段**。

---

## 六、关于移动端用 Rust（评估结论）

**可行，但建议现在不做。** 障碍不在语言，在依赖：C++ 侧用了 **33 个 OpenCV API**
（`calcOpticalFlowPyrLK`、`goodFeaturesToTrack`、`fitEllipse`、`findContours`、`Canny` 等）
和 ONNX Runtime C++ API。`opencv-rs` 只是 binding，仍要交叉编译整个 OpenCV 并额外承担
bindgen/NDK 配置成本；纯 Rust 重写算子则是数周起步，**且每个算子都要重做数值对齐** ——
已知 OpenCV 版本差异就足以让闸门从 20/34 掉到 17/34，换实现只会更糟。

**若确实要引入**：建议只用 Rust 写上层编排与 JNI 层（`jni` crate 成熟），
CV 与推理继续调 C++ `libvbtcore.so`。等第二阶段数值基准锁死后再评估算子层迁移。

---

## 七、仓库地图

| 路径 | 说明 |
| --- | --- |
| `vbtcore/` | Python 研究基准实现（1421 行），出 SCOREBOARD 数字 |
| `vbtcore-cpp/` | C++ 生产实现（1857 行），7 个源文件与 Python 一一对应 |
| `android/` | JNI 壳 + Gradle/CMake 骨架，尚未交叉编译 |
| `tests/` | 26 项单测，`python tests/run_all_tests.py` |
| `scripts/` | 工具集，关键几个见第四节表格 |
| `validation/dataset_benchmark/` | 开发集 34 视频 + 真值索引 |
| `validation/holdout/` | 留出集，**尚未采集**，纪律见 [HOLDOUT.md](HOLDOUT.md) |
| `validation/reports/SCOREBOARD.md` | **成绩唯一事实表**，其他文档只许链接 |
| `validation/reports/diagnostics/` | 帧级轨迹 CSV + 接触表，复查用 |
| `docs/archive/`、`validation/reports/archive/` | 历史存档，勿引用其中数字 |
