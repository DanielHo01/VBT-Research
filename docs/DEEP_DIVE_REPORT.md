# VBT-Research 深度探索报告

> 探索日期：2026-09-10 ｜ 方法：全量代码走读 + 34 视频数据集核验 + 本环境（Linux/ONNX-CPU）实跑 3 个代表性视频 + 竞品调研
> 目的：为「做一个类似 Metric / VBTgo / QwikVBT 的 App 产品」摸清现有技术资产的真实成色

---

## 一、项目基本情况（一页速览）

| 维度 | 现状 |
|---|---|
| **定位** | 手机竖屏侧视深蹲视频 → 杠铃速度估计（MCV/mCV）+ 计次的计算机视觉研究仓库 |
| **技术栈** | Python + OpenCV + ONNX Runtime (YOLO 检测) + SciPy（SG 滤波/峰值检测） |
| **数据资产** | 34 条验证视频（720×1280@30fps，约 104 MB，已入 Git），负荷 20–140 kg，金标准来自 GymAware（LPT 线性位移传感器），共 130+ 个 rep 的 MCV 真值 |
| **模型资产** | 3 个 ONNX：`barbell_v4.onnx`（416px，README 称 RMSE=0.88）、`yolo11_plate.onnx`（640px，未验证）、`plate_v1.onnx`（640px，已知差） |
| **算法资产** | 5 条对比管线（baseline_sg / subpixel_spline / kalman_sg / global_smoothing / associator）+ 1 个新引擎（AnchorTemplateEngine）+ 学术级评估器（RMSE/MAE/Bias/Pearson/ICC/Bland-Altman） |
| **阶段状态** | Phase 1 完成（基线）；Phase 1.5 失败归档（OWL-ViT 伪标签）；Phase 2 未开始 |
| **工程质量** | ⚠️ 硬编码 Windows 路径（`D:\EasyVBT-Research\...`）、无单元测试、无 CI、无包结构；声称的最优模型 `barbell_v4.onnx.backup` **不在仓库中**，头号指标 RMSE=0.88 严格意义上不可复现 |

**一句话结论**：这是一个「算法假设验证」性质的研究仓库，四层架构（检测→关联→平滑→结算）方向正确、验证数据集有真实价值，但代码存在多处会在产品化中爆雷的实际 bug（下文第三节的实跑证据），距离 App 产品还缺「实时性、移动端部署、产品功能层」三大块。

---

## 二、代码与架构深度解析

### 2.1 文件地图（60 个文件中真正重要的 ~10 个）

```
validation/dataset_benchmark/
├── algorithms/
│   ├── common.py        ← 检测器 + 亚像素精修 + rep 分段 + 标定（182 行，核心原语）
│   └── pipelines.py     ← 5 条管线 + TargetAssociator 状态机（725 行，核心逻辑）
├── engines/
│   └── anchor_template_engine.py   ← yolo11_plate + 首帧锚定 + NCC 模板（500 行，新一代引擎）
├── metrics_evaluator.py            ← RMSE/MAE/Bias/Pearson/ICC(2,1)/Bland-Altman
├── run_benchmark.py                ← 4 管线批量对比 + 出图
├── dataset_index.json              ← 34 视频索引（load_kg + gt_reps_mcv）
├── ground_truth.csv                ← GymAware 金标准
└── raw_videos/                     ← 34 条 mp4（20–140kg，720×1280@30fps）
scripts/
├── run_full_benchmark.py           ← 34 视频全量基准（读 dataset_index）
├── interactive_label.py            ← OWL-ViT 交互式标注（Phase 1.5 遗产，有复用价值）
└── extract_owlvit_pseudo.py 等     ← OWL-ViT 伪标签（已证伪，归档级）
models/  barbell_v4.onnx(12MB) / yolo11_plate.onnx(10.6MB) / plate_v1.onnx(12.3MB)
```

### 2.2 数据流：四层架构（docs/ARCHITECTURE.md 与代码一致）

```
视频帧 ─→ Layer 0 检测（YOLO，每帧 bbox）
       ─→ Layer 1 关联（TargetAssociator 状态机 / DBSCAN 聚类去误检）
       ─→ Layer 2 平滑（中值滤波 + Savitzky-Golay；α-β 外推短时丢失）
       ─→ Layer 3 结算（segment_reps 峰谷配对 → MCV/PV/ROM/计次）
```

**Layer 0 检测**——两条实现路线并存：

| | `common.py::YoloPlateDetector` | `engines/anchor_template_engine.py` |
|---|---|---|
| 输入预处理 | ❌ 直接 resize（竖屏被压扁） | ✅ letterbox + 竖屏旋转90°再旋回 |
| 置信度解析 | ❌ 对已是概率的输出**再套一次 sigmoid** | ✅ 直接使用（注释明确"不要重复 sigmoid"） |
| 输出格式假设 | [1,5,8400]（但 barbell_v4 实际是 [1,6,3549]，靠 argmax 侥幸工作） | [1,5,8400]（与 yolo11_plate 匹配） |
| 多目标选择 | 取全图 argmax（单目标） | conf×尺寸×居中度打分，返回 top-2 |

**Layer 1 关联**——`TargetAssociator` 状态机（专家 B 方案）：
`INIT → ANCHORING(前5帧中值建锚) → TRACKING(距离门控400px) → LOST(无检测,外推≤500ms) → RECOVERED / DRIFTED(重建锚点)`
关键设计决策：不再用圆形度一票否决（遮挡时框被拉长仍可用中心点）；圆形检测只用于 mpp 标定；跨大 gap 不插值（防 BatchAnalyzer 误判）。

**Layer 2 平滑**——3 帧中值 + SG(window=5, poly=2) 保留峰值；`pipeline_associator` 里还有一段 α-β 匀速外推（仅补 1–3 帧短丢失）。

**Layer 3 结算**——`segment_reps()`：三信号自适应（选 Y/X/框高中变化最大者）→ `find_peaks(distance=0.5s, prominence=12px)` → peak 向后找 trough → 物理过滤（时长 0.2–7.5s，ROM≥8cm）→ MCV=concentric 段正速度均值。`AnchorTemplateEngine` 用另一套 bottom→top 事件配对，MCV 取**中点瞬时速度**（`v[mid]`，且 clip 到 0.10–2.50）——两种 MCV 定义并存，这是潜在的不一致性。

**标定（mpp）**：用 45cm 标准杠铃片直径反推 米/像素。`calibrate_scale()` 里有个 `scale_factor=1.15` 的神秘系数（补偿透视/片厚），`AnchorTemplateEngine` 则直接 `0.45/median_h` 不加系数——同样不一致，标定逻辑需要统一。

### 2.3 评估体系（做得好的部分）

`metrics_evaluator.py` 是学术级的：ICC(2,1) two-way mixed 手写实现正确，Bland-Altman LoA、Pearson r、逐 rep 配对（truncate/贪心最近邻两种策略）。`run_benchmark.py` 能出 1:1 散点图 + BA 图。达标线定为 RMSE<0.063 m/s（对标 GymAware 的工程目标）。

---

## 三、本环境实跑结果（重要：暴露的真实问题）

在 Linux + ONNX CPU 环境对 3 个代表性视频（轻 20kg / 中 80kg / 重 102.5kg）实跑两条引擎路线：

### 3.1 `pipeline_associator`（barbell_v4.onnx，416px）

| 视频 | GT (m/s) | 预测 (m/s) | 覆盖率 | RMSE |
|---|---|---|---|---|
| 20kg_0.87_0.88_0.89_0.91 | 0.87/0.88/0.89/0.91 | **[2.68, 13.94]** ← 荒谬值 | 16% | 9.32 |
| 80kg_0.88_0.88_0.94_0.90 | 0.88/0.88/0.94/0.90 | [0.81,0.66,0.24,0.66,0.34,0.71]（6 个 vs 4 个） | 33% | 0.39（r=**-0.95**） |
| 102.5kg_0.51_0.49_0.42_0.30 | 0.51/0.49/0.42/0.30 | [0.50,0.37,0.62,0.79]（后两个反向） | 68% | 0.27（r=**-0.92**） |

### 3.2 `AnchorTemplateEngine`（yolo11_plate.onnx，640px）

三个视频**全部输出 0 个 rep**，静默失败。根因诊断（实锤）：
- 每一帧都有检测（379/379），但首帧锁定后，60px 最近邻门禁把后续所有检测**全部拒绝**；
- 最终轨迹 y 恒等于 153（range=0），峰谷检测自然一无所获；
- 引擎没有告警日志，直接返回空列表——**产品里这种静默失败是致命的**。

### 3.3 定位到的 5 个具体 Bug / 风险

1. **置信度双重 sigmoid（common.py）**：plate_v1/yolo11 的导出输出已是概率（0–1），代码再套 `1/(1+exp(-x))`，任何非负概率都会被映射到 ≥0.5 → `conf<0.25` 的门控**永远失效**。实测 plate_v1 在测试帧上真实最高置信度只有 **0.03**（基本是噪声），双重 sigmoid 后显示 0.508 蒙混过关。这解释了 plate_v1 RMSE≈1.4 的"差"，也让"用 conf≥0.5 帧做标定"的防线失效（3.1 表中 20kg 视频 scale=0.00616，是正常值 ~0.002 的 3 倍，标定错对象 → 速度荒谬）。
2. **无 letterbox 直 resize（common.py）**：720×1280 竖屏压成 416×416，宽高被不均匀压缩（水平 1.73× vs 垂直 3.08×），真实圆形杠铃片测量长宽比被系统性放大约 1.78×→ **圆形度门控 1.4 对竖屏视频几乎必然误杀**，这正是覆盖率只有 16–33% 的直接原因之一。
3. **60px 门禁冻结（anchor_template_engine.py）**：杠铃相心运动时常超过 60px/帧（快速离心段可达 100px+），固定门禁 + 不更新模板 → 锁死在首帧位置。NCC 模板字段建了但主流程根本没用上（`_lock_template` 是死代码）。
4. **声称的生产模型不在仓库**：`barbell_v4.onnx.backup` 被 .gitignore 排除，仓库里的 `barbell_v4.onnx` 输出是 [1,6,3549]（4+2 类），与代码假设的 [1,5,8400] 不符，靠"只读通道 4"侥幸工作。**RMSE=0.88 无法用仓库现有内容复现**（实测单视频 RMSE 0.27–9.3，且负相关说明 rep 配对后速度排序都是错的）。
5. **MCV 定义不统一 + 静默失败**：两套引擎对 MCV 的计算（正速度均值 vs 中点瞬时值+clip）、标定系数（1.15× vs 1.0×）、rep 方向（peak→trough vs bottom→top）各说各话；所有失败路径都返回空而不报错。

> ⚠️ 对产品的启示：**当前仓库的算法离"可用"还有明显距离，但失败模式全部可解释、可修**——门控、标定、letterbox、置信度解析都是工程修复而非科学难题。真正的科学短板是：没有侧视角训练数据（Phase 1.5 的教训：顶视角模型 ≠ 侧视角场景）、mPP 标定只有单点（一片直径）、rep 配对策略太弱。

---

## 四、竞品调研（Metric / VBTgo / QwikVBT）

| | **Metric**（metric.coach） | **VBTgo** | **Qwik VBT** |
|---|---|---|---|
| 平台 | iOS + Android，200k 下载，200 万组记录 | iOS + Android + 鸿蒙 Beta | iOS + Android |
| 核心交互 | **实时**：架好手机→录制→逐 rep 实时语音报数 | **实时**：逐 rep 语音播报 + 实时反馈 | **事后分析**：导入视频→点选杠铃片→批量结算 |
| 指标 | 13+ 项/rep：MV、PV、ROM、离心/向心节奏、功率、time-to-peak | 速度、功率、位移、velocity loss、1RM e1RM | MV、PV、ROM、暂停时长、bar path、velocity loss |
| 特色功能 | 杠铃轨迹叠加回放、负荷-速度画像（LVP）、e1RM、训练计划器、RPE 记录、速度损失阈值提醒 | Jump Center（6 种跳跃：SJ/CMJ/RSI）、LVP、数据中心、教练端 | rep 对比工具（自动水平/透视校正/同步向心段）、4K、云存储、CSV/JSON 导出、剪辑 |
| 模式 | 免费（全量追踪）+ Pro（计划/e1RM/语音/导出）+ Pro Coach（队伍管理） | 免费 + 订阅（3 组/天限制的付费墙被用户吐槽） | 免费 + 订阅（同样有"锁功能"差评） |
| 精度背书 | 独立同行评审：rep 检出 >95%，vs 3D motion capture ICC 0.98，2026 vs GymAware CCC 0.982（95 reps） | 官方称科学验证 | LPT 对比验证，学术引用较多 |
| 技术要点（可借鉴） | 450mm 标准片标定、60fps 采集、on-device 推理（无网络依赖）、0–25° 机位容差、unrack 幻影 rep 手动隐藏 | 跳跃模块横向扩展（同一套 CV 栈复用） | 2/3 出画仍可测、低分辨率鲁棒、 Olympic 举重分段 |

**共同的产品骨架**（即我们的功能清单基线）：
① 实时或准实时 rep 检测 + 速度指标 ② 逐 set/rep 的历史与趋势 ③ LVP + e1RM ④ 语音/音频实时反馈 ⑤ 视频回放 + bar path 叠加 ⑥ 数据导出 ⑦ 教练/多运动员（商业化天花板）

**差异化缝隙**（竞品痛点）：
- Metric：ROM/速度 LoA 偏宽（PeerJ 2024 指出深蹲 ROM LoA 达 ±10cm），unrack 幻影 rep 靠手动删——**计次鲁棒性 + 自动去 unrack 是可打的点**；
- Qwik/Metric 都没有中文市场的深度本地化；国内健身/体科所场景（体测、举重队、高校实验室）基本空白；
- VBTgo 的跳跃测试模块证明了「一套 CV 栈、多运动能力测试」的扩张路径。

---

## 五、GAP 分析：研究仓库 → App 产品

| 能力 | 现状 | 产品要求 | 差距量级 |
|---|---|---|---|
| 检测/跟踪精度 | 3 视频实测覆盖率 16–68%，2/3 视频 rep 数配不齐 | Metric 级别：>95% rep 检出 | 🔴 大（但主要是工程修复+侧视数据） |
| 处理模式 | 批处理（一段视频 ~30–90s，CPU） | 实时逐 rep 反馈（Metric/VBTgo 的灵魂） | 🔴 大（需移动端 NPU/量化/跳帧策略） |
| 部署形态 | Python 脚本 + Windows 硬编码路径 | iOS/Android App，on-device | 🔴 大（ONNX → CoreNN/TFLite/CoreML 转换） |
| 标定 | 单点：45cm 片直径 + 神秘 1.15 系数 | 多片径识别/用户点选标定（Qwik 模式） | 🟡 中 |
| 指标面 | MCV/PV/ROM/计次 | 13+ 指标（功率、节奏、e1RM、LVP…） | 🟡 中（数学上不难，数据管道要重构） |
| 产品层 | 无 | 训练日志、计划、语音、导出、教练端、账号云同步 | 🔴 大（全新工程，但与算法解耦） |
| 评估/验证 | ✅ 34 视频 + GymAware 金标准 + 学术指标（**最有价值的资产**） | 持续回归 + 分场景（机位/光照/负荷）测试集 | 🟢 小（扩展即可） |

**结论**：仓库的真实价值 = **验证数据集 + 评估体系 + 四层架构蓝图 + 已证伪的教训清单**，而非某个具体模型。以当前代码直接套壳做 App 是不可行的；但以「四层架构 + 该数据集」为内核启动产品，路线是清晰的。

---

## 六、建议：产品化路线（Phase 2 → MVP）

### 6.1 技术路线（先修内核，再做壳）

**Step 1 — 算法止血（1–2 周量级，纯工程）**
1. 统一检测器：全面切到 `AnchorTemplateEngine` 的预处理路线（letterbox + 竖屏旋转 + 正确的置信度解析），删除 common.py 的双重 sigmoid 与直 resize；
2. 修 60px 冻结：门禁改为**速度自适应**（基于最近 N 帧位移的中值外推 ±k·MAD），或直接上 IoU/特征级关联；
3. 统一 MCV 定义与标定流程（去掉魔法系数或标定其物理意义）；
4. 所有失败路径必须返回结构化诊断（覆盖率/失败原因码），杜绝静默空列表。

**Step 2 — 用现有 benchmark 回归验证**：修完跑满 34 视频，目标：rep 检出率 >90%、配对 RMSE <0.10、无荒谬值（速度 sanity 界 0.05–2.5 m/s）。这是「能不能进 App」的硬门槛。

**Step 3 — 模型侧（与 Step 2 并行）**：用 `interactive_label.py`（仓库已有的最好遗产）积累 300–500 帧侧视标注 → YOLOv8n/YOLO11n 640px 单类微调 → 专项解决 110kg+ 深蹲底部遮挡问题（Phase 1.5 已证明：伪标签路线不行，人工+建议引擎可行）。

**Step 4 — 移动端化**：ONNX → CoreML / TFLite（int8 量化，640→416 或 320 输入），逐帧推理下沉到 NPU；音频反馈、set 管理、LVP、e1RM 全部是端上逻辑。实时形态：跳帧检测（每 N 帧 YOLO + 帧间光流/模板插值）→ 30fps 手机可达。

**Step 5 — 产品 MVP 功能栈**（对标竞品骨架，按依赖排序）：
`录制+标定向导 → 实时 rep/速度 → set 总结（velocity loss 曲线）→ 历史与趋势 → LVP/e1RM → CSV 导出 → 视频回放+bar path → （商业化）教练端/多运动员`

### 6.2 差异化建议
1. **先做「事后分析+导入视频」形态冷启动**（Qwik 路线）：对实时性要求低，正好匹配当前算法成熟度，且健身房里「拍完传 App」的用户习惯已存在；实时形态作为第二阶段。
2. **把验证做成护城河**：仓库已有 GymAware 对标数据集，持续扩充并公开精度报告（Metric 靠两篇论文打天下），中文市场没人做这件事。
3. **中文 VBT 空白**：三个竞品均无深度中文本地化；国内体能训练/体科所/高校力量实验室是天然早期用户。
4. **计次鲁棒性 + 自动 unrack 剔除**作为主打卖点（Metric 被论文和用户同时诟病的点）。

---

## 附：本次探索的实测记录

- 环境：Linux / Python 3 / onnxruntime CPU / opencv 4.10
- `pipeline_associator` ×3 视频 ≈ 90s；`AnchorTemplateEngine` ×3 视频 ≈ 208s（640px 输入更慢）
- 模型输入输出实测：`barbell_v4 [1,3,416,416]→[1,6,3549]`；`plate_v1 [1,3,640,640]→[1,5,8400]`（测试帧最高真实置信度 0.030）；`yolo11_plate [1,3,640,640]→[1,5,8400]`（0.564）
- AnchorTemplateEngine 冻结证据：轨迹 y 全程=153（range=0），379/379 帧有检测、仅 29 次亚像素级抖动被接受
- 数据集：34 视频 104MB 已入 Git（.git 128MB），命名即真值（`{负荷}kg_{mcv1}_..._{mcvN}.mp4`）
