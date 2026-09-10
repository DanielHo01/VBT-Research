# 公开数据集审计（M3 数据闭环 · 公开数据优先策略）

> 日期：2026-09-10 ｜ 范围：Roboflow Universe + Kaggle 检索 ｜ 状态：**桌面审计**（类列表/规模来自项目页，下载后需逐条核实）
> 背景：M3 原本计划"自标 300–500 帧"，本次审计验证"公开数据优先"能否大幅压缩自标工作量。
> 结论先行：**公开数据比预想的多，值得先做公开微调实验（Tier-1）；但"蹲底遮挡/片堆干扰/侧视手机竖屏"三个硬 case 大概率仍需自标兜底（Tier-3）。**

---

## 一、高贴合候选（含片/杠端类，直接对口引擎需求）

| # | 项目 | 规模 | 类 | 模型 | 状态 | 贴合点 |
|---|---|---|---|---|---|---|
| 1 | [vbt-barbell-detection](https://universe.roboflow.com/morgans-workspace-1tho7/vbt-barbell-detection) | 43 图 | **plate / barbell end** / objects | Roboflow Instant ×1（2026-03） | ✅ 页面已确认 | **就是为 VBT 建的**，类名与引擎需求 1:1；可惜太小，只够做"域参考/类名对齐" |
| 2 | [Weight Plate Detector](https://universe.roboflow.com/bar-ozhei/weight-plate-detector) | 652 图（v1=32/v2=86/v3=410，v3 带 3× 增强） | 标注图可见 **plate_25_red** 等**按 IPF 色系重量分类**（红25/蓝20/黄15/绿10） | Roboflow 3.0 Fast ×1（v2, COCO 预训练） | ⚠️ 高置信待核实 | **"各种重量"的片**；颜色分重对我们"外层片直径查表"是天然增强 |
| 3 | [olympic weightlifting tracking](https://universe.roboflow.com/bar-path/olympic-weightlifting-tracking-jjni3) | 550 图 | **barbell-cap**（杠端"杆眼"） | **3 个模型** | ✅ 页面已确认 | 工作室名 bar-path——**杠端帽直接对标"杆眼"标注**；举重场景含大片 |
| 4 | [BarbellDetection2](https://universe.roboflow.com/thecoachbarbelldetection/barbelldetection2) | 165 图 | Barbell / **End** | **2 个模型** | ✅ 页面已确认 | End = 杠端；TheCoach 系 VBT 工具项目 |
| 5 | [barbell tracking](https://universe.roboflow.com/kakann/barbell-tracking-olmop) | 1.91k 图 | Bar / Barbell / **End** | 无（仅数据） | ✅ 页面已确认 | 规模最大且带 End 类；kakann 另有 4542 图整杠集 |
| 6 | [Barbell Detection](https://universe.roboflow.com/techtitans-pcchf/barbell-detection-8phtm) | 1.34k 图 | Barbell | **6 个模型** | ✅ 页面已确认 | 模型多，可做零成本 baseline 试跑 |
| 7 | [barbell-end-tracker](https://universe.roboflow.com/barbellendtracker/barbell-end-tracker) | 90 图 | **barbell-end** / barbells | 无 | ✅ 页面已确认 | 纯杠端实例分割，最小而精准 |
| 8 | [Barbells Detector](https://universe.roboflow.com/yolo-project-c2bfs/barbells-detector) | 92 图 | Barbell / **End** | **7 个模型** | ✅ 页面已确认 | 模型库全（含各 YOLO 版本） |
| 9 | [thesis/IAP](https://universe.roboflow.com/iap/thesis) | 1781 图 | person / barbell / dumbbell / **plates** | — | ⚠️ 待核实 | 论文级标注质量通常更高 |
| 10 | [Group 3 Workout/Exercises](https://universe.roboflow.com/technological-institute-of-the-philippines/group-3-workoutexercises) | 2229 图 | Deadlift 等 / **Weight plates** / barbell | — | ⚠️ 待核实 | 力量举动作场景 |
| 11 | [barbell_detection](https://universe.roboflow.com/test/barbell_detection) | 435 图 | Barbell / **Plates** | — | ⚠️ 待核实 | 双类 |

## 二、Kaggle 候选

| 项目 | 规模 | 内容 | 备注 |
|---|---|---|---|
| [Barbell Detection-Pose Estimation (YOLOv8)](https://www.kaggle.com/datasets/danishghaffar786/barbell-detection-pose-estimation-yolov8-format) | 156 MB | **杠铃两端检测**（YOLOv8 格式） | 2025-05，与"杆眼"需求直接对口 |
| [Gym Data COCO for YOLO-pose](https://www.kaggle.com/datasets/pratapdevs11/gym-data-coco-format-for-yolo-pose) | 523 MB | 深蹲等动作 + 杠铃关键点 | MIT 许可；可作 R3 关键点备选 |

## 三、对照失败清单的覆盖映射（桌面级预判）

| 失败模式（开发集实证） | 公开数据覆盖预判 | 依据 |
|---|---|---|
| 侧视手机竖屏（工作片小、透视） | 部分：vbt-barbell-detection / bar-path 系疑似侧视实拍 | 需下载核实机位 |
| 蹲底遮挡（28 次盲区） | **大概率缺口**——公开集多为摆拍/清晰帧 | 无一项目声称含遮挡帧 |
| 工作片 vs 背景片堆（身份锁定） | 缺口——公开集很少同帧含多组片 | 类列表无法体现，需抽帧核实 |
| 10/15kg 小铁片（非 450mm） | 有：Weight Plate Detector 色系分类含小片 | 高置信 |
| 130kg 大片堆叠 | 有：bar-path 举重场景 | 高置信 |

## 四、执行计划（三级漏斗，量化 gate）

1. **Tier-1 公开微调实验**（零自标成本）：合并候选 1–11 → 类名映射为单类 `plate`（+保留 `barbell_end` 做身份辅助）→ 云端 GPU 微调 YOLOv8n/11n → 34 视频开发集打分。**Gate：假拒绝从 6 条降到 ≤3 条且无新增回归 → 进入 Tier-2，否则直接 Tier-3。**
2. **Tier-2 留出集验收**：holdout-v0 复测（双报纪律），验证跨健身房泛化。
3. **Tier-3 自标兜底**：只对 Tier-1 救不了的失败模式（预判：蹲底遮挡 + 片堆干扰）用 `interactive_label.py` 自标 200–300 帧，混合再训。

## 五、沙箱网络实测与数据获取路径（2026-09-10）

**实测结论：本沙箱无法直连 Roboflow/Kaggle/HuggingFace**——出站网络白名单仅放行
`github.com`/`api.github.com`/`pypi.org`；roboflow.com 全系域名 TCP 可连但 TLS 握手被断；
且 Roboflow 下载页本身要求登录（免费账号即可，网站下载不需要 API key，key 仅供 API 访问）。
因此数据获取路径为：

1. **用户浏览器下载**（推荐）：Roboflow 候选页 → Download Dataset → YOLOv11 格式 → 免费登录下载
   → 把 zip 丢进 `m3_data/downloads/`（入口文档：`docs/M3_DATA_INTAKE.md`）；
2. **GitHub 直连**（已核实，收获有限）：oscaragren/VBT 的 `data/` 在 git 里只有空目录
   （本体在 Roboflow）；simonkosina/vbt 有 **1088 张整杠 XML 标注 @416px** 在仓内（整杠框，
   非片框，价值有限，暂不拉取）。其余候选仓库均无现成标注数据。

## 六、待办（需要外部条件）

- [ ] **下载候选 zip**：用户浏览器操作（见第五节路径 1；免费 Roboflow 账号即可，无需 API key）
- [ ] **Kaggle 或 Colab 免费 GPU**（训练执行地；沙箱无 GPU，本地 GTX 1650 只做推理/标注）
- [ ] 下载后逐条核实：机位是否侧视、是否有遮挡帧、`Weight Plate Detector` 完整类列表（确认 plate_25_red 是否含蓝/黄/绿全系）

> 相关文档：TECH_ROUTE.md（M3 数据闭环）· HOLDOUT.md（评估纪律）· PLATE_LAYER_REVIEW.md（域差实证）
> 工具：scripts/m3_merge_datasets.py（合并+类映射+审计，已自测）· docs/M3_DATA_INTAKE.md（数据入口）
