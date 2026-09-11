# M3 检测器换血训练计划（plate_v2）

> 版本：v1.0（2026-09-11）｜ 状态：脚手架完成，待数据（公开下载 + 自标 196 帧）
> 上游：TECH_ROUTE.md（M3 数据闭环）· HOLDOUT.md（评估纪律）· PUBLIC_DATASET_AUDIT.md（源审计）

---

## 0. 决策摘要（一页纸）

- **路线 (b)**：COCO 预训练 YOLO11n + 全新单类检测头，全层可训。**不是**在 `yolo11_plate` 上微调（§1）。
- **单类 `plate`**，严格片语义；整杠/人/卡箍/End 全 drop（§2）。
- **数据配方**：公开 8–11k 图作底（~75%）+ 自标 196 帧种子 ×10 加权（~25%）（§3–4）。
- **训练**：150 epochs / 640px / batch 32 / Colab T4（§5–6）。
- **验收**：格式门 `[1,5,N]` + 34 视频门（假拒绝 6→≤2、无回归）+ holdout 双报（§7）。

---

## 1. 为什么重训，不微调

| 选项 | 结论 |
|---|---|
| (a) 从 `yolo11_plate` 权重继续训 | ❌ 它的权重编码的就是错误域（边缘 0.9 conf 误检 + 工作片漏检）；且仓库只有 `.onnx`（无梯度不能训），`.pt` 还得去找 |
| **(b) COCO 预训练 YOLO11n + 全新检测头** | ✅ **采用**：任务层面就是重训（新类语义/新数据/新头），只借通用视觉先验 |
| (c) 全随机初始化从零训 | ❌ 数据量（~10k）+ 免费 GPU 下期望更差、收敛慢 2–3 倍；留作烧蚀 |

`train_plate_v2.py --no-pretrained` 是 (c) 的逃生舱（沙箱 smoke 已验证可跑通）。

---

## 2. 类别设计：单类 `plate`

- **KEEP**：任何重量的片盘框（含背景片堆、≥50% 可见半截片）；空标签图（负样本：架子/链条/墙/光杆）。
- **DROP**：整杠框、人、卡箍/锁扣、`End`/杆眼框。
- **v2a 暂 drop End 的理由**：End 框圈杠头，中心比片心偏外半个片垛、尺寸小一圈；当 plate 喂会系统性带偏框高→mpp→速度。有 5.6k 干净片框在手，不掺噪。
- **v2b 再议**：End 当噪声 plate 掺入 vs End 独立第二类，用 34 视频基准烧蚀说话。
- 不做重量/颜色多类：重量由用户输入 + 外层片直径查表解决，不需要视觉分类。

实现：注册表 `public_registry.py`（keep/drop 规则）→ `audit_public.py`（审计门）→ `build_dataset.py --drop-name-contains end,cap`（v2a 策略，默认开）。

---

## 3. 数据源（按真实可用性排序）

### P0 基座（必有）

| 源 | 规模 | 类 | 备注 |
|---|---|---|---|
| `weightlifting-plates-v11` | 5.6k 图 / ~24k 框 | 10 重量类 + 卡箍 | **先查本地 `datasets/weightlifting-plates/*.zip`**，没有再下；10 类→单类，删卡箍 |

### P1 有 plate 框（v2a 主力，audit 通过就合）

| 源 | 规模 | plate 类 | 备注 |
|---|---|---|---|
| `iap-thesis` | 1781 图 | plates | 论文级，待核实 |
| `group3-workout` | 2229 图 | Weight plates | 力量举场景，动作类名按 unlisted 复核 |
| `weight-plate-detector` | 652 图 | IPF 色系重量类 | **唯一小片来源**；v3 带 3× 增强（train 可用） |
| `test-barbell_detection` | 435 图 | Plates | 双类留 Plates |
| `vbt-barbell-detection` | 43 图 | plate | 量小但为 VBT 而建，域参考价值高 |

### P2 End/杆眼为主（v2a 先审后存，v2b 再定）

`barbell-tracking`(1.91k) · `olympic-weightlifting-tracking`(550) · `BarbellDetection2`(165) · `barbell-end-tracker`(90) · `barbells-detector`(92) · Kaggle `barbell-pose`(156MB)。

### P3 大概率 REJECT

`techtitans-barbell`(1.34k，单类疑似整杠，audit 见分晓）。

### 合规与风险

- Roboflow Universe 多为 CC BY 4.0：商用需署名，研究/内测无碍；`dataset_card.json` 记录来源，LICENSE 合规可查。
- Roboflow 版本号会漂移：`fetch_public.py` 无版本时取 latest 并告警，audit 核对内容；`dataset_card.json` 锁定实际下载版。
- 现实产量预估：P0 5.6k + P1 通过 ~2–4k ≈ **8–11k 图 / ~30k 框**。

---

## 4. 自标规范（196 帧种子 + R1）

- **语义**：所有可见片全标（含背景片堆，身份归引擎运动探针）；<50% 可见不标；**框贴边**（片高进 mpp 标定）。
- **预算**：R0 种子 196 帧（挖掘队列 top6/视频：140 失败 + 56 多样性）；R1 按 v2a 残余失败定点补 100–200 帧；封顶 ~500（再多不如加新健身房）。
- **划分**：按视频留 val（≥3 条，覆盖轻/重/易/难），`build_dataset.py --auto-val 4` 建议 + 人工目检。
- **空帧**：20kg 光杆等无片帧正常存空标签（负样本），不要跳过。

---

## 5. 管线命令（Windows 本机顺序）

```bat
:: 0. 环境（二选一套装）
pip install opencv-python numpy onnxruntime scipy roboflow
pip install -r requirements-train.txt   :: 仅训练机（Colab 装，见 notebook）

:: 1. 公开下载（需 ROBOFLOW_API_KEY）+ 先审后合
set ROBOFLOW_API_KEY=xxxx
python scripts/fetch_public.py --source weightlifting-plates-v11,weight-plate-detector,iap-thesis
python scripts/audit_public.py --src datasets/public/weightlifting-plates-v11 --source weightlifting-plates-v11
:: → 看 *_montage.jpg 目检，verdict PASS 才合

:: 2. 挖掘 + 标注（~2 小时）
python scripts/mine_frames.py --out datasets/mining/queue_r0.json
python scripts/interactive_label.py --mine datasets/mining/queue_r0.json
python scripts/interactive_label.py --status --mine datasets/mining/queue_r0.json

:: 3. 校验 + 组装
python scripts/verify_labels.py --labels-dir datasets/interactive_labels --render-dir datasets/interactive_labels/_qa
python scripts/build_dataset.py --public datasets/public/weightlifting-plates-v11 --public datasets/public/weight-plate-detector --auto-val 4
:: → datasets/plate_v2/（data.yaml + dataset_card.json），打包 plate_v2.zip 传 Colab

:: 4. 训练（Colab，见 notebooks/colab_train_plate_v2.ipynb）
:: 5. 回传 models/plate_v2a.onnx → 跑 §7 三道门
```

沙箱注意：`ultralytics` 会拖入 `opencv-python`（非 headless），无 libGL 容器需换 `opencv-python-headless`；`ultralytics>=8.3` 必需（8.2.x 无 YOLO11）。

---

## 6. 训练配置（plate_v2a）

| 项 | 值 | 备注 |
|---|---|---|
| 基座 | `yolo11n.pt`（官方 COCO） | ultralytics 自动下载；Colab 网络正常 |
| imgsz / epochs / batch | 640 / 150 / 32（T4 16GB） | 1650 备选：batch 8，慢 5–8 倍 |
| seed / patience / workers | 42 / 30 / 8 | `deterministic=True` |
| single_cls / flipud | True / **0.0** | 竖屏禁垂直翻转；mosaic/mixup 用默认 |
| 导出 | ONNX opset 12 | Gate A 自动校验 `[1,5,N]` |

脚本：`scripts/train_plate_v2.py`（沙箱 CPU smoke 已验证：2 epochs + 导出 + Gate A/B 全过，见 §8 首条记录）。

---

## 7. 验收门（按顺序）

- **Gate A 格式**（自动）：ONNX 输入 `[1,3,640,640]` → 输出 `[1,5,N]`，零输入推理断言。
- **Gate B 烟雾**（自动）：`--smoke-img` 一帧开发集图，vbtcore 能跑通（检出数仅供参考）。
- **Gate C 基准**（手动，真门）：
  ```bat
  python scripts/run_benchmark_v0.py --tag plate_v2a --model models/plate_v2a.onnx --engine "vbtcore v1.5 + plate_v2a"
  ```
  通过标准：**假拒绝 6→≤2 且零回归**（任一原通过视频变失败即否决）；另跑 `--no-regrind` 烧蚀复测 regrind 增益（M1.5 时被检测器 gating）。
- **Gate D 留出**（纪律）：holdout-v0 基线 + 双报 `dev｜held-out`（HOLDOUT.md 第五节）。
- val mAP 仅看趋势，不设门（真理在 34 视频 + holdout）。

---

## 8. 运行记录（追加制）

| 日期 | run | 数据 | 配置 | val mAP | Gate C | 备注 |
|---|---|---|---|---|---|---|
| 2026-09-11 | smoke | 合成 24/8 图 @320 | 8.4.147/2ep/CPU/随机初始化 | — | — | 脚本链路验证：训练→导出→GateA(1,5,2100)→GateB 全过 |
| … | plate_v2a | … | … | … | … | … |

模板（新 run 复制一行）：`日期 ｜ run名 ｜ 数据（公开源+自标帧数+K） ｜ 基座/epochs/batch ｜ mAP50 ｜ 计数/RMSE ｜ 结论`
