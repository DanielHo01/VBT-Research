# m3_data — 训练数据入口（M3 数据闭环）

> 本目录**不进 git**（视频/图片按仓库惯例另议），是 M3 训练数据的 staging 区。
> 处理工具：`scripts/m3_merge_datasets.py`（合并 + 类映射 + 审计）。

## 目录约定

```text
/home/user/m3_data/
  downloads/     ← 把 Roboflow 下载的 zip 直接丢这里（YOLOv8/v11 格式）
  extra/         ← 手标数据（同类目录结构 + data.yaml，可选）
  merged/        ← 工具输出：合并后的统一训练集（images/ labels/ data.yaml audit.json）
  work/          ← 工具临时解压区（可随时删）
  raw/           ← 从 GitHub 直接克隆的原始数据集（可选）
```

## 下载 Roboflow 数据（浏览器操作，不需要 API key）

1. 打开候选数据集页（清单见 `docs/PUBLIC_DATASET_AUDIT.md`），点 **Download Dataset**；
2. 格式选 **YOLOv11**（或 YOLOv8——都带 data.yaml + 三目录，工具同样吃）；
3. 免费注册/登录 Roboflow 账号（社区数据集 CC BY 许可，下载免费）；
4. 把 zip 上传/拷贝到 `/home/user/m3_data/downloads/`。

## 合并与审计

```bash
python3 scripts/m3_merge_datasets.py          # 输出到 m3_data/merged/
```

工具自动做三件事：
1. **类映射**：`plate_25_red`/`Weight plates`/`Plate`… → `plate`；`End`/`barbell end`/`barbell-cap` → `barbell_end`；`Barbell`/`Bar` → `barbell`；`person` → `person`；其余类忽略并记入审计；
2. **文件名加源前缀**防冲突（`源名__原文件名.jpg`）；
3. **审计报告** `merged/audit.json`：每源图数/标数/忽略类/缺标文件 + 合并后类分布。

## 统一 schema（为什么这么定）

| id | 类 | 引擎用途 |
|---|---|---|
| 0 | plate | 主检测目标（锚定/跟踪/标定的身份来源） |
| 1 | barbell_end | "杆眼"：杆端高对比目标，抗遮挡的辅助身份 |
| 2 | barbell | bar 级跟踪辅助 / 背景负样本 |
| 3 | person | 动作区域先验（辅助 gate） |

## 当前状态（2026-09-10）

- [x] 工具与目录就绪
- [ ] downloads/ 空——等 Roboflow 候选 zip 到位（11 个候选见审计文档）
- [ ] GitHub 直连源已核实：无现成 plate 标注数据集（oscaragren/VBT 的 data/ 在 git 里只有空目录，本体在 Roboflow；simonkosina/vbt 有 1088 张整杠 XML 标注 @416px，价值有限未拉取）
- [ ] 云端 GPU 训练（Kaggle/Colab）待排期——沙箱无 GPU
