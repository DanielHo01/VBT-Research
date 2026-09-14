# 留出集（Held-out Set）

> **状态：框架已就绪，视频待采集（当前 0 条）。**
> 规范见 [`../../docs/HOLDOUT.md`](../../docs/HOLDOUT.md)。

---

## 为什么它是空的，以及为什么这很重要

开发集 34 视频的当前成绩（计数 31/34、rep RMSE 0.0664、r 0.9423，见
[`../reports/SCOREBOARD.md`](../reports/SCOREBOARD.md)）是 **dev-only 成绩**。

开发集有三个已知偏态：**同一健身房、同一机位、无小铁片**。
在这样的数据上继续调参，本质是在过拟合 34 条视频。
留出集就是用来打破这层幻觉的——**没有它，后续任何「提升」都无从验证**。

---

## 目录结构

```text
validation/holdout/
├── raw_videos/            视频本体
├── holdout_index.json     索引（schema 镜像开发集 + 采集元信息）
├── ground_truth.csv       真值表（格式同开发集）
└── README.md              本文件
```

---

## 采集工作流

```bash
# 1. 按 docs/HOLDOUT.md 第三节覆盖清单采集视频，放入 raw_videos/
#    命名：{总负荷}kg_{rep1}_{rep2}_....mp4    例：100kg_0.55_0.50.mp4

# 2. 从文件名生成索引骨架
python3 scripts/holdout_intake.py --scaffold

# 3. 人工补齐每条的采集元信息
#    gym / camera / angle / lighting / background / device / outer_plate / date

# 4. 复验（命名、三方一致性、体积、覆盖度）
python3 scripts/holdout_intake.py --check

# 5. 随时查看覆盖缺口
python3 scripts/holdout_intake.py --coverage
```

### 元信息字段取值

| 字段 | 取值 | 说明 |
| --- | --- | --- |
| `gym` | 自由文本 | 需 ≥2 个不同场地 |
| `angle` | `side_near` / `side_far` / `landscape` | 三者都要有 |
| `lighting` | `bright` / `dim` / `backlit` | 各 ≥2 条 |
| `background` | `plate_stack` / `clean` | 片堆是身份锁定头号敌人，必须有硬 case |
| `device` | 手机型号 | 需 ≥2 种 |
| `outer_plate` | `20kg` / `15kg` / `10kg` / `45lb` … | **轻负荷必须含小铁片**（非 450mm），验证标定查表 |
| `date` | `YYYY-MM-DD` | |

`load_band`（light ≤40kg / medium / heavy ≥100kg）由工具自动从 `load_kg` 派生，无需填写。

---

## 使用红线（摘自 docs/HOLDOUT.md 第五节）

1. **永不在留出集上调参、选模型、定阈值。** 调参只看 dev，留出集只跑最终版。
2. **先基线后调参。** M3 动手前先跑 `holdout-v0` 留底，否则后面一切「提升」无从谈起。
3. **双报。** 任何成绩汇报写 `dev X/Y ｜ held-out A/B` 两组数字。只报 dev = 耍流氓。
4. **永不毕业。** 留出集视频永不并入开发集调参。

```bash
# 留出集基线（采集完成、M3 动手前跑一次留底）
python3 scripts/gen_cpp_baseline.py \
    --bench-dir validation/holdout \
    --out-dir validation/reports/holdout_baseline \
    --workers 4
```

> 注：`gen_cpp_baseline.py` 的 `--bench-dir` 期望目录下有 `dataset_index.json`。
> 留出集索引名为 `holdout_index.json`，跑基线前需软链或改名，
> 或使用 `scripts/run_benchmark_v0.py --bench-dir validation/holdout`。

---

## 体积约定

- 单条 < 15 MB，总量 < 100 MB
- 超过 200 MB 时切 GitHub Release 分发，索引 JSON 永远进 git
