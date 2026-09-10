# 留出集（Held-out Set）采集与使用规范

> 地位：执行 `TECH_ROUTE.md` 第九节"评估纪律"的落地文档。
> **M3 数据闭环调参之前，必须先有留出集并跑出基线**——否则新成绩不可信。

---

## 一、开发集冻结声明

- `validation/dataset_benchmark/` 的 **34 视频 = 开发集（dev）**，冻结其用途：诊断失败、调参、报 dev 成绩。
- 此后**任何新视频一律先进留出集**，绝不直接进开发集凑数。
- 此后所有成绩**双报**：`dev 计数/RMSE ｜ held-out 计数/RMSE`。只报 dev = 耍流氓。

## 二、目录与索引规范（镜像开发集，工具可复用）

```text
validation/holdout/
  raw_videos/            ← 视频本体（命名见第四节）
  holdout_index.json     ← 索引（schema 同 dataset_index.json + 采集元信息）
  ground_truth.csv       ← 真值表（同开发集格式）
```

`holdout_index.json` 条目（required 4 项与开发集一致，extra 用于覆盖度审计）：

```json
{
  "video_id": "100kg_0.55_0.50.mp4",
  "load_kg": 100.0,
  "gt_reps_mcv": [0.55, 0.50],
  "fps_override": null,
  "gym": "Gym-B 朝阳",
  "camera": "iPhone 14 / 脚架 / 侧视竖屏",
  "angle": "side",
  "lighting": "bright",
  "outer_plate": "20kg",
  "date": "2026-09-15"
}
```

## 三、覆盖清单（首批 10–20 条，缺一格就记一笔技术债）

开发集的已知偏态：**同一健身房、同一机位、无小铁片**。留出集专治偏态：

| 维度 | 必须覆盖 | 说明 |
|---|---|---|
| 场景 | ≥2 个不同健身房 | 灯光/地面/背景杂物都不同才算 |
| 机位 | 侧视近 / 侧视远 / 横屏 | 当前以竖屏侧视为多；距离影响片表观尺寸（锚定尺寸域的考验） |
| 光照 | 明亮 / 偏暗 / 逆光（至少各 2 条） | 检测器域差的主因之一 |
| 负荷 | 轻（总重 ≤40kg）/ 中 / 重（≥100kg） | 轻负荷必须含**外层小铁片**（10/15kg 非 450mm）：标定查表的验证集 |
| 背景难度 | 片堆可见（硬）/ 干净（易）各半 | 片堆是身份锁定的头号敌人，必须有硬 case |
| 动作 | 高杠深蹲为主 + 至少 2 条暂停蹲 | 卧推/硬拉暂不收（引擎未验证，进扩展集） |
| 设备 | ≥2 种手机 + 脚架固定 | **第一批只收固定机位**；手持视频单独标注进"扩展集"，不进主留出分数（引擎暂假设固定机位，防抖是 backlog） |

每条 1–8 reps，必须带 GymAware（或同级 LPT）逐 rep MCV 真值。

## 四、命名与真值规范

- 文件名沿用开发集约定：`{总负荷}kg_{rep1mcv}_{rep2mcv}_….mp4`（如 `100kg_0.55_0.50.mp4`）。
- GT 对齐：按**顺序+数量**与视频 rep 配对；采集时数清 reps 数，发现对不上当场重录，别留到标注时猜。
- `fps_override`：手机可变帧率导致 cap 读错时填实测 fps，否则 null。

## 五、使用纪律（红线）

1. **永不在留出集上调参、选模型、定阈值**。调参只看 dev，留出集只跑最终版。
2. **先基线后调参**：M3 动手前，用当前引擎跑 `holdout-v0` 留底（命令见下）。没有 v0，后面一切"提升"都无从谈起。
3. **双报**：任何成绩汇报写 `dev X/Y ｜ held-out A/B` 两组数字。
4. **永不毕业**：留出集视频永不并入开发集调参（M3 的帧级标注是另一回事：标注帧可进训练，但视频级成绩永远双报）。
5. 工具缺口（M3 顺手补）：`run_benchmark_v0.py` 的 `BENCH` 路径目前硬编码开发集，需加 `--bench-dir validation/holdout` 参数化。

```bash
# 留出集基线（M3 前跑一次留底）
python3 scripts/run_benchmark_v0.py --tag holdout-v0 --bench-dir validation/holdout
```

## 六、存储约定

- 沿用仓库惯例：视频直接进 git（开发集 34 条 ~100MB 前例）。
- 留出集首批控制在 **<100MB**；总量超过 200MB 时切 GitHub Release 分发（参考 `v1.0-videos` tag 前例），索引 JSON 永远进 git。

## 七、新视频 intake 检查单

```text
[ ] 命名符合第四节（负荷_真值序列.mp4）
[ ] holdout_index.json 有条目（含 gym/camera/angle/lighting/outer_plate）
[ ] ground_truth.csv 有对应行，rep 数与视频一致
[ ] 覆盖清单（三）打勾：场景/机位/光照/负荷/背景/动作/设备
[ ] 体积：单条 <15MB，总量 <100MB（超了走 Release）
[ ] 跑过引擎不崩溃（状态码正常，非 VIDEO_ERROR）
```
