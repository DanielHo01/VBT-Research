# Benchmark Validation Dataset

## 目录结构

```
dataset_benchmark/
├── raw_videos/          # 原始视频文件（MP4）
├── ground_truth.csv      # 金标准数据（视频 → GymAware MCV 配对）
└── README.md
```

## ground_truth.csv 格式

```csv
source,filename,load_kg,rep1,rep2,rep3,rep4,rep5,rep6
wechat,30kg_1.03_0.89_0.76_0.65.mp4,30,1.03,0.89,0.76,0.65,,
```

- **source**: wechat（微信）/ sportSci_Pro / unknown（待确认）
- **filename**: 视频文件名
- **load_kg**: 杠铃总重量（kg）
- **rep1...rep6**: GymAware MCV 值（单位 m/s），无数据则留空

## 当前数据（2026-09-01）

| 来源 | 负荷 | Reps | GymAware MCV (m/s) |
|------|------|------|-------------------|
| wechat | 30kg | 4 | 1.03, 0.89, 0.76, 0.65 |
| wechat | 20kg | 4 | 0.87, 0.88, 0.89, 0.91 |
| wechat | 30kg | 5 | 0.84, 0.87, 0.93, 0.88, 0.53 |
| sportSci_Pro | 80kg | 4 | 0.88, 0.88, 0.94, 0.90 |
| sportSci_Pro | 50kg | 4 | 0.89, 1.09, 1.12, 1.11 |
| sportSci_Pro | 110kg | 2 | 0.69, 0.69 |
| sportSci_Pro | 110kg | 2 | 0.71, 0.73 |
| sportSci_Pro | 130kg | 1 | 0.52 |
| sportSci_Pro | 140kg | 1 | 0.41 |
| unknown | 105kg | 6 | 0.69, 0.64, 0.65, 0.60, 0.56, 0.45 |
| unknown | 105kg | 6 | 0.60, 0.54, 0.55, 0.54, 0.51, 0.37 |
| unknown | 105kg | 4 | 0.61, 0.59, 0.56, 0.32 |
| unknown | 105kg | 4 | 0.62, 0.58, 0.53, 0.44 |
| unknown | 105kg | 3 | 0.59, 0.56, 0.44 |

**当前总计：14 个视频，41 个 Rep**

## 添加新视频步骤

1. 把视频文件复制到 `raw_videos/`
2. 重命名为：`{负荷}kg_{mcv1}_{mcv2}_..._{mcvN}.mp4`
   - 示例：`60kg_0.95_1.02_0.98.mp4`（3个Rep）
3. 在 `ground_truth.csv` 末尾追加一行
4. source 填：wechat / sportSci_Pro / other

## 命名参考（GymAware MCV 顺序对应 Rep 1, 2, 3...）

用户通过手机拍摄 → GymAware 同步采集 → 导出 MCV 峰值速度
文件名直接编码 GA 数据，便于批量处理脚本读取。
