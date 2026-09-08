# DATASET

训练数据来源、清洗规则、class 映射、为什么保留/删除某些数据集。

---

## 数据来源总览

| 数据集 | 来源 | 图片数 | 标注数 | 类型 | 决策 |
|---|---|---|---|---|---|
| weightlifting-plates-v11 | Roboflow | 5638 | ~29800 | 单 plate | ✅ 保留 |
| barbell-detection-yolov5 | Roboflow | 4149 | 5373 | 整杠+人 | ❌ 删除 |
| barbell-small-v1 | Roboflow | 1090 | 1730 | 整体 bar | ❌ 删除 |
| barbell-v1 | Roboflow | 958 | 1495 | 小目标 bar | ❌ 删除 |

**最终保留**：weightlifting-plates-v11（5638 张图）

---

## 第一性原理：为什么删除 barbell-*？

### VBT 真正需要什么？
- 每帧中**单 plate 的中心点**（用于算 mCV）
- 不需要分类（重量已知）
- 不需要检测"人 + 整杠"

### barbell-* 教的是什么？

```
barbell-detection-yolov5: 
  bbox = 0.34 × 0.22 (1280×1280)
  → 框住"人 + 整杠 + 片"整体
  → 中心点在杠的中间
  
  这不是 plate 中心！
```

```
weightlifting-plates-v11:
  bbox = 0.05 × 0.47 (336×640)
  → 单个 plate
  → 中心点 = plate 中心 ✅
```

### 决策逻辑

```
训练数据 → 模型学到"找什么"
barbell-*  → 模型学"找整杠"
weightlifting-*  → 模型学"找 plate"

我们要 plate，所以只留 weightlifting-*
```

**结论**：barbell-* 数据集用于**其他应用**（gym 照片里有没有杠铃），不适合 plate 检测。混入会让模型混乱。

---

## weightlifting-plates-v11 详情

**来源**：https://universe.roboflow.com/projektciezary/weightlifting-plates

**原始 11 类**（按 Roboflow 顺序）：
```
0: 0.5kg, 1: 1.5kg, 2: 10kg, 3: 15kg, 4: 1kg,
5: 2.5kg, 6: 20kg, 7: 25kg, 8: 2kg, 9: 5kg,
10: zacisk (卡箍)
```

### 标注分布（原始）
```
class  7 (25kg      ):  8265  ← 最多
class  0 (barbell   ):  8437
class 11 (zacisk    ):  5569  ← 删除
class  6 (2.5kg     ):  4314
class  5 (1.25kg?   ):  4480
... 其他类  1000-1500
```

### 清洗决策

| 类 | 决策 | 理由 |
|---|---|---|
| zacisk (卡箍) | ❌ 删除 | 不是 plate，VBT 不用 |
| 0.5kg ~ 25kg (10 个) | ✅ 合并为 plate | VBT 不需要分类重量 |
| (整杠类) | ❌ 删除 | 不同的视觉任务 |

### 清洗后

```
保留：~24300 个 plate 标注
类别：1 类 = plate
图片：5638 张
平均标注数：4.3 个 plate / 图
```

---

## 单标签 vs 多标签：经验法则

```
Object Detection 数据量经验法则：
  每类最少:    1000-2000 张
  每类推荐:    1500+
  最低门槛:    10x 类别数
  推荐量:      50-100x 类别数

我们的状态：
  1 类:   5638 张 → 5638 张/类  ✅ 数据过剩（5.6x 推荐量）
  11 类:  5638 张 → 512 张/类   ⚠️ 刚过最低线，有失衡风险
  
选择：1 类（最稳、最快、模型最简单）
```

---

## 为什么不用自标注数据？

**理论上**：自标注 > 公开数据（域匹配最好）

**实际约束**：
1. 标注成本：300 张图需要 5-10 小时人工
2. 工具不熟：用 labelImg / CVAT 还得学习
3. vLLM 自动标注：本地机器（GTX 1650 4GB）跑不动，CPU 太慢

**结论**：先用公开数据（5 分钟下载），等模型训练出来再评估是否需要自标注补充。

---

## 训练/验证/测试 划分

**计划**：
```
train: 4510 张 (80%)
val:    564 张 (10%)   ← 训练时验证
test:   564 张 (10%)   ← 训练后最终评估
```

**种子**：random.seed(42) 保证可复现

**注意**：所有图片文件名带数据集前缀（如 `weightlifting-plates-v11_*.jpg`），避免重名冲突。

---

## 数据增强（在训练时做）

训练时 Ultralytics 默认增强：
- HSV 抖动（hsv_h=0.015, hsv_s=0.7, hsv_v=0.4）
- 几何变换（translate=0.1, scale=0.5）
- 水平翻转（fliplr=0.5）
- Mosaic（mosaic=1.0）
- MixUp（mixup=0.1）
- Copy-Paste（copy_paste=0.1）

**为什么不开垂直翻转**（flipud）：竖屏视频不能垂直翻转。

---

## 文件位置

```
源数据：
  datasets/weightlifting-plates/Weightlifting Plates.v11i.yolov8.zip
  解压后：datasets/weightlifting-plates/weightlifting-plates-v11/{train,valid,test}

训练时：
  datasets/source/  ← 解压后的源数据
  datasets/final/   ← 80/10/10 划分后的数据
    ├── data.yaml
    ├── train/{images,labels}/
    ├── val/{images,labels}/
    └── test/{images,labels}/
```

**YAML 格式**（单类版）：
```yaml
path: D:/EasyVBT-Research/datasets/final
train: train/images
val: val/images
test: test/images
nc: 1
names:
  '0': plate
```