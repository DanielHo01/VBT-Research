"""public_registry — M3 公开数据源注册表（stdlib only）
========================================================
单一事实来源：fetch_public.py（下载）/ audit_public.py（审计）/
build_dataset.py（合并）共用此表，类映射规则只写一次。

来源：docs/PUBLIC_DATASET_AUDIT.md（一、桌面审计）+ DATASET.md（Phase 1 基座）。
版本说明：Roboflow 下载需要 version 号；凡标注时审计页未明确版本的，
暂填 None，下载时默认取 latest 并告警（audit 环节再核对内容是否相符）。

类映射规则（match_class）：
  1. keep 优先（精确意图，如 'end' 命中 'barbell-end'）；
  2. 再看 drop（如 'barbell' 命中整杠 'Barbell'）；
  3. 都不命中 → unlisted（审计时强制人工复核，合并时丢弃）。
大小写不敏感，子串匹配。整杠/人/卡箍一律 drop——DATASET.md 血训：
整杠框教模型"找杠"，与"找片"冲突，混入即污染。
"""
from __future__ import annotations

SOURCES: dict[str, dict] = {
    # —— Tier-0：Phase 1 基座（5.6k 图，已知语义，可信度最高）——
    "weightlifting-plates-v11": {
        "provider": "roboflow",
        "workspace": "projektciezary",
        "project": "weightlifting-plates",
        "version": 11,
        "format": "yolov8",
        "keep": ["kg", "plate"],
        "drop": ["zacisk", "clamp", "collar", "barbell", "person"],
        "notes": "Phase 1 基座：10 个重量类→单类 plate；删卡箍/整杠（见 DATASET.md）",
    },
    # —— Tier-1：审计候选（按 PUBLIC_DATASET_AUDIT.md 表号）——
    "vbt-barbell-detection": {
        "provider": "roboflow",
        "workspace": "morgans-workspace-1tho7",
        "project": "vbt-barbell-detection",
        "version": None,
        "format": "yolov8",
        "keep": ["plate", "barbell end", "end"],
        "drop": ["object", "barbell", "person"],
        "notes": "43 图，为 VBT 建的；objects 类含义不明，默认 drop（audit 复核）",
    },
    "weight-plate-detector": {
        "provider": "roboflow",
        "workspace": "bar-ozhei",
        "project": "weight-plate-detector",
        "version": 3,
        "format": "yolov8",
        "keep": ["plate"],
        "drop": ["barbell", "person"],
        "notes": "652 图，按 IPF 色系分重（plate_25_red 等）；v3 带 3× 增强",
    },
    "olympic-weightlifting-tracking": {
        "provider": "roboflow",
        "workspace": "bar-path",
        "project": "olympic-weightlifting-tracking-jjni3",
        "version": None,
        "format": "yolov8",
        "keep": ["cap", "plate", "end"],
        "drop": ["barbell", "bar", "person"],
        "notes": "550 图，barbell-cap（杆眼）；举重场景含大片",
    },
    "BarbellDetection2": {
        "provider": "roboflow",
        "workspace": "thecoachbarbelldetection",
        "project": "barbelldetection2",
        "version": None,
        "format": "yolov8",
        "keep": ["end"],
        "drop": ["barbell", "bar", "person"],
        "notes": "165 图，End=杠端；整杠 Barbell 类 drop",
    },
    "barbell-tracking": {
        "provider": "roboflow",
        "workspace": "kakann",
        "project": "barbell-tracking-olmop",
        "version": None,
        "format": "yolov8",
        "keep": ["end"],
        "drop": ["barbell", "bar", "person"],
        "notes": "1.91k 图，规模最大；只要 End 类，Bar/Barbell 全 drop",
    },
    "techtitans-barbell": {
        "provider": "roboflow",
        "workspace": "techtitans-pcchf",
        "project": "barbell-detection-8phtm",
        "version": None,
        "format": "yolov8",
        "keep": ["plate", "end"],
        "drop": ["barbell", "bar", "person"],
        "notes": "1.34k 图，单类 Barbell 疑似整杠——audit 时重点看框语义，不对则整源 REJECT",
    },
    "barbell-end-tracker": {
        "provider": "roboflow",
        "workspace": "barbellendtracker",
        "project": "barbell-end-tracker",
        "version": None,
        "format": "yolov8",
        "keep": ["barbell-end", "end"],
        "drop": ["barbell", "person"],
        "notes": "90 图，纯杠端实例分割；是分割格式→导出 yolov8 时转框（fetch 用 yolov8 格式）",
    },
    "barbells-detector": {
        "provider": "roboflow",
        "workspace": "yolo-project-c2bfs",
        "project": "barbells-detector",
        "version": None,
        "format": "yolov8",
        "keep": ["end", "plate"],
        "drop": ["barbell", "person"],
        "notes": "92 图，Barbell/End 双类；只留 End/plate",
    },
    "iap-thesis": {
        "provider": "roboflow",
        "workspace": "iap",
        "project": "thesis",
        "version": None,
        "format": "yolov8",
        "keep": ["plate"],
        "drop": ["person", "barbell", "dumbbell"],
        "notes": "1781 图，论文级标注；只要 plates",
    },
    "group3-workout": {
        "provider": "roboflow",
        "workspace": "technological-institute-of-the-philippines",
        "project": "group-3-workoutexercises",
        "version": None,
        "format": "yolov8",
        "keep": ["plate"],
        "drop": ["barbell", "deadlift", "person", "squat", "bench"],
        "notes": "2229 图，力量举场景；动作类名未知→ unlisted 复核",
    },
    "test-barbell-detection": {
        "provider": "roboflow",
        "workspace": "test",
        "project": "barbell_detection",
        "version": None,
        "format": "yolov8",
        "keep": ["plate"],
        "drop": ["barbell", "person"],
        "notes": "435 图，Barbell/Plates 双类；只留 Plates",
    },
    # —— Kaggle（手动下载或 kaggle API，见 fetch_public.py）——
    "kaggle-barbell-pose": {
        "provider": "kaggle",
        "slug": "danishghaffar786/barbell-detection-pose-estimation-yolov8-format",
        "keep": ["plate", "end", "cap"],
        "drop": ["barbell", "person", "pose"],
        "notes": "156MB，杠铃两端检测 YOLOv8 格式；action 类按 unlisted 复核",
    },
}


def match_class(name: str, keep: list[str], drop: list[str]) -> str:
    """类名 → keep / drop / unlisted（keep 优先，默认保守丢弃）。"""
    low = name.strip().lower()
    for k in keep:
        if k.lower() in low:
            return "keep"
    for d in drop:
        if d.lower() in low:
            return "drop"
    return "unlisted"


def mapping_table(names: list[str], keep: list[str], drop: list[str]
                  ) -> list[tuple[int, str, str]]:
    """data.yaml 类名表 → [(id, name, 决策)]。"""
    return [(i, n, match_class(n, keep, drop)) for i, n in enumerate(names)]
