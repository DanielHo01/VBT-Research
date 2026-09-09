"""
config.py — 仓库内相对路径与可复现路径解析
===========================================

不再硬编码 Windows 绝对路径：所有脚本默认在**仓库内**查找数据/模型。
视频、模型等大文件可以放在任何本地位置，用环境变量覆盖即可
（本地验证，全程不上传）：

    VBT_MODEL_PATH    ONNX 模型路径（默认 models/barbell_v4.onnx）
    VBT_VIDEOS_DIR    原始视频目录（可指向任意本地视频文件夹）
    VBT_INDEX_PATH    dataset_index.json 路径
    VBT_OUTPUT_DIR    基准/诊断结果输出目录
    VBT_DATASETS_DIR  数据集根目录（标注产物用）

示例（视频在 E 盘 / 家目录，模型用 plate_v1）：
    export VBT_VIDEOS_DIR=/media/me/videos/raw_videos
    export VBT_MODEL_PATH=~/vbt/models/plate_v1.onnx
"""
from __future__ import annotations

import os
from pathlib import Path

# 仓库根：validation/dataset_benchmark/config.py -> 上两级
REPO_ROOT = Path(__file__).resolve().parents[2]

# 以下常量仅当环境变量未设置时用作默认值
_DEFAULT_MODEL       = REPO_ROOT / "models" / "barbell_v4.onnx"
_DEFAULT_VIDEOS_DIR  = REPO_ROOT / "validation" / "dataset_benchmark" / "raw_videos"
_DEFAULT_INDEX       = REPO_ROOT / "validation" / "dataset_benchmark" / "dataset_index.json"
_DEFAULT_OUTPUT      = REPO_ROOT / "validation" / "dataset_benchmark" / "results"
_DEFAULT_DATASETS    = REPO_ROOT / "datasets"


def _env_path(key: str, default: Path) -> Path:
    val = os.environ.get(key)
    return Path(val).expanduser() if val else default


def default_model_path() -> Path:
    """当前使用的 ONNX 模型（VBT_MODEL_PATH 可覆盖）。"""
    return _env_path("VBT_MODEL_PATH", _DEFAULT_MODEL)


def raw_videos_dir() -> Path:
    """原始视频目录（VBT_VIDEOS_DIR 可覆盖）。"""
    return _env_path("VBT_VIDEOS_DIR", _DEFAULT_VIDEOS_DIR)


def dataset_index_path() -> Path:
    """数据集索引（VBT_INDEX_PATH 可覆盖）。"""
    return _env_path("VBT_INDEX_PATH", _DEFAULT_INDEX)


def output_dir() -> Path:
    """基准/诊断结果输出（VBT_OUTPUT_DIR 可覆盖）。"""
    return _env_path("VBT_OUTPUT_DIR", _DEFAULT_OUTPUT)


def datasets_dir() -> Path:
    """数据集根目录，用于标注产物（VBT_DATASETS_DIR 可覆盖）。"""
    return _env_path("VBT_DATASETS_DIR", _DEFAULT_DATASETS)
