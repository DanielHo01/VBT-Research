#!/usr/bin/env bash
# ============================================================================
# run_local_validation.sh — 本地一键验证（视频不出本机）
# ============================================================================
# 用法:
#   bash scripts/run_local_validation.sh               # 全部步骤
#   bash scripts/run_local_validation.sh --skip-download  # 视频已就位时
#
# 做的事:
#   1. 准备 .venv 并安装依赖（已存在则跳过）
#   2. 从 Release v1.0-videos 下载 raw_videos.zip 并解压到
#      validation/dataset_benchmark/raw_videos/（若没有视频文件）
#   3. 冒烟自检 self_test.py
#   4. 跑完整基准 run_full_benchmark.py
#   5. 逐视频诊断 diagnose_benchmark.py
#
# 隐私: 所有文件只在本机处理，不上传。
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=""
if command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else echo "[错误] 未找到 python3/python，请先安装 Python 3.10+"; exit 1; fi

VENV=".venv"
VIDEOS_DIR="validation/dataset_benchmark/raw_videos"
ZIP_PATH="validation/dataset_benchmark/raw_videos.zip"

# ── 1. venv + 依赖 ─────────────────────────────────────────────────────────
if [ ! -x "$VENV/bin/python" ] && [ ! -x "$VENV/Scripts/python.exe" ]; then
    echo "==> [1/5] 创建虚拟环境 $VENV ..."
    "$PY" -m venv "$VENV"
fi
VENV_PY="$VENV/bin/python"; [ -x "$VENV/Scripts/python.exe" ] && VENV_PY="$VENV/Scripts/python.exe"

echo "==> [1/5] 检查依赖 ..."
if ! "$VENV_PY" -c "import cv2, numpy, onnxruntime, scipy, pandas" >/dev/null 2>&1; then
    echo "    安装依赖（仅首次，约 1-2 分钟）..."
    "$VENV_PY" -m pip install --quiet -r requirements.txt
else
    echo "    依赖已就绪"
fi

# ── 2. 视频 ─────────────────────────────────────────────────────────────────
HAVE_VIDEO=0
if [ -d "$VIDEOS_DIR" ] && ls "$VIDEOS_DIR"/*.mp4 >/dev/null 2>&1; then
    HAVE_VIDEO=1
fi

if [ "$HAVE_VIDEO" = "0" ] && [ "${1:-}" != "--skip-download" ]; then
    echo "==> [2/5] 下载验证视频（Release v1.0-videos, 98MB, 一次性）..."
    if ! command -v gh >/dev/null 2>&1; then
        echo "    [错误] 未找到 gh CLI，请安装且已登录（gh auth login）"
        echo "    或者手动下载: https://github.com/DanielHo01/VBT-Research/releases/tag/v1.0-videos"
        echo "    然后解压到 $VIDEOS_DIR/"
        exit 1
    fi
    mkdir -p "$VIDEOS_DIR"
    gh release download v1.0-videos -p raw_videos.zip -D validation/dataset_benchmark/ --clobber
    "$VENV_PY" - "$ZIP_PATH" "$VIDEOS_DIR" <<'PYEOF'
import sys, zipfile
from pathlib import Path
zip_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(out_dir)
print(f"    解压完成: {sum(1 for p in out_dir.rglob('*.mp4'))} 个视频 -> {out_dir}")
PYEOF
    HAVE_VIDEO=1
fi

if [ "$HAVE_VIDEO" = "0" ]; then
    echo "==> [2/5] 未发现视频，按 --skip-download 跳过下载"
else
    echo "==> [2/5] 视频已就绪: $VIDEOS_DIR"
fi

# ── 3. 自检 ─────────────────────────────────────────────────────────────────
echo "==> [3/5] 冒烟自检（无视频回归测试）..."
"$VENV_PY" scripts/self_test.py

# ── 4. 基准 ─────────────────────────────────────────────────────────────────
echo "==> [4/5] 运行完整基准 ..."
"$VENV_PY" scripts/run_full_benchmark.py

# ── 5. 诊断 ─────────────────────────────────────────────────────────────────
echo "==> [5/5] 逐视频诊断 ..."
"$VENV_PY" scripts/diagnose_benchmark.py

echo ""
echo "✅ 全部完成！结果目录: $REPO_ROOT/validation/dataset_benchmark/results/"
echo "   把 diagnose_benchmark.py 输出的汇总（flags 分布）发给我，即可定位下一步优化点。"
