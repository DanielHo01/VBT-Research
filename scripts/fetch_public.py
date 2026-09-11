"""fetch_public.py — M3 公开数据下载（注册表驱动）
==================================================
数据源与类映射见 scripts/public_registry.py（单一事实来源）。

Roboflow 需要免费 API key（https://app.roboflow.com/settings/api），
通过环境变量传入（不要写进命令行历史/代码）：

    Windows (cmd):  set ROBOFLOW_API_KEY=xxxx
    Windows (PS):   $env:ROBOFLOW_API_KEY=\"xxxx\"
    Linux/macOS:    export ROBOFLOW_API_KEY=xxxx
    pip install roboflow

用法：
    python scripts/fetch_public.py --list                 # 看注册表（无需 key）
    python scripts/fetch_public.py --source weightlifting-plates-v11
    python scripts/fetch_public.py --source barbell-tracking --version 2
    python scripts/fetch_public.py --all --out datasets/public

下载后必须先跑 audit（先审后合）：
    python scripts/audit_public.py --src datasets/public/<name>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from public_registry import SOURCES  # noqa: E402


def cmd_list() -> int:
    print(f"{'name':32s} {'provider':9s} size/notes")
    for name, cfg in SOURCES.items():
        ver = cfg.get("version", "?")
        if cfg.get("provider") == "roboflow":
            loc = f"{cfg['workspace']}/{cfg['project']} v{ver}"
        else:
            loc = cfg.get("slug", "?")
        print(f"{name:32s} {cfg['provider']:9s} {loc}")
        print(f"{'':32s}  keep={cfg['keep']} drop={cfg['drop']}")
    print("\nKaggle 源需手动下载（或配 kaggle.json 后用 kaggle API），"
          "解压到 datasets/public/<name>/ 即可进 audit。")
    return 0


def fetch_roboflow(name: str, cfg: dict, out_dir: Path,
                   version_override: int | None) -> int:
    try:
        from roboflow import Roboflow  # noqa: PLC0415
    except ImportError:
        print("缺 roboflow 包：pip install roboflow")
        return 1
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        print("缺 ROBOFLOW_API_KEY 环境变量（见本文件头注释）。")
        return 1
    rf = Roboflow(api_key=key)
    project = rf.workspace(cfg["workspace"]).project(cfg["project"])
    ver = version_override or cfg.get("version")
    if ver is None:
        versions = project.versions()
        ver = versions[-1] if isinstance(versions, list) else project.version()
        print(f"  [{name}] 注册表无版本号，用 latest={ver}（audit 时核对内容）")
        version_obj = project.version(ver) if isinstance(ver, int) else ver
    else:
        version_obj = project.version(ver)
    dest = out_dir / name
    print(f"  [{name}] 下载 {cfg['workspace']}/{cfg['project']} v{ver} → {dest}")
    version_obj.download(cfg.get("format", "yolov8"), location=str(dest))
    print(f"  [{name}] 完成。下一步：audit_public.py --src {dest}")
    return 0


def fetch_kaggle(name: str, cfg: dict, out_dir: Path) -> int:
    dest = out_dir / name
    print(f"  [{name}] Kaggle 源：https://www.kaggle.com/datasets/{cfg['slug']}")
    try:
        from kaggle import api as kaggle_api  # noqa: PLC0415
    except ImportError:
        print(f"  请手动下载解压到 {dest}/（或 pip install kaggle 并配 kaggle.json 后重跑）")
        return 2
    try:
        kaggle_api.authenticate()
        dest.mkdir(parents=True, exist_ok=True)
        kaggle_api.dataset_download_files(cfg["slug"], path=str(dest), unzip=True)
    except Exception as e:  # noqa: BLE001 — 给出手动兜底
        print(f"  kaggle API 失败（{e}），请手动下载到 {dest}/")
        return 2
    print(f"  [{name}] 完成。下一步：audit_public.py --src {dest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="M3 公开数据下载")
    ap.add_argument("--list", action="store_true", help="列注册表（无需 key）")
    ap.add_argument("--source", default=None, help="注册表名（逗号分隔多个）")
    ap.add_argument("--all", action="store_true", help="下载全部 Roboflow 源")
    ap.add_argument("--version", type=int, default=None, help="覆盖版本号")
    ap.add_argument("--out", default="datasets/public")
    args = ap.parse_args()

    if args.list or (not args.source and not args.all):
        return cmd_list()

    if args.all:
        names = [n for n, c in SOURCES.items() if c["provider"] == "roboflow"]
    else:
        names = [n.strip() for n in args.source.split(",") if n.strip()]
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        print(f"未知源 {unknown}，--list 查看注册表")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for n in names:
        cfg = SOURCES[n]
        if cfg["provider"] == "roboflow":
            rc |= fetch_roboflow(n, cfg, out_dir, args.version)
        else:
            rc |= fetch_kaggle(n, cfg, out_dir)
    return rc


if __name__ == "__main__":
    sys.exit(main())
