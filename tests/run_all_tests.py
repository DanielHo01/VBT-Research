"""run_all_tests.py — 汇总测试入口（CI / 本地共用）
====================================================
测试文件为纯函数风格（test_*()），非 unittest 类；
按模块逐个执行并计数，失败打印 traceback 后以非零码退出。

用法:
    python3 tests/run_all_tests.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# 与各测试文件自身插入的路径一致（仓库根），保证 vbtcore 可导入
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import test_anchor  # noqa: E402
import test_geometry  # noqa: E402
import test_regrind  # noqa: E402
import test_segment  # noqa: E402

MODULES = [test_geometry, test_segment, test_anchor, test_regrind]


def main() -> int:
    total = 0
    failed = 0
    for mod in MODULES:
        tests = sorted(
            name for name in dir(mod) if name.startswith("test_") and callable(getattr(mod, name))
        )
        for name in tests:
            total += 1
            label = f"{mod.__name__}.{name}"
            try:
                getattr(mod, name)()
                print(f"PASS {label}")
            except Exception:
                failed += 1
                print(f"FAIL {label}")
                traceback.print_exc()
    print(f"\n{total - failed}/{total} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
