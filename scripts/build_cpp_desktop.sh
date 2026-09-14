#!/bin/bash
# scripts/build_cpp_desktop.sh — Linux 桌面端 C++ 编译（铁律闸门 Step 1）
# ==========================================================================
# 用途：在 Linux 桌面端编译 vbtcore-cpp，跑通 34 视频 golden_test。
#       验收后才允许启动 NDK 交叉编译（铁律闸门）。
#
# 适用系统：Ubuntu 22.04+ / Debian 12+（apt 包管理）
# 预计耗时：首次安装 15 分钟，编译 5 分钟，golden_test 30 分钟。
# 预计产出：build/vbtcore-cpp/ 静态库 + golden_test 可执行。
#
# 用法：
#   chmod +x scripts/build_cpp_desktop.sh
#   ./scripts/build_cpp_desktop.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/vbtcore-cpp/build"
CPP_DIR="${REPO_ROOT}/vbtcore-cpp"
VBT_BASELINE_DIR="${REPO_ROOT}/validation/reports/cpp_baseline"

echo "================================================================"
echo " vbtcore-cpp 桌面端编译（铁律闸门 Step 1）"
echo " REPO: ${REPO_ROOT}"
echo "================================================================"

# ── 0. 系统依赖检查 ───────────────────────────────────────────────
echo ">>> [0/6] 检查系统依赖..."
if ! command -v cmake &>/dev/null; then
    echo "    ✗ cmake 缺失，正在安装..."
    sudo apt update && sudo apt install -y cmake ninja-build
fi
if ! command -v g++ &>/dev/null && ! command -v clang++ &>/dev/null; then
    echo "    ✗ C++ 编译器缺失，正在安装..."
    sudo apt install -y g++ clang
fi

CMAKE_VER=$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+')
echo "    ✓ cmake: ${CMAKE_VER}"
if command -v clang++ &>/dev/null; then
    CXX=clang++
    echo "    ✓ clang++: $(clang++ --version | head -1)"
else
    CXX=g++
    echo "    ✓ g++: $(g++ --version | head -1)"
fi

# ── 1. 系统库依赖（OpenCV + nlohmann/json）─────────────────────
echo ">>> [1/6] 安装系统库（OpenCV 4 + nlohmann/json）..."
if ! pkg-config --modversion opencv4 2>/dev/null; then
    if ! pkg-config --modversion opencv 2>/dev/null; then
        echo "    ✗ OpenCV 缺失，正在安装..."
        sudo apt install -y libopencv-dev
    fi
fi
if ! dpkg -l | grep -q nlohmann-json3-dev; then
    echo "    ✗ nlohmann/json 缺失，正在安装..."
    sudo apt install -y nlohmann-json3-dev
fi

OPENCV_VER=""
OPENCV_VER=$(pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv)
echo "    ✓ OpenCV: ${OPENCV_VER}"
echo "    ✓ nlohmann/json: header-only（apt nlohmann-json3-dev）"

# ── 2. ONNX Runtime CPU 包（apt 或下载预编译）─────────────────
echo ">>> [2/6] 安装 ONNX Runtime..."
if [ ! -f /usr/local/onnxruntime/lib/libonnxruntime.so ]; then
    echo "    ✗ ONNX Runtime 缺失，正在下载预编译..."
    ONNX_VER="1.17.1"
    ONNX_TARBALL="onnxruntime-linux-x64-${ONNX_VER}.tgz"
    wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VER}/${ONNX_TARBALL}"
    sudo tar -xzf "${ONNX_TARBALL}" -C /usr/local/
    sudo mv "/usr/local/onnxruntime-linux-x64-${ONNX_VER}" /usr/local/onnxruntime
    sudo ldconfig
    rm -f "${ONNX_TARBALL}"
else
    echo "    ✓ ONNX Runtime 已存在: /usr/local/onnxruntime"
fi
echo "    ✓ ONNX Runtime: /usr/local/onnxruntime"

# ── 3. 准备 Python 端 baseline JSON（供 golden_test 对比）───
echo ">>> [3/6] 生成 Python baseline JSON..."
mkdir -p "${VBT_BASELINE_DIR}"
cd "${REPO_ROOT}"
# 必须用 gen_cpp_baseline.py：它产出 golden_test 期望的 per-video JSON
# （<video_id>.mp4.json，含 reps/mpp/diagnostics）。
# run_benchmark_v0.py 产出的是聚合 Markdown 报告，格式不匹配（历史错误）。
# stride=1 = 第一阶段「Make it Right」口径，不得改为稀疏检测。
python3 scripts/gen_cpp_baseline.py \
    --workers "$(nproc)" \
    --copy-to-tmp \
    --out-dir "${VBT_BASELINE_DIR}" || {
    echo "    ⚠ Python baseline 生成失败，将以 NO_BASELINE 模式跑 golden_test"
}

# ── 4. CMake 配置 ─────────────────────────────────────────────
echo ">>> [4/6] CMake 配置（Release + ASan）..."
mkdir -p "${BUILD_DIR}" && cd "${BUILD_DIR}"
cmake "${CPP_DIR}" \
    -GNinja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" \
    -DONNXRUNTIME_ROOT=/usr/local/onnxruntime \
    -DJSON_ROOT=/usr/include \
    -DVBT_BUILD_TESTS=ON

# ── 5. 编译 ───────────────────────────────────────────────────
echo ">>> [5/6] 编译 vbtcore 静态库 + golden_test..."
ninja -j"$(nproc)" vbtcore golden_test

# ── 6. 跑 golden_test ─────────────────────────────────────────
echo ">>> [6/6] 跑 golden_test（34 视频 vs Python baseline）..."
echo "============================================================"
# ASan 泄漏抑制：libonnxruntime 自身有 ~318 B 的静态分配未释放，
# 属第三方库行为，不应判为我方内存错误。
if [ ! -f "${REPO_ROOT}/scripts/lsan.supp" ]; then
    printf 'leak:libonnxruntime\n' > "${REPO_ROOT}/scripts/lsan.supp"
fi

set +e
ASAN_OPTIONS=detect_leaks=1 \
LSAN_OPTIONS="suppressions=${REPO_ROOT}/scripts/lsan.supp" \
./tests/golden_test \
    "${REPO_ROOT}/validation/dataset_benchmark/raw_videos" \
    "${VBT_BASELINE_DIR}" \
    2>&1 | tee golden_test.log
GOLDEN_RC=${PIPESTATUS[0]}
set -e

echo "============================================================"
echo " 铁律闸门验收"
echo "============================================================"

# 直接解析「通过: N」为整数，避免用正则碰运气（旧写法 "通过: [2-3][0-9]"
# 既会漏判也会误判，且完全忽略 golden_test 的退出码）。
N_PASS=$(grep -oP '通过:\s*\K[0-9]+' golden_test.log | head -1)
N_PASS=${N_PASS:-0}
N_TOTAL=$(grep -oP '总计:\s*\K[0-9]+' golden_test.log | head -1)
N_TOTAL=${N_TOTAL:-34}

# ASan 真实内存错误（泄漏已 suppress，这里抓的是越界/UAF/SEGV）
if grep -qE "ERROR: AddressSanitizer|SEGV|heap-buffer-overflow|use-after-free" golden_test.log; then
    echo "  ✗ 检测到 ASan 内存错误 → 严禁启动 NDK 编译"
    grep -nE "ERROR: AddressSanitizer|SEGV|heap-buffer-overflow|use-after-free" golden_test.log | head
    exit 1
fi
echo "  ✓ ASan：无内存越界/UAF/SEGV"

echo "  通过: ${N_PASS}/${N_TOTAL}（要求 ≥ 20）"
if [ "${N_PASS}" -ge 20 ] && [ "${GOLDEN_RC}" -eq 0 ]; then
    echo "  ✅ Step 1-3 通过：≥ 20/34 视频 MCV 偏差 ≤ 0.005 m/s"
    echo "  → 可进入 Step 4：NDK 交叉编译"
    echo ""
    echo "  下一步："
    echo "    export ANDROID_NDK_HOME=/path/to/ndk/25.2.9519653"
    echo "    ./scripts/build_cpp_android.sh"
    exit 0
else
    echo "  ✗ Step 1-3 未通过：< 20/34 视频或偏差超阈值"
    echo "  → 严禁启动 NDK 编译（铁律闸门）"
    echo "  → 请检查 golden_test.log 中的 mcv_err，定位数值漂移"
    echo "  → 提示：确保 Python 端 opencv 版本与 C++ 端一致"
    echo "     （goodFeaturesToTrack / calcOpticalFlowPyrLK 跨大版本实现有差异）"
    exit 1
fi
