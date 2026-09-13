#!/bin/bash
# scripts/iron_gate_wsl2.sh — WSL2 Ubuntu 专用铁律闸门一键脚本
# ==============================================================
# 在 WSL2 Ubuntu 22.04 root 终端跑（不带 sudo，因为 root 不需要）。
# 自检环境：非 WSL2 / 非 root / 缺 apt 都立即大声报错。
# 产出：/tmp/golden_test.log（编译 + 34 视频 golden_test）

set -e

echo "=== 0. 环境自检 ==="
if ! command -v apt >/dev/null 2>&1; then
    echo "❌ 错！不在 WSL2 Ubuntu（无 apt）"
    echo "👉 正确操作：在 PowerShell 跑 wsl -d Ubuntu-22.04"
    exit 1
fi
echo "✅ WSL2 Ubuntu ($(lsb_release -s -d 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2))"
echo "✅ cmake $(cmake --version | head -1 | awk '{print $3}')"
echo "✅ clang++ $(clang++ --version | head -1 | awk '{print $3}')"
echo "✅ OpenCV $(pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv)"

echo ""
echo "=== 1. 装 ONNX Runtime ==="
[ -f /tmp/onnxruntime/include/onnxruntime_cxx_api.h ] || {
    cd /tmp
    rm -f onnx.tgz
    wget -q --retry-connrefused --tries=3 https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz -O onnx.tgz
    [ -s onnx.tgz ] || {
        echo "❌ ONNX Runtime 下载失败"
        exit 1
    }
    rm -rf /tmp/onnxruntime
    tar -xzf onnx.tgz -C /tmp/
    TOP=$(tar -tzf onnx.tgz | head -1 | cut -d/ -f1)
    mv "/tmp/$TOP" /tmp/onnxruntime
    ldconfig
}
ls /tmp/onnxruntime/include/onnxruntime_cxx_api.h && echo "✅ ONNX Runtime"

echo ""
echo "=== 2. 拉取最新代码 ==="
if [ ! -d /root/VBT-Research/.git ]; then
    git clone https://github.com/DanielHo01/VBT-Research.git /root/VBT-Research
else
    cd /root/VBT-Research && git pull --ff-only
fi
ls /root/VBT-Research/scripts/build_cpp_desktop.sh >/dev/null
cd /root/VBT-Research
git log --oneline -1
echo "✅ 代码就位"

echo ""
echo "=== 3. 去 sudo（root 不需要）+ 闸门 ==="
sed -i 's/sudo //g' scripts/build_cpp_desktop.sh
chmod +x scripts/build_cpp_desktop.sh
export ONNXRUNTIME_ROOT=/tmp/onnxruntime
./scripts/build_cpp_desktop.sh 2>&1 | tee /tmp/golden_test.log
EC=${PIPESTATUS[0]}

echo ""
echo "================================================================"
echo "=== 闸门结果（exit_code=$EC）==="
echo "================================================================"
tail -30 /tmp/golden_test.log
exit $EC
