# EasyVBT Demo 构建与运行指南

> Phase 4 交付：2 周 Demo 完整构建步骤 + 验收清单

## 0. 前置条件（一次性）

### 0.1 开发主机

- Windows 10/11 / macOS 13+ / Ubuntu 22.04+
- Git 2.30+
- Python 3.10+（仅 vbtcore Python 端需要；C++ 端不依赖）
- CMake 3.22.1+
- Android Studio Hedgehog (2023.1.1) 或更新
- Android NDK r25c（侧载 SDK Manager → SDK Tools）
- JDK 17（Temurin 推荐）

### 0.2 Python 依赖（仅 baseline 对比需要）

```bash
pip install -r requirements.txt
# onnxruntime, opencv-python, numpy, scipy 等
```

### 0.3 模型与视频

- 模型：`models/best.onnx`（10.1 MB，YOLOv11n，已 commit）
- 视频：`validation/dataset_benchmark/raw_videos/`（34 个 mp4，已 commit）
- 或从 GitHub Release `v1.0-videos` 下载

### 0.4 C++ 第三方库

| 库 | 版本 | 用途 |
| --- | --- | --- |
| OpenCV Android SDK | 4.5+ | 视频解码 + 图像预处理 |
| ONNX Runtime Mobile | 1.17+ | 模型推理（CPU EP） |
| nlohmann/json | 3.11+ | analyze_video_json 输出 |

## 1. 验证 Python baseline（5 分钟）

```bash
python tests/run_all_tests.py
# 期望：26/26 tests passed
```

```bash
python scripts/run_benchmark_v0.py --tag baseline_check
# 期望：与 BENCHMARK_v5.md 数字一致（≥ 24/34 OK）
```

## 2. 编译 C++ 中间层（30 分钟）

### 2.1 桌面环境验证（可选）

```bash
cd vbtcore-cpp/
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel
./tests/golden_test ../validation/dataset_benchmark/raw_videos
# 期望：≥ 20/34 视频与 Python baseline 偏差 ≤ 阈值
```

### 2.2 NDK 交叉编译（生产构建）

```bash
export ANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/25.2.9519653
cd vbtcore-cpp/
mkdir -p build-android && cd build-android
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-24 \
      -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel
# 产出：libvbtcore.a（NDK 静态库，给 libvbtcore_jni.so 链接）
```

## 3. 构建 Android APK（10 分钟）

```bash
cd android/

# 模型放入 assets
cp ../models/best.onnx app/src/main/assets/

# 构建 Debug APK
./gradlew assembleDebug
# 产出：android/app/build/outputs/apk/debug/app-debug.apk
```

## 4. 安装与测试（5 分钟）

```bash
# 连接 Android 设备（USB 调试已启用）
adb devices

# 安装 APK
adb install -r app/build/outputs/apk/debug/app-debug.apk

# 启动应用
adb shell am start -n com.easyvbt.demo.debug/com.easyvbt.demo.MainActivity
```

### 4.1 设备最低要求

- Android 7.0+（API 24+）
- arm64-v8a（推荐）或 armeabi-v7a
- ≥ 200 MB 可用存储

### 4.2 测试视频

- 准备 3 个 mp4（30kg / 80kg / 130kg 各一个，≤ 60 秒）
- 通过系统文件选择器导入

## 5. 验收清单（DoD）

### 5.1 Python baseline ✅

- [ ] `tests/run_all_tests.py` 26/26 全过
- [ ] 34 视频 benchmark ≥ 24/34 OK
- [ ] `BENCHMARK_v5.md` 与上次数字一致

### 5.2 C++ 中间层 ✅

- [ ] `libvbtcore.a` 在桌面环境编译通过
- [ ] `libvbtcore.a` 在 NDK r25c 编译通过（arm64-v8a）
- [ ] golden test ≥ 20/34 视频与 Python baseline 一致
  - 位置偏差 ≤ 1px
  - 速度偏差 ≤ 0.001 m/s

### 5.3 Android APK ✅

- [ ] APK 在 Android 12+ 真机安装成功
- [ ] 选 mp4 → 进度条走完 → 看到 rep 列表 + 速度曲线
- [ ] 3 个真实视频跑出 rep 列表（30kg / 80kg / 130kg）
- [ ] 崩溃日志无 ERROR 级条目
- [ ] APK size < 25 MB（含 best.onnx）

### 5.4 文档 ✅

- [ ] `docs/BRANCH_POLICY.md` 写完
- [ ] `docs/VBTCORE_PUBLIC_API.md` 写完
- [ ] `docs/DEMO_GUIDE.md` 写完（本文件）
- [ ] `android/README.md` 写完
- [ ] PR review 至少 1 人通过

## 6. 常见问题排查

### 6.1 编译错误

| 错误 | 原因 | 解决 |
| --- | --- | --- |
| `cannot find -lonnxruntime` | 未指定 ONNX Runtime 路径 | `export ONNXRUNTIME_ROOT=/path/to/onnxruntime-android` |
| `OpenCV not found` | 未配置 OpenCV Android SDK | `export OpenCV_DIR=/path/to/OpenCV-android-sdk/sdk/native/jni` |
| `nlohmann/json.hpp not found` | 未安装 nlohmann/json | `apt install nlohmann-json3-dev` 或 vendoring |

### 6.2 运行时错误

| 错误 | 原因 | 解决 |
| --- | --- | --- |
| `System.loadLibrary failed` | .so 与 ABI 不匹配 | 确认 APK 内含 arm64-v8a/libvbtcore_jni.so |
| `NO_PLATE_DETECTED` 全部视频 | 模型路径错误 / 模型损坏 | 检查 best.onnx 文件大小（10.1 MB） |
| 进度条卡死 | OpenCV 视频解码失败 | 转码为 H.264 mp4（HEVC 在某些设备失败） |
| MCV 全为 0 | mpp 未锁定 | 检查视频开头是否有 20+ 帧静止准备期 |

### 6.3 NDK 调试

```bash
# NDK 工具链日志
export CMAKE_VERBOSE_MAKEFILE=1
./gradlew assembleDebug --info

# APK ABI 检查
$ANDROID_NDK_HOME/.../aapt dump badging app-debug.apk | grep native-code
```

## 7. 下一步（Demo 通过后启动）

- **Stage 4 — 实时推理 + 摄像头采集**（5 天）
  - CameraX 集成
  - 实时帧流处理（每 N 帧跑 ONNX + LK 光流）
  - UI 加摄像头预览 + 实时 MCV 叠加
- **Stage 5 — iOS 移植**（3 天）
  - Xcode 项目 + Objective-C++ 桥接 vbtcore-cpp
  - SwiftUI 界面（参照 Compose）
- **Stage 6 — 模型量化**（2 天）
  - ONNX Runtime int8 量化
  - APK size 降至 < 15 MB

---

*Demo 阶段目标：2 周内可演示的 Android APK，可选 34 视频中 ≥ 20 个通过*
