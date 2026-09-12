# EasyVBT Demo — Android 应用

> Phase 3 交付：Android Compose UI + JNI 桥接 vbtcore-cpp + 离线 mp4 导入分析

## 目录结构

```
android/
├── app/                                 # Kotlin 应用模块
│   └── src/main/
│       ├── kotlin/com/easyvbt/demo/
│       │   ├── MainActivity.kt           # Compose UI 入口（视频选择 + 进度 + 结果）
│       │   ├── EngineBridge.kt          # JNI 封装（System.loadLibrary + nativeAnalyze）
│       │   └── (ResultViewModel 内联于 MainActivity.kt)
│       ├── res/
│       │   └── values/
│       │       ├── strings.xml          # app_name
│       │       └── themes.xml           # Theme.EasyVBT
│       └── AndroidManifest.xml          # READ_MEDIA_VIDEO 权限
├── cpp/                                 # JNI 桥接层
│   ├── CMakeLists.txt                   # NDK 构建脚本
│   └── vbtcore_jni.cpp                  # JNI 函数（Java_com_easyvbt_demo_...）
├── build.gradle.kts                     # 根 Gradle 配置
├── settings.gradle.kts                  # 子项目声明（含 vbtcore-cpp）
├── gradle.properties                    # AndroidX 启用
└── README.md (本文件)
```

## 构建步骤

### 前置依赖

| 依赖 | 版本 | 来源 |
| --- | --- | --- |
| Android Studio | Hedgehog (2023.1.1) 或更新 | <https://developer.android.com/studio> |
| Android NDK | r25c (Side by side) | SDK Manager → SDK Tools |
| Android SDK | 34 (compileSdk = 34) | SDK Manager |
| CMake | 3.22.1+ | SDK Manager → SDK Tools |
| JDK | 17 (Temurin) | <https://adoptium.net/> |
| OpenCV Android SDK | 4.5+ | <https://opencv.org/releases/> |
| ONNX Runtime Mobile | 1.17+ | <https://onnxruntime.ai/docs/install/android.html> |

### 模型文件

将 `models/best.onnx` 拷贝到 `android/app/src/main/assets/best.onnx`：

```bash
mkdir -p android/app/src/main/assets
cp models/best.onnx android/app/src/main/assets/
```

### 构建命令

```bash
cd android/
./gradlew assembleDebug
# 输出：android/app/build/outputs/apk/debug/app-debug.apk
```

### 安装到设备

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.easyvbt.demo.debug/com.easyvbt.demo.MainActivity
```

## JNI 调用约定

### Kotlin 端

```kotlin
import com.easyvbt.demo.EngineBridge

val result = EngineBridge.analyzeVideo(
    srcVideoPath = "/sdcard/Download/input.mp4",
    modelAssetPath = "best.onnx",          // 从 assets 拷贝后路径
    exerciseType = "squat_bench",          // 或 "deadlift"
    plateDiameterM = 0.45,
    outerPlate = "20kg",
)
```

### C++ 端（vbtcore-cpp）

```cpp
#include "vbt/analyze.hpp"

AnalyzeOptions opts;
opts.exercise_type = "squat_bench";
opts.plate_diameter_m = 0.45;
std::string json_str = vbt::analyze_video_json("/sdcard/Download/input.mp4",
                                                "best.onnx", opts);
```

### JNI 函数签名

```cpp
JNIEXPORT jstring JNICALL
Java_com_easyvbt_demo_EngineBridge_nativeAnalyze(
    JNIEnv* env,
    jobject thiz,
    jstring videoPath,    // 视频绝对路径
    jstring modelPath,    // ONNX 模型绝对路径
    jstring optionsJson,  // 可选，null 表示默认
);
```

## UI 流程

1. **主屏**：`选择视频` 按钮 → `ACTION_OPEN_DOCUMENT` (类型 video/mp4)
2. **加载**：拷贝视频到应用 cache 目录，调 `EngineBridge.analyzeVideo`（IO/Default 调度器）
3. **进度**：CircularProgressIndicator + 视频名
4. **结果**：状态卡片（OK/NO_PLATE_DETECTED）+ 速度折线图 + Rep 列表 + MCV/ROM/峰值

## 权限说明

| 权限 | 用途 | 必需？ |
| --- | --- | --- |
| `READ_MEDIA_VIDEO` | Android 13+ 读取相册视频 | ✅ Demo 必需 |
| `READ_EXTERNAL_STORAGE` | Android 12 及以下（maxSdk=32） | ✅ Demo 必需 |
| `INTERNET` | ❌ 不需要（Demo 离线分析） | 已移除 |

## 已知限制

- **离线导入**：Demo 阶段仅支持用户选择 mp4 文件，不做实时摄像头采集
- **模型未量化**：直接使用 best.onnx（10.1 MB），未做 int8 量化
- **单 ABI**：当前默认 arm64-v8a，可扩展 armeabi-v7a
- **无 LVP / e1RM / 历史记录**：保留给后续版本

## 下一步

- Stage 4（实时推理 + 摄像头采集）
- iOS 端 Swift UI（共用 vbtcore-cpp）
- 模型 int8 量化
- 多动作类型（硬拉已支持；待加卧推/深蹲专属 UI）

---

*构建脚本由 AI 助手（受 Daniel 委托）生成；Demo 阶段 Android 7.0+ 兼容。*
