# android/cpp/CMakeLists.txt
# =========================
# JNI 桥接层：将 vbtcore-cpp 打包为 libvbtcore_jni.so 给 Kotlin 调用。

cmake_minimum_required(VERSION 3.22.1)
project(vbtcore_jni CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── 引入 vbtcore-cpp 子项目 ──────────────────────────────────────────
set(VBTCORE_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../vbtcore-cpp"
    CACHE PATH "vbtcore-cpp 根目录")

# 包含头文件
include_directories(${VBTCORE_CPP_DIR}/include)
include_directories(${VBTCORE_CPP_DIR})

# 静态库 vbtcore（来自 vbtcore-cpp/CMakeLists.txt）
add_subdirectory(${VBTCORE_CPP_DIR} vbtcore-cpp-build EXCLUDE_FROM_ALL)

# ── JNI 桥接源 ──────────────────────────────────────────────────────
add_library(vbtcore_jni SHARED
    vbtcore_jni.cpp
)

# ── 链接依赖 ────────────────────────────────────────────────────────
find_library(log-lib log)
target_link_libraries(vbtcore_jni
    PRIVATE
        vbtcore
        ${log-lib}
)

# ── 安装到 APK ──────────────────────────────────────────────────────
# Gradle 自动处理 .so 打包路径，无需 install 命令