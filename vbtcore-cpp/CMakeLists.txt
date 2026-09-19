cmake_minimum_required(VERSION 3.18)
project(vbtcore_cpp
    VERSION 0.1.0
    DESCRIPTION "C++ mid-layer for VBT-Research (Android Demo Phase 2)"
    LANGUAGES CXX)

# ── C++ 标准 ──────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── 编译选项 ──────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(VBT_BUILD_TESTS "Build golden tests" ON)
option(VBT_BUILD_ANDROID "Build for Android NDK" OFF)

# ── 公共配置 ──────────────────────────────────────────────────────
if(NOT DEFINED VBT_DEPLOY_DIR)
    set(VBT_DEPLOY_DIR "${CMAKE_BINARY_DIR}/deploy")
endif()

# ── 依赖 ──────────────────────────────────────────────────────────
if(VBT_BUILD_ANDROID)
    # Android: OpenCV4Android SDK（由 build.gradle.kts 传入 -DOpenCV_DIR）
    # opencv-mobile 只提供 core/imgproc/video，imgcodecs/videoio 可能缺失 → QUIET
    find_package(OpenCV QUIET COMPONENTS core imgproc video)
    if(OpenCV_FOUND)
        message(STATUS "OpenCV on Android: ${OpenCV_VERSION} (${OpenCV_INCLUDE_DIRS})")
        # opencv-mobile creates lowercase IMPORTED targets (opencv_core, opencv_imgproc, opencv_video).
        # Create uppercase namespace aliases so target_link_libraries(OpenCV::core ...) works.
        if(NOT TARGET OpenCV::core AND TARGET opencv_core)
            add_library(OpenCV::core ALIAS opencv_core)
        endif()
        if(NOT TARGET OpenCV::imgproc AND TARGET opencv_imgproc)
            add_library(OpenCV::imgproc ALIAS opencv_imgproc)
        endif()
        if(NOT TARGET OpenCV::video AND TARGET opencv_video)
            add_library(OpenCV::video ALIAS opencv_video)
        endif()
    else()
        message(WARNING "OpenCV not found on Android. Using stub headers from cmake/opencv_stub.")
        set(OpenCV_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/opencv_stub")
        # 为 stub 模式创建空 target
        if(NOT TARGET OpenCV::core)
            add_library(OpenCV::core INTERFACE IMPORTED)
            set_target_properties(OpenCV::core PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}")
        endif()
        if(NOT TARGET OpenCV::imgproc)
            add_library(OpenCV::imgproc INTERFACE IMPORTED)
            set_target_properties(OpenCV::imgproc PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}")
        endif()
        if(NOT TARGET OpenCV::video)
            add_library(OpenCV::video INTERFACE IMPORTED)
            set_target_properties(OpenCV::video PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}")
        endif()
    endif()
else()
    find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs videoio video)
    # Ubuntu 22.04 apt OpenCV 4.5.4 不导出 OpenCV::core/imgproc 现代目标
    # → 找不到时手动建 fallback
    if(NOT TARGET OpenCV::core)
        add_library(OpenCV::core INTERFACE IMPORTED)
        set_target_properties(OpenCV::core PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}")
        if(OpenCV_LIBS)
            set_target_properties(OpenCV::core PROPERTIES
                INTERFACE_LINK_LIBRARIES "${OpenCV_LIBS}")
        endif()
    endif()
    if(NOT TARGET OpenCV::imgproc)
        add_library(OpenCV::imgproc INTERFACE IMPORTED)
        set_target_properties(OpenCV::imgproc PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}")
    endif()
    if(NOT TARGET OpenCV::imgcodecs)
        add_library(OpenCV::imgcodecs INTERFACE IMPORTED)
        set_target_properties(OpenCV::imgcodecs PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}")
    endif()
    if(NOT TARGET OpenCV::videoio)
        add_library(OpenCV::videoio INTERFACE IMPORTED)
        set_target_properties(OpenCV::videoio PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}")
    endif()
    if(NOT TARGET OpenCV::video)
        add_library(OpenCV::video INTERFACE IMPORTED)
        set_target_properties(OpenCV::video PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}")
    endif()
endif()

# ONNX Runtime C++ API（必须预编译 NDK 版本）
# Android: 使用从 Maven 下载的 AAR 中提取的 ONNX Runtime
if(VBT_BUILD_ANDROID)
    set(ORT_ANDROID_DIR "D:/AndroidSdk/onnxruntime-android")
    IF(EXISTS "${ORT_ANDROID_DIR}/include/onnxruntime_cxx_api.h")
        set(ONNXRUNTIME_INCLUDE_DIR "${ORT_ANDROID_DIR}/include")
        message(STATUS "ONNX Runtime headers found at ${ORT_ANDROID_DIR}/include")
    ENDIF()
    set(ORT_LIB_PATH "${ORT_ANDROID_DIR}/extracted/jni/${ANDROID_ABI}/libonnxruntime.so")
    IF(EXISTS "${ORT_LIB_PATH}")
        set(ONNXRUNTIME_LIB "${ORT_LIB_PATH}")
        message(STATUS "ONNX Runtime lib found at ${ORT_LIB_PATH}")
    ENDIF()
else()
    # 桌面构建
    find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
        PATHS ${ONNXRUNTIME_ROOT}/onnxruntime/include /usr/local/onnxruntime/include)
    find_library(ONNXRUNTIME_LIB NAMES onnxruntime
        PATHS ${ONNXRUNTIME_ROOT}/onnxruntime/lib /usr/local/onnxruntime/lib)
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(WARNING "ONNX Runtime not found. Build ONNX-less demo with stub detector.")
    set(VBT_HAS_ONNXRUNTIME OFF)
else()
    set(VBT_HAS_ONNXRUNTIME ON)
endif()

# nlohmann/json (header-only)
# Android: use the include dir bundled in vbtcore-cpp directly
if(VBT_BUILD_ANDROID)
    set(JSON_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
    message(STATUS "Android: using bundled nlohmann/json at ${JSON_INCLUDE_DIR}")
else()
    find_path(JSON_INCLUDE_DIR
        NAMES nlohmann/json.hpp
        PATHS ${JSON_ROOT} /usr/include /usr/local/include
        DOC "nlohmann/json single-header")
    if(NOT JSON_INCLUDE_DIR)
        message(FATAL_ERROR "nlohmann/json.hpp not found. Install or set JSON_ROOT.")
    endif()
endif()

# ── 库 ──────────────────────────────────────────────────────────
# 新管线（Phase 4 架构）：剔除 tracker/kalman/旧 geometry，
# 保留纯 C++ 几何 + OpenCV 核心模块，由 geometry_cv.cpp 提供 OpenCV 依赖实现。
add_library(vbtcore STATIC
    src/calibrator.cpp
    src/kalman.cpp
    src/segmenter.cpp
    src/segmenter_v2.cpp   # v2 Physical-Aligned Trajectory Segmenter
    src/analyze.cpp
    src/detector.cpp
    src/tracker.cpp
    src/frame_analyzer.cpp
    src/geometry.cpp       # resolve_plate_diameter + fit_plate_ellipse
    src/geometry_cv.cc     # OpenCV 依赖：fitEllipse / preprocess / canvas_to_orig
)

# 向编译器传递 VBT_BUILD_ANDROID（默认 0=桌面，Android 子项目通过 -DVBT_BUILD_ANDROID=ON/1/OFF/0 设置）
if(DEFINED VBT_BUILD_ANDROID)
    # 外部已定义（Android 默认 ON）：保留原值语义
    if(VBT_BUILD_ANDROID STREQUAL "ON" OR VBT_BUILD_ANDROID STREQUAL "on" OR VBT_BUILD_ANDROID STREQUAL "1" OR VBT_BUILD_ANDROID STREQUAL "TRUE" OR VBT_BUILD_ANDROID STREQUAL "true")
        set(VBT_BUILD_ANDROID_INT 1)
    else()
        set(VBT_BUILD_ANDROID_INT 0)
    endif()
else()
    set(VBT_BUILD_ANDROID_INT 0)
endif()
target_compile_definitions(vbtcore PUBLIC VBT_BUILD_ANDROID=${VBT_BUILD_ANDROID_INT})

target_include_directories(vbtcore
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${JSON_INCLUDE_DIR}
)
target_link_libraries(vbtcore
    PUBLIC OpenCV::core OpenCV::imgproc OpenCV::video
)
# videoio / imgcodecs 仅 desktop 可用（opencv-mobile 上未提供）
if(NOT VBT_BUILD_ANDROID_FLAG)
    if(TARGET OpenCV::videoio)
        target_link_libraries(vbtcore PUBLIC OpenCV::videoio)
    endif()
    if(TARGET OpenCV::imgcodecs)
        target_link_libraries(vbtcore PUBLIC OpenCV::imgcodecs)
    endif()
endif()
# OpenCV::dnn only when OpenCV DNN is available
if(TARGET OpenCV::dnn)
    target_link_libraries(vbtcore PRIVATE OpenCV::dnn)
endif()
# ONNX Runtime: always link when found (independent of OpenCV::dnn)
if(VBT_HAS_ONNXRUNTIME)
    target_include_directories(vbtcore PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(vbtcore PRIVATE ${ONNXRUNTIME_LIB})
    target_compile_definitions(vbtcore PUBLIC VBT_HAS_ONNXRUNTIME=1)
endif()

# ── Golden Test ──────────────────────────────────────────────────
if(VBT_BUILD_TESTS)
    enable_testing()
    add_executable(golden_test tests/golden_test.cpp)
    target_link_libraries(golden_test PRIVATE vbtcore)
    target_include_directories(golden_test PRIVATE
        ${JSON_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_test(NAME golden_test COMMAND golden_test)

    add_executable(test_direct tests/test_direct.cpp)
    target_link_libraries(test_direct PRIVATE vbtcore)
    target_include_directories(test_direct PRIVATE
        ${JSON_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests)

    add_executable(seg_compare tests/seg_compare.cpp)
    target_link_libraries(seg_compare PRIVATE vbtcore)
    target_include_directories(seg_compare PRIVATE
        ${JSON_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()

# ── 部署 ──────────────────────────────────────────────────────────
install(TARGETS vbtcore
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/vbt DESTINATION include)

message(STATUS "vbtcore-cpp ${PROJECT_VERSION} configured")
message(STATUS "  OpenCV: ${OpenCV_VERSION}")
message(STATUS "  ONNX Runtime: ${VBT_HAS_ONNXRUNTIME}")
message(STATUS "  nlohmann/json: ${JSON_INCLUDE_DIR}")