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
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs videoio)

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
if(NOT TARGET OpenCV::dnn)
    add_library(OpenCV::dnn INTERFACE IMPORTED)
    set_target_properties(OpenCV::dnn PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}")
    if(OpenCV_DNN_LIBRARIES)
        set_target_properties(OpenCV::dnn PROPERTIES
            INTERFACE_LINK_LIBRARIES "${OpenCV_DNN_LIBRARIES}")
    endif()
endif()

# ONNX Runtime C++ API（必须预编译 NDK 版本）
# ONNX Runtime 安装在 ${ONNXRUNTIME_ROOT}/onnxruntime/{include,lib}
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS ${ONNXRUNTIME_ROOT}/onnxruntime/include ${ANDROID_NDK}/../onnxruntime/include
    DOC "ONNX Runtime C++ API header")
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_ROOT}/onnxruntime/lib ${ANDROID_NDK}/../onnxruntime/lib
    DOC "ONNX Runtime shared library")

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(WARNING "ONNX Runtime not found. Build ONNX-less demo with stub detector.")
    set(VBT_HAS_ONNXRUNTIME OFF)
else()
    set(VBT_HAS_ONNXRUNTIME ON)
endif()

# nlohmann/json (header-only)
find_path(JSON_INCLUDE_DIR
    NAMES nlohmann/json.hpp
    PATHS ${JSON_ROOT} /usr/include /usr/local/include
    DOC "nlohmann/json single-header")
if(NOT JSON_INCLUDE_DIR)
    message(FATAL_ERROR "nlohmann/json.hpp not found. Install or set JSON_ROOT.")
endif()

# ── 库 ──────────────────────────────────────────────────────────
add_library(vbtcore STATIC
    src/geometry.cpp
    src/kalman.cpp
    src/segmenter.cpp
    src/tracker.cpp
    src/calibrator.cpp
    src/analyze.cpp
    src/detector.cpp
)
target_include_directories(vbtcore
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${JSON_INCLUDE_DIR}
)
target_link_libraries(vbtcore
    PUBLIC OpenCV::core OpenCV::imgproc OpenCV::dnn
)
if(VBT_HAS_ONNXRUNTIME)
    target_include_directories(vbtcore PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(vbtcore PRIVATE ${ONNXRUNTIME_LIB})
    target_compile_definitions(vbtcore PRIVATE VBT_HAS_ONNXRUNTIME=1)
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