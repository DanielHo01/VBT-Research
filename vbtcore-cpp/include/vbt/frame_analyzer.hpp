#pragma once
// vbt/frame_analyzer.hpp — Android 帧输入分析器
// ================================================
// Kotlin 层用 MediaCodec 解码视频，通过 JNI 喂 RGBA 帧给 C++ 引擎。
// 引擎内部维护标定/跟踪/分段状态，无需 OpenCV VideoCapture。

#include <memory>
#include <string>
#include <vector>

namespace vbt {

// 前向声明（避免引入 cv::Mat 到此头文件）
struct PlateDetectionResult;

/// 帧输入分析器（内部维护标定/跟踪状态）。
class FrameAnalyzer {
public:
    /// fps: 视频帧率（用于时间戳计算）
    /// model_path: ONNX 模型路径
    /// plate_diameter_m: 外层片直径（米）
    /// exercise_type: "squat_bench" 或 "deadlift"
    /// redet_every: YOLO 重检测间隔
    /// conf_thresh: 检测置信度门限
    FrameAnalyzer(double fps,
                  const std::string& model_path,
                  double plate_diameter_m,
                  const std::string& exercise_type,
                  int redet_every,
                  double conf_thresh);
    ~FrameAnalyzer();

    // 禁用拷贝
    FrameAnalyzer(const FrameAnalyzer&) = delete;
    FrameAnalyzer& operator=(const FrameAnalyzer&) = delete;

    /// 喂入一帧 RGBA 数据（来自 Android MediaCodec）。
    /// rgba_data: RGBA 像素数据（4 字节/像素）
    /// width, height: 帧尺寸
    /// stride_bytes: 每行字节数（通常 width*4）
    /// frame_idx: 当前帧序号（从 0 开始）
    /// timestamp_ms: 当前帧时间戳（毫秒）
    void add_frame(const uint8_t* rgba_data,
                   int width, int height, int stride_bytes,
                   int frame_idx, double timestamp_ms);

    /// 检查是否已完成标定。
    bool is_calibrated() const;

    /// 获取当前比例尺（米/像素），标定完成前返回 0。
    double get_mpp() const;

    /// 获取已处理帧数。
    int frame_count() const;

    /// 完成分析，返回 JSON 结果。
    /// 调用此方法后对象不可再用。
    std::string finalize();

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace vbt
