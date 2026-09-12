#pragma once
// vbt/detector.hpp — ONNX Runtime 检测器（C++ 实现）
// ====================================================
// 对应 Python 端 vbtcore/detector.py::PlateDetector。
// 输入：mp4 帧（cv::Mat BGR）
// 输出：Detection 列表
//
// 编译时若未链接 ONNX Runtime，则降级为 stub（返回空检测，仅供接口测试）。

#include <memory>
#include <string>
#include <vector>

namespace cv {
class Mat;
}

namespace vbt {

/// 检测结果。
struct Detection {
    double cx = 0.0;
    double cy = 0.0;
    double w = 0.0;
    double h = 0.0;
    double conf = 0.0;
    double ratio = 0.0;  ///< w/h
};

/// 检测器抽象接口（生产用 PlateDetectorImpl，测试用 StubDetector）。
class Detector {
public:
    virtual ~Detector() = default;
    /// 在 BGR 帧上跑检测。
    /// conf_thresh: 置信度门限
    /// 返回按 conf 降序排列的检测列表。
    virtual std::vector<Detection> detect(const cv::Mat& frame_bgr,
                                          double conf_thresh) = 0;
    virtual bool is_stub() const = 0;
};

/// 生产检测器（ONNX Runtime）。`vbt_HAS_ONNXRUNTIME` 宏决定是否可用。
/// providers: ["CPUExecutionProvider"] / ["CUDAExecutionProvider"]
class PlateDetector : public Detector {
public:
    PlateDetector(const std::string& model_path,
                  const std::vector<std::string>& providers = {"CPUExecutionProvider"});
    ~PlateDetector() override;

    std::vector<Detection> detect(const cv::Mat& frame_bgr,
                                  double conf_thresh) override;
    bool is_stub() const override { return false; }

private:
    struct Impl;
    Impl* impl_;
};

/// Stub 检测器（无 ONNX Runtime 时使用，返回空列表）。
class StubDetector : public Detector {
public:
    std::vector<Detection> detect(const cv::Mat& /*frame_bgr*/,
                                  double /*conf_thresh*/) override {
        return {};
    }
    bool is_stub() const override { return true; }
};

/// 工厂：自动选择 PlateDetector 或 StubDetector。
std::unique_ptr<Detector> make_detector(const std::string& model_path);

}  // namespace vbt