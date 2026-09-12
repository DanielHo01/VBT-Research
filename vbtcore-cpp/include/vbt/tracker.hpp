#pragma once
// vbt/tracker.hpp — 密集视觉跟踪器（LK 光流 + 卡尔曼融合）
// ========================================================
// 对应 Python 端 vbtcore/tracker.py::DenseVisualTracker +
// vbtcore/tracker.py::KinematicKalmanTracker。
//
// 工作流：
//   1. 锚定期：每 N 帧跑 YOLO + 物理先验，标定 mpp 后启动 tracker
//   2. 跟踪期：
//      - 关键帧（每 redet_every 帧）：YOLO + 卡尔曼 update
//      - 中间帧：LK 光流 + 卡尔曼 predict
//   3. 输出：每帧的位置（米，向上为正）和速度（米/秒）

#include <array>
#include <string>
#include <vector>

namespace cv {
class Mat;
}

namespace vbt {

struct Detection;

/// 跟踪器诊断（每帧状态）。
struct TrackStep {
    double y_m = 0.0;      ///< 位置（米，向上为正）
    double v_mps = 0.0;    ///< 速度（米/秒，向上为正）
    double confidence = 0.0; ///< 0-1，可信度
};

/// 标定后的跟踪器。
/// 调用流程：
///   - construct(initial_y, initial_v, mpp, bbox_xyxy, gray)
///   - 每帧 step(gray, current_time_s, optional_detection) → TrackStep
class DenseVisualTracker {
public:
    /// initial_bbox: (x1, y1, x2, y2) 原图坐标
    DenseVisualTracker(double mpp,
                       std::array<double, 4> initial_bbox,
                       const cv::Mat& initial_gray,
                       double initial_time_s,
                       double dt = 1.0 / 30.0);
    ~DenseVisualTracker();

    /// 关键帧：调用 YOLO 检测 + 卡尔曼 update。
    TrackStep step_keyframe(const cv::Mat& gray,
                            std::array<double, 4> bbox_xyxy,
                            double current_time_s);

    /// 中间帧：LK 光流 + 卡尔曼 predict。
    TrackStep step_interframe(const cv::Mat& gray, double current_time_s);

private:
    struct Impl;
    Impl* impl_;
};

/// 1D 物理空间卡尔曼（无 LK 光流，对应 vbtcore.kalman.BarbellKalmanTracker）。
class KinematicKalmanTracker {
public:
    KinematicKalmanTracker(double initial_y,
                           double dt = 1.0 / 30.0,
                           double Q = 0.01,
                           double R = 4.0);

    std::array<double, 2> predict();
    std::array<double, 2> update(double observed_y);

    double y() const;
    double v() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace vbt