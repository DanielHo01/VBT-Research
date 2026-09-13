// vbt/tracker.cpp — 密集视觉跟踪器（LK 光流 + 卡尔曼融合）
// ===========================================================
// 对应 Python 端 vbtcore/tracker.py::DenseVisualTracker +
// vbtcore/tracker.py::KinematicKalmanTracker（vbtcore.kalman.BarbellKalmanTracker）。
//
// 在没有完整 OpenCV + LK 光流的环境（stub）下，使用简化实现：
//   - LK 光流降级为 box 中心恒速度预测
//   - 卡尔曼使用 vbt::BarbellKalmanTracker（vbt/kalman.hpp）
// 生产构建（NDK + 完整 OpenCV）会替换为 cv::calcOpticalFlowPyrLK。

#include "vbt/tracker.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

#include "vbt/kalman.hpp"

#include <opencv2/core.hpp>

namespace vbt {

// ──────────────────────────────────────────────────────────────────────
// DenseVisualTracker::Impl
// ──────────────────────────────────────────────────────────────────────

struct DenseVisualTracker::Impl {
    double mpp = 0.0;          ///< 米/像素（标定后）
    BarbellKalmanTracker kf;   ///< 1D 卡尔曼（y 像素）
    double prev_y = 0.0;       ///< 上一帧位置（像素）
    double prev_t = 0.0;       ///< 上一帧时间
    double dt = 1.0 / 30.0;    ///< 帧间间隔
    std::array<double, 4> bbox = {0.0, 0.0, 0.0, 0.0};  ///< 当前框 (x1,y1,x2,y2)
    double y_m = 0.0;          ///< 输出位置（米，向上为正）
    double v_mps = 0.0;        ///< 输出速度（米/秒，向上为正）
    int initialized = 0;
};

DenseVisualTracker::DenseVisualTracker(double mpp,
                                       std::array<double, 4> initial_bbox,
                                       const cv::Mat& initial_gray,
                                       double initial_time_s,
                                       double dt)
    : impl_(new Impl()) {
    impl_->mpp = mpp;
    impl_->dt = dt;
    impl_->bbox = initial_bbox;
    // 初始 y = 框中心（图像 y 向下为正）
    const double cy_px = 0.5 * (initial_bbox[1] + initial_bbox[3]);
    impl_->prev_y = cy_px;
    impl_->prev_t = initial_time_s;
    impl_->kf = BarbellKalmanTracker(cy_px, dt, /*Q=*/0.01, /*R=*/4.0);
    impl_->initialized = 1;
    (void)initial_gray;
}

DenseVisualTracker::~DenseVisualTracker() { delete impl_; }

TrackStep DenseVisualTracker::step_keyframe(const cv::Mat& gray,
                                            std::array<double, 4> bbox_xyxy,
                                            double current_time_s) {
    TrackStep step{};
    if (impl_ == nullptr) {
        return step;
    }
    impl_->bbox = bbox_xyxy;
    const double cy_px = 0.5 * (bbox_xyxy[1] + bbox_xyxy[3]);

    // 时间步进卡尔曼
    if (impl_->initialized && current_time_s > impl_->prev_t) {
        const double frame_dt = current_time_s - impl_->prev_t;
        // 用当前 dt 临时重置（生产代码应支持可变 dt）
        BarbellKalmanTracker local(impl_->prev_y, frame_dt, 0.01, 4.0);
        local.predict();
        const auto upd = local.update(cy_px);
        impl_->y_m = -upd[0] * impl_->mpp;       // 图像 y 向下 → 取反为向上
        impl_->v_mps = -upd[1] * impl_->mpp;     // 同上
        impl_->prev_y = cy_px;
        impl_->prev_t = current_time_s;
    } else {
        impl_->y_m = -cy_px * impl_->mpp;
        impl_->v_mps = 0.0;
    }
    step.y_m = impl_->y_m;
    step.v_mps = impl_->v_mps;
    step.confidence = 1.0;
    (void)gray;
    return step;
}

TrackStep DenseVisualTracker::step_interframe(const cv::Mat& gray,
                                               double current_time_s) {
    TrackStep step{};
    if (impl_ == nullptr || !impl_->initialized) {
        return step;
    }
    // 简化：LK 光流降级为恒速预测（生产代码用 cv::calcOpticalFlowPyrLK）
    if (current_time_s > impl_->prev_t) {
        const auto pred = impl_->kf.predict();
        const double cy_px = pred[0];
        impl_->y_m = -cy_px * impl_->mpp;
        impl_->v_mps = -pred[1] * impl_->mpp;
        impl_->prev_t = current_time_s;
    }
    step.y_m = impl_->y_m;
    step.v_mps = impl_->v_mps;
    step.confidence = 0.7;
    (void)gray;
    return step;
}

// ──────────────────────────────────────────────────────────────────────
// KinematicKalmanTracker (无 LK 光流，纯 1D 卡尔曼)
// ──────────────────────────────────────────────────────────────────────

struct KinematicKalmanTracker::Impl {
    BarbellKalmanTracker kf;
};

KinematicKalmanTracker::KinematicKalmanTracker(double initial_y,
                                               double dt, double Q, double R)
    : impl_(new Impl()) {
    impl_->kf = BarbellKalmanTracker(initial_y, dt, Q, R);
}

std::array<double, 2> KinematicKalmanTracker::predict() {
    return impl_->kf.predict();
}

std::array<double, 2> KinematicKalmanTracker::update(double observed_y) {
    return impl_->kf.update(observed_y);
}

double KinematicKalmanTracker::y() const { return impl_->kf.y(); }
double KinematicKalmanTracker::v() const { return impl_->kf.v(); }

}  // namespace vbt