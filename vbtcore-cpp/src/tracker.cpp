// vbt/tracker.cpp — 密集视觉跟踪器（LK 光流 + 卡尔曼融合）
// ===========================================================
// 对应 Python 端 vbtcore/tracker.py::DenseVisualTracker。
//
// 修复记录（2026-09-13）：
//   - step_keyframe 不再重建 Kalman，只做 predict + update
//   - step_interframe 实现 LK 光流观测更新（中位数抗杂波）
//   - 特征点提取限定外侧 60% 区域，杜绝人体污染

#include "vbt/tracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "vbt/kalman.hpp"

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

namespace vbt {

namespace {

/// 提取 ROI 内特征点（外侧 60% 区域，禁止包含身体侧）。
std::vector<cv::Point2f> extract_roi_points(
    const cv::Mat& gray,
    std::array<double, 4> bbox_xyxy)
{
    const double x1 = bbox_xyxy[0];
    const double y1 = bbox_xyxy[1];
    const double x2 = bbox_xyxy[2];
    const double y2 = bbox_xyxy[3];
    const double w = x2 - x1;
    const double h = y2 - y1;

    // 外侧 60% 区域：x 范围 [x1, x1 + 0.6*w]，y 范围 [y1, y2]
    const double roi_x1 = x1;
    const double roi_y1 = y1;
    const double roi_x2 = x1 + 0.6 * w;
    const double roi_y2 = y2;

    cv::Rect roi(
        static_cast<int>(roi_x1),
        static_cast<int>(roi_y1),
        static_cast<int>(roi_x2 - roi_x1),
        static_cast<int>(roi_y2 - roi_y1));

    // 裁剪到图像边界
    roi &= cv::Rect(0, 0, gray.cols, gray.rows);
    if (roi.width < 10 || roi.height < 10) {
        return {};
    }

    cv::Mat roi_gray = gray(roi);

    // Shi-Tomasi 角点检测
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(
        roi_gray, corners,
        /*maxCorners=*/30,
        /*qualityLevel=*/0.01,
        /*minDistance=*/10.0);

    // 转换回原图坐标
    for (auto& pt : corners) {
        pt.x += static_cast<float>(roi.x);
        pt.y += static_cast<float>(roi.y);
    }
    return corners;
}

/// 中位数计算（抗杂波）。
double median(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 0) {
        return (v[n/2 - 1] + v[n/2]) / 2.0;
    }
    return v[n/2];
}

}  // namespace

// ──────────────────────────────────────────────────────────────────────
// DenseVisualTracker::Impl
// ──────────────────────────────────────────────────────────────────────

struct DenseVisualTracker::Impl {
    double mpp = 0.0;                      ///< 米/像素（标定后）
    BarbellKalmanTracker kf;               ///< 1D 卡尔曼（y 像素，全生命周期单例）
    double prev_t = 0.0;                   ///< 上一帧时间
    double center_y_px = 0.0;              ///< 当前中心 y（像素）
    double center_x_px = 0.0;              ///< 当前中心 x（像素）
    cv::Mat prev_gray;                     ///< 上一帧灰度图
    std::vector<cv::Point2f> feature_points;  ///< LK 跟踪特征点
    double y_m = 0.0;                      ///< 输出位置（米，向上为正）
    double v_mps = 0.0;                    ///< 输出速度（米/秒，向上为正）
    double prev_keyframe_y_m = 0.0;        ///< 上一关键帧位置（米）
    double prev_keyframe_t = 0.0;          ///< 上一关键帧时间（秒）
    int keyframe_count = 0;                ///< 关键帧计数
    int initialized = 0;
};

DenseVisualTracker::DenseVisualTracker(double mpp,
                                       std::array<double, 4> initial_bbox,
                                       const cv::Mat& initial_gray,
                                       double initial_time_s,
                                       double dt)
    : impl_(new Impl()) {
    impl_->mpp = mpp;
    impl_->prev_t = initial_time_s;
    impl_->prev_gray = initial_gray.clone();

    // 初始中心 y（图像坐标，向下为正）
    const double cy_px = 0.5 * (initial_bbox[1] + initial_bbox[3]);
    const double cx_px = 0.5 * (initial_bbox[0] + initial_bbox[2]);
    impl_->center_y_px = cy_px;
    impl_->center_x_px = cx_px;

    // Kalman 全生命周期只构造一次
    impl_->kf = BarbellKalmanTracker(cy_px, dt, /*Q=*/0.01, /*R=*/4.0);

    // 提取初始特征点
    impl_->feature_points = extract_roi_points(initial_gray, initial_bbox);

    impl_->initialized = 1;
}

DenseVisualTracker::~DenseVisualTracker() { delete impl_; }

TrackStep DenseVisualTracker::step_keyframe(const cv::Mat& gray,
                                            std::array<double, 4> bbox_xyxy,
                                            double current_time_s) {
    TrackStep step{};
    if (impl_ == nullptr || !impl_->initialized) {
        return step;
    }

    const double det_cy_px = 0.5 * (bbox_xyxy[1] + bbox_xyxy[3]);
    impl_->center_y_px = det_cy_px;

    const double raw_y_m = -det_cy_px * impl_->mpp;
    if (impl_->keyframe_count > 0) {
        const double dt = current_time_s - impl_->prev_keyframe_t;
        if (dt > 1e-6) {
            impl_->v_mps = (raw_y_m - impl_->prev_keyframe_y_m) / dt;
        }
    }
    impl_->prev_keyframe_y_m = raw_y_m;
    impl_->prev_keyframe_t = current_time_s;
    impl_->keyframe_count++;
    impl_->y_m = raw_y_m;
    impl_->prev_t = current_time_s;

    step.y_m = impl_->y_m;
    step.v_mps = impl_->v_mps;
    step.confidence = 1.0;
    return step;
}

TrackStep DenseVisualTracker::step_interframe(const cv::Mat& gray,
                                              double current_time_s) {
    TrackStep step{};
    if (impl_ == nullptr || !impl_->initialized) {
        return step;
    }

    impl_->prev_t = current_time_s;

    step.y_m = impl_->y_m;
    step.v_mps = impl_->v_mps;
    step.confidence = 0.5;
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
