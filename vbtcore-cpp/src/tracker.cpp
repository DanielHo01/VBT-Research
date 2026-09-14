// vbt/tracker.cpp — 密集视觉跟踪器（LK 光流 + 卡尔曼融合）
// ===========================================================
// 对应 Python 端 vbtcore/tracker.py::DenseVisualTracker。
//
// 修复记录（2026-09-13）：
//   - step_keyframe 不再重建 Kalman，只做 predict + update
//   - step_interframe 实现 LK 光流观测更新（中位数抗杂波）
//   - 特征点提取参数与 Python 端严格对齐
//
// 【2026-09-13 重写 · 与 Python 端严格对齐】
// ------------------------------------------------------------------
// 此前 C++ 实现与 Python 是两套不同架构，导致 golden_test 无法对齐：
//   · 旧 C++ step_interframe 完全不做光流，只把上一帧的 y/v 原样返回
//     （速度呈阶梯保持），且 step_keyframe 用两关键帧差分算速度，
//     Kalman 构造后从未参与输出；
//   · Python 则是「物理空间 CA 卡尔曼（状态 [y, v, a]，单位 米/秒）
//     + 每帧 LK 光流位移观测 + 关键帧 YOLO 绝对位置校正」。
// 现按 Python 语义重写：
//   · PhysicalKalman：状态 [y(m), v(m/s), a(m/s²)]，F 由真实 PTS dt 构造，
//     Q = G·Gᵀ·q_a（q_a=25.0），R = 0.0004（2cm 观测噪声），
//     P0 = diag(0.001, 0.1, 1.0) —— 与 vbtcore/tracker.py 数值逐项一致；
//   · 特征点：mask 为 bbox 外侧 70%、上下各内缩 5%，
//     goodFeaturesToTrack(maxCorners=25, qualityLevel=0.02, minDistance=6)；
//   · LK：winSize=(21,21), maxLevel=3，≥5 个跟踪成功点才接受，
//     取 dy 中位数累加到 center_y_px 后喂给 Kalman update。
// 注意符号约定：Python 内部 Kalman 跑在「图像 y（向下为正）」的米制空间，
// 由 pipeline 统一取反输出；C++ 在 analyze.cpp 直接输出，故此处在返回时取反，
// 保证对外语义一致（向上为正）。

#include "vbt/tracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "vbt/kalman.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace vbt {

namespace {

/// 提取 ROI 内特征点 —— 与 Python `_extract_roi_points` 严格一致。
/// mask：bbox 外侧 70%（x: [x1, x1+0.70w]），上下各内缩 5%（y: [y1+0.05h, y2-0.05h]），
/// 避开内侧手腕/杠铃杆区域，防止人体干扰。
/// goodFeaturesToTrack 在**全图 + mask** 上跑（与 Python 一致），
/// 而非在裁剪子图上跑 —— 后者会改变角点响应的归一化基准。
std::vector<cv::Point2f> extract_roi_points(
    const cv::Mat& gray,
    std::array<double, 4> bbox_xyxy)
{
    const int x1 = static_cast<int>(bbox_xyxy[0]);
    const int y1 = static_cast<int>(bbox_xyxy[1]);
    const int x2 = static_cast<int>(bbox_xyxy[2]);
    const int y2 = static_cast<int>(bbox_xyxy[3]);
    const int bw = x2 - x1;
    const int bh = y2 - y1;

    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
    const int sx1 = x1;
    const int sx2 = static_cast<int>(x1 + bw * 0.70);
    const int sy1 = static_cast<int>(y1 + bh * 0.05);
    const int sy2 = static_cast<int>(y2 - bh * 0.05);
    if (sx2 > sx1 && sy2 > sy1) {
        cv::Rect r(sx1, sy1, sx2 - sx1, sy2 - sy1);
        r &= cv::Rect(0, 0, gray.cols, gray.rows);
        if (r.width > 0 && r.height > 0) {
            mask(r).setTo(255);
        }
    }

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(
        gray, corners,
        /*maxCorners=*/25,
        /*qualityLevel=*/0.02,
        /*minDistance=*/6.0,
        mask);
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

/// 物理空间恒加速度（CA）卡尔曼 —— 对应 Python KinematicKalmanTracker。
/// 状态 x = [y(米), v(米/秒), a(米/秒²)]ᵀ，dt 来自真实 PTS（可变帧率安全）。
class PhysicalKalman {
public:
    PhysicalKalman() = default;
    PhysicalKalman(double initial_y_m, double initial_time_s)
        : last_t_(initial_time_s) {
        x_ = {initial_y_m, 0.0, 0.0};
        // P0 = diag(0.001, 0.1, 1.0)
        for (auto& row : P_) row.fill(0.0);
        P_[0][0] = 0.001; P_[1][1] = 0.1; P_[2][2] = 1.0;
    }

    /// 预测步（时间更新）。返回 (y, v)。
    std::array<double, 2> predict(double current_time_s) {
        double dt = current_time_s - last_t_;
        if (dt <= 0.0) dt = 1.0 / 60.0;   // 与 Python 一致的退化保护
        last_t_ = current_time_s;

        const double dt2 = 0.5 * dt * dt;
        // F（牛顿运动学）
        const double F[3][3] = {
            {1.0, dt,  dt2},
            {0.0, 1.0, dt },
            {0.0, 0.0, 1.0}
        };
        // x = F x
        std::array<double, 3> nx{};
        for (int i = 0; i < 3; ++i) {
            double acc = 0.0;
            for (int j = 0; j < 3; ++j) acc += F[i][j] * x_[j];
            nx[i] = acc;
        }
        x_ = nx;

        // Q = (G Gᵀ) q_a,  G = [0.5dt², dt, 1]ᵀ
        const double G[3] = {dt2, dt, 1.0};
        double FP[3][3]{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double acc = 0.0;
                for (int k = 0; k < 3; ++k) acc += F[i][k] * P_[k][j];
                FP[i][j] = acc;
            }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double acc = 0.0;
                for (int k = 0; k < 3; ++k) acc += FP[i][k] * F[j][k];  // (FP)Fᵀ
                P_[i][j] = acc + G[i] * G[j] * kQa;
            }
        return {x_[0], x_[1]};
    }

    /// 位置观测更新（H = [1,0,0]，R = 0.0004 = (2cm)²）。返回 (y, v)。
    std::array<double, 2> update_position(double observed_y_m) {
        const double S = P_[0][0] + kR;
        if (S <= 0.0) return {x_[0], x_[1]};
        const double resid = observed_y_m - x_[0];
        // K = P Hᵀ / S  → 取 P 第 0 列
        const double K[3] = {P_[0][0] / S, P_[1][0] / S, P_[2][0] / S};
        for (int i = 0; i < 3; ++i) x_[i] += K[i] * resid;
        // P = (I - K H) P，H 只作用在第 0 行
        double NP[3][3]{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                NP[i][j] = P_[i][j] - K[i] * P_[0][j];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) P_[i][j] = NP[i][j];
        return {x_[0], x_[1]};
    }

    double y() const { return x_[0]; }
    double v() const { return x_[1]; }

private:
    static constexpr double kQa = 25.0;     ///< 过程噪声（~5 m/s² 冲击）
    static constexpr double kR  = 0.0004;   ///< 观测噪声 (2cm)²
    std::array<double, 3> x_{};
    std::array<std::array<double, 3>, 3> P_{};
    double last_t_ = 0.0;
};

struct DenseVisualTracker::Impl {
    double mpp = 0.0;                         ///< 米/像素（标定后冻结）
    PhysicalKalman kf;                        ///< 物理空间 CA 卡尔曼
    double center_y_px = 0.0;                 ///< 当前中心 y（图像坐标，向下为正）
    double center_x_px = 0.0;
    cv::Mat prev_gray;
    std::vector<cv::Point2f> feature_points;
    int initialized = 0;
};

DenseVisualTracker::DenseVisualTracker(double mpp,
                                       std::array<double, 4> initial_bbox,
                                       const cv::Mat& initial_gray,
                                       double initial_time_s,
                                       double dt)
    : impl_(new Impl()) {
    (void)dt;  // dt 由真实 PTS 决定，构造参数保留仅为 ABI 兼容
    impl_->mpp = mpp;
    impl_->prev_gray = initial_gray.clone();

    impl_->center_y_px = 0.5 * (initial_bbox[1] + initial_bbox[3]);
    impl_->center_x_px = 0.5 * (initial_bbox[0] + initial_bbox[2]);

    // Kalman 初始化在「图像 y 米制」空间（与 Python 一致），对外输出时取反
    impl_->kf = PhysicalKalman(impl_->center_y_px * mpp, initial_time_s);
    impl_->feature_points = extract_roi_points(initial_gray, initial_bbox);
    impl_->initialized = 1;
}

DenseVisualTracker::~DenseVisualTracker() { delete impl_; }

TrackStep DenseVisualTracker::step_keyframe(const cv::Mat& gray,
                                            std::array<double, 4> bbox_xyxy,
                                            double current_time_s) {
    TrackStep step{};
    if (impl_ == nullptr || !impl_->initialized) return step;

    // 与 Python 一致：先 predict，再用 YOLO 绝对位置 update
    impl_->kf.predict(current_time_s);

    const double det_cy_px = 0.5 * (bbox_xyxy[1] + bbox_xyxy[3]);
    const double det_cx_px = 0.5 * (bbox_xyxy[0] + bbox_xyxy[2]);
    impl_->kf.update_position(det_cy_px * impl_->mpp);

    impl_->center_y_px = det_cy_px;
    impl_->center_x_px = det_cx_px;
    impl_->prev_gray = gray.clone();
    impl_->feature_points = extract_roi_points(gray, bbox_xyxy);

    // 图像 y 向下为正 → 取反为物理向上为正
    step.y_m = -impl_->kf.y();
    step.v_mps = -impl_->kf.v();
    step.confidence = 1.0;
    return step;
}

TrackStep DenseVisualTracker::step_interframe(const cv::Mat& gray,
                                              double current_time_s) {
    TrackStep step{};
    if (impl_ == nullptr || !impl_->initialized) return step;

    impl_->kf.predict(current_time_s);

    if (impl_->feature_points.size() >= 5 && !impl_->prev_gray.empty()) {
        std::vector<cv::Point2f> next_pts;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(
            impl_->prev_gray, gray,
            impl_->feature_points, next_pts,
            status, err,
            cv::Size(21, 21), 3);

        std::vector<double> dys;
        std::vector<cv::Point2f> good_new;
        dys.reserve(next_pts.size());
        good_new.reserve(next_pts.size());
        for (std::size_t i = 0; i < next_pts.size() && i < status.size(); ++i) {
            if (status[i]) {
                dys.push_back(static_cast<double>(next_pts[i].y) -
                              static_cast<double>(impl_->feature_points[i].y));
                good_new.push_back(next_pts[i]);
            }
        }
        if (good_new.size() >= 5) {
            const double dy_px = median(dys);
            impl_->center_y_px += dy_px;
            impl_->kf.update_position(impl_->center_y_px * impl_->mpp);
            impl_->feature_points = good_new;
        }
    }

    impl_->prev_gray = gray.clone();

    step.y_m = -impl_->kf.y();
    step.v_mps = -impl_->kf.v();
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
