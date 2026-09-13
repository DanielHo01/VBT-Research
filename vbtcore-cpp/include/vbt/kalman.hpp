#pragma once
// vbt/kalman.hpp — 1D 匀加速卡尔曼轨迹滤波器
// ============================================
// 对应 Python 端 vbtcore/kalman.py::BarbellKalmanTracker。
// 状态向量: [y, v_y, a_y]^T（牛顿匀加速模型）
// 观测向量: [y]（每 N 帧 YOLO+椭圆圆心提供一次）

#include <array>

namespace vbt {

/// BarbellKalmanTracker —— 1D 卡尔曼（Newton 匀加速）。
/// 与 Python 端 vbtcore.kalman.BarbellKalmanTracker 行为对齐：
///   - initial_y: 初始位置（px 或 m，依调用方约定）
///   - dt: 时间步长（秒）
///   - Q: 过程噪声（位置噪声，0.01 默认）
///   - R: 观测噪声（4.0 默认）
class BarbellKalmanTracker {
public:
    BarbellKalmanTracker(double initial_y = 0.0,
                         double dt = 1.0 / 120.0,
                         double Q = 0.01,
                         double R = 4.0);

    /// 每帧调用：推进状态。
    /// 返回 (y_pred, v_pred)。
    std::array<double, 2> predict();

    /// 检测到目标时调用：用观测 y 更新状态。
    /// 返回 (y_filt, v_filt)。
    std::array<double, 2> update(double observed_y);

    double y() const { return x_[0]; }
    double v() const { return x_[1]; }
    double a() const { return x_[2]; }

private:
    double dt_;
    double Q_;
    double R_;
    std::array<std::array<double, 3>, 3> F_;  ///< 状态转移矩阵
    std::array<double, 3> H_;                 ///< 观测矩阵（[1, 0, 0]）
    std::array<double, 3> x_;                 ///< 状态向量
    std::array<std::array<double, 3>, 3> P_;  ///< 协方差矩阵
    std::array<std::array<double, 3>, 3> Qm_; ///< 过程噪声协方差
    double Rv_;                               ///< 观测噪声方差
};

}  // namespace vbt