// vbt/kalman.cpp — 1D 匀加速卡尔曼
// =================================
// 对应 Python 端 vbtcore/kalman.py::BarbellKalmanTracker。

#include "vbt/kalman.hpp"

#include <cmath>
#include <stdexcept>

namespace vbt {

BarbellKalmanTracker::BarbellKalmanTracker(double initial_y,
                                           double dt,
                                           double Q,
                                           double R)
    : dt_(dt), Q_(Q), R_(R), Rv_(R) {
    if (dt <= 0.0) {
        throw std::invalid_argument("BarbellKalmanTracker: dt must be > 0");
    }
    // 状态向量 [y, v, a]
    x_ = {initial_y, 0.0, 0.0};

    // 状态转移矩阵 F（牛顿匀加速）
    const double dt2_2 = 0.5 * dt * dt;
    F_ = {{{1.0, dt, dt2_2},
           {0.0, 1.0, dt},
           {0.0, 0.0, 1.0}}};

    // 观测矩阵 H = [1, 0, 0]
    H_ = {1.0, 0.0, 0.0};

    // 过程噪声协方差（位置分量）
    Qm_ = {{{Q_, 0.0, 0.0},
            {0.0, Q_, 0.0},
            {0.0, 0.0, Q_}}};

    // 初始协方差（较大不确定）
    P_ = {{{10.0, 0.0, 0.0},
           {0.0, 10.0, 0.0},
           {0.0, 0.0, 10.0}}};
}

std::array<double, 2> BarbellKalmanTracker::predict() {
    // x' = F @ x
    std::array<double, 3> x_new = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            x_new[i] += F_[i][j] * x_[j];
        }
    }
    x_ = x_new;

    // P' = F @ P @ F^T + Q
    std::array<std::array<double, 3>, 3> FP = {{{0.0}}};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                FP[i][j] += F_[i][k] * P_[k][j];
            }
        }
    }
    std::array<std::array<double, 3>, 3> FPFt = {{{0.0}}};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                FPFt[i][j] += FP[i][k] * F_[j][k];
            }
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            P_[i][j] = FPFt[i][j] + Qm_[i][j];
        }
    }

    return {x_[0], x_[1]};
}

std::array<double, 2> BarbellKalmanTracker::update(double observed_y) {
    // 卡尔曼增益 K = P @ H^T / (H @ P @ H^T + R)
    double PHt[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) {
        PHt[i] = P_[i][0] * H_[0];
    }
    double HPHR = R_;
    for (int i = 0; i < 3; ++i) {
        HPHR += H_[0] * P_[0][i] * H_[i];
    }
    if (HPHR < 1e-12) {
        HPHR = 1e-12;  // avoid divide by zero
    }
    double K[3];
    for (int i = 0; i < 3; ++i) {
        K[i] = PHt[i] / HPHR;
    }

    // 创新 innovation = z - H @ x
    double hx = 0.0;
    for (int i = 0; i < 3; ++i) {
        hx += H_[i] * x_[i];
    }
    const double innov = observed_y - hx;

    // x = x + K * innov
    for (int i = 0; i < 3; ++i) {
        x_[i] += K[i] * innov;
    }

    // P = (I - K @ H) @ P
    std::array<std::array<double, 3>, 3> newP = {{{0.0}}};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double kH_ij = K[i] * H_[j];
            double I_kH = (i == j ? 1.0 : 0.0) - kH_ij;
            for (int k = 0; k < 3; ++k) {
                newP[i][j] += I_kH * P_[i][k];
            }
        }
    }
    P_ = newP;

    return {x_[0], x_[1]};
}

}  // namespace vbt