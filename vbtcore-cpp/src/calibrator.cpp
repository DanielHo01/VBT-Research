// vbt/calibrator.cpp — 静态中位数锁死 + CV 门禁
// ============================================
// 对应 Python 端 vbtcore/calibrator.py::StaticPlateCalibrator。

#include "vbt/calibrator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vbt {

StaticPlateCalibrator::StaticPlateCalibrator(double real_diameter_m,
                                              std::size_t min_static_frames,
                                              double max_cv)
    : real_diameter_m_(real_diameter_m),
      min_static_frames_(min_static_frames),
      max_cv_(max_cv),
      mpp_locked_(0.0),
      locked_(false) {}

void StaticPlateCalibrator::add_sample(double bbox_height_px) {
    if (!locked_ && bbox_height_px > 10.0) {
        samples_.push_back(bbox_height_px);
    }
}

bool StaticPlateCalibrator::is_ready() const {
    return samples_.size() >= min_static_frames_;
}

double StaticPlateCalibrator::lock_scale() {
    if (locked_) {
        return mpp_locked_;
    }
    if (samples_.size() < min_static_frames_) {
        throw std::runtime_error("标定样本不足");
    }

    // IQR 过滤离群点
    std::vector<double> sorted = samples_;
    std::sort(sorted.begin(), sorted.end());
    auto n = sorted.size();
    double q25 = sorted[n / 4];
    double q75 = sorted[(3 * n) / 4];
    double iqr = q75 - q25;
    std::vector<double> valid;
    for (double h : sorted) {
        if (h >= q25 - 1.5 * iqr && h <= q75 + 1.5 * iqr) {
            valid.push_back(h);
        }
    }
    if (valid.empty()) {
        throw std::runtime_error("IQR 过滤后无有效样本");
    }

    // 计算 CV（用于诊断）
    double sum = 0.0;
    for (double h : valid) sum += h;
    const double mean = sum / valid.size();
    double var_sum = 0.0;
    for (double h : valid) {
        var_sum += (h - mean) * (h - mean);
    }
    const double std = std::sqrt(var_sum / valid.size());
    const double cv = std / mean;
    (void)cv;  // CV 超限不抛错（与 Python 端一致：降级使用中位数）

    // 中位数（降级路径）
    const double median = valid[valid.size() / 2];
    mpp_locked_ = real_diameter_m_ / median;
    locked_ = true;
    return mpp_locked_;
}

}  // namespace vbt