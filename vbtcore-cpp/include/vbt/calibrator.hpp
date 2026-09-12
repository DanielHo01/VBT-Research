#pragma once
// vbt/calibrator.hpp — 静态中位数锁死 + CV 门禁
// ===============================================
// 对应 Python 端 vbtcore/calibrator.py::StaticPlateCalibrator。

#include <cstddef>
#include <vector>

namespace vbt {

/// 静态中位数锁死 + CV 变异系数门禁标定器。
/// 在起始静止期采样检测框高度，完成门禁核验后永久冻结 mpp。
class StaticPlateCalibrator {
public:
    /// real_diameter_m : 杠铃片标准物理外径（默认 450 mm）
    /// min_static_frames : 计算标定所需的最小静止有效帧数
    /// max_cv : 变异系数门禁阈值；超过则降级使用中位数（不抛错）
    StaticPlateCalibrator(double real_diameter_m = 0.45,
                          std::size_t min_static_frames = 20,
                          double max_cv = 0.015);

    /// 仅在杠铃静止期输入每帧检测到的边界框高度（像素）。
    /// 锁死后调用无效（仅保留返回值中的 mpp_locked）。
    void add_sample(double bbox_height_px);

    /// 是否已采集足够样本（≥ min_static_frames）。
    bool is_ready() const;

    /// 完成门禁核验并永久冻结 mpp。返回米/像素。
    double lock_scale();

    double mpp_locked() const { return mpp_locked_; }

private:
    double real_diameter_m_;
    std::size_t min_static_frames_;
    double max_cv_;
    std::vector<double> samples_;
    double mpp_locked_;
    bool locked_;
};

}  // namespace vbt