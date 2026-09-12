#pragma once
// vbt/segmenter.hpp — 运动学分段器（速度 FSM）
// ===============================================
// 对应 Python 端 vbtcore/segmenter.py::BiomechanicalRepSegmenter。
//
// 输入：timestamps(秒)、positions(米, 向上为正)、velocities(米/秒, 向上为正)
// 输出：list[Rep]

#include <cstdint>
#include <string>
#include <vector>

namespace vbt {

/// 单个 rep 结果（与 Python 端 segmenter.Rep 对齐）。
struct Rep {
    int start_idx = 0;        ///< 起始帧（闭区间）
    int end_idx = 0;          ///< 结束帧
    double start_time = 0.0;  ///< 起始时间（秒）
    double end_time = 0.0;    ///< 结束时间（秒）
    double duration_s = 0.0;  ///< 向心段时间
    double rom_m = 0.0;       ///< 位移幅值（米）
    double mcv_mps = 0.0;     ///< 向心段平均速度（主指标 = ROM / duration）
    double pcv_mps = 0.0;     ///< 向心段峰值速度
};

/// 运动学分段器（速度 FSM）。
///
/// exercise_type:
///   - "squat_bench"：SSC 模式（深蹲、卧推）
///       IDLE → ECCENTRIC → (底部换向) → CONCENTRIC → TOP → IDLE
///   - "deadlift"：无 SSC（直接向心）
///       IDLE → CONCENTRIC → (速度归零) → IDLE
///
/// 速度约定：输入 velocities 必须 向上为正（+），向下为负（-）
/// （即图像 y 取反后的物理坐标）。
class BiomechanicalRepSegmenter {
public:
    BiomechanicalRepSegmenter(
        const std::string& exercise_type = "squat_bench",
        double min_rom_m = 0.12,
        double min_dur_s = 0.20,
        double v_thresh_start = 0.08,
        double v_zero_band = 0.02);

    /// 主入口。t_arr/y_arr/v_arr 长度必须一致。
    /// 输入均为物理量（米、秒）。
    std::vector<Rep> segment(
        const double* t_arr, const double* y_arr, const double* v_arr,
        std::size_t n);

private:
    std::string exercise_type_;
    double min_rom_m_;
    double min_dur_s_;
    double v_thresh_;
    double v_band_;

    /// 内部：构造并追加 Rep（带物理门禁）。
    void validate_and_append(std::vector<Rep>& reps,
                             std::size_t start_idx, std::size_t end_idx,
                             const double* t_arr, const double* y_arr,
                             const double* v_arr) const;
};

}  // namespace vbt