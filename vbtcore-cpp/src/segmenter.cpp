// vbt/segmenter.cpp — 运动学 FSM 分段器
// ======================================
// 对应 Python 端 vbtcore/segmenter.py::BiomechanicalRepSegmenter。

#include "vbt/segmenter.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vbt {

BiomechanicalRepSegmenter::BiomechanicalRepSegmenter(
    const std::string& exercise_type,
    double min_rom_m,
    double min_dur_s,
    double v_thresh_start,
    double v_zero_band)
    : exercise_type_(exercise_type),
      min_rom_m_(min_rom_m),
      min_dur_s_(min_dur_s),
      v_thresh_(v_thresh_start),
      v_band_(v_zero_band) {}

void BiomechanicalRepSegmenter::validate_and_append(
    std::vector<Rep>& reps,
    std::size_t start_idx, std::size_t end_idx,
    const double* t_arr, const double* y_arr, const double* v_arr) const
{
    if (end_idx <= start_idx || end_idx - start_idx < 2) {
        return;
    }
    const double duration = t_arr[end_idx] - t_arr[start_idx];
    const double rom = std::abs(y_arr[end_idx] - y_arr[start_idx]);
    if (duration < min_dur_s_ || rom < min_rom_m_) {
        return;
    }
    // 计算 mcv = ROM / duration（位移积分主指标）
    const double mcv = duration > 0.0 ? rom / duration : 0.0;
    // 计算 pcv（峰值速度）
    double pcv = 0.0;
    for (std::size_t i = start_idx; i <= end_idx; ++i) {
        if (v_arr[i] > pcv) {
            pcv = v_arr[i];
        }
    }
    Rep rep;
    rep.start_idx = static_cast<int>(start_idx);
    rep.end_idx = static_cast<int>(end_idx);
    rep.start_time = t_arr[start_idx];
    rep.end_time = t_arr[end_idx];
    rep.duration_s = std::round(duration * 1000.0) / 1000.0;
    rep.rom_m = std::round(rom * 10000.0) / 10000.0;
    rep.mcv_mps = std::round(mcv * 1000.0) / 1000.0;
    rep.pcv_mps = std::round(pcv * 1000.0) / 1000.0;
    reps.push_back(rep);
}

std::vector<Rep> BiomechanicalRepSegmenter::segment(
    const double* t_arr, const double* y_arr, const double* v_arr,
    std::size_t n)
{
    std::vector<Rep> reps;
    if (t_arr == nullptr || y_arr == nullptr || v_arr == nullptr || n < 10) {
        return reps;
    }

    enum class State { IDLE, ECCENTRIC, CONCENTRIC };
    State state = State::IDLE;
    std::size_t rep_start_idx = 0;

    const bool is_deadlift = (exercise_type_ == "deadlift");

    // 【2026-09-14 与 Python 对齐】状态跃迁需连续 kConfirmFrames 帧同向确认。
    // 旧实现仅看 (i, i+1)，蹲底停顿的单帧速度抖动会触发「早产的 CONCENTRIC」，
    // 该 rep 随即以极小 rom 收尾被门禁拒绝，真正的上行冲程再无状态机接管。
    // 详见 vbtcore/segmenter.py 中 confirm_frames 的推导与实测数据。
    constexpr std::size_t kConfirmFrames = 3;

    // i+1 .. i+kConfirmFrames 连续同向？positive=true 要求全部 > +v_band_。
    auto sustained = [&](std::size_t i, bool positive) -> bool {
        for (std::size_t k = 1; k <= kConfirmFrames; ++k) {
            const double vk = v_arr[i + k];
            if (positive) {
                if (vk <= v_band_) return false;
            } else {
                if (vk >= -v_band_) return false;
            }
        }
        return true;
    };

    for (std::size_t i = 1; i + kConfirmFrames < n; ++i) {
        const double v = v_arr[i];
        const double v_next = v_arr[i + 1];
        (void)v_next;

        if (is_deadlift) {
            // 硬拉拓扑（无 SSC，直接向心）
            switch (state) {
                case State::IDLE:
                    if (v > v_thresh_ && v_next > v_thresh_) {
                        state = State::CONCENTRIC;
                        rep_start_idx = (i >= 1) ? i - 1 : 0;
                    }
                    break;
                case State::CONCENTRIC:
                    if (v < v_band_ && v_next < v_band_) {
                        validate_and_append(reps, rep_start_idx, i,
                                            t_arr, y_arr, v_arr);
                        state = State::IDLE;
                    }
                    break;
                default:
                    state = State::IDLE;
                    break;
            }
        } else {
            // SSC 拓扑（深蹲/卧推）：ECCENTRIC → 底部换向 → CONCENTRIC
            switch (state) {
                case State::IDLE:
                    if (v < -v_thresh_) {
                        state = State::ECCENTRIC;
                    }
                    break;
                case State::ECCENTRIC:
                    // 底部换向：v 由负转正过零点
                    if (v >= -v_band_ && sustained(i, /*positive=*/true)) {
                        state = State::CONCENTRIC;
                        rep_start_idx = i;  // 向心起点 = 底部换向点
                    }
                    break;
                case State::CONCENTRIC:
                    // 向心结束：速度归零（顶部停顿）
                    if (v <= v_band_ && sustained(i, /*positive=*/false)) {
                        validate_and_append(reps, rep_start_idx, i,
                                            t_arr, y_arr, v_arr);
                        state = State::IDLE;
                    }
                    break;
                default:
                    state = State::IDLE;
                    break;
            }
        }
    }

    return reps;
}

}  // namespace vbt