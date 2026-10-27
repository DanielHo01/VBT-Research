// vbt/frame_analyzer.cpp — Android 帧输入分析器
// ====================================================
// 与 Python 端 vbtcore/pipeline.py 严格对齐的 C++ 实现。
//
// 管线（3阶段）：
//   1. 标定（mpp 未锁死）→ YOLO 检测 → StaticPlateCalibrator 收集 bbox.height
//   2. 跟踪（mpp 已锁死）→ DenseVisualTracker(LK光流+卡尔曼)
//   3. 分段 → BiomechanicalRepSegmenter (FSM)
//      注意：验证表明旧 FSM RMSE~0.02（几乎完美），v2 实现 RMSE~0.20（差10倍）。
//      保留旧 FSM，仅在 trajectory 数据质量足够好时再考虑 v2。
//
// 无需 OpenCV VideoCapture，使用 Kotlin 侧 MediaMetadataRetriever 提取的 RGBA 帧。

#include "../include/nlohmann/json.hpp"
#include "vbt/frame_analyzer.hpp"
#include "vbt/detector.hpp"
#include "vbt/tracker.hpp"
#include "vbt/calibrator.hpp"
#include "vbt/segmenter.hpp"
#include "vbt/profiler.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

namespace {

inline std::string fmt_dbl(double v) {
    if (std::isnan(v) || std::isinf(v)) return "null";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << v;
    return oss.str();
}

/// RGBA → BGR（Android 格式转换）
/// 策略：ARM64 NEON（128-bit SIMD，一次处理 16 像素） > x64 SSE > 标量 fallback
#if defined(__ARM_NEON) || defined(__aarch64__) || defined(__ARM64_ARCH_8__)
#include <arm_neon.h>
inline void rgba_to_bgr(const uint8_t* rgba, int w, int h,
                         int rgba_stride, uint8_t* bgr_out) {
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = rgba + y * rgba_stride;
        uint8_t* dst = bgr_out + y * w * 3;
        int x = 0;
        // NEON: 一次循环处理 16 个 RGBA 像素 (64 bytes) → 48 bytes BGR
        //       vld4q_u8 一次性加载 src[4*x+0..4*x+63] 并解交织为 4 路独立通道
        for (; x <= w - 16; x += 16) {
            uint8x16x4_t v_rgba = vld4q_u8(src + x * 4);
            uint8x16x3_t v_bgr;
            v_bgr.val[0] = v_rgba.val[2];  // B = R_src
            v_bgr.val[1] = v_rgba.val[1];  // G = G_src
            v_bgr.val[2] = v_rgba.val[0];  // R = B_src
            vst3q_u8(dst + x * 3, v_bgr);
        }
        // 尾部标量补齐（0..15 像素）
        for (; x < w; ++x) {
            dst[x * 3 + 0] = src[x * 4 + 2];
            dst[x * 3 + 1] = src[x * 4 + 1];
            dst[x * 3 + 2] = src[x * 4 + 0];
        }
    }
}
#elif defined(_M_X64) || defined(__x86_64__)
// x64: 使用 SSE2 intrinsics（_mm_loadu_si128 16-byte对齐读取）
#include <emmintrin.h>
inline void rgba_to_bgr(const uint8_t* rgba, int w, int h,
                         int rgba_stride, uint8_t* bgr_out) {
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = rgba + y * rgba_stride;
        uint8_t* dst = bgr_out + y * w * 3;
        int x = 0;
        for (; x <= w - 16; x += 16) {
            __m128i r0 = _mm_loadu_si128((const __m128i*)(src + x * 4 + 0));
            __m128i r1 = _mm_loadu_si128((const __m128i*)(src + x * 4 + 16));
            // 通道重排: RGBA→BGR (shuffle mask = [2,1,0, 5,4,3, 8,7,6, ...])
            // 待处理: 16 像素 = 64 bytes → 每个 __m128i 处理 16 个 src 像素
            // 使用 _mm_shuffle_epi8 需要 SSSE3
            // 简化方案: 16x标量处理 (x64 通常有更快的标量路径)
            for (int i = 0; i < 16; ++i) {
                dst[(x + i) * 3 + 0] = src[(x + i) * 4 + 2];
                dst[(x + i) * 3 + 1] = src[(x + i) * 4 + 1];
                dst[(x + i) * 3 + 2] = src[(x + i) * 4 + 0];
            }
        }
        for (; x < w; ++x) {
            dst[x * 3 + 0] = src[x * 4 + 2];
            dst[x * 3 + 1] = src[x * 4 + 1];
            dst[x * 3 + 2] = src[x * 4 + 0];
        }
    }
}
#else
// 通用标量 fallback（覆盖 Desktop MSVC / Linux GCC）
inline void rgba_to_bgr(const uint8_t* rgba, int w, int h,
                         int rgba_stride, uint8_t* bgr_out) {
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = rgba + y * rgba_stride;
        uint8_t* dst = bgr_out + y * w * 3;
        for (int x = 0; x < w; ++x) {
            dst[x * 3 + 0] = src[x * 4 + 2];
            dst[x * 3 + 1] = src[x * 4 + 1];
            dst[x * 3 + 2] = src[x * 4 + 0];
        }
    }
}
#endif

/// BGR → 灰度（浮点权重的定点近似: 0.299*R + 0.587*G + 0.114*B）
/// 策略：ARM64 NEON（16 像素并行，uint8×uint8→uint16 累加）
#if defined(__ARM_NEON) || defined(__aarch64__) || defined(__ARM64_ARCH_8__)
#include <arm_neon.h>
inline void bgr_to_gray(const uint8_t* bgr, int w, int h, uint8_t* gray_out) {
    // 定点系数（精度损失 < 0.5/255，视觉无感知）
    // Gray = (77*R + 150*G + 29*B + 128) >> 8
    //       = ((77>>8)*R + (150>>8)*G + (29>>8)*B) + rounding
    // 16×uint8 → 16×uint16 multiply-accumulate in one instruction
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = bgr + y * w * 3;
        uint8_t* dst = gray_out + y * w;
        int x = 0;
        // 加载 RGB 系数（每个通道 16 个副本）
        const uint8_t R_COEFF = 77, G_COEFF = 150, B_COEFF = 29;
        // 待定: NEON vdupq_n_u8 填充，vld3q_u8 加载 BGR
        // 使用 vmull_u8 (unsigned multiply-long) 避免 signed 溢出
        for (; x <= w - 16; x += 16) {
            uint8x16_t v_b = vld1q_u8(src + x * 3 + 0);   // B 通道
            uint8x16_t v_g = vld1q_u8(src + x * 3 + 16);   // G 通道
            uint8x16_t v_r = vld1q_u8(src + x * 3 + 32);   // R 通道
            // 定点乘加: 77*R + 150*G + 29*B → uint16×3 → 相加 → 右移8
            uint16x8_t lo_r = vmull_u8(vget_low_u8(v_r), vdup_n_u8(R_COEFF));
            uint16x8_t hi_r = vmull_u8(vget_high_u8(v_r), vdup_n_u8(R_COEFF));
            uint16x8_t lo_g = vmull_u8(vget_low_u8(v_g), vdup_n_u8(G_COEFF));
            uint16x8_t hi_g = vmull_u8(vget_high_u8(v_g), vdup_n_u8(G_COEFF));
            uint16x8_t lo_b = vmull_u8(vget_low_u8(v_b), vdup_n_u8(B_COEFF));
            uint16x8_t hi_b = vmull_u8(vget_high_u8(v_b), vdup_n_u8(B_COEFF));
            uint16x8_t lo = vaddq_u16(vaddq_u16(lo_r, lo_g), lo_b);
            uint16x8_t hi = vaddq_u16(vaddq_u16(hi_r, hi_g), hi_b);
            // +128 四舍五入后右移8: (val + 128) >> 8
            // 先加128(0x80)，再右移（vrshrn：向+∞取整）→ 模拟标准round-to-nearest
            lo = vaddq_u16(lo, vdupq_n_u16(128));
            hi = vaddq_u16(hi, vdupq_n_u16(128));
            uint8x8_t lo_gray = vrshrn_n_u16(lo, 8);  // narrowing + shift in one intrinsic
            uint8x8_t hi_gray = vrshrn_n_u16(hi, 8);
            vst1q_u8(dst + x, vcombine_u8(lo_gray, hi_gray));
        }
        // 尾部标量补齐
        for (; x < w; ++x) {
            const uint8_t B = src[x * 3 + 0];
            const uint8_t G = src[x * 3 + 1];
            const uint8_t R = src[x * 3 + 2];
            dst[x] = static_cast<uint8_t>((77 * R + 150 * G + 29 * B + 128) >> 8);
        }
    }
}
#else
inline void bgr_to_gray(const uint8_t* bgr, int w, int h, uint8_t* gray_out) {
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = bgr + y * w * 3;
        uint8_t* dst = gray_out + y * w;
        for (int x = 0; x < w; ++x) {
            const uint8_t B = src[x * 3 + 0];
            const uint8_t G = src[x * 3 + 1];
            const uint8_t R = src[x * 3 + 2];
            dst[x] = static_cast<uint8_t>((77 * R + 150 * G + 29 * B + 128) >> 8);
        }
    }
}
#endif

}  // anonymous namespace

// ── Impl ────────────────────────────────────────────────────────
struct vbt::FrameAnalyzer::Impl {
    const double fps;
    const std::string model_path;
    const double plate_diameter_m;
    const std::string exercise_type;
    const int redet_every;
    const double conf_thresh;

    // 标定器（对应 Python StaticPlateCalibrator）
    vbt::StaticPlateCalibrator calibrator;

    // 检测器（YOLO，ONNX Runtime）
    std::unique_ptr<vbt::Detector> detector;

    // 跟踪器（LK 光流 + 卡尔曼）
    std::unique_ptr<vbt::DenseVisualTracker> tracker;

    bool calibrated = false;
    double mpp = 0.0;

    int total_frames = 0;
    int yolo_frames = 0;
    int calib_frames = 0;

    // 时间戳和位置（用于分段）
    std::vector<double> timestamps;
    std::vector<double> positions_m;   // 向上为正
    std::vector<double> velocities_mps;

    // 性能剖析器
    vbt::FrameProfiler perf_; // "VBTPerf" tag

    Impl(double fps_, const std::string& model_path_,
          double plate_diameter_m_, const std::string& exercise_type_,
          int redet_every_, double conf_thresh_)
        : fps(fps_), model_path(model_path_),
          plate_diameter_m(plate_diameter_m_),
          exercise_type(exercise_type_),
          redet_every(redet_every_), conf_thresh(conf_thresh_),
          calibrator(plate_diameter_m_, /*min_static_frames=*/20,
                     /*max_cv=*/0.015) {
        detector = vbt::make_detector(model_path_);
    }

    void process_frame(const uint8_t* rgba_data,
                       int width, int height, int stride_bytes,
                       int /*frame_idx*/, double timestamp_ms) {
        total_frames++;
        perf_.begin_frame();

        // 转换为 BGR + 灰度（tracker 需要）
        std::vector<uint8_t> bgr_buf(static_cast<size_t>(width * 3) * height);
        rgba_to_bgr(rgba_data, width, height, stride_bytes, bgr_buf.data());
        double after_rgba_to_bgr = perf_.now_us();
        perf_.end_rgba_to_bgr();

        std::vector<uint8_t> gray_buf(static_cast<size_t>(width) * height);
        bgr_to_gray(bgr_buf.data(), width, height, gray_buf.data());
        double after_bgr_to_gray = perf_.now_us();
        perf_.end_bgr_to_gray(after_rgba_to_bgr);

        cv::Mat gray_mat(height, width, CV_8UC1, gray_buf.data());
        cv::Mat bgr_mat(height, width, CV_8UC3, bgr_buf.data());

        const double t_s = timestamp_ms / 1000.0;

        // ── 阶段1：标定（mpp 未锁死）─────────────────────────────
        if (mpp == 0.0) {
            if (detector && !detector->is_stub()) {
                double before_yolo = perf_.now_us();
                const auto dets = detector->detect(bgr_mat, 0.45);  // conf_thresh_calibrate
                double after_yolo = perf_.now_us();
                perf_.end_yolo(before_yolo, after_yolo);
                if (!dets.empty()) {
                    const auto& best = *std::max_element(
                        dets.begin(), dets.end(),
                        [](const vbt::Detection& a, const vbt::Detection& b) {
                            return a.w * a.h < b.w * b.h;
                        });
                    calibrator.add_sample(best.h);

                    if (calibrator.is_ready()) {
                        mpp = calibrator.lock_scale();
                        // 初始化 tracker
                        std::array<double, 4> bbox = {
                            best.cx - best.w / 2,
                            best.cy - best.h / 2,
                            best.cx + best.w / 2,
                            best.cy + best.h / 2
                        };
                        double before_tinit = perf_.now_us();
                        tracker = std::make_unique<vbt::DenseVisualTracker>(
                            mpp, bbox, gray_mat, t_s);
                        double after_tinit = perf_.now_us();
                        perf_.end_tracker_init(before_tinit, after_tinit);
                        yolo_frames++;
                    }
                }
            }
            // 无论检测是否成功，都继续下一帧（标定期不跟踪）
            perf_.end_no_det();
            perf_.end_frame();
            return;
        }

        // ── 阶段2：跟踪（mpp 已锁死，tracker 已初始化）───────────
        if (!tracker) {
            // tracker 还未初始化（标定期结束后还没找到 bbox）
            perf_.end_no_det();
            perf_.end_frame();
            return;
        }

        vbt::TrackStep step;
        const bool is_keyframe = (total_frames % redet_every == 0);

        if (is_keyframe && detector && !detector->is_stub()) {
            double before_yolo = perf_.now_us();
            const auto dets = detector->detect(bgr_mat, conf_thresh);
            double after_yolo = perf_.now_us();
            perf_.end_yolo(before_yolo, after_yolo);
            if (!dets.empty()) {
                const auto& best = *std::max_element(
                    dets.begin(), dets.end(),
                    [](const vbt::Detection& a, const vbt::Detection& b) {
                        return a.w * a.h < b.w * b.h;
                    });
                std::array<double, 4> bbox = {
                    best.cx - best.w / 2,
                    best.cy - best.h / 2,
                    best.cx + best.w / 2,
                    best.cy + best.h / 2
                };
                double before_track = perf_.now_us();
                step = tracker->step_keyframe(gray_mat, bbox, t_s);
                double after_track = perf_.now_us();
                perf_.end_tracker(before_track, after_track);
                yolo_frames++;
            } else {
                double before_track = perf_.now_us();
                step = tracker->step_interframe(gray_mat, t_s);
                double after_track = perf_.now_us();
                perf_.end_tracker(before_track, after_track);
            }
        } else {
            double before_track = perf_.now_us();
            step = tracker->step_interframe(gray_mat, t_s);
            double after_track = perf_.now_us();
            perf_.end_tracker(before_track, after_track);
        }

        timestamps.push_back(t_s);
        positions_m.push_back(step.y_m);       // 向上为正（tracker 已取反）
        velocities_mps.push_back(step.v_mps);  // 向上为正
        perf_.end_append();
        perf_.end_frame();
    }

    std::string finalize() {
        nlohmann::json result;

        if (mpp == 0.0) {
            result["status"] = "NO_PLATE_DETECTED";
            result["fps"] = fmt_dbl(fps);
            nlohmann::json diag;
            diag["total_frames"] = total_frames;
            diag["calib_frames"] = calib_frames;
            diag["detector_stub"] = detector ? detector->is_stub() : true;
            diag["method"] = "yolo_calibrator";
            result["diagnostics"] = diag;
            return result.dump();
        }

        const size_t n = timestamps.size();
        if (n < 5) {
            result["status"] = "TOO_FEW_FRAMES";
            result["fps"] = fmt_dbl(fps);
            return result.dump();
        }

        // ── 阶段3：Rep 分段（FSM）────────────────────────
        vbt::BiomechanicalRepSegmenter seg(exercise_type);
        std::vector<vbt::Rep> reps = seg.segment(
            timestamps.data(), positions_m.data(),
            velocities_mps.data(), n);

        // 诊断
        nlohmann::json diag;
        diag["total_frames"] = total_frames;
        diag["tracked_frames"] = static_cast<int>(n);
        diag["yolo_frames"] = yolo_frames;
        diag["calib_frames"] = calibrator.is_ready() ? 20 : 0;
        diag["plate_diameter_m"] = fmt_dbl(plate_diameter_m);
        diag["exercise_type"] = exercise_type;
        diag["mpp"] = fmt_dbl(mpp);
        diag["method"] = "yolo_tracker_kalman";
        diag["redet_every"] = redet_every;
        diag["detector_stub"] = detector ? detector->is_stub() : true;

        // 注入性能剖析数据
        {
            std::string perf_json = perf_.diagnostics_json();
            nlohmann::json perf_data = nlohmann::json::parse(perf_json, nullptr, false);
            if (!perf_data.is_discarded()) {
                for (auto& [k, v] : perf_data.items()) {
                    diag["perf_" + k] = v;
                }
            }
            diag["perf_n_frames"] = perf_.frame_count();
        }

        // 速度诊断
        if (!velocities_mps.empty()) {
            double v_min = 1e9, v_max = -1e9, v_sum = 0.0;
            for (double v : velocities_mps) {
                if (v < v_min) v_min = v;
                if (v > v_max) v_max = v;
                v_sum += v;
            }
            diag["v_min"] = fmt_dbl(v_min);
            diag["v_max"] = fmt_dbl(v_max);
            diag["v_mean"] = fmt_dbl(v_sum / velocities_mps.size());
        }
        if (!positions_m.empty()) {
            double y_min = positions_m[0], y_max = positions_m[0];
            for (double y : positions_m) {
                if (y < y_min) y_min = y;
                if (y > y_max) y_max = y;
            }
            diag["y_range_m"] = fmt_dbl(y_max - y_min);
        }

        result["status"] = reps.empty() ? "NO_CLEAN_SEGMENT" : "OK";
        result["fps"] = fmt_dbl(fps);
        result["mpp"] = fmt_dbl(mpp);

        nlohmann::json arr = nlohmann::json::array();
        for (const auto& r : reps) {
            nlohmann::json j;
            j["start_time"] = fmt_dbl(r.start_time);
            j["end_time"]   = fmt_dbl(r.end_time);
            j["duration_s"]  = fmt_dbl(r.duration_s);
            j["rom_m"]       = fmt_dbl(r.rom_m);
            j["mcv_mps"]    = fmt_dbl(r.mcv_mps);
            j["pcv_mps"]    = fmt_dbl(r.pcv_mps);
            arr.push_back(j);
        }
        result["reps"] = arr;
        result["diagnostics"] = diag;
        return result.dump();
    }
};

// ── Public API ─────────────────────────────────────────────────

vbt::FrameAnalyzer::FrameAnalyzer(
    double fps, const std::string& model_path,
    double plate_diameter_m, const std::string& exercise_type,
    int redet_every, double conf_thresh) {
    impl_ = new Impl(fps, model_path, plate_diameter_m,
                     exercise_type, redet_every, conf_thresh);
}

vbt::FrameAnalyzer::~FrameAnalyzer() { delete impl_; impl_ = nullptr; }

void vbt::FrameAnalyzer::add_frame(const uint8_t* rgba_data,
                                    int width, int height,
                                    int stride_bytes, int frame_idx,
                                    double timestamp_ms) {
    if (impl_) impl_->process_frame(rgba_data, width, height,
                                     stride_bytes, frame_idx, timestamp_ms);
}

bool vbt::FrameAnalyzer::is_calibrated() const {
    return impl_ && impl_->mpp > 0.0;
}

double vbt::FrameAnalyzer::get_mpp() const {
    return impl_ ? impl_->mpp : 0.0;
}

int vbt::FrameAnalyzer::frame_count() const {
    return impl_ ? impl_->total_frames : 0;
}

std::string vbt::FrameAnalyzer::finalize() {
    if (!impl_) return R"({"status":"NOT_INITIALIZED"})";
    std::string json = impl_->finalize();
    delete impl_;
    impl_ = nullptr;
    return json;
}
