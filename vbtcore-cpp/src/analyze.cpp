// vbt/analyze.cpp — 端到端视频分析入口
// =======================================
// 对应 Python 端 vbtcore/pipeline.py::analyze_video。
// 输入：mp4 路径 + ONNX 模型路径 + 选项
// 输出：JSON 字符串

#include "vbt/analyze.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "vbt/calibrator.hpp"
#include "vbt/detector.hpp"
#include "vbt/geometry.hpp"
#include "vbt/segmenter.hpp"
#include "vbt/tracker.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// nlohmann/json 头文件（生产构建用真实库，存放在 vbtcore-cpp/include/nlohmann/）
#include "../include/nlohmann/json.hpp"

namespace vbt {

namespace {

/// 把 double 安全格式化为 6 位精度字符串（避免 -nan/inf）。
inline std::string fmt(double v) {
    if (std::isnan(v)) return "null";
    if (std::isinf(v)) return "null";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

/// 把 Rep 列表写入 JSON。
nlohmann::json reps_to_json(const std::vector<Rep>& reps) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& r : reps) {
        nlohmann::json item;
        item["start_idx"] = r.start_idx;
        item["end_idx"] = r.end_idx;
        item["start_time"] = fmt(r.start_time);
        item["end_time"] = fmt(r.end_time);
        item["duration_s"] = fmt(r.duration_s);
        item["rom_m"] = fmt(r.rom_m);
        item["mcv_mps"] = fmt(r.mcv_mps);
        item["pcv_mps"] = fmt(r.pcv_mps);
        arr.push_back(item);
    }
    return arr;
}

/// 安全从 std::string 构造 std::string*（resolve_plate_diameter 接受 const std::string*）。
inline const std::string* str_or_null(const char* s) {
    static thread_local std::string buf;
    buf = (s == nullptr) ? std::string() : std::string(s);
    return buf.empty() ? nullptr : &buf;
}

}  // namespace

std::string analyze_video_json(const std::string& video_path,
                               const std::string& model_path,
                               const AnalyzeOptions& opts) {
    nlohmann::json result;
    result["video"] = video_path;
    result["model"] = model_path;

    // 打开视频
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        result["status"] = "VIDEO_ERROR";
        result["error"] = "无法打开视频: " + video_path;
        return result.dump();
    }

    const double fps = cap.get(cv::CAP_PROP_FPS);
    const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    if (fps <= 0.0 || total_frames < 30) {
        cap.release();
        result["status"] = "VIDEO_ERROR";
        result["error"] = "视频过短或 fps 异常";
        return result.dump();
    }

    auto t0 = std::chrono::steady_clock::now();

    // 标定器
    StaticPlateCalibrator calibrator(/*real_diameter_m=*/opts.plate_diameter_m);
    const std::string* outer_plate_str = str_or_null(opts.outer_plate);
    const double outer_diam = (outer_plate_str != nullptr)
        ? resolve_plate_diameter(outer_plate_str)
        : opts.plate_diameter_m;

    // 创建检测器（若未链接 ONNX Runtime，返回 Stub）
    auto detector = make_detector(model_path);
    const bool det_is_stub = detector->is_stub();

    // 跟踪器状态
    std::unique_ptr<DenseVisualTracker> tracker;
    double mpp = 0.0;

    std::vector<double> timestamps;
    std::vector<double> positions_m;  // 向上为正
    std::vector<double> velocities_mps;

    int frame_idx = 0;
    int n_yolo = 0;
    cv::Mat frame;

    while (cap.read(frame)) {
        if (frame.empty()) break;
        const double pts_ms = cap.get(cv::CAP_PROP_POS_MSEC);
        const double t_s = (pts_ms > 0.0) ? pts_ms / 1000.0
                                          : static_cast<double>(frame_idx) / fps;
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // 标定期
        if (mpp == 0.0) {
            const auto dets = detector->detect(frame, opts.conf_thresh_calibrate);
            if (!dets.empty()) {
                const auto& best = *std::max_element(
                    dets.begin(), dets.end(),
                    [](const Detection& a, const Detection& b) {
                        return a.w * a.h < b.w * b.h;
                    });
                calibrator.add_sample(best.h);
                if (calibrator.is_ready()) {
                    mpp = calibrator.lock_scale();
                    std::array<double, 4> bbox = {
                        best.cx - best.w / 2, best.cy - best.h / 2,
                        best.cx + best.w / 2, best.cy + best.h / 2};
                    tracker = std::make_unique<DenseVisualTracker>(
                        mpp, bbox, gray, t_s);
                    n_yolo++;
                }
            }
            frame_idx++;
            continue;
        }

        // 跟踪期
        if (tracker == nullptr) {
            frame_idx++;
            continue;
        }
        TrackStep step;
        const bool is_keyframe = (frame_idx % opts.redet_every == 0);
        if (is_keyframe) {
            const auto dets = detector->detect(frame, opts.conf_thresh);
            if (!dets.empty()) {
                const auto& best = *std::max_element(
                    dets.begin(), dets.end(),
                    [](const Detection& a, const Detection& b) {
                        return a.w * a.h < b.w * b.h;
                    });
                std::array<double, 4> bbox = {
                    best.cx - best.w / 2, best.cy - best.h / 2,
                    best.cx + best.w / 2, best.cy + best.h / 2};
                step = tracker->step_keyframe(gray, bbox, t_s);
                n_yolo++;
            } else {
                step = tracker->step_interframe(gray, t_s);
            }
        } else {
            step = tracker->step_interframe(gray, t_s);
        }
        timestamps.push_back(t_s);
        positions_m.push_back(step.y_m);
        velocities_mps.push_back(step.v_mps);
        frame_idx++;
    }
    cap.release();

    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    if (mpp == 0.0) {
        result["status"] = "NO_PLATE_DETECTED";
        result["fps"] = fps;
        nlohmann::json diag;
        diag["n_frames"] = frame_idx;
        diag["elapsed_s"] = fmt(elapsed);
        diag["detector_stub"] = det_is_stub;
        result["diagnostics"] = diag;
        return result.dump();
    }

    // Rep 分段
    BiomechanicalRepSegmenter seg(opts.exercise_type);
    const std::size_t n = timestamps.size();
    std::vector<Rep> reps;
    if (n > 0) {
        reps = seg.segment(timestamps.data(), positions_m.data(),
                           velocities_mps.data(), n);
    }

    result["status"] = reps.empty() ? "NO_CLEAN_SEGMENT" : "OK";
    result["fps"] = fps;
    result["mpp"] = fmt(mpp);
    result["reps"] = reps_to_json(reps);

    nlohmann::json diag;
    diag["n_frames"] = frame_idx;
    diag["n_yolo"] = n_yolo;
    diag["elapsed_s"] = fmt(elapsed);
    diag["detector_stub"] = det_is_stub;
    diag["plate_diameter_m"] = fmt(opts.plate_diameter_m);
    diag["outer_plate"] = opts.outer_plate ? std::string(opts.outer_plate) : std::string();
    diag["exercise_type"] = std::string(opts.exercise_type);
    diag["redet_every"] = opts.redet_every;
    diag["tracker"] = "dense_visual_kalman";
    diag["calibrator"] = "static_cv_gate";
    result["diagnostics"] = diag;
    return result.dump();
}

extern "C" char* vbt_analyze_video(const char* video_path,
                                   const char* model_path,
                                   const char* options_json) {
    if (video_path == nullptr || model_path == nullptr) {
        return nullptr;
    }
    AnalyzeOptions opts;
    if (options_json != nullptr) {
        try {
            auto j = nlohmann::json::parse(options_json);
            if (j.contains("redet_every")) {
                opts.redet_every = j.value("redet_every", opts.redet_every);
            }
            if (j.contains("plate_diameter_m")) {
                opts.plate_diameter_m = j.value("plate_diameter_m", opts.plate_diameter_m);
            }
            if (j.contains("outer_plate") && j["outer_plate"].is_string()) {
                const std::string s = j.value("outer_plate", std::string());
                if (!s.empty()) {
                    opts.outer_plate = strdup(s.c_str());
                }
            }
            if (j.contains("exercise_type") && j["exercise_type"].is_string()) {
                const std::string s = j.value("exercise_type", std::string());
                if (!s.empty()) {
                    opts.exercise_type = strdup(s.c_str());
                }
            }
            if (j.contains("conf_thresh")) {
                opts.conf_thresh = j.value("conf_thresh", opts.conf_thresh);
            }
        } catch (...) {
            // 解析失败：使用默认选项
        }
    }
    std::string json_str = analyze_video_json(video_path, model_path, opts);
    // 释放 strdup
    if (opts.outer_plate && opts.outer_plate != "squat_bench") {
        std::free(const_cast<char*>(opts.outer_plate));
    }
    char* out = strdup(json_str.c_str());
    return out;
}

}  // namespace vbt