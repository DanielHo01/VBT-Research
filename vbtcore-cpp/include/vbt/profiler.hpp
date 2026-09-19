/*
 * vbt/profiler.hpp — 轻量级跨平台微秒级性能剖析器
 * ==============================================
 * 使用 std::chrono::high_resolution_clock，支持 Android logcat 和桌面 stderr。
 */
#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cmath>

#ifdef __ANDROID__
#include <android/log.h>
#define PROFILER_LOG(tag, level, msg) __android_log_print(level, tag, "%s", msg)
#else
#define PROFILER_LOG(tag, level, msg) ((void)0)
#endif

namespace vbt {

struct PerfStats {
    double sum_us = 0.0, sum_sq_us = 0.0;
    double min_us = 1e12, max_us = -1e12;
    int count = 0;
    void record(double us) {
        sum_us += us; sum_sq_us += us * us;
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
        ++count;
    }
    double mean() const { return count > 0 ? sum_us / count : 0.0; }
    double std() const {
        if (count <= 1) return 0.0;
        double m = mean();
        return std::sqrt(std::max(0.0, sum_sq_us / count - m * m));
    }
};

class ScopedTimer {
public:
    explicit ScopedTimer(PerfStats& out) : out_(out), start_(now()) {}
    ~ScopedTimer() { out_.record(now() - start_); }
private:
    PerfStats& out_;
    double start_;
    static double now() {
        return std::chrono::duration<double, std::micro>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
};

class FrameProfiler {
public:
    explicit FrameProfiler(const char* tag = "VBTPerf") : tag_(tag) {}

    // 开始一帧
    void begin_frame() {
        frame_start_ = now();
    }

    // 阶段 1: RGBA→BGR
    void end_rgba_to_bgr() {
        rgba_to_bgr_.record(now() - frame_start_);
    }

    // 公开当前时间戳（微秒），供外部调用者记录阶段起止点
    double now_us() { return now(); }
    // 阶段 2: BGR→Gray
    void end_bgr_to_gray(double prev_end_us) {
        bgr_to_gray_.record(now() - prev_end_us);
        last_t_ = now();
    }
    // 阶段 3: YOLO 检测
    void end_yolo(double start_us, double end_us) { yolo_.record(end_us - start_us); prev_t_ = end_us; }
    // 阶段 4: 跟踪
    void end_tracker(double start_us, double end_us) { tracker_.record(end_us - start_us); prev_t_ = end_us; }
    // 阶段 5: 追加数据
    void end_append() { last_t_ = now(); append_.record(last_t_ - prev_t_); prev_t_ = last_t_; }
    // 阶段 6: 无检测帧
    void end_no_det() { last_t_ = now(); no_det_.record(last_t_ - prev_t_); prev_t_ = last_t_; }
    // 阶段 7: tracker_init
    void end_tracker_init(double start_us, double end_us) { tracker_init_.record(end_us - start_us); prev_t_ = end_us; }
    // 结束整帧
    void end_frame() {
        frame_total_.record(now() - frame_start_);
    }

    // JNI 层计时（外部调用）
    void record_jni(double us) { jni_.record(us); }

    // 输出汇总到 logcat
    void report() const {
        char buf[1024];
        int pos = 0;
        auto append = [&](const char* name, const PerfStats& s) {
            if (s.count == 0) return;
            int n = std::snprintf(buf + pos, sizeof(buf) - pos,
                " %s=%.1f±%.1fμs(%d)",
                name, s.mean(), s.std(), s.count);
            if (n > 0) pos += n;
        };
        append("Frame", frame_total_);
        append("JNI", jni_);
        append("rgba2bgr", rgba_to_bgr_);
        append("bgr2gray", bgr_to_gray_);
        append("yolo", yolo_);
        append("tracker", tracker_);
        append("no_det", no_det_);
        append("t_init", tracker_init_);
        append("append", append_);
        buf[pos] = '\0';
        PROFILER_LOG(tag_, ANDROID_LOG_INFO, buf);
    }

    // 生成 JSON 诊断字符串
    std::string diagnostics_json() const {
        std::ostringstream oss;
        oss << "{";
        bool first = true;
        auto write = [&](const char* name, const PerfStats& s) {
            if (s.count == 0) return;
            if (!first) oss << ","; first = false;
            oss << "\"" << name << "_mean_us\":" << std::fixed << std::setprecision(2) << s.mean()
                << ",\"" << name << "_std_us\":" << s.std()
                << ",\"" << name << "_min_us\":" << s.min_us
                << ",\"" << name << "_max_us\":" << s.max_us
                << ",\"" << name << "_count\":" << s.count;
        };
        write("frame_total", frame_total_);
        write("jni", jni_);
        write("rgba_to_bgr", rgba_to_bgr_);
        write("bgr_to_gray", bgr_to_gray_);
        write("yolo", yolo_);
        write("tracker", tracker_);
        write("no_det", no_det_);
        write("tracker_init", tracker_init_);
        write("append", append_);
        oss << "}";
        return oss.str();
    }

    int frame_count() const { return frame_total_.count; }

private:
    const char* tag_;
    double frame_start_ = 0.0, prev_t_ = 0.0, last_t_ = 0.0;
    PerfStats frame_total_, jni_, rgba_to_bgr_, bgr_to_gray_,
              yolo_, tracker_, no_det_, tracker_init_, append_;

    static double now() {
        return std::chrono::duration<double, std::micro>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
};

}  // namespace vbt
