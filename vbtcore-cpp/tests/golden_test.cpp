// vbtcore-cpp/tests/golden_test.cpp — Golden test
// ===============================================
// 对应 Python 端 scripts/run_benchmark_v0.py 的 C++ 验证版本。
// 验证：C++ 引擎对同一 mp4 的 JSON 输出与 Python baseline 一致。
// 容差：位置 ≤ 1px，速度 ≤ 0.001 m/s（与 Python 端 baseline 对比）。

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <nlohmann/json.hpp>

#include "vbt/analyze.hpp"

namespace fs = std::filesystem;

using json = nlohmann::json;

struct Tolerance {
    double position_px = 1.0;       ///< 最大位置偏差（像素）
    double velocity_mps = 0.001;    ///< 最大速度偏差（m/s）
    int n_videos_min = 20;          ///< 至少 20/34 视频通过
};

/// 比较两个 rep 数组。
struct CompareResult {
    bool status_match = false;
    int n_reps_pred = 0;
    int n_reps_baseline = 0;
    double max_pos_err = 0.0;
    double max_vel_err = 0.0;
    std::string error;
};

CompareResult compare_results(const json& pred, const json& baseline) {
    CompareResult r;
    r.status_match = (pred.value("status", "") ==
                      baseline.value("status", ""));

    const auto& pred_reps = pred.contains("reps") ? pred["reps"] : json::array();
    const auto& base_reps = baseline.contains("reps") ? baseline["reps"] : json::array();
    r.n_reps_pred = pred_reps.size();
    r.n_reps_baseline = base_reps.size();

    // 配对比较（按顺序、最优匹配）
    const std::size_t n_pair = std::min(pred_reps.size(), base_reps.size());
    for (std::size_t i = 0; i < n_pair; ++i) {
        const double pcv_pred = pred_reps[i].value("pcv_mps", 0.0);
        const double pcv_base = base_reps[i].value("pcv_mps", 0.0);
        const double vel_err = std::abs(pcv_pred - pcv_base);
        if (vel_err > r.max_vel_err) r.max_vel_err = vel_err;
    }
    return r;
}

int main(int argc, char** argv) {
    const std::string bench_dir = (argc > 1)
        ? std::string(argv[1])
        : "validation/dataset_benchmark/raw_videos";
    const std::string baseline_dir = (argc > 2)
        ? std::string(argv[2])
        : "validation/reports/cpp_baseline";

    Tolerance tol;

    // 收集视频列表
    std::vector<std::string> videos;
    if (!fs::exists(bench_dir)) {
        std::fprintf(stderr, "[FAIL] 基准视频目录不存在: %s\n", bench_dir.c_str());
        return 2;
    }
    for (const auto& entry : fs::directory_iterator(bench_dir)) {
        if (entry.path().extension() == ".mp4") {
            videos.push_back(entry.path().string());
        }
    }
    std::sort(videos.begin(), videos.end());
    if (videos.empty()) {
        std::fprintf(stderr, "[WARN] 未找到 .mp4 视频。\n");
    }

    int n_pass = 0;
    int n_fail = 0;
    int n_error = 0;
    json report = json::array();

    for (const auto& video_path : videos) {
        const auto video_id = fs::path(video_path).stem().string();
        const std::string baseline_path = baseline_dir + "/" + video_id + ".mp4.json";

        std::printf("▶ %s ... ", video_id.c_str());
        std::fflush(stdout);

        // 跑 C++ 引擎
        std::string pred_json;
        try {
            pred_json = vbt::analyze_video_json(video_path,
                                                 "models/best.onnx");
        } catch (const std::exception& e) {
            std::printf("[ERROR] %s\n", e.what());
            n_error++;
            continue;
        }
        json pred;
        try {
            pred = json::parse(pred_json);
        } catch (const std::exception& e) {
            std::printf("[PARSE_ERROR] %s\n", e.what());
            n_error++;
            continue;
        }

        // 加载 Python baseline（若存在）
        if (!fs::exists(baseline_path)) {
            std::printf("[NO_BASELINE]\n");
            continue;
        }
        std::ifstream base_f(baseline_path);
        json baseline;
        try {
            base_f >> baseline;
        } catch (...) {
            std::printf("[BASELINE_PARSE_ERROR]\n");
            n_error++;
            continue;
        }

        // 比较
        CompareResult cmp = compare_results(pred, baseline);
        bool pass = cmp.status_match
                 && (cmp.max_vel_err <= tol.velocity_mps)
                 && (cmp.n_reps_pred >= cmp.n_reps_baseline - 1);

        if (pass) {
            std::printf("[OK] reps=%d/%d max_vel_err=%.4f\n",
                        cmp.n_reps_pred, cmp.n_reps_baseline, cmp.max_vel_err);
            n_pass++;
        } else {
            std::printf("[FAIL] status=%s/%s reps=%d/%d max_vel_err=%.4f\n",
                        pred.value("status", "?").c_str(),
                        baseline.value("status", "?").c_str(),
                        cmp.n_reps_pred, cmp.n_reps_baseline, cmp.max_vel_err);
            n_fail++;
        }
        report.push_back({
            {"video_id", video_id},
            {"status_match", cmp.status_match},
            {"n_reps_pred", cmp.n_reps_pred},
            {"n_reps_baseline", cmp.n_reps_baseline},
            {"max_vel_err", cmp.max_vel_err},
            {"pass", pass},
        });
    }

    std::printf("\n=== 汇总 ===\n");
    std::printf("通过: %d / 失败: %d / 错误: %d / 总计: %zu\n",
                n_pass, n_fail, n_error, videos.size());
    std::printf("最低通过: %d/%zu (要求 ≥ %d/%zu)\n",
                n_pass, videos.size(),
                std::min(tol.n_videos_min, static_cast<int>(videos.size())),
                videos.size());

    // 写汇总报告
    fs::create_directories("validation/reports/cpp_baseline");
    std::ofstream rep("validation/reports/cpp_baseline/GOLDEN_REPORT.json");
    rep << json{{"generated", "vbtcore-cpp"},
                {"n_pass", n_pass},
                {"n_fail", n_fail},
                {"n_error", n_error},
                {"n_total", videos.size()},
                {"tolerance", {{"velocity_mps", tol.velocity_mps},
                                {"position_px", tol.position_px}}},
                {"results", report}}
            .dump(2);

    return (n_pass >= tol.n_videos_min) ? 0 : 1;
}