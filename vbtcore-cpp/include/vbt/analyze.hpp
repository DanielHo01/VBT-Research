#pragma once
// vbt/analyze.hpp — C++ 端到端视频分析入口
// ==========================================
// 对应 Python 端 vbtcore/pipeline.py::analyze_video。
// 输入：mp4 路径 + ONNX 模型路径
// 输出：JSON 字符串（包含 rep 列表 + 诊断）

#include <string>

namespace vbt {

/// 分析选项。
struct AnalyzeOptions {
    int redet_every = 15;              ///< YOLO 重检测间隔（帧）
    double plate_diameter_m = 0.45;    ///< 外层片直径（米）
    const char* outer_plate = nullptr; ///< 外层片规格（如 "20kg"，nullptr=默认）
    const char* exercise_type = "squat_bench";  ///< "squat_bench" 或 "deadlift"
    double conf_thresh = 0.40;         ///< YOLO 置信度门限
    double conf_thresh_calibrate = 0.45;  ///< 标定期 YOLO 置信度门限（更高）
};

/// 端到端视频分析（mp4 → JSON）。
///
/// 返回 JSON 字符串（UTF-8）。格式：
/// {
///   "video": "input.mp4",
///   "status": "OK" | "NO_PLATE_DETECTED" | ...,
///   "fps": 30.0,
///   "mpp": 0.0045,
///   "reps": [
///     {"start_idx": 30, "end_idx": 60, "mcv_mps": 0.45, "pcv_mps": 0.72, ...},
///     ...
///   ],
///   "diagnostics": {"n_frames": 120, "elapsed_s": 3.2, ...}
/// }
///
/// 失败时 JSON 包含 "error" 字段。
std::string analyze_video_json(const std::string& video_path,
                               const std::string& model_path,
                               const AnalyzeOptions& opts = {});

/// C-linkage 入口（供 JNI 调用）。
extern "C" {
    /// JNI 调用约定：传入 mp4/onnx 路径 + 选项 JSON，返回 JSON 字符串。
    /// 调用方负责 free 返回的字符串（caller 用 free() 释放）。
    /// 返回 nullptr 表示参数错误。
    char* vbt_analyze_video(const char* video_path,
                            const char* model_path,
                            const char* options_json);
}

}  // namespace vbt