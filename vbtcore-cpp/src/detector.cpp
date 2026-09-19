// vbt/detector.cpp — ONNX Runtime 检测器实现
// ===========================================
// 对应 Python 端 vbtcore/detector.py::PlateDetector。
// 编译时若 VBT_HAS_ONNXRUNTIME == 1 则链接 ONNX Runtime，
// 否则为 StubDetector 模式（返回空检测，仅供接口冒烟测试）。

#include "vbt/detector.hpp"

#if defined(VBT_HAS_ONNXRUNTIME) && VBT_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace vbt {

#if defined(VBT_HAS_ONNXRUNTIME) && VBT_HAS_ONNXRUNTIME
#include <algorithm>
#include <cmath>

namespace {
/// 【已修复 · 禁止复活】置信度双重 sigmoid
/// ------------------------------------------------------------------
/// Ultralytics 导出的 YOLOv11 ONNX，输出 [1,5,8400] 的 row[4] **已经是概率**
/// （实测 models/best.onnx 首帧：min=0.000000 max=0.930283 mean=0.002112）。
/// 旧实现在此再套一层 sigmoid，把 [0,1] 概率映射到 [0.5, 0.731]，
/// 导致全部 8400 个 anchor 都 > conf_thresh(0.40)（实测 8400/8400 通过，
/// 正确解析应只有 20 个），max-area 选框于是锁定到无意义的巨大噪声框，
/// 标定得到 mpp=0.000565539（正确值 0.002471，虚小 4.37×）。
///
/// 这与 Python 端 vbtcore/detector.py 的一号历史 bug 完全同源
/// （见该文件 docstring「置信度双重 sigmoid」），Python 已修复，
/// C++ 端此前遗留未同步。
///
/// 铁律：row[4] 直接用作概率，禁止再套 sigmoid。
inline float score_as_prob(float x) {
    return x;
}

/// 计算两个框的 IoU。
inline double iou(double cx1, double cy1, double w1, double h1,
                  double cx2, double cy2, double w2, double h2) {
    const double x1a = cx1 - w1 / 2, y1a = cy1 - h1 / 2;
    const double x2a = cx1 + w1 / 2, y2a = cy1 + h1 / 2;
    const double x1b = cx2 - w2 / 2, y1b = cy2 - h2 / 2;
    const double x2b = cx2 + w2 / 2, y2b = cy2 + h2 / 2;
    const double xi1 = std::max(x1a, x1b);
    const double yi1 = std::max(y1a, y1b);
    const double xi2 = std::min(x2a, x2b);
    const double yi2 = std::min(y2a, y2b);
    const double iw = std::max(0.0, xi2 - xi1);
    const double ih = std::max(0.0, yi2 - yi1);
    const double inter = iw * ih;
    const double union_ = w1 * h1 + w2 * h2 - inter;
    return union_ > 0.0 ? inter / union_ : 0.0;
}

/// canvas 坐标 → 原图坐标（letterbox 逆变换，纯 C++ 无需 OpenCV DNN）。
inline void canvas_to_orig_coords(
    double cx_c, double cy_c, double w_c, double h_c,
    double scale, int pad_x, int pad_y,
    double& out_cx, double& out_cy, double& out_w, double& out_h)
{
    out_cx = (cx_c - pad_x) / scale;
    out_cy = (cy_c - pad_y) / scale;
    out_w  = w_c / scale;
    out_h  = h_c / scale;
}

/// 解析单帧 YOLO 输出 (1, N, 5+) →  Detection 列表。
/// 模型格式：YOLOv11/v8 推理输出 (1, num_anchors, 5+num_classes)，
/// 每行 = [cx, cy, w, h, conf, class_probs...]（conf 已是概率，禁止再套 sigmoid）。
std::vector<Detection> parse_yolo_output(
    const float* data, std::size_t num_boxes,
    int box_stride,
    float conf_thresh,
    double scale, int pad_x, int pad_y)
{
    std::vector<Detection> dets;
    dets.reserve(num_boxes);
    for (std::size_t i = 0; i < num_boxes; ++i) {
        const float* row = data + i * box_stride;
        const float obj_conf = score_as_prob(row[4]);
        if (obj_conf < conf_thresh) continue;
        const double cx_c = row[0], cy_c = row[1], w_c = row[2], h_c = row[3];
        double cx, cy, w, h;
        canvas_to_orig_coords(cx_c, cy_c, w_c, h_c, scale, pad_x, pad_y, cx, cy, w, h);
        Detection d; d.cx = cx; d.cy = cy; d.w = w; d.h = h;
        d.conf = obj_conf;
        d.ratio = w / std::max(h, 1e-6);
        dets.push_back(d);
    }
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
    return dets;
}

/// 解析 YOLOv11 输出 (1, 5, num_anchors) 布局（Ultralytics YOLOv11/v8 转置格式）。
/// 性能要点（Daniel 反馈 2026-09-12）：
///   1. 数据在内存中是连续的 5 个通道，各 8400 元素。
///   2. 用指针步长直接寻址（避免 cv::transpose 额外 0.2-0.4ms 内存搬运 + Cache Miss）。
///   3. 单次扫描完成 conf 过滤 + letterbox 逆映射，输出按 conf 降序。
///
/// 耗时：< 0.03ms（vs transpose 路径 0.2-0.4ms）。
std::vector<Detection> parse_yolov11_transposed(
    const float* data, int num_anchors,
    float conf_thresh,
    double scale, int pad_x, int pad_y)
{
    std::vector<Detection> dets;
    dets.reserve(64);
    const float* row_cx    = data + 0 * num_anchors;
    const float* row_cy    = data + 1 * num_anchors;
    const float* row_w     = data + 2 * num_anchors;
    const float* row_h    = data + 3 * num_anchors;
    const float* row_score = data + 4 * num_anchors;

    for (int i = 0; i < num_anchors; ++i) {
        const float score = score_as_prob(row_score[i]);
        if (score < conf_thresh) continue;
        const double cx_c = row_cx[i], cy_c = row_cy[i], w_c = row_w[i], h_c = row_h[i];
        double cx, cy, w, h;
        canvas_to_orig_coords(cx_c, cy_c, w_c, h_c, scale, pad_x, pad_y, cx, cy, w, h);
        Detection d; d.cx = cx; d.cy = cy; d.w = w; d.h = h;
        d.conf = score;
        d.ratio = w / std::max(h, 1e-6);
        dets.push_back(d);
    }
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
    return dets;
}
}  // namespace
#endif  // VBT_HAS_ONNXRUNTIME

// ──────────────────────────────────────────────────────────────────
// PlateDetector Impl
// ──────────────────────────────────────────────────────────────────

struct PlateDetector::Impl {
#if defined(VBT_HAS_ONNXRUNTIME) && VBT_HAS_ONNXRUNTIME
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
#endif

    Impl(const std::string& model_path,
         const std::vector<std::string>& providers)
#if defined(VBT_HAS_ONNXRUNTIME) && VBT_HAS_ONNXRUNTIME
        : env(ORT_LOGGING_LEVEL_WARNING, "vbtcore-cpp")
#endif
    {
#if defined(VBT_HAS_ONNXRUNTIME) && VBT_HAS_ONNXRUNTIME
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(0);  // 0 = auto (all cores)
        // GPU 加速（需要 CUDA toolkit + cuBLAS 库，WSL 需额外安装）
        // 预期路径: LD_LIBRARY_PATH 包含 /usr/local/cuda/lib64
        // #OrtCUDAProviderOptions cuda_opts;
        // #session_options.AppendExecutionProvider_CUDA(cuda_opts);
        session = std::make_unique<Ort::Session>(
            env, model_path.c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;
        const size_t num_inputs = session->GetInputCount();
        const size_t num_outputs = session->GetOutputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = session->GetInputNameAllocated(i, allocator);
            input_names.emplace_back(name.get());
        }
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session->GetOutputNameAllocated(i, allocator);
            output_names.emplace_back(name.get());
        }
        (void)providers;
#else
        (void)model_path;
        (void)providers;
#endif
    }
};

PlateDetector::PlateDetector(const std::string& model_path,
                             const std::vector<std::string>& providers)
    : impl_(new Impl(model_path, providers)) {}

PlateDetector::~PlateDetector() { delete impl_; }

std::vector<Detection> PlateDetector::detect(const cv::Mat& frame_bgr,
                                              double conf_thresh) {
#if defined(VBT_HAS_ONNXRUNTIME) && VBT_HAS_ONNXRUNTIME
    if (frame_bgr.empty() || impl_ == nullptr || impl_->session == nullptr) {
        return {};
    }

    // ── Letterbox 预处理（纯 C++，不依赖 OpenCV DNN）──────────────
    constexpr int INPUT_SIZE = 640;
    const int orig_h = frame_bgr.rows, orig_w = frame_bgr.cols;
    const double scale = std::min(
        static_cast<double>(INPUT_SIZE) / orig_h,
        static_cast<double>(INPUT_SIZE) / orig_w);
    const int canvas_w = static_cast<int>(orig_w * scale);
    const int canvas_h = static_cast<int>(orig_h * scale);
    const int pad_x = (INPUT_SIZE - canvas_w) / 2;
    const int pad_y = (INPUT_SIZE - canvas_h) / 2;

    // resize + pad
    cv::Mat resized, canvas(INPUT_SIZE, INPUT_SIZE, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(frame_bgr, resized, {canvas_w, canvas_h});
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, canvas_w, canvas_h)));

    // BGR→RGB 并 1/255 归一化，手动填充 float blob (1,3,640,640)
    // 等价于 cv::dnn::blobFromImage(canvas, 1.0/255.0)，但不依赖 OpenCV DNN
    std::vector<float> blob_data(INPUT_SIZE * INPUT_SIZE * 3);
    for (int y = 0; y < INPUT_SIZE; ++y) {
        const cv::Vec3b* row = canvas.ptr<cv::Vec3b>(y);
        for (int x = 0; x < INPUT_SIZE; ++x) {
            const float b = row[x][0] * (1.0f / 255.0f);
            const float g = row[x][1] * (1.0f / 255.0f);
            const float r = row[x][2] * (1.0f / 255.0f);
            blob_data[0 * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x] = r;
            blob_data[1 * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x] = g;
            blob_data[2 * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x] = b;
        }
    }
    (void)orig_h; (void)orig_w;  // suppress unused warnings


    // 创建输入 tensor
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, 3, INPUT_SIZE, INPUT_SIZE};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, blob_data.data(), blob_data.size() * sizeof(float),
        input_shape.data(), input_shape.size());

    const char* input_name = impl_->input_names.empty()
        ? "images" : impl_->input_names[0].c_str();
    const char* output_name = impl_->output_names.empty()
        ? "output0" : impl_->output_names[0].c_str();
    auto outputs = impl_->session->Run(
        Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1,
        &output_name, 1);

    if (outputs.empty() || !outputs[0].IsTensor()) {
        return {};
    }
    const float* out_data = outputs[0].GetTensorData<float>();
    const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    // 检测输出张量布局：
    //   YOLOv11 / Ultralytics 转置格式: (1, 5, num_anchors) — 首维=5 (cx, cy, w, h, score)
    //   YOLOv5 原始格式:                (1, num_anchors, 5+nc) — 中维=num_anchors
    if (out_shape.size() == 3) {
        const int64_t a = out_shape[1], b = out_shape[2];
        if (a == 5 || a == 6) {
            // YOLOv11 转置格式 → 指针步长寻址（避免 cv::transpose 额外 0.2-0.4ms 拷贝）
            const int num_anchors = static_cast<int>(b);
            return parse_yolov11_transposed(out_data, num_anchors,
                                            static_cast<float>(conf_thresh),
                                            scale, pad_x, pad_y);
        }
        // YOLOv5 原始格式: (1, num_anchors, 5)
        const std::size_t num_boxes = static_cast<std::size_t>(a);
        const int box_stride = static_cast<int>(b);
        return parse_yolo_output(out_data, num_boxes, box_stride,
                                 static_cast<float>(conf_thresh),
                                 scale, pad_x, pad_y);
    }
    return {};
#else
    (void)frame_bgr;
    (void)conf_thresh;
    return {};  // Stub: no detection when ONNX Runtime unavailable
#endif
}

std::unique_ptr<Detector> make_detector(const std::string& model_path) {
#if defined(VBT_HAS_ONNXRUNTIME) && VBT_HAS_ONNXRUNTIME
    try {
        auto det = std::make_unique<PlateDetector>(model_path);
        return det;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[WARN] PlateDetector init failed, falling back to stub: %s\n", e.what());
        return std::make_unique<StubDetector>();
    } catch (...) {
        std::fprintf(stderr, "[WARN] PlateDetector init failed: unknown exception\n");
        return std::make_unique<StubDetector>();
    }
#else
    (void)model_path;
    return std::make_unique<StubDetector>();
#endif
}

}  // namespace vbt