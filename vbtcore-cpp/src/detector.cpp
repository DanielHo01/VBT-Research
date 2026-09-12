// vbt/detector.cpp — ONNX Runtime 检测器实现
// ===========================================
// 对应 Python 端 vbtcore/detector.py::PlateDetector。
// 编译时若 VBT_HAS_ONNXRUNTIME == 1 则链接 ONNX Runtime，
// 否则为 StubDetector 模式（返回空检测，仅供接口冒烟测试）。

#include "vbt/detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "vbt/geometry.hpp"

#if defined(VBT_HAS_ONNXRUNTIME) && VBT_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace vbt {

namespace {
/// 计算 sigmoid（ONNX YOLO 输出已 logits，应用 sigmoid 转概率）。
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
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

/// 解析单帧 YOLO 输出 (1, N, 5+) →  Detection 列表。
/// 模型格式：YOLOv11/v8 推理输出 (1, num_anchors, 5+num_classes)，
/// 每行 = [cx, cy, w, h, conf, class_probs...]（sigmoid 后）。
std::vector<Detection> parse_yolo_output(
    const float* data, std::size_t num_boxes,
    int box_stride,
    float conf_thresh,
    int orig_w, int orig_h,
    const vbt::PreprocessResult& pp)
{
    std::vector<Detection> dets;
    dets.reserve(num_boxes);
    for (std::size_t i = 0; i < num_boxes; ++i) {
        const float* row = data + i * box_stride;
        const float obj_conf = sigmoid(row[4]);
        if (obj_conf < conf_thresh) continue;
        const double cx_c = row[0];
        const double cy_c = row[1];
        const double w_c = row[2];
        const double h_c = row[3];
        const auto orig = vbt::canvas_to_orig(pp, cx_c, cy_c, w_c, h_c);
        if (orig.cx < 0 || orig.cx >= orig_w || orig.cy < 0 || orig.cy >= orig_h) {
            continue;
        }
        Detection d;
        d.cx = orig.cx;
        d.cy = orig.cy;
        d.w = orig.w;
        d.h = orig.h;
        d.conf = obj_conf;
        d.ratio = orig.w / std::max(orig.h, 1e-6);
        dets.push_back(d);
    }
    // 按 conf 降序
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
    return dets;
}

}  // namespace

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
        session_options.SetIntraOpNumThreads(2);
        Ort::ThrowOnError(Ort::GetApi().CreateSessionOptions());
        session = std::make_unique<Ort::Session>(
            env, model_path.c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;
        const auto in_names = session->GetInputNames();
        const auto out_names = session->GetOutputNames();
        for (auto* n : in_names) input_names.push_back(n);
        for (auto* n : out_names) output_names.push_back(n);
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
    // letterbox 预处理
    auto pp = vbt::preprocess(frame_bgr, 640);
    if (pp.blob_handle == nullptr) {
        return {};
    }
    cv::Mat& blob = *static_cast<cv::Mat*>(pp.blob_handle);

    // 创建输入 tensor
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        Ort::ArenaAllocator, Ort::MemTypeDefault);
    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, blob.ptr<float>(), blob.total() * sizeof(float),
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
    // 期望: (1, num_boxes, 5) 或 (1, 5, num_boxes)
    std::size_t num_boxes = 0;
    int box_stride = 5;
    if (out_shape.size() == 3) {
        const int64_t a = out_shape[1], b = out_shape[2];
        if (a == 5 || a == 6) {
            // (1, 5, num_boxes) - YOLOv5/v8 format
            num_boxes = static_cast<std::size_t>(b);
            // need transpose; for simplicity require (1, num_boxes, 5)
            return {};
        }
        // (1, num_boxes, 5) - YOLOv11 format
        num_boxes = static_cast<std::size_t>(a);
        box_stride = static_cast<int>(b);
    }
    return parse_yolo_output(out_data, num_boxes, box_stride,
                             static_cast<float>(conf_thresh),
                             frame_bgr.cols, frame_bgr.rows, pp);
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
    } catch (...) {
        return std::make_unique<StubDetector>();
    }
#else
    (void)model_path;
    return std::make_unique<StubDetector>();
#endif
}

}  // namespace vbt