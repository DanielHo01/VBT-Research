// vbt/geometry.cpp — 帧预处理、椭圆拟合、比例尺计算
// ====================================================
// 对应 Python 端 vbtcore/geometry.py。

#include "../include/vbt/geometry.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>

// OpenCV core（cv::Mat / cv::cvtColor / cv::fitEllipse 等）
// imgcodecs/imgproc 在 stub 中已通过 core.hpp 转发，生产环境链接真实 OpenCV。
#include <opencv2/core.hpp>

namespace vbt {

double resolve_plate_diameter(const std::string* outer_plate) {
    if (outer_plate == nullptr || outer_plate->empty()) {
        return kDefaultPlateDiameterM;
    }
    // Lowercase + remove whitespace
    std::string key;
    key.reserve(outer_plate->size());
    for (char c : *outer_plate) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            key.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
    }
    const auto& table = plate_diameters_m();
    auto it = table.find(key);
    if (it == table.end()) {
        return kDefaultPlateDiameterM;
    }
    return it->second;
}

bool fit_plate_ellipse(
    const uint8_t* bgr_data, int width, int height, int row_stride_bytes,
    double crop_cx, double crop_cy, EllipseFit& out)
{
    out = EllipseFit{};
    if (bgr_data == nullptr || width < 5 || height < 5) {
        return false;
    }
    // Wrap raw BGR buffer as cv::Mat (no copy).
    cv::Mat crop(height, width, cv::CV_8UC3,
                 const_cast<uint8_t*>(bgr_data), row_stride_bytes);

    cv::Mat gray, blurred, binary, edges, closed;
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);

    int k = std::max(3, (std::min(width, height) / 16) * 2 + 1);
    cv::GaussianBlur(gray, blurred, cv::Size(k, k), 0);

    cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Canny(blurred, edges, 50, 150);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(edges, closed, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        // Fallback: use binary mask
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    }
    if (contours.empty()) {
        return false;
    }
    // Largest contour by area
    auto best = std::max_element(contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });
    if (cv::contourArea(*best) < 20.0) {
        return false;
    }
    cv::RotatedRect ellipse;
    try {
        ellipse = cv::fitEllipse(*best);
    } catch (const cv::Exception&) {
        return false;
    }
    double major = std::max(ellipse.size.width, ellipse.size.height);
    double minor = std::min(ellipse.size.width, ellipse.size.height);
    if (major < 3.0) {
        return false;
    }
    out.major_axis_px = major;
    out.cx_px = ellipse.center.x;
    out.cy_px = ellipse.center.y;
    out.angle_deg = ellipse.angle;
    (void)minor;  // 当前未使用
    (void)crop_cx;
    (void)crop_cy;
    return true;
}

bool compute_mpp(double major_axis_px, double plate_diameter_m, double& out_mpp) {
    if (major_axis_px < 2.0) {
        out_mpp = 0.0;
        return false;
    }
    out_mpp = plate_diameter_m / major_axis_px;
    return true;
}

PreprocessResult preprocess(const cv::Mat& frame, int size) {
    PreprocessResult result;
    if (frame.empty()) {
        return result;
    }
    const int h = frame.rows;
    const int w = frame.cols;
    result.orig_h = h;
    result.orig_w = w;
    result.rotated = false;  // v5: 恒为 false，纯 letterbox

    const double scale = std::min(static_cast<double>(size) / h,
                                   static_cast<double>(size) / w);
    const int nh = static_cast<int>(h * scale);
    const int nw = static_cast<int>(w * scale);
    const int yo = (size - nh) / 2;
    const int xo = (size - nw) / 2;

    // Letterbox canvas (gray 114, BGR)
    cv::Mat canvas(size, size, cv::CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(nw, nh));

    // Copy resized into canvas ROI（避免 cv::Mat::operator()(Rect) 在 stub 中未实现）
    // 逐像素拷贝（生产代码可换 cv::Mat::copyTo(ROI)）
    for (int y = 0; y < nh; ++y) {
        const uint8_t* src_row = resized.data + y * resized.step;
        uint8_t* dst_row = canvas.data + (yo + y) * canvas.step + xo * 3;
        std::memcpy(dst_row, src_row, nw * 3);
    }

    // Convert to (1, 3, S, S) float32 via blobFromImage
    cv::Mat chw;
    cv::dnn::blobFromImage(canvas, chw, 1.0 / 255.0);

    result.scale = scale;
    result.xo = xo;
    result.yo = yo;
    // Hold blob via opaque handle (caller must not free).
    static thread_local cv::Mat last_blob;
    last_blob = chw;
    result.blob_handle = &last_blob;
    return result;
}

OrigBox canvas_to_orig(const PreprocessResult& pp,
                       double cx_c, double cy_c,
                       double w_c, double h_c)
{
    OrigBox out;
    if (pp.scale <= 0.0) {
        return out;
    }
    out.cx = (cx_c - pp.xo) / pp.scale;
    out.cy = (cy_c - pp.yo) / pp.scale;
    out.w = w_c / pp.scale;
    out.h = h_c / pp.scale;
    return out;
}

}  // namespace vbt