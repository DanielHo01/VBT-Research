#pragma once
// vbt/geometry.hpp — 帧预处理、椭圆拟合、比例尺计算
// ====================================================
// 对应 Python 端 vbtcore/geometry.py（PLATE_DIAMETERS_M / fit_plate_ellipse /
// compute_mpp / resolve_plate_diameter / preprocess / canvas_to_orig）

#include <cstdint>
#include <string>
#include <unordered_map>

namespace cv {
class Mat;  // 前置声明，避免本头文件引入完整 OpenCV
}

namespace vbt {

/// 杠铃片外径查表（米）。
inline const std::unordered_map<std::string, double>& plate_diameters_m() {
    static const std::unordered_map<std::string, double> table = {
        {"45lb", 0.450}, {"25kg", 0.450}, {"20kg", 0.450},
        {"35lb", 0.420}, {"25lb", 0.400}, {"15kg", 0.380},
        {"10lb", 0.280}, {"10kg", 0.320}};
    return table;
}

/// 默认外径（bumper 假设）：0.45 m。
constexpr double kDefaultPlateDiameterM = 0.45;

/// 解析外层片规格 → 物理直径（米）。
/// 未知规格 / nullptr 返回默认值。
double resolve_plate_diameter(const std::string* outer_plate);

/// 亚像素椭圆拟合结果（major_axis_px == 0 表示拟合失败）。
struct EllipseFit {
    double major_axis_px = 0.0;  ///< 长轴像素（对应物理直径方向）
    double cy_px = 0.0;          ///< 圆心 y（相对裁剪）
    double cx_px = 0.0;          ///< 圆心 x（相对裁剪）
    double angle_deg = 0.0;      ///< 长轴角度
};

/// 在铃片裁剪上做亚像素椭圆拟合（OpenCV::fitEllipse）。
/// 拟合失败返回 false，out.major_axis_px == 0。
bool fit_plate_ellipse(
    const uint8_t* bgr_data, int width, int height, int row_stride_bytes,
    double crop_cx, double crop_cy, EllipseFit& out);

/// 计算米/像素比例尺。major_axis_px < 2 返回 false。
bool compute_mpp(double major_axis_px, double plate_diameter_m, double& out_mpp);

/// Letterbox 预处理结果。
struct PreprocessResult {
    void* blob_handle = nullptr;  ///< 指向 cv::Mat 的不透明指针
    double scale = 0.0;           ///< canvas → 原图缩放
    int xo = 0;                   ///< canvas x 偏移
    int yo = 0;                   ///< canvas y 偏移
    bool rotated = false;         ///< 是否旋转（v5: 恒为 false）
    int orig_h = 0;
    int orig_w = 0;
};

/// YOLO letterbox 预处理（与 Python 端 preprocess() 对齐）。
PreprocessResult preprocess(const cv::Mat& frame, int size = 640);

/// canvas 坐标 → 原图坐标（letterbox 分支）。
struct OrigBox {
    double cx = 0.0, cy = 0.0, w = 0.0, h = 0.0;
};
OrigBox canvas_to_orig(const PreprocessResult& pp,
                       double cx_c, double cy_c,
                       double w_c, double h_c);

}  // namespace vbt