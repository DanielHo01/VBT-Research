// Minimal OpenCV stub for LSP validation ONLY.
// Real OpenCV is linked at build time via CMake find_package.
// This file exists ONLY so clangd / clang-tidy can validate vbtcore-cpp structure
// without requiring a full OpenCV install. DO NOT compile against this stub.

#pragma once
#include <cstddef>
#include <vector>

namespace cv {

using uchar = unsigned char;

class Scalar {
public:
    Scalar(double v0 = 0, double v1 = 0, double v2 = 0, double v3 = 0);
};

class Point {
public:
    Point() = default;
    Point(int x_, int y_);
    int x = 0, y = 0;
};

class Point2f {
public:
    Point2f() = default;
    Point2f(float x_, float y_);
    float x = 0, y = 0;
};

class Size2f {
public:
    Size2f() = default;
    Size2f(float w, float h);
    float width = 0, height = 0;
};

class Size {
public:
    Size() = default;
    Size(int w, int h);
    int width = 0, height = 0;
};

class Rect {
public:
    Rect() = default;
    Rect(int x_, int y_, int w, int h);
    int x = 0, y = 0, width = 0, height = 0;
};

class RotatedRect {
public:
    RotatedRect() = default;
    cv::Point2f center;
    cv::Size2f size;
    float angle = 0.0f;
};

class Mat {
public:
    Mat() = default;
    Mat(int rows, int cols, int type);
    Mat(int rows, int cols, int type, const Scalar& s);
    Mat(int rows, int cols, int type, void* data, std::size_t step = 0);
    Mat(const Mat& m);
    Mat& operator=(const Mat& m);
    ~Mat();
    bool empty() const;
    int rows = 0;
    int cols = 0;
    int type() const;
    std::size_t step = 0;
    uchar* data = nullptr;

    template <typename T> T* ptr() { return reinterpret_cast<T*>(data); }
    std::size_t total() const;
};

class Exception : public std::exception {
public:
    const char* what() const noexcept override;
};

// Type constants
constexpr int CV_8UC3 = 16;
constexpr int CV_32F = 5;

// Color conversion codes
constexpr int COLOR_BGR2GRAY = 8;

// Threshold types
constexpr int THRESH_BINARY = 0;
constexpr int THRESH_OTSU = 8;

// Morphology operations
constexpr int MORPH_ELLIPSE = 2;
constexpr int MORPH_CLOSE = 3;

// Contour retrieval modes
constexpr int RETR_EXTERNAL = 0;

// Contour approximation methods
constexpr int CHAIN_APPROX_SIMPLE = 1;
constexpr int CHAIN_APPROX_NONE = 2;

// Functions (declarations only; not implemented in stub)
void cvtColor(const Mat& src, Mat& dst, int code);
void GaussianBlur(const Mat& src, Mat& dst, const Size& ksize, double sigmaX);
void threshold(const Mat& src, Mat& dst, double thresh, double maxval, int type);
void Canny(const Mat& src, Mat& dst, double threshold1, double threshold2);
Mat getStructuringElement(int shape, const Size& ksize);
void morphologyEx(const Mat& src, Mat& dst, int op, const Mat& kernel);
void findContours(const Mat& src, std::vector<std::vector<Point>>& contours,
                  int mode, int method);
double contourArea(const std::vector<Point>& contour);
RotatedRect fitEllipse(const std::vector<Point>& points);
void resize(const Mat& src, Mat& dst, const Size& dsize);

namespace dnn {
void blobFromImage(const Mat& src, Mat& dst, double scalefactor = 1.0);
}

}  // namespace cv