// Minimal OpenCV videoio stub for LSP validation ONLY.
// Real OpenCV is linked at build time.

#pragma once
#include "core.hpp"

namespace cv {

class VideoCapture {
public:
    VideoCapture() = default;
    VideoCapture(const std::string& filename);
    VideoCapture(int index);
    ~VideoCapture();
    bool open(const std::string& filename);
    bool isOpened() const;
    bool read(Mat& frame);
    void release();
    double get(int prop_id) const;
    bool set(int prop_id, double value);
};

// VideoCapture properties
constexpr int CAP_PROP_FPS = 5;
constexpr int CAP_PROP_FRAME_COUNT = 7;
constexpr int CAP_PROP_POS_MSEC = 0;

}  // namespace cv