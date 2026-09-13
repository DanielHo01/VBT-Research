#include <iostream>
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include "vbt/detector.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: diag_pure_detect <video> <model> [conf_thresh]\n";
        return 1;
    }
    const std::string video_path = argv[1];
    const std::string model_path = argv[2];
    const double conf_thresh = (argc >= 4) ? std::atof(argv[3]) : 0.35;

    auto detector = vbt::make_detector(model_path);
    if (detector->is_stub()) {
        std::cerr << "ERROR: detector is STUB (ONNX Runtime not linked)\n";
        return 1;
    }

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: cannot open " << video_path << "\n";
        return 1;
    }

    const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const double fps = cap.get(cv::CAP_PROP_FPS);

    int frame_idx = 0;
    int detect_count = 0;
    int nan_count = 0;
    int empty_count = 0;
    double first_cy = -1.0;
    double last_cy = -1.0;
    double min_cy = 1e9, max_cy = -1e9;

    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;
        auto dets = detector->detect(frame, conf_thresh);
        if (dets.empty()) {
            empty_count++;
        } else {
            auto best = std::max_element(dets.begin(), dets.end(),
                [](const vbt::Detection& a, const vbt::Detection& b) {
                    return a.w * a.h < b.w * b.h;
                });
            double cy = best->cy;
            if (std::isnan(cy) || std::isinf(cy)) {
                nan_count++;
            } else {
                detect_count++;
                if (first_cy < 0) first_cy = cy;
                last_cy = cy;
                if (cy < min_cy) min_cy = cy;
                if (cy > max_cy) max_cy = cy;
            }
        }
        frame_idx++;
    }
    cap.release();

    std::cout << "=== Pure Detection Baseline ===\n";
    std::cout << "Video:       " << video_path << "\n";
    std::cout << "Total frames: " << frame_idx << "\n";
    std::cout << "FPS:         " << fps << "\n";
    std::cout << "Conf thresh: " << conf_thresh << "\n";
    std::cout << "Detect (OK): " << detect_count << " (" << (100.0 * detect_count / frame_idx) << "%)\n";
    std::cout << "Empty (miss):" << empty_count << " (" << (100.0 * empty_count / frame_idx) << "%)\n";
    std::cout << "NaN:         " << nan_count << "\n";
    std::cout << "cy range:    [" << min_cy << ", " << max_cy << "] px\n";
    std::cout << "cy span:     " << (max_cy - min_cy) << " px\n";
    std::cout << "First cy:    " << first_cy << "\n";
    std::cout << "Last cy:     " << last_cy << "\n";

    if (detect_count > 0) {
        double detect_rate = 100.0 * detect_count / frame_idx;
        if (nan_count == 0 && detect_rate >= 95.0) {
            std::cout << "\n>>> PASS: zero NaN, detect rate >= 95% <<<\n";
        } else if (nan_count > 0) {
            std::cout << "\n>>> FAIL: " << nan_count << " NaN frames <<<\n";
        } else {
            std::cout << "\n>>> WARN: detect rate " << detect_rate << "% < 95% <<<\n";
        }
    } else {
        std::cout << "\n>>> FAIL: zero detections <<<\n";
    }

    return 0;
}
