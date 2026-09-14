// segmenter_parity.cpp — Python/C++ 分段器数值对拍工具
//
// 用途：绕开完整流水线（无需 OpenCV / ONNX Runtime），直接把一条已导出的
// 帧级轨迹 CSV 喂给 C++ 分段器，与 Python 端输出逐位比对。
// 这是在无法构建全链路时验证两端语义一致性的最小手段。
//
// 输入 CSV 由 scripts/diagnose_missed_reps.py --dump-csv 生成，
// 列格式：idx,t_s,y_m,v_mps,det_h_px,det_conf
//
// 构建（只需 g++，不依赖 OpenCV）：
//   g++ -std=c++17 -O2 -Ivbtcore-cpp/include \
//       vbtcore-cpp/tools/segmenter_parity.cpp vbtcore-cpp/src/segmenter.cpp \
//       -o /tmp/segmenter_parity
//
// 运行：
//   /tmp/segmenter_parity validation/reports/diagnostics/<id>_trajectory.csv
//
// 对照 Python：
//   python - <<'PY'
//   import csv, numpy as np
//   from vbtcore.segmenter import BiomechanicalRepSegmenter
//   rows = list(csv.DictReader(open(PATH)))
//   t = np.array([float(r['t_s']) for r in rows])
//   y = np.array([float(r['y_m']) for r in rows])
//   v = np.array([float(r['v_mps']) for r in rows])
//   print([r.mcv_mps for r in BiomechanicalRepSegmenter().segment(t, y, v)])
//   PY
//
// 预期：两端 rep 数与每个 MCV 逐位相同。不同即说明语义已漂移 ——
// 修一端分段器必须同步另一端，否则 Iron Gate 会掉点。

#include <cstdio>
#include <vector>
#include "vbt/segmenter.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <trajectory.csv>\n", argv[0]);
        return 2;
    }
    std::FILE* f = std::fopen(argv[1], "r");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }

    char line[512];
    std::vector<double> t, y, v;
    if (std::fgets(line, sizeof line, f) == nullptr) {  // 跳过表头
        std::fclose(f);
        return 2;
    }
    while (std::fgets(line, sizeof line, f) != nullptr) {
        double idx, tt, yy, vv;
        if (std::sscanf(line, "%lf,%lf,%lf,%lf", &idx, &tt, &yy, &vv) == 4) {
            t.push_back(tt);
            y.push_back(yy);
            v.push_back(vv);
        }
    }
    std::fclose(f);

    // 与 vbtcore/segmenter.py 默认值保持一致
    vbt::BiomechanicalRepSegmenter seg("squat_bench", 0.12, 0.20, 0.08, 0.02);
    const auto reps = seg.segment(t.data(), y.data(), v.data(), t.size());

    std::printf("%zu reps:", reps.size());
    for (const auto& r : reps) std::printf(" %.3f", r.mcv_mps);
    std::printf("\n");
    return 0;
}
