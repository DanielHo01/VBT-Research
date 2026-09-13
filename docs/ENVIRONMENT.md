# VBT-Research 本地环境配置

> 一次性配置文档。模型已在 `models/best.onnx` / `models/best.pt`。

## 1. 系统信息

| 项目 | 版本/路径 |
| --- | --- |
| Python | 3.12.12（位于 `D:\.venv`） |
| pip | 25.3 |
| OS | Windows 11（NTFS） |
| 模型 | `D:\EasyVBT-Research\models\best.onnx`（10.1 MB）<br>`D:\EasyVBT-Research\models\best.pt`（5.2 MB） |
| 视频 | `validation/dataset_benchmark/raw_videos/`（34 个 mp4） |

## 2. 关键 Python 依赖

| 包 | 版本 | 验证 |
| --- | --- | --- |
| onnxruntime | 1.30.0 | ✅ 模型推理 OK（YOLOv11 输入 [1,3,640,640] → 输出 [1,5,8400]） |
| opencv-python | 5.0.0 | ✅ 视频解码 OK |
| numpy | 2.3.5 | ✅ 数组操作 |
| scipy | 1.17.1 | ✅ Savitzky-Golay 滤波 |
| ultralytics | 8.4.138 | ✅ YOLO 工具链 |
| filterpy | 1.4.5 | ✅ Kalman 滤波（已装） |
| tqdm | 4.67.1 | ✅ 进度条 |

## 3. 验证步骤

### 3.1 单元测试

```bash
cd D:\EasyVBT-Research
python tests/run_all_tests.py
# 期望：26/26 tests passed
```

### 3.2 模型加载

```bash
python -c "
import onnxruntime as ort
sess = ort.InferenceSession('models/best.onnx', providers=['CPUExecutionProvider'])
print('Input:', sess.get_inputs()[0].name, sess.get_inputs()[0].shape)
print('Output:', sess.get_outputs()[0].name, sess.get_outputs()[0].shape)
"
# 期望：Input: images [1, 3, 640, 640] / Output: output0 [1, 5, 8400]
```

### 3.3 端到端烟雾测试

```bash
python -c "
from vbtcore import analyze_video
r = analyze_video(
    'validation/dataset_benchmark/raw_videos/30kg_1.03_0.89_0.76_0.65.mp4',
    'models/best.onnx',
    exercise_type='squat_bench',
)
print(f'Status: {r.status}, Reps: {len(r.reps)}, ms/frame: {r.diagnostics[\"ms_per_frame\"]}')
"
# 期望：Status: OK, Reps: 4, ms/frame: ~14
```

实测（2026-09-12）：

```
Status: OK, Reps: 4, ms/frame: 14.5
Rep 1: MCV=0.731 m/s, ROM=0.633 m, dur=0.87s
Rep 2: MCV=0.971 m/s, ROM=0.550 m, dur=0.57s
Rep 3: MCV=0.996 m/s, ROM=0.631 m, dur=0.63s
```

## 4. 已知小问题

- **CUDA Provider 警告**：`Specified provider 'CUDAExecutionProvider' is not in available provider names`。
  - 原因：本机无 CUDA，自动 fallback 到 CPU。功能不受影响。
  - 解决（可选）：安装 `onnxruntime-gpu` + CUDA 12.x。

## 5. 故障排查

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'X'` | 依赖缺失 | `pip install X` |
| `onnxruntime 加载慢` | 首次调用 JIT 编译 | 正常现象，第二次调用会快 |
| `mp4 解码失败` | HEVC 编码 | 重新转码为 H.264 |
| `best.onnx 不可读` | 路径错误 | `ls -la models/best.onnx` 确认 |

## 6. 后续构建（C++ 端 / Android）

详见 `docs/DEMO_GUIDE.md`：

- C++ 中间层：`vbtcore-cpp/CMakeLists.txt`（需 OpenCV + nlohmann/json + ONNX Runtime）
- Android Demo：`android/app/build.gradle.kts`（需 NDK r25c + 真机）

当前 **Python 端已就绪**，可直接运行 `scripts/run_benchmark_v0.py` 出基准报告。

---

*配置日期：2026-09-12 | 配置人：AI 助手（受 Daniel 委托）*
