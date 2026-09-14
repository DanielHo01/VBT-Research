# 环境配置

> 跨平台（Linux / WSL2 / macOS / Windows）。所有路径均相对仓库根目录，
> 不要在代码或文档中写死 `D:\...` 之类的绝对路径。

---

## 1. Python 环境

### 1.1 最小依赖（单测 + 基准）

单元测试与基准脚本只需要四个包，**不需要** ultralytics / torch 等重型依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install opencv-python-headless numpy scipy onnxruntime
```

> **注意（PEP 668）**：Debian/Ubuntu 的系统 Python 是 externally-managed，
> 直接 `pip install` 会被拒绝。必须用虚拟环境（如上），
> 或显式加 `--break-system-packages`（不推荐）。

### 1.2 完整依赖（含训练/标注工具链）

```bash
pip install -r requirements.txt
```

### 1.3 版本一致性要求 ⚠️

**Python 端与 C++ 端必须使用同一大版本的 OpenCV。**

`goodFeaturesToTrack`（Shi-Tomasi 角点）与 `calcOpticalFlowPyrLK` 的实现
在 OpenCV 4 → 5 之间有差异，会导致两端特征点集合不同、LK 观测序列分叉，
Iron Gate 因此无法对齐（实测：其他条件相同，仅版本不同就使闸门从预期
20+/34 掉到 17/34）。

```bash
# 若 C++ 端用 OpenCV 4.10，Python 端应对齐：
pip install opencv-python-headless==4.10.0.84
```

---

## 2. 验证步骤

### 2.1 单元测试

```bash
python3 tests/run_all_tests.py
# 期望：26/26 tests passed
```

### 2.2 模型加载

```bash
python3 -c "
import onnxruntime as ort
s = ort.InferenceSession('models/best.onnx', providers=['CPUExecutionProvider'])
print('Input :', s.get_inputs()[0].name, s.get_inputs()[0].shape)
print('Output:', s.get_outputs()[0].name, s.get_outputs()[0].shape)
"
# 期望：Input: images [1,3,640,640] / Output: output0 [1,5,8400]
```

### 2.3 端到端冒烟（~74 秒）

```bash
python3 scripts/gen_cpp_baseline.py --smoke --workers 2
```

覆盖 5 个代表性场景：20kg 空杆（应正确拒绝）、30kg 轻片、80kg 中片、
110kg 多片、140kg 极限负荷。

### 2.4 物理校验基准

`110kg_0.71_0.73.mp4`（GymAware 真值 0.71 / 0.73 m/s）应稳定给出：

| 量 | 期望值 |
| --- | --- |
| mpp | ≈ 0.002471 |
| y_range | ≈ 0.664 m |
| MCV | ≈ [0.705, 0.762] |

**任何改动后此三项若偏离，说明几何骨架被破坏，应立即回滚。**

---

## 3. C++ 桌面端环境

### 3.1 标准路径（apt 可用）

```bash
sudo apt update
sudo apt install -y cmake ninja-build g++ libopencv-dev nlohmann-json3-dev pkg-config
./scripts/build_cpp_desktop.sh      # 自动下载 ONNX Runtime 并跑 Iron Gate
```

### 3.2 降级路径（apt 不可用，如受限沙箱）

2026-09-13 在 apt 源不可达的环境下验证过的自建流程：

| 组件 | 获取方式 |
| --- | --- |
| cmake / ninja | `pip install cmake ninja` |
| pkg-config | `pip install pkgconf`（OpenCV 检测 FFmpeg 时必需） |
| FFmpeg 6.1.2 | 源码编译，`--disable-x86asm`，仅启用 h264/hevc/mp4 |
| OpenCV 4.10.0 | 源码编译，`BUILD_LIST=core,imgproc,imgcodecs,videoio,video,dnn` |
| ONNX Runtime | 头文件取自 v1.17.1 源码 tarball；`libonnxruntime.so` 取自 PyPI wheel |

关键点：
- OpenCV 必须检测到 **FFMPEG: YES**，否则 `videoio` 无法解码 mp4，
  golden_test 会读到 0 帧；
- CMake ≥ 4 时需 `-DOPENCV_GENERATE_PKGCONFIG=OFF`（`.pc` 生成器与 CMake 4 不兼容）；
- ONNX Runtime 需要 `libonnxruntime.so.1` 软链接。

### 3.3 ASan 泄漏抑制

`libonnxruntime` 自身有约 318 B 的静态分配未释放，属第三方库行为。
仓库已提供 `scripts/lsan.supp`：

```bash
export LSAN_OPTIONS=suppressions=scripts/lsan.supp
```

---

## 4. 性能参考

| 环境 | stride=1 单视频耗时 | 34 视频全量 |
| --- | --- | --- |
| 2 vCPU 沙箱（2 进程） | 10–300 s | ~34.5 min |
| 8 核桌面（4 进程，预估） | — | ~3 min |

提速手段：
- `--workers N` 多进程并发（每进程复用 ONNX session）
- `--copy-to-tmp` 把视频复制到 Linux 原生盘，消除 WSL2 跨盘 9P 协议开销

---

## 5. 常见问题

| 症状 | 原因与处理 |
| --- | --- |
| `ModuleNotFoundError: No module named 'cv2'` | 未激活 venv，或系统 Python 被 PEP 668 拦截 |
| golden_test 全部 `NO_BASELINE` | baseline 目录为空，先跑 `gen_cpp_baseline.py` |
| golden_test 读到 0 帧 | OpenCV 未启用 FFMPEG，检查 CMake 配置输出 |
| `ASan runtime does not come first` | 不要用 `stdbuf` 等包装器启动 ASan 程序 |
| Iron Gate 卡在 17/34 左右 | 检查两端 OpenCV 版本是否一致（见 1.3） |
