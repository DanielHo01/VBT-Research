# VBT-Research

Vertical Bar tracking research — a computer vision pipeline for estimating barbell velocity and repetition count from smartphone videos of squats.

## Project Status

**Phase 1 ✅** — Baseline model (`barbell_v4.onnx`) achieves RMSE = 0.88 on MCV prediction across 34 validation videos.

**Phase 1.5 ❌** — Attempted OWL-ViT zero-shot pseudo-labeling; training produced a worse model (RMSE = 2.20). [Lessons learned](docs/TRAINING.md)

**Phase 2 🔲** — Not yet started.

## Quick Start

```bash
# Run full benchmark (34 videos)
python3 scripts/run_full_benchmark.py

# Run interactive labeling tool (OWL-ViT suggests, you click to accept/reject)
python3 scripts/interactive_label.py <video_file.mp4>

# Extract pseudo-labels via OWL-ViT
python3 scripts/extract_owlvit_pseudo.py

# Download the 34 validation videos (98 MB) from GitHub Releases
gh release download v1.0-videos --pattern "raw_videos.zip" --dir .
unzip raw_videos.zip   # produces validation/dataset_benchmark/raw_videos/
```

## Validation Dataset

The 34 benchmark videos (`validation/dataset_benchmark/raw_videos/`) are **not** tracked in git due to size.

Download them from the GitHub Release:

- **[Release `v1.0-videos`](https://github.com/DanielHo01/VBT-Research/releases/tag/v1.0-videos)** — `raw_videos.zip` (98 MB)

After downloading, verify integrity:

```bash
python3 -c "
import json, os
idx = json.load(open('validation/dataset_benchmark/dataset_index.json'))
files = {f for f in os.listdir('validation/dataset_benchmark/raw_videos') if f.endswith('.mp4')}
assert len(idx) == 34 and {v['video_id'] for v in idx} == files
print('OK: 34 videos match dataset_index.json')
"
```

## Project Structure

```
docs/                  # Architecture, dataset, training, roadmap
models/                # Trained ONNX models
  barbell_v4.onnx      ← Production model (RMSE=0.88)
scripts/               # Core scripts
  run_full_benchmark.py
  interactive_label.py
  extract_owlvit_pseudo.py
  verify_labels.py
validation/            # 34 benchmark videos + ground truth
                    # (videos NOT in git — download from Releases)
```

## Models

| Model | RMSE (MCV) | Notes |
| ------- | ------------- | ------- |
| `barbell_v4.onnx` | **0.88** | Best — production use |
| `yolo11_plate.onnx` | ? | Untested — needs benchmark |
| `plate_v1.onnx` | ~1.4 | Poor on side-view |

## Key Findings

- barbell_v4 generalizes well across weight loads (20–50 kg) and body weights
- Side-view (lateral camera angle) is harder than top-view for plate detection
- OWL-ViT zero-shot detection finds plates but with low confidence (0.05–0.08); pseudo-labels from it are too noisy to train a better model
- Manual labeling with OWL-ViT as a suggestion engine is the most practical path forward

## Requirements

```
opencv-python>=4.10
numpy>=1.26
scipy>=1.13
matplotlib>=3.9
ultralytics>=8.2
transformers>=4.57
onnxruntime>=1.19
pandas>=2.2
```

See `requirements.txt` for full pinned versions.
