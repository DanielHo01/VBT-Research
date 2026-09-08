"""
OWL-ViT 可视化：标注视频帧 + 提取 plate crops
"""
import os, cv2, numpy as np, torch, matplotlib.pyplot as plt
from transformers import OwlViTForObjectDetection, AutoImageProcessor, AutoTokenizer

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", local_files_only=True)
processor = AutoImageProcessor.from_pretrained("google/owlvit-base-patch32", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("google/owlvit-base-patch32", local_files_only=True)
model.eval()

TEXTS = ["a weight plate", "barbell plate", "gym weight",
          "a red plate", "a blue plate", "a yellow plate", "a green plate", "a white plate"]


def detect(frame, score_thresh=0.05, ratio_thresh=1.5):
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_inputs = processor(images=rgb, return_tensors="pt")
    txt_inputs = tokenizer(TEXTS, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(
            pixel_values=img_inputs['pixel_values'],
            input_ids=txt_inputs['input_ids'],
            attention_mask=txt_inputs['attention_mask']
        )
    results = processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([[H, W]]), threshold=score_thresh
    )[0]
    dets = []
    for i in range(len(results['scores'])):
        box = results['boxes'][i].numpy()
        bw, bh = float(box[2]-box[0]), float(box[3]-box[1])
        ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 999
        if ratio <= ratio_thresh:
            dets.append({
                'cx': float(box[0]+bw/2), 'cy': float(box[1]+bh/2),
                'w': bw, 'h': bh,
                'score': float(results['scores'][i]), 'ratio': ratio
            })
    return dets


def nms(dets, iou_thresh=0.4):
    if not dets:
        return dets
    dets = sorted(dets, key=lambda x: -x['score'])
    keep = []
    for d in dets:
        suppressed = False
        for k in keep:
            xi1 = max(d['cx']-d['w']/2, k['cx']-k['w']/2)
            yi1 = max(d['cy']-d['h']/2, k['cy']-k['h']/2)
            xi2 = min(d['cx']+d['w']/2, k['cx']+k['w']/2)
            yi2 = min(d['cy']+d['h']/2, k['cy']+k['h']/2)
            inter = max(0, xi2-xi1) * max(0, yi2-yi1)
            union = d['w']*d['h'] + k['w']*k['h'] - inter
            if inter / (union + 1e-9) > iou_thresh:
                suppressed = True
                break
        if not suppressed:
            keep.append(d)
    return keep


def draw_dets(frame, dets):
    """Draw detections with color coding by confidence."""
    annotated = frame.copy()
    for d in dets:
        x1 = max(0, int(d['cx'] - d['w']/2))
        y1 = max(0, int(d['cy'] - d['h']/2))
        x2 = min(frame.shape[1], int(d['cx'] + d['w']/2))
        y2 = min(frame.shape[0], int(d['cy'] + d['h']/2))

        conf = d['score']
        if conf > 0.08:
            color = (0, 255, 0)     # bright green = high conf
        elif conf > 0.06:
            color = (0, 200, 255)   # yellow = medium
        else:
            color = (0, 130, 255)    # orange = low conf

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = f"{d['score']:.3f}  r={d['ratio']:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (x1, max(0, y1-lh-10)), (x1+lw+6, y1), color, -1)
        cv2.putText(annotated, label, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    return annotated


def annotate_video(vpath, times_s, out_path, title):
    """Create annotated key-frame montage for a video."""
    cap = cv2.VideoCapture(vpath)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for idx, t in enumerate(times_s):
        fi = min(int(t * fps), fc - 1)
        cap = cv2.VideoCapture(vpath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue

        dets = nms(detect(frame))
        ann = draw_dets(frame, dets)
        ax = axes[idx // 3, idx % 3]
        ax.imshow(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
        ax.set_title(f"t={t:.1f}s  frame={fi}  |  {len(dets)} plates detected", fontsize=10)
        ax.axis('off')

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# 1. Annotated montages for two videos
# ============================================================
vpath_20 = r'D:\EasyVBT-Research\validation\dataset_benchmark\raw_videos\20kg_0.87_0.88_0.89_0.91.mp4'
vpath_30 = r'D:\EasyVBT-Research\validation\dataset_benchmark\raw_videos\30kg_1.03_0.89_0.76_0.65.mp4'
out_dir = r'D:\EasyVBT-Research\datasets\owlvit_vis'
os.makedirs(out_dir, exist_ok=True)

print("Generating annotated montages...")

annotate_video(
    vpath_20,
    times_s=[0, 2, 4, 6, 8, 10],
    out_path=os.path.join(out_dir, '20kg_keyframes.jpg'),
    title="OWL-ViT 零样本检测 — 20kg 侧视图\n绿色=高置信(>0.08) 黄色=中(>0.06) 橙色=低(<0.06)"
)

annotate_video(
    vpath_30,
    times_s=[0, 1.5, 3.0, 4.5, 6.0, 7.5],
    out_path=os.path.join(out_dir, '30kg_keyframes.jpg'),
    title="OWL-ViT 零样本检测 — 30kg 侧视图\n绿色=高置信(>0.08) 黄色=中(>0.06) 橙色=低(<0.06)"
)

# ============================================================
# 2. Extract and save large plate crops
# ============================================================
print("\nExtracting large plate crops...")
crop_dir = os.path.join(out_dir, 'crops_big')
os.makedirs(crop_dir, exist_ok=True)

cap = cv2.VideoCapture(vpath_30)
fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Collect all detections across all frames (sample every 3 frames)
all_dets = []
for fi in range(0, fc, 3):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        continue
    dets = nms(detect(frame))
    for d in dets:
        all_dets.append({'frame': frame.copy(), 'fi': fi, **d})
cap.release()

print(f"  Total raw detections: {len(all_dets)}")

# Cluster by position
clusters = []
used = set()
for i, di in enumerate(all_dets):
    if i in used:
        continue
    cl = [i]
    used.add(i)
    for j, dj in enumerate(all_dets):
        if j in used:
            continue
        if np.hypot(di['cx']-dj['cx'], di['cy']-dj['cy']) < 80:
            cl.append(j)
            used.add(j)
    clusters.append(cl)

# Keep stable clusters (detected in >= 3 frames)
stable = []
for cl in clusters:
    ds = [all_dets[i] for i in cl]
    n_frames = len(set(d['fi'] for d in ds))
    if n_frames < 3:
        continue
    best = max(ds, key=lambda x: x['score'])
    stable.append({
        'd': best,
        'n_frames': n_frames,
        'total_dets': len(ds),
        'avg_score': float(np.mean([dd['score'] for dd in ds])),
        'cx_std': float(np.std([dd['cx'] for dd in ds])),
        'cy_std': float(np.std([dd['cy'] for dd in ds])),
    })

stable.sort(key=lambda x: -x['n_frames'])
print(f"  Stable clusters (>=3 frames): {len(stable)}")

for rank, info in enumerate(stable[:8]):
    d = info['d']
    frame = d['frame']

    # Extract crop with padding
    pad = 0.25
    x1 = max(0, int(d['cx'] - d['w']/2 - d['w']*pad))
    y1 = max(0, int(d['cy'] - d['h']/2 - d['h']*pad))
    x2 = min(W, int(d['cx'] + d['w']/2 + d['w']*pad))
    y2 = min(H, int(d['cy'] + d['h']/2 + d['h']*pad))
    crop = frame[y1:y2, x1:x2]

    # Draw bbox on crop (relative coordinates)
    rel_x1 = int(d['cx'] - d['w']/2 - x1)
    rel_y1 = int(d['cy'] - d['h']/2 - y1)
    rel_x2 = int(d['cx'] + d['w']/2 - x1)
    rel_y2 = int(d['cy'] + d['h']/2 - y1)
    cv2.rectangle(crop, (rel_x1, rel_y1), (rel_x2, rel_y2), (0, 255, 0), 2)

    # Label
    label = f"#{rank+1}  n={info['n_frames']}f  score={info['avg_score']:.3f}  r={d['ratio']:.2f}"
    cv2.putText(crop, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Upscale 4x for visibility
    h_c, w_c = crop.shape[:2]
    crop_big = cv2.resize(crop, (w_c*4, h_c*4), interpolation=cv2.INTER_LINEAR)

    fname = f"plate_r{rank+1}_n{info['n_frames']}_score{info['avg_score']:.3f}_cx{d['cx']:.0f}_cy{d['cy']:.0f}.jpg"
    out_crop = os.path.join(crop_dir, fname)
    cv2.imwrite(out_crop, crop_big, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"  {fname}")
    print(f"    cluster: n_frames={info['n_frames']}, score={info['avg_score']:.3f}, "
          f"CX={d['cx']:.0f}±{info['cx_std']:.1f}, CY={d['cy']:.0f}±{info['cy_std']:.1f}, "
          f"crop={crop.shape[1]}x{crop.shape[0]}")

print(f"\nAll outputs: {out_dir}")
