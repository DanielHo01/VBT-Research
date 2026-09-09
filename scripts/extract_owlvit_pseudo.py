"""
OWL-ViT Pseudo-Label Extractor v2 — fixed YOLO label coordinates.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse, time, json, hashlib, sys
from pathlib import Path
import numpy as np
import cv2, torch
from transformers import OwlViTForObjectDetection, AutoImageProcessor, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'validation' / 'dataset_benchmark'))
import config as cfg

VIDEO_DIR = str(cfg.raw_videos_dir())
OUTPUT_DIR = str(cfg.datasets_dir() / 'owlvit_pseudo')
INDEX_PATH = str(cfg.dataset_index_path())
TEXTS = [
    "a weight plate", "barbell plate", "gym weight",
    "a red plate", "a blue plate", "a yellow plate",
    "a green plate", "a white plate"
]
CROP_PADDING = 0.30

_model = _processor = _tokenizer = None


def load_model():
    global _model, _processor, _tokenizer
    if _model is not None:
        return
    t0 = time.time()
    print("  Loading OWL-ViT...")
    _model = OwlViTForObjectDetection.from_pretrained(
        "google/owlvit-base-patch32", local_files_only=True)
    _processor = AutoImageProcessor.from_pretrained(
        "google/owlvit-base-patch32", local_files_only=True)
    _tokenizer = AutoTokenizer.from_pretrained(
        "google/owlvit-base-patch32", local_files_only=True)
    _model.eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s")


def detect(frame, score_thresh=0.05, ratio_thresh=1.5):
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_inputs = _processor(images=rgb, return_tensors="pt")
    txt_inputs = _tokenizer(TEXTS, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(
            pixel_values=img_inputs['pixel_values'],
            input_ids=txt_inputs['input_ids'],
            attention_mask=txt_inputs['attention_mask']
        )
    results = _processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([[H, W]]), threshold=score_thresh
    )[0]
    dets = []
    for i in range(len(results['scores'])):
        box = results['boxes'][i].numpy()
        bw, bh = float(box[2]-box[0]), float(box[3]-box[1])
        ratio = max(bw,bh)/min(bw,bh) if min(bw,bh) > 0 else 999
        cx, cy = float(box[0]+bw/2), float(box[1]+bh/2)
        score = float(results['scores'][i])
        label_idx = int(results['labels'][i])
        label = TEXTS[label_idx] if label_idx < len(TEXTS) else f"idx_{label_idx}"
        if ratio <= ratio_thresh:
            dets.append({'cx':cx,'cy':cy,'w':bw,'h':bh,'score':score,'ratio':ratio,'label':label})
    return dets


def nms(dets, iou_thresh=0.4):
    if not dets:
        return dets
    dets = sorted(dets, key=lambda x: -x['score'])
    keep = []
    for d in dets:
        ok = True
        for k in keep:
            xi1 = max(d['cx']-d['w']/2, k['cx']-k['w']/2)
            yi1 = max(d['cy']-d['h']/2, k['cy']-k['h']/2)
            xi2 = min(d['cx']+d['w']/2, k['cx']+k['w']/2)
            yi2 = min(d['cy']+d['h']/2, k['cy']+k['h']/2)
            inter = max(0, xi2-xi1) * max(0, yi2-yi1)
            union = d['w']*d['h'] + k['w']*k['h'] - inter
            if inter / (union + 1e-9) > iou_thresh:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


def extract_crop_coords(det, pad=CROP_PADDING):
    """Extract crop from full frame. Returns (crop, crop_x1, crop_y1)."""
    H, W = det['frame'].shape[:2]
    pad_x = det['w'] * pad
    pad_y = det['h'] * pad
    x1 = int(max(0, det['cx'] - det['w']/2 - pad_x))
    y1 = int(max(0, det['cy'] - det['h']/2 - pad_y))
    x2 = int(min(W, det['cx'] + det['w']/2 + pad_x))
    y2 = int(min(H, det['cy'] + det['h']/2 + pad_y))
    return det['frame'][y1:y2, x1:x2], x1, y1


def temporal_filter(cap, frame_indices, pos_dist=100, min_dets=3, score_thresh=0.05, ratio_thresh=1.4):
    """
    Run OWL-ViT across sampled frames, return temporally consistent detections.
    Each output: {frame, frame_idx, cx, cy, w, h, score, ratio, label, n_frames}
    """
    all_dets = []
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        dets = nms(detect(frame, score_thresh, ratio_thresh))
        for d in dets:
            all_dets.append({'frame': frame.copy(), 'frame_idx': fi, **d})
    
    if not all_dets:
        return []
    
    # Spatial clustering
    clusters = []
    used = set()
    for i, di in enumerate(all_dets):
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        for j, dj in enumerate(all_dets):
            if j in used:
                continue
            if np.hypot(di['cx']-dj['cx'], di['cy']-dj['cy']) < pos_dist:
                cluster.append(j)
                used.add(j)
        clusters.append(cluster)
    
    stable = []
    for cluster in clusters:
        if len(cluster) < min_dets:
            continue
        dets_in = [all_dets[i] for i in cluster]
        
        # Average position
        avg_cx = np.mean([d['cx'] for d in dets_in])
        avg_cy = np.mean([d['cy'] for d in dets_in])
        avg_w  = np.mean([d['w']  for d in dets_in])
        avg_h  = np.mean([d['h']  for d in dets_in])
        
        # Best-scoring detection → use its frame for crop
        best = max(dets_in, key=lambda x: x['score'])
        frame = best['frame']
        
        # Extract crop in original image coords
        crop, crop_x1, crop_y1 = extract_crop_coords({
            **best, 'cx': avg_cx, 'cy': avg_cy, 'w': avg_w, 'h': avg_h
        })
        
        # Compute YOLO label relative to CROP (not global image)
        # The detection bbox in crop coords:
        det_x1 = avg_cx - avg_w / 2 - crop_x1
        det_y1 = avg_cy - avg_h / 2 - crop_y1
        det_x2 = avg_cx + avg_w / 2 - crop_x1
        det_y2 = avg_cy + avg_h / 2 - crop_y1
        
        # Clip to crop boundaries
        H_c, W_c = crop.shape[:2]
        det_x1 = np.clip(det_x1, 0, W_c)
        det_x2 = np.clip(det_x2, 0, W_c)
        det_y1 = np.clip(det_y1, 0, H_c)
        det_y2 = np.clip(det_y2, 0, H_c)
        
        # Normalize to [0,1] for YOLO format
        cx_norm = ((det_x1 + det_x2) / 2) / W_c
        cy_norm = ((det_y1 + det_y2) / 2) / H_c
        bw_norm = (det_x2 - det_x1) / W_c
        bh_norm = (det_y2 - det_y1) / H_c
        
        stable.append({
            'frame_idx': best['frame_idx'],
            'cx': avg_cx, 'cy': avg_cy, 'w': avg_w, 'h': avg_h,
            'score': float(np.mean([d['score'] for d in dets_in])),
            'ratio': float(np.mean([d['ratio'] for d in dets_in])),
            'label': best['label'],
            'n_frames': len(set(d['frame_idx'] for d in dets_in)),
            'crop': crop,
            'yolo': (cx_norm, cy_norm, bw_norm, bh_norm)
        })
    
    return stable


def save_item(det, out_dir, videoid, idx):
    """Save crop image + YOLO label file."""
    crop = det['crop']
    H, W = crop.shape[:2]
    cx, cy, bw, bh = det['yolo']
    
    # Validate
    assert 0 <= cx <= 1, f"cx={cx} out of [0,1]"
    assert 0 <= cy <= 1, f"cy={cy} out of [0,1]"
    assert 0 < bw <= 1, f"bw={bw} out of (0,1]"
    assert 0 < bh <= 1, f"bh={bh} out of (0,1]"
    
    hash_str = hashlib.md5(
        f"{videoid}_{det['frame_idx']}_{det['cx']:.0f}_{det['cy']:.0f}".encode()
    ).hexdigest()[:8]
    basename = f"owlvit_{videoid}_{det['frame_idx']}_{hash_str}"
    
    img_path = os.path.join(out_dir, 'images', basename + '.jpg')
    label_path = os.path.join(out_dir, 'labels', basename + '.txt')
    
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'labels'), exist_ok=True)
    
    ok = cv2.imwrite(img_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        print(f"  WARNING: failed to write {img_path}")
        return False
    
    with open(label_path, 'w') as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    
    return True


def main():
    global VIDEO_DIR
    ap = argparse.ArgumentParser(description='OWL-ViT 伪标签提取（本地）')
    ap.add_argument('--videos-dir', default=VIDEO_DIR)
    ap.add_argument('--output', default=OUTPUT_DIR)
    ap.add_argument('--index', default=INDEX_PATH)
    ap.add_argument('--max-fps', type=float, default=1.0)
    ap.add_argument('--pos-dist', type=int, default=100)
    ap.add_argument('--min-dets', type=int, default=3)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    VIDEO_DIR = args.videos_dir

    load_model()
    
    with open(args.index) as f:
        dataset = json.load(f)
    if args.limit:
        dataset = dataset[:args.limit]
    
    os.makedirs(os.path.join(args.output, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'labels'), exist_ok=True)
    
    total_crops = 0
    print(f"\nProcessing {len(dataset)} videos at max {args.max_fps} fps...")
    print(f"Output: {args.output}\n")
    
    for vidx, item in enumerate(dataset):
        video_path = os.path.join(VIDEO_DIR, item['video_id'])
        videoid = os.path.splitext(item['video_id'])[0]
        
        cap = cv2.VideoCapture(video_path)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        
        interval = max(1, int(fps / args.max_fps))
        frame_indices = list(range(0, fc, interval))
        
        t0 = time.time()
        cap = cv2.VideoCapture(video_path)
        stable = temporal_filter(cap, frame_indices,
                                 pos_dist=args.pos_dist,
                                 min_dets=args.min_dets)
        cap.release()
        elapsed = time.time() - t0
        
        saved = 0
        for i, det in enumerate(stable):
            if save_item(det, args.output, videoid, i):
                saved += 1
        
        total_crops += saved
        marker = " ✅" if saved > 0 else ""
        print(f"[{vidx+1:2d}/{len(dataset)}] {item['video_id']:<40} "
              f"{fc:4d}fr {saved:2d} crops {elapsed:5.1f}s{marker}")
    
    print(f"\n{'='*60}")
    print(f"Total: {total_crops} crops saved → {args.output}")
    
    # Write data.yaml
    yaml_content = f"path: {args.output.replace(chr(92), '/')}\ntc: 1\nval: images\nnames:\n  0: plate\n"
    with open(os.path.join(args.output, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    print(f"Written: {os.path.join(args.output, 'data.yaml')}")


if __name__ == '__main__':
    main()
