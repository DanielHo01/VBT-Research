"""
OWL-ViT Interactive Labeler v2
=============================
Two modes:
  --preview  : Save annotated frames and crops, no GUI needed
  --label   : Interactive GUI labeling (default if no args)

Preview mode:
  python3 interactive_label.py --preview [video.mp4]

GUI mode:
  python3 interactive_label.py 20kg_0.87_0.88_0.89_0.91.mp4
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import cv2, numpy as np, torch, json, time, argparse
from transformers import OwlViTForObjectDetection, AutoImageProcessor, AutoTokenizer

VIDEO_DIR = r'D:\EasyVBT-Research\validation\dataset_benchmark\raw_videos'
OUTPUT_DIR = r'D:\EasyVBT-Research\datasets\interactive_labels'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEXTS = [
    "a weight plate", "barbell plate", "gym weight",
    "a red plate", "a blue plate", "a yellow plate",
    "a green plate", "a white plate"
]

print("Loading OWL-ViT...", flush=True)
_model = OwlViTForObjectDetection.from_pretrained(
    "google/owlvit-base-patch32", local_files_only=True)
_processor = AutoImageProcessor.from_pretrained(
    "google/owlvit-base-patch32", local_files_only=True)
_tokenizer = AutoTokenizer.from_pretrained(
    "google/owlvit-base-patch32", local_files_only=True)
_model.eval()
print("Ready!", flush=True)


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
        ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 999
        if ratio <= ratio_thresh:
            dets.append({
                'cx': float(box[0]+bw/2), 'cy': float(box[1]+bh/2),
                'w': bw, 'h': bh,
                'score': float(results['scores'][i]),
                'ratio': ratio,
                'accepted': None,
            })
    return dets


def draw_frame(frame, dets, frame_idx, total_frames, videoname, scale=1.0):
    H, W = frame.shape[:2]
    display = cv2.resize(frame, (int(W*scale), int(H*scale)))

    for i, d in enumerate(dets):
        cx_s = d['cx'] * scale; cy_s = d['cy'] * scale
        w_s  = d['w']  * scale; h_s  = d['h']  * scale
        x1 = int(cx_s - w_s/2); y1 = int(cy_s - h_s/2)
        x2 = int(cx_s + w_s/2); y2 = int(cy_s + h_s/2)

        if d['accepted'] is True:
            color = (0, 255, 0); thick = 4
        elif d['accepted'] is False:
            color = (0, 0, 255); thick = 2
        else:
            color = (0, 200, 255); thick = 2

        cv2.rectangle(display, (x1,y1), (x2,y2), color, thick)
        label = f"#{i+1}  score={d['score']:.3f}  r={d['ratio']:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, max(0.4, 0.5*scale), 2)
        cv2.rectangle(display, (x1, max(0,y1-lh-10)), (x1+lw+6, y1), color, -1)
        cv2.putText(display, label, (x1+3, y1-4),
                     cv2.FONT_HERSHEY_SIMPLEX, max(0.4, 0.5*scale), (0,0,0), 2)

    # Header
    bar_h = int(60 * scale)
    cv2.rectangle(display, (0,0), (display.shape[1], bar_h), (20,20,20), -1)
    accepted = sum(1 for d in dets if d['accepted'] is True)
    rejected = sum(1 for d in dets if d['accepted'] is False)
    pending = len(dets) - accepted - rejected
    header = (
        f"{videoname}  |  Frame {frame_idx+1}/{total_frames}  "
        f"|  ACCEPTED={accepted}  REJECTED={rejected}  PENDING={pending}  "
        f"|  [click]=toggle  [A]=accept all  [R]=reject all  [S]=save  [ESC]=quit"
    )
    cv2.putText(display, header[:130], (int(8*scale), int(40*scale)),
                 cv2.FONT_HERSHEY_SIMPLEX, max(0.4, 0.45*scale), (220,220,220), 1)
    return display


# ── GUI Mode ──────────────────────────────────────────────────────────────
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        scale = param['scale']
        for i, d in enumerate(param['dets']):
            cx_s = d['cx'] * scale; cy_s = d['cy'] * scale
            w_s  = d['w']  * scale; h_s  = d['h']  * scale
            if (abs(x - cx_s) < w_s/2 + 10 and abs(y - cy_s) < h_s/2 + 10):
                if d['accepted'] is None:   d['accepted'] = True
                elif d['accepted'] is True: d['accepted'] = False
                else:                         d['accepted'] = None
                param['dirty'] = True
                break


def label_video(videopath, videoname, frame_step=20, max_frames=30):
    cap = cv2.VideoCapture(videopath)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    frame_indices = list(range(0, fc, frame_step))[:max_frames]
    print(f"Video: {videoname}  {fc} frames  sampling {len(frame_indices)} frames")

    print("Running OWL-ViT detection...", flush=True)
    t0 = time.time()
    video_data = []
    for fi in frame_indices:
        cap = cv2.VideoCapture(videopath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        cap.release()
        if not ret: continue
        dets = detect(frame)
        video_data.append({'frame_idx': fi, 'frame': frame, 'dets': dets})
    print(f"Detection done in {time.time()-t0:.1f}s")

    cv2.namedWindow('OWL-ViT Labeler', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('OWL-ViT Labeler', 1280, 720)

    win_w, win_h = 1280, 720
    scale = min(win_w / W, win_h / H)

    state = {'idx': 0, 'dirty': True}
    cv2.setMouseCallback('OWL-ViT Labeler', on_mouse,
                         {'dets': [], 'dirty': state, 'scale': scale})

    while True:
        item = video_data[state['idx']]
        display = draw_frame(item['frame'], item['dets'],
                            state['idx'], len(video_data), videoname, scale)
        cv2.imshow('OWL-ViT Labeler', display)
        state['dirty'] = False

        key = cv2.waitKey(0) & 0xFF
        if key == 27: break
        elif key == ord(' '):
            if state['idx'] < len(video_data)-1:
                state['idx'] += 1; state['dirty'] = True
        elif key == ord('b') or key == ord('B'):
            if state['idx'] > 0:
                state['idx'] -= 1; state['dirty'] = True
        elif key == ord('a') or key == ord('A'):
            for d in item['dets']: d['accepted'] = True
            state['dirty'] = True
        elif key == ord('r') or key == ord('R'):
            for d in item['dets']: d['accepted'] = False
            state['dirty'] = True
        elif key == ord('s') or key == ord('S'):
            save_labels(video_data, videoname, H, W)
        elif 49 <= key <= 57:  # 1-9
            i = key - 49
            if i < len(item['dets']):
                d = item['dets'][i]
                d['accepted'] = not d['accepted'] if d['accepted'] is not None else True
                state['dirty'] = True

    cv2.destroyAllWindows()
    return video_data


def save_labels(video_data, videoname, H, W):
    out_img = os.path.join(OUTPUT_DIR, 'images')
    out_lbl = os.path.join(OUTPUT_DIR, 'labels')
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    n = 0
    for item in video_data:
        frame_idx = item['frame_idx']
        frame = item['frame']
        for i, d in enumerate(item['dets']):
            if d['accepted'] is not True:
                continue
            pad = 0.3
            x1 = max(0, int(d['cx'] - d['w']/2 - d['w']*pad))
            y1 = max(0, int(d['cy'] - d['h']/2 - d['h']*pad))
            x2 = min(W, int(d['cx'] + d['w']/2 + d['w']*pad))
            y2 = min(H, int(d['cy'] + d['h']/2 + d['h']*pad))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            bn = f"vid{videoname}_fi{frame_idx}_i{i}_s{d['score']:.3f}_cx{d['cx']:.0f}_cy{d['cy']:.0f}"
            bn = bn.replace(' ', '_').replace('/', '_')
            img_path = os.path.join(out_img, bn + '.jpg')
            lbl_path = os.path.join(out_lbl, bn + '.txt')

            if not cv2.imwrite(img_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                continue

            cx_n = (d['cx'] - x1) / (x2 - x1)
            cy_n = (d['cy'] - y1) / (y2 - y1)
            bw_n = d['w'] / (x2 - x1)
            bh_n = d['h'] / (y2 - y1)
            cx_n = float(np.clip(cx_n, 0, 1))
            cy_n = float(np.clip(cy_n, 0, 1))
            bw_n = float(np.clip(bw_n, 0.001, 1))
            bh_n = float(np.clip(bh_n, 0.001, 1))

            with open(lbl_path, 'w') as f:
                f.write(f"0 {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}\n")
            n += 1

    print(f"  Saved {n} accepted labels")


# ── Preview Mode ────────────────────────────────────────────────────────────
def preview_video(videopath, videoname, frame_step=20, max_frames=30):
    cap = cv2.VideoCapture(videopath)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    frame_indices = list(range(0, fc, frame_step))[:max_frames]
    print(f"Preview: {videoname}  {fc} frames  sampling {len(frame_indices)} frames")

    out_dir = os.path.join(OUTPUT_DIR, 'preview', videoname)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'crops'), exist_ok=True)

    print("Running OWL-ViT detection...", flush=True)
    t0 = time.time()
    video_data = []
    for fi in frame_indices:
        cap = cv2.VideoCapture(videopath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        cap.release()
        if not ret: continue
        dets = detect(frame)
        video_data.append({'frame_idx': fi, 'frame': frame, 'dets': dets})
    print(f"Detection done in {time.time()-t0:.1f}s")

    # Save full annotated montage
    cols = 4
    rows = (len(video_data) + cols - 1) // cols
    cell_h = H // 2; cell_w = W // 2
    montage = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    crop_data = []
    for idx, item in enumerate(video_data):
        row = idx // cols; col = idx % cols
        display = draw_frame(item['frame'], item['dets'],
                             idx, len(video_data), videoname, scale=0.5)
        cell = cv2.resize(display, (cell_w, cell_h))
        montage[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w] = cell

        # Save individual frame
        frame_path = os.path.join(out_dir, f"frame_{item['frame_idx']:04d}.jpg")
        cv2.imwrite(frame_path, display)
        print(f"  Frame {item['frame_idx']:4d}: {len(item['dets'])} detections")

        for i, d in enumerate(item['dets']):
            pad = 0.3
            x1 = max(0, int(d['cx'] - d['w']/2 - d['w']*pad))
            y1 = max(0, int(d['cy'] - d['h']/2 - d['h']*pad))
            x2 = min(W, int(d['cx'] + d['w']/2 + d['w']*pad))
            y2 = min(H, int(d['cy'] + d['h']/2 + d['h']*pad))
            crop = item['frame'][y1:y2, x1:x2]
            if crop.size == 0: continue

            bn = f"frame{item['frame_idx']:04d}_det{i}_s{d['score']:.3f}_cx{d['cx']:.0f}_cy{d['cy']:.0f}_r{d['ratio']:.2f}"
            crop_path = os.path.join(out_dir, 'crops', bn + '.jpg')
            cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            crop_data.append({
                'frame': item['frame_idx'], 'det': i,
                'score': d['score'], 'ratio': d['ratio'],
                'cx': d['cx'], 'cy': d['cy'], 'w': d['w'], 'h': d['h'],
                'path': crop_path
            })

    montage_path = os.path.join(out_dir, 'montage.jpg')
    cv2.imwrite(montage_path, montage, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n  Montage: {montage_path}")
    print(f"  Crops:   {os.path.join(out_dir, 'crops')}")
    print(f"  Total detections: {sum(len(item['dets']) for item in video_data)}")
    return video_data


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true', help='Preview mode (no GUI)')
    parser.add_argument('video', nargs='?', help='Video filename (e.g. 20kg_0.87_0.88_0.89_0.91.mp4)')
    args = parser.parse_args()

    videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')])

    if args.video:
        vn = args.video if args.video.endswith('.mp4') else args.video + '.mp4'
        vp = os.path.join(VIDEO_DIR, vn)
    else:
        print("\nAvailable videos:")
        for i, v in enumerate(videos):
            print(f"  {i+1:2d}. {v}")
        print(f"\nUsage:")
        print(f"  Preview: python3 {__file__} --preview 20kg_0.87_0.88_0.89_0.91.mp4")
        print(f"  Label:   python3 {__file__} 20kg_0.87_0.88_0.89_0.91.mp4")
        print(f"\nStarting preview of first video...")
        vn = videos[0]; vp = os.path.join(VIDEO_DIR, vn)

    if not os.path.exists(vp):
        print(f"Video not found: {vp}")
        exit(1)

    videoname = os.path.splitext(vn)[0]

    if args.preview:
        preview_video(vp, videoname)
    else:
        label_video(vp, videoname)
