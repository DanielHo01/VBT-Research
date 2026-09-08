import os, cv2, numpy as np

img_dir = r'D:\EasyVBT-Research\datasets\owlvit_pseudo\images'
label_dir = r'D:\EasyVBT-Research\datasets\owlvit_pseudo\labels'

files = sorted(os.listdir(img_dir))
print(f"Verifying {len(files)} crops:\n")

all_ok = True
for f in files:
    img = cv2.imread(os.path.join(img_dir, f))
    H, W = img.shape[:2]

    with open(os.path.join(label_dir, f.replace('.jpg', '.txt'))) as lf:
        line = lf.read().strip()
    parts = [float(x) for x in line.split()]
    cls, cx, cy, bw, bh = parts

    # Check range
    ok = all(0 <= v <= 1 for v in [cx, cy, bw, bh])
    ratio = max(bw,bh)/min(bw,bh) if min(bw,bh) > 0 else 999

    # Draw bbox
    x1 = int((cx - bw/2) * W)
    y1 = int((cy - bh/2) * H)
    x2 = int((cx + bw/2) * W)
    y2 = int((cy + bh/2) * H)
    img_draw = img.copy()
    cv2.rectangle(img_draw, (x1,y1), (x2,y2), (0,255,0), 2)
    status = "OK" if ok else "BAD"
    print(f"  {status} {f}: crop={W}x{H} label=({cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f}) "
          f"ratio={ratio:.2f} bbox=[{x1},{y1},{x2},{y2}]")
    if not ok:
        all_ok = False

print(f"\n{'All labels OK!' if all_ok else 'SOME LABELS HAVE ISSUES!'}")
