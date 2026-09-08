"""
Full 34-video benchmark with patched detector.
"""
import sys, os, json, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'validation' / 'dataset_benchmark'))

from algorithms.common import YoloPlateDetector
import algorithms.common as cm
from algorithms.pipelines import pipeline_associator
from metrics_evaluator import MetricsEvaluator

VIDEO_DIR = Path(r'D:\EasyVBT-Research\validation\dataset_benchmark\raw_videos')
INDEX_FILE = Path(r'D:\EasyVBT-Research\validation\dataset_benchmark\dataset_index.json')

models = {
    'barbell_v4_baseline':  r'D:\EasyVBT-Research\models\barbell_v4.onnx.backup',
    'combined_v2_epoch3':   r'D:\EasyVBT-Research\runs\detect\combined_v2_run\weights\last.onnx',
}

with open(INDEX_FILE) as f:
    dataset = json.load(f)

def run_model(name, model_path):
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"Path:  {model_path}")
    
    # Patch default
    orig_init = YoloPlateDetector.__init__
    YoloPlateDetector.__init__ = lambda self, mp=None: orig_init(self, model_path)
    
    t0 = time.time()
    errors = []
    passed = 0
    no_pred = 0
    results_detail = []
    
    for item in dataset:
        vid = item['video_id']
        gt_mcvs = item.get('gt_reps_mcv', [])
        vp = VIDEO_DIR / vid
        
        try:
            result = pipeline_associator(str(vp))
            reps = result.get('reps', [])
            
            if reps and gt_mcvs:
                ev = MetricsEvaluator.evaluate_video(vid, gt_mcvs, reps)
                err = float(ev.rmse) if not np.isnan(ev.rmse) else None
                if err is not None:
                    errors.append(err)
                    results_detail.append({'vid': vid, 'err': err, 'n_gt': len(gt_mcvs), 'n_pred': len(reps)})
                    if err <= 2.0:
                        passed += 1
                else:
                    no_pred += 1
            elif not reps and gt_mcvs:
                no_pred += 1
        except Exception as e:
            no_pred += 1
    
    YoloPlateDetector.__init__ = orig_init
    
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")
    
    if errors:
        mean_rmse = float(np.mean(errors))
        median_rmse = float(np.median(errors))
        print(f"  Videos: {len(errors)}/{len(dataset)} with valid RMSE")
        print(f"  Passed (RMSE ≤ 2.0): {passed}/{len(errors)} ({passed/len(errors)*100:.0f}%)")
        print(f"  No predictions: {no_pred}")
        print(f"  RMSE mean:   {mean_rmse:.4f}")
        print(f"  RMSE median: {median_rmse:.4f}")
        
        # Worst videos
        worst = sorted(results_detail, key=lambda x: -x['err'])[:3]
        print(f"  Worst videos:")
        for r in worst:
            print(f"    {r['vid']:<40} RMSE={r['err']:.4f} (GT={r['n_gt']}, pred={r['n_pred']})")
        
        return {'mean': mean_rmse, 'median': median_rmse, 'n': len(errors), 'passed': passed, 'no_pred': no_pred}
    else:
        print(f"  No valid results (no_pred={no_pred})")
        return {'mean': None}

results = {}
for name, mp in models.items():
    results[name] = run_model(name, mp)

print(f"\n\n{'='*60}")
print("FINAL COMPARISON (RMSE of MCV, lower is better)")
print(f"{'='*60}")
for name, r in results.items():
    if r['mean'] is not None:
        print(f"  {name:<25} RMSE={r['mean']:.4f}  median={r['median']:.4f}  passed={r['passed']}/{r['n']}")
    else:
        print(f"  {name:<25} NO RESULTS")
