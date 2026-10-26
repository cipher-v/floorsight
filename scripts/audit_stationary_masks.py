"""GCP-only camera-2 detector ablation; manually selected pixel ROI, no truth."""
import json, os
from pathlib import Path
import cv2
import numpy as np
from floorsight.edge.privacy import blur_faces
from floorsight.vision.detect import CameraDetector

assert os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED') == 'yes'
cv2.setNumThreads(1)
out=Path('eval/diagnostics/stationary_masks_20261017_02')
out.mkdir(parents=True,exist_ok=False)
cap=cv2.VideoCapture('data/sim/demo_001/cam2.mp4')
plate=cv2.imread('data/sim/demo_001/setup_empty_cam2.png')
detectors={k:CameraDetector(960,540,10) for k in ['baseline','no_ghost_heal']}
for det in detectors.values(): det.set_plate(plate)
detectors['no_ghost_heal']._is_ghost=lambda *args:False
events=[]
base=detectors['baseline']
original=base._is_ghost
current_t=0.
def audited(roi,x,y,w,h,gray):
    decision=original(roi,x,y,w,h,gray)
    if decision:
        events.append({'t':current_t,'bbox_full':[2*x,2*y,2*w,2*h],'edges':list(base.last_ghost_test)})
    return decision
base._is_ghost=audited
rows=[]
targets={0,100,500,1000,1400,1500,1530,1540,1570,1600}
for i in range(1601):
    ok,image=cap.read()
    assert ok,i
    image,_=blur_faces(image)
    current_t=i/10
    results={k:det.process(image,current_t) for k,det in detectors.items()}
    if i%10==0:
        row={'t':current_t}
        # Orange worker's visible camera-2 box, selected from source pixels.
        for k,det in detectors.items():
            roi=(slice(40,96),slice(294,314))
            row[k]={'diff_fraction':float(det.diff_mask[roi].mean()) if det.diff_mask is not None else None,
                    'static_fraction':float(det.static_mask[roi].mean()) if det.static_mask is not None else None,
                    'recent_motion_fraction':float(det.recent[roi].mean()) if hasattr(det,'recent') else None,
                    'static_count_median':float(np.median(det.static_count[roi])),
                    'moving_fraction':float(det.moving_mask[roi].mean()) if det.moving_mask is not None else None,
                    'blobs':[{'bbox':b.bbox,'kind':b.kind,'stationary_s':b.stationary_s} for b in results[k]]}
        rows.append(row)
        with (out/'samples.jsonl').open('a') as f:
            f.write(json.dumps(row,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')
    if i in targets and i>0:
        tiles=[]
        for k,det in detectors.items():
            masks=[('source',image),('background',cv2.convertScaleAbs(det.bg)),('difference',det.diff_mask),('static',det.static_mask),('recent motion',det.recent),('moving',det.moving_mask)]
            for label,data in masks:
                if data is None: continue
                if data.ndim==2: data=cv2.cvtColor(data.astype('uint8')*255,cv2.COLOR_GRAY2BGR)
                tile=cv2.resize(data,(480,270))
                cv2.rectangle(tile,(294,40),(314,96),(0,0,255),1)
                cv2.putText(tile,f'{k} {label} t={current_t}',(6,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,220,255),1)
                tiles.append(tile)
        mosaic=np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:6]),np.hstack(tiles[6:9]),np.hstack(tiles[9:])])
        cv2.imwrite(str(out/f't{current_t:.1f}.jpg'),mosaic)
cap.release()
(out/'audit.json').write_text(json.dumps({'version':'stationary-mask-ablation-v2','camera':'cam2','roi_full':[588,80,40,112],'limitation':'Single camera detector replay, not full pipeline or field validation; no-ghost-heal is diagnostic only.','ghost_heal_events':events,'samples':rows},indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))
print(json.dumps({'out':str(out),'last':rows[-1]},indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))
