"""GCP-only stage trace; source instrumentation does not change detector decisions."""
import json, os, sys, types
from pathlib import Path
import cv2
import numpy as np
from floorsight.edge.privacy import blur_faces
assert os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED') == 'yes'
cv2.setNumThreads(1)
out = Path('eval/diagnostics/stationary_stage_trace_20261017_01')
out.mkdir(parents=True, exist_ok=False)
source = Path('floorsight/vision/detect.py').read_text()
needle = '        keep[0] = False'
assert source.count(needle) == 1
source = source.replace(needle, needle + '\n        self.stage = dict(now=now.copy(), fg=fg.copy(), static=static.copy(), settled=settled.copy(), labels=lab.copy(), stats=stats.copy(), motion_fraction=frac.copy(), mog_fraction=ffrac.copy(), keep=keep.copy())')
module = types.ModuleType('diagnostic_detector'); sys.modules[module.__name__] = module
exec(compile(source, 'instrumented-detect.py', 'exec'), module.__dict__)
det = module.CameraDetector(960,540,10)
det.set_plate(cv2.imread('data/sim/demo_001/setup_empty_cam2.png'))
cap=cv2.VideoCapture('data/sim/demo_001/cam2.mp4')
rows=[]
roi=(slice(40,96),slice(294,314))
for i in range(1601):
    ok, frame=cap.read(); assert ok
    frame,_=blur_faces(frame)
    blobs=det.process(frame,i/10)
    if i not in {1500,1530,1540,1550,1560,1570,1580,1590,1600}: continue
    s=det.stage
    components=[]
    for label in np.unique(s['labels'][roi]):
        if label == 0: continue
        x,y,w,h,area=map(int,s['stats'][label])
        reasons=[]
        if s['motion_fraction'][label] < .08: reasons.append('immediate_motion_below_0.08')
        if s['mog_fraction'][label] < .15: reasons.append('mog_foreground_below_0.15')
        if area < det.min_area: reasons.append('area_below_minimum')
        components.append(dict(bbox=[2*x,2*y,2*w,2*h], area=area,
            motion_fraction=float(s['motion_fraction'][label]), mog_fraction=float(s['mog_fraction'][label]),
            moving_component_selected=bool(s['keep'][label]), rejection_reasons=reasons))
    n, labels, stats,_=cv2.connectedComponentsWithStats(s['static'],8)
    static_components=[]
    for label in np.unique(labels[roi]):
        if label==0: continue
        x,y,w,h,area=map(int,stats[label]); static_components.append(dict(bbox=[2*x,2*y,2*w,2*h],area=area,below_static_minimum=area < det.min_area*.6))
    row=dict(t=i/10,roi_full=[588,80,40,112],components=components,static_components=static_components,
        roi=dict(diff=float(det.diff_mask[roi].mean()),recent=float(det.recent[roi].mean()),now=float(s['now'][roi].mean()),mog=float(s['fg'][roi].mean()),static_count_median=float(np.median(det.static_count[roi]))),
        raw_blobs=[dict(bbox=list(map(int,b.bbox)),kind=b.kind) for b in blobs])
    rows.append(row)
    if i==1570:
        tiles=[]
        for label,data in [('source',frame),('difference',det.diff_mask),('immediate motion',s['now']),('MOG foreground',s['fg']),('recent motion',det.recent),('static before area gate',s['static'])]:
            if data.ndim==2: data=cv2.cvtColor(data.astype('uint8')*255,cv2.COLOR_GRAY2BGR)
            tile=cv2.resize(data,(480,270)); cv2.rectangle(tile,(294,40),(314,96),(0,0,255),1)
            cv2.putText(tile,label,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,220,255),1); tiles.append(tile)
        cv2.imwrite(str(out/'stages157.jpg'),np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]))
cap.release()
(out/'trace.json').write_text(json.dumps(dict(version='pixel-stage-trace-v1',source='cam2 filtered pixels',limitation='single-camera replay; no floor projection occurs when no raw worker blob survives',samples=rows),indent=2))
print(json.dumps(rows,indent=2))
