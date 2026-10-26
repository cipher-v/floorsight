"""GCP-only region-history diagnostic and synchronized saved-map evidence."""
import os,json,subprocess
from pathlib import Path
from collections import deque
import cv2,numpy as np
from floorsight.edge.privacy import blur_faces
from floorsight.vision.detect import CameraDetector
assert os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED')=='yes'
cv2.setNumThreads(1)
out=Path('eval/diagnostics/region_history_20261017_01');out.mkdir(parents=True,exist_ok=False)
worlds=[json.loads(x)['world'] for x in Path('data/runtime/gcp_temporal_clearance_20261017_02/worlds.jsonl').read_text().splitlines()]
site=json.loads(Path('data/sim/demo_001/site.json').read_text())
det=CameraDetector(960,540,10);det.set_plate(cv2.imread('data/sim/demo_001/setup_empty_cam2.png'))
cap=cv2.VideoCapture('data/sim/demo_001/cam2.mp4');history=deque(maxlen=11);rows=[]
writer=cv2.VideoWriter(str(out/'source-masks-map-original.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1440,540));assert writer.isOpened()
for i in range(1601):
    ok,frame=cap.read();assert ok
    frame,_=blur_faces(frame);blobs=det.process(frame,i/10)
    if det.diff_mask is None: continue
    n,lab,stats,_=cv2.connectedComponentsWithStats(det.diff_mask.astype('uint8'),8)
    labels,count=np.unique(lab[40:96,294:314],return_counts=True)
    candidates=[(int(c),int(k)) for k,c in zip(labels,count) if k>0]
    label=max(candidates)[1] if candidates else None
    mask=lab==label if label is not None else np.zeros_like(lab,dtype=bool)
    row=dict(t=i/10,bbox_full=None,history=[],static_fraction=0.)
    if label is not None:
        x,y,w,h,area=map(int,stats[label]);row.update(bbox_full=[2*x,2*y,2*w,2*h],area=area,static_fraction=float(det.static_mask[mask].mean()))
        for lag in (5,10):
            if len(history)<lag: continue
            old=history[-lag]; intersections=np.bincount(old[mask]);intersections[0]=0
            best=int(intersections.argmax());om=old==best if best else np.zeros_like(mask)
            overlap=int(np.count_nonzero(mask&om));union=int(np.count_nonzero(mask|om))
            cy,cx=np.where(mask);py,px=np.where(om)
            row['history'].append(dict(lag_s=lag/10,mask_iou=overlap/max(union,1),centroid_displacement_px=float(np.hypot(cx.mean()-px.mean(),cy.mean()-py.mean())) if len(px) else None,previous_components_overlapped=int(np.count_nonzero(intersections))))
    history.append(lab.copy())
    if i>=1450:
        rows.append(row)
        source=cv2.resize(frame,(480,270));cv2.rectangle(source,(294,40),(314,96),(0,0,255),1)
        diff=cv2.cvtColor(det.diff_mask.astype('uint8')*255,cv2.COLOR_GRAY2BGR)
        static=cv2.cvtColor(det.static_mask.astype('uint8')*255,cv2.COLOR_GRAY2BGR)
        for image,title in [(source,'FILTERED SOURCE'),(diff,'FOREGROUND DIFFERENCE'),(static,'STATIC OUTPUT')]:
            cv2.putText(image,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,255),1)
        floor=np.full((270,480,3),25,np.uint8)
        def point(x,y):return (int(20+x/site['width_m']*440),int(245-y/site['depth_m']*210))
        for z in site['zones']:
            cv2.polylines(floor,[np.array([point(x,y) for x,y in z['polygon']])],True,(90,90,90),1)
        world=min(worlds,key=lambda w:abs(w['t']-i/10))
        for tr in world['tracks']:
            p=point(tr['x'],tr['y']);color=(240,180,20) if tr['state']=='seen' else (0,190,255)
            cv2.circle(floor,p,4,color,1);cv2.putText(floor,tr['track_id'],(p[0]+5,p[1]),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
        cv2.putText(floor,'ACCEPTED SAVED MAP (no truth)',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(220,220,220),1)
        detail=np.full((270,960,3),25,np.uint8)
        lines=[f'SIMULATED / cam2 / t={i/10:.1f}s',f'Pixel-selected region: {row["bbox_full"]}',f'Static coverage: {row["static_fraction"]:.3f}']
        lines += [f'{v["lag_s"]:.1f}s history: mask IoU={v["mask_iou"]:.3f}, center shift={v["centroid_displacement_px"]} px, overlaps={v["previous_components_overlapped"]}' for v in row['history']]
        lines += ['Region association is geometric, not an identity label.','Masks: single-camera diagnostic replay; map: accepted full run.','This tests stability evidence; no production behavior was changed.']
        for j,line in enumerate(lines):cv2.putText(detail,line,(12,26+29*j),cv2.FONT_HERSHEY_SIMPLEX,.48,(230,230,230),1)
        mosaic=np.vstack([np.hstack([source,diff,static]),np.hstack([floor,detail])]);writer.write(mosaic)
        if i in (1540,1570,1580,1590):cv2.imwrite(str(out/f't{i/10:.1f}.jpg'),mosaic)
writer.release();cap.release()
(out/'history.json').write_text(json.dumps(dict(version='pixel-region-history-v1',input='cam2.mp4 with blur_faces',map_run='gcp_temporal_clearance_20261017_02',limitation='Fixed pixel ROI selects largest overlapping component; not truth identity, full pipeline causal proof or field validation.',samples=rows),indent=2))
subprocess.run(['ffmpeg','-nostdin','-n','-i',str(out/'source-masks-map-original.mp4'),'-c:v','libvpx-vp9','-threads','2','-crf','34','-b:v','0',str(out/'source-masks-map.webm')],check=True)
print(json.dumps([r for r in rows if r['t'] in [154.,157.,158.,159.]],indent=2))
