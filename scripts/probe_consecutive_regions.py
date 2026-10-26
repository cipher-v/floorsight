"""GCP-only experimental stability predicate; never used by production."""
import os, json
from collections import deque
from pathlib import Path
import cv2
import numpy as np
from floorsight.edge.privacy import blur_faces

def stable_regions(history, min_area=39):
    result=np.zeros_like(history[-1],dtype=np.uint8)
    if len(history)<11: return result
    current=history[-1]
    for label in np.unique(current):
        if label==0: continue
        mask=current==label
        area=int(mask.sum())
        if area<min_area: continue
        cy,cx=np.where(mask)
        valid=True
        for old in list(history)[:-1]:
            ids,counts=np.unique(old[mask],return_counts=True)
            significant=[int(k) for k,n in zip(ids,counts) if k and n>=max(3,.02*area)]
            if len(significant)!=1: valid=False;break
            om=old==significant[0]
            # Bidirectional one-to-one correspondence rejects splits and merges.
            reverse,rc=np.unique(current[om],return_counts=True)
            if sum(k!=0 and n>=max(3,.02*int(om.sum())) for k,n in zip(reverse,rc))!=1:
                valid=False;break
            union=np.count_nonzero(mask|om)
            oy,ox=np.where(om)
            if np.count_nonzero(mask&om)/union<.90 or np.hypot(cx.mean()-ox.mean(),cy.mean()-oy.mean())>2:
                valid=False;break
        if valid: result[mask]=1
    return result

def labels(mask): return cv2.connectedComponents(mask.astype('uint8'),8)[1]

def controls():
    def rect(x):
        m=np.zeros((80,140),np.uint8);m[20:60,x:x+20]=1;return m
    cases={
        'stationary':[rect(40) for _ in range(11)],
        'continuous_translation':[rect(20+i) for i in range(11)],
        'out_and_back':[rect(30+min(i,10-i)*2) for i in range(11)],
        'brief_pause':[rect(20+min(i,6)*2) for i in range(11)],
        'disappearance':[rect(40) for _ in range(10)]+[np.zeros((80,140),np.uint8)],
        'interrupted_visibility':[rect(40) for _ in range(5)]+[np.zeros((80,140),np.uint8)]+[rect(40) for _ in range(5)],
    }
    split=rect(40);split[:,49:51]=0
    cases['merge']=[split for _ in range(10)]+[rect(40)]
    cases['split']=[rect(40) for _ in range(10)]+[split]
    rows=[]
    for name,frames in cases.items():
        accepted=int(stable_regions([labels(m) for m in frames]).sum())
        expected=name=='stationary'
        rows.append(dict(case=name,accepted_pixels=accepted,expected=expected,passed=bool(accepted)>0 if expected else accepted==0))
    assert all(r['passed'] for r in rows),rows
    return rows

def main():
    from floorsight.vision.detect import CameraDetector
    assert os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED')=='yes'
    cv2.setNumThreads(1)
    out=Path('eval/diagnostics/consecutive_regions_20261017_01');out.mkdir(parents=True,exist_ok=False)
    checks=controls();(out/'controls.json').write_text(json.dumps(checks,indent=2))
    det=CameraDetector(960,540,10);det.set_plate(cv2.imread('data/sim/demo_001/setup_empty_cam2.png'))
    cap=cv2.VideoCapture('data/sim/demo_001/cam2.mp4');history=deque(maxlen=11);rows=[]
    for i in range(1601):
        ok,frame=cap.read();assert ok
        frame,_=blur_faces(frame);det.process(frame,i/10)
        if det.diff_mask is None: continue
        history.append(labels(det.diff_mask))
        if i<1450: continue
        support=stable_regions(history)
        roi=support[40:96,294:314]
        rows.append(dict(t=i/10,support_roi_pixels=int(roi.sum()),baseline_static_roi_pixels=int(det.static_mask[40:96,294:314].sum()),support_total_pixels=int(support.sum())))
        if i in (1540,1570,1580,1590):
            panels=[cv2.resize(frame,(480,270)),cv2.cvtColor(det.static_mask.astype('uint8')*255,cv2.COLOR_GRAY2BGR),cv2.cvtColor(support*255,cv2.COLOR_GRAY2BGR)]
            for panel,title in zip(panels,['FILTERED SOURCE','BASELINE STATIC','EXPERIMENTAL STABILITY ONLY']):cv2.putText(panel,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,220,255),1)
            cv2.imwrite(str(out/f't{i/10:.1f}.jpg'),np.hstack(panels))
    cap.release()
    result=dict(version='consecutive-region-probe-v1',controls=checks,samples=rows,limitation='Pixel stability is not person identity or foreground validity. Ghost rejection and production integration untested; no tracking improvement claim.')
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps([r for r in rows if r['t'] in (154,157,158,159)],indent=2))

if __name__=='__main__':main()
