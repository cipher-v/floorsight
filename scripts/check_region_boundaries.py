"""GCP-only foreground and lifecycle controls for experimental region support."""
import os,json
from pathlib import Path
from collections import deque
import cv2,numpy as np
from scripts.probe_consecutive_regions import stable_regions,labels

class TimedRegions:
    def __init__(self): self.reset()
    def reset(self): self.history=deque(maxlen=11);self.times=deque(maxlen=11)
    def update(self,mask,t):
        if self.times and (t<=self.times[-1] or t-self.times[-1]>.15): self.reset()
        self.history.append(labels(mask));self.times.append(t)
        if len(self.times)<11 or self.times[-1]-self.times[0]<.99: return np.zeros_like(mask)
        return stable_regions(self.history)

def main():
    from floorsight.vision.detect import CameraDetector
    assert os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED')=='yes'
    cv2.setNumThreads(1)
    out=Path('eval/diagnostics/region_boundaries_20261017_01');out.mkdir(parents=True,exist_ok=False)
    mask=np.zeros((120,160),np.uint8);mask[30:90,60:90]=1
    rows=[]
    for name,times,reset_at,expected in [('normal',[i/10 for i in range(11)],None,True),('duplicate_timestamp',[0.]*11,None,False),('reversed_timestamp',[i/10 for i in range(10)]+[.8],None,False),('camera_gap',[i/10 for i in range(10)]+[2.],None,False),('compressed_time',[i/20 for i in range(11)],None,False),('reset',[i/10 for i in range(11)],6,False)]:
        history=TimedRegions()
        for i,t in enumerate(times):
            if i==reset_at: history.reset()
            result=history.update(mask,t)
        accepted=bool(result.any());rows.append(dict(case=name,accepted=accepted,expected=expected,passed=accepted==expected))
    # Same persistent difference silhouette: current object versus departed object.
    floor=np.full((120,160,3),50,np.uint8);object_frame=floor.copy();object_frame[mask>0]=220
    panels=[]
    for name,bg,current,expected in [('present_object',floor,object_frame,False),('departed_object_ghost',object_frame,floor,True)]:
        detector=CameraDetector(160,120,10,scale=1);detector.bg=bg.astype(np.float32)
        ghost=bool(detector._is_ghost(mask[30:90,60:90]>0,60,30,30,60,cv2.cvtColor(current,cv2.COLOR_BGR2GRAY)))
        rows.append(dict(case=name,ghost=ghost,expected_ghost=expected,edges=list(detector.last_ghost_test),passed=ghost==expected))
        panel=cv2.resize(current,(480,360));cv2.putText(panel,name,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,210,255),1);panels.append(panel)
    # Quantify a known resolution limit rather than asserting all motion is rejected.
    frames=[]
    for i in range(11):
        m=np.zeros_like(mask);m[30:90,60+i//5:90+i//5]=1;frames.append(labels(m))
    slow=stable_regions(frames)
    rows.append(dict(case='slow_two_pixel_translation',accepted_pixels=int(slow.sum()),limitation='Engineering tolerance may accept slow motion; not proof of stationary state.'))
    (out/'result.json').write_text(json.dumps(dict(version='region-boundaries-v1',checks=rows),indent=2))
    cv2.imwrite(str(out/'foreground-ghost.jpg'),np.hstack(panels))
    print(json.dumps(rows,indent=2))
    assert all(r.get('passed',True) for r in rows)

if __name__=='__main__':main()
