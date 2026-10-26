"""Summarize pixel-derived diagnostics only; no simulator truth input."""
import json, os
from pathlib import Path
import numpy as np
from floorsight.vision.tracker import FloorDet, cluster
assert os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED') == 'yes'
root=Path('eval/diagnostics/recall_pixels_20261017_01')
summary=[]
for t in [51.,112.,154.,157.]:
    record=json.loads((root/f'{t:.1f}.json').read_text())
    dets=[]
    for d in record['detections']:
        d['xy']=np.asarray(d['xy'])
        dets.append(FloorDet(**d))
    groups=[]
    for kind,radius in [('moving',.9),('static',1.2)]:
        for c in cluster([d for d in dets if d.kind==kind],radius,1.6 if kind=='moving' else None):
            groups.append({'kind':kind,'xy':c.xy.tolist(),'sigma':c.sigma,'height':c.height_m,'footprint':c.footprint,'person_like':c.person_like,'stationary_s':c.stationary_s,'cameras':c.camera_ids,
                           'detections':[{'camera':d.camera_id,'xy':d.xy.tolist(),'bbox':d.bbox,'height':d.height_m,'footprint':d.footprint} for d in c.dets]})
    summary.append({'t':t,'groups':groups,'tracks':record['world']['tracks']})
out=root/'pixel-summary.json'
with out.open('x') as f: json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
