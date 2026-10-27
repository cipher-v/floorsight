"""Exact frame equality and sampling latency; never writes raw image pixels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.real_clip_perception import background_samples

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--clip', type=Path, action='append', required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
rows=[]
for clip in a.clip:
    cap=cv2.VideoCapture(str(clip))
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for repeat, order in enumerate([['seek','auto'],['auto','seek']]):
        measured={}
        for mode in order:
            start=time.perf_counter(); frames=background_samples(cap,n,mode=mode)
            elapsed=time.perf_counter()-start
            measured[mode]={'seconds':elapsed,'samples':len(frames),
                'sample_sha256':hashlib.sha256(np.stack(frames).tobytes()).hexdigest()}
        rows.append({'clip':clip.stem,'frames':n,'repeat':repeat,'runs':measured,
                     'identical':measured['seek']['sample_sha256']==measured['auto']['sample_sha256']})
    cap.release()
a.out.write_text(json.dumps({'schema':'floorsight.background-bench.v1','opencv':cv2.__version__,'runs':rows},indent=2))
print(json.dumps(rows,indent=2))
raise SystemExit(0 if all(r['identical'] and r['runs']['auto']['samples']==25 for r in rows) else 1)
