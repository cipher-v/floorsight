"""Isolated GCP experiment; always restores accepted detector after evaluation."""
import os,sys,subprocess,hashlib,json
from pathlib import Path
assert os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED')=='yes'
run='gcp_region_support_20261017_01'
out=Path('eval/evidence')/run;out.mkdir(parents=True,exist_ok=False)
target=Path('floorsight/vision/detect.py');before=target.read_bytes()
assert hashlib.sha256(before).hexdigest()=='e52e85a42493297a4abe59883d37c994e6d9cb4b7b0ca1f8ee2f539bea67be19'
(out/'detect-before.py').write_bytes(before)
source=before.decode()
source=source.replace('from dataclasses import dataclass','from dataclasses import dataclass\nfrom scripts.check_region_boundaries import TimedRegions')
# Avoid circular import: boundary module imports detector only inside its CLI.
source=source.replace('        self.frames = 0','        self.frames = 0\n        self.region_history = TimedRegions()')
source=source.replace('        if self.global_change:\n            return []','        if self.global_change:\n            self.region_history.reset()\n            return []')
needle='        settled = cnt >= min_frames'
assert source.count(needle)==1
source=source.replace(needle,'''        # Experimental additional support, with original static logic retained.
        support = self.region_history.update(d, t) if abs(self.fps-10)<.01 else np.zeros_like(d)
        sn, sl, ss, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
        for si in range(1, sn):
            sx, sy, sw, sh, sa = ss[si]
            sr = sl[sy:sy+sh, sx:sx+sw] == si
            if not self._is_ghost(sr, sx, sy, sw, sh, gray):
                sc = cnt[sy:sy+sh, sx:sx+sw]
                sc[sr] = np.maximum(sc[sr], min_frames)
        settled = cnt >= min_frames''')
(out/'detect-candidate.py').write_text(source)
def call(args,name):
    with (out/name).open('w') as log:
        subprocess.run([sys.executable]+args,stdout=log,stderr=subprocess.STDOUT,check=True)
try:
    target.write_text(source)
    call(['-m','pytest','tests','-q','-p','no:cacheprovider','-o','tmp_path_retention_policy=all','-o','tmp_path_retention_count=1000000'],'tests.log')
    call(['-m','eval.e2e.run_e2e','--name',run,'--port','8013','--no-screenshot','--no-drop'],'replay.log')
    call(['-m','scripts.audit_confirmation_trace','--run',run],'confirmation.log')
    call(['-m','scripts.review_saved_run','--run',run,'--times','51','104','157','158'],'review.log')
finally:
    target.write_bytes(before)
    restored=hashlib.sha256(target.read_bytes()).hexdigest()
    (out/'restoration.json').write_text(json.dumps(dict(restored_sha256=restored,exact=target.read_bytes()==before)))
print(out)
