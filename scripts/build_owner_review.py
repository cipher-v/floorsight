"""GCP-only, private review bundle from saved simulation observations."""
import json, os, shutil, subprocess
from pathlib import Path
from datetime import datetime, timezone
import cv2
import numpy as np
from floorsight.site import load_site
from floorsight.vision.calib import MarkerDetector, marker_floor_corners, fit_calibration

assert os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED') == 'yes'
root = Path('owner-review')
root.mkdir(exist_ok=False)
media = root/'media'
media.mkdir()
run = 'gcp_temporal_clearance_20261017_02'
for name, src in [('crossing.jpg','eval/reviews/gcp_recall_gaps_20261017_01/t51.0.jpg'),
                  ('stationary.jpg','eval/reviews/gcp_recall_gaps_20261017_01/t157.0.jpg'),
                  ('uncertainty.jpg',f'eval/reviews/{run}/t104.0.jpg')]:
    shutil.copyfile(src, media/name)
site = load_site('data/sim/demo_001/site.json')
info = next(c for c in site.cameras if c.camera_id == 'cam2')
detector = MarkerDetector(site)
cap = cv2.VideoCapture('data/sim/demo_001/cam2.mp4')
cap.set(cv2.CAP_PROP_POS_MSEC, 145000)
ok, frame = cap.read()
assert ok
calib = fit_calibration(detector.detect(frame, fine=True), marker_floor_corners(site), info)
assert calib is not None
worlds = [json.loads(x)['world'] for x in Path(f'data/runtime/{run}/worlds.jsonl').read_text().splitlines()]
raw = Path('eval/evidence/owner-review-01-overlay.mp4')
assert not raw.exists()
writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame.shape[1],frame.shape[0]))
for t in np.arange(145.,161.,.1):
    cap.set(cv2.CAP_PROP_POS_MSEC,float(t*1000))
    ok, frame = cap.read()
    assert ok
    world = min(worlds,key=lambda w:abs(w['t']-t))
    for track in world['tracks']:
        uv, z = calib.project(np.array([[track['x'],track['y'],0.]]))
        if z[0] <= 0: continue
        x,y = np.round(uv[0]).astype(int)
        color = (240,180,20) if track['state']=='seen' else (0,190,255)
        cv2.circle(frame,(int(x),int(y)),8,color,2)
        cv2.putText(frame,f"{track['track_id']} {track['state']}",(int(x)+10,int(y)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1)
    cv2.rectangle(frame,(0,0),(frame.shape[1],38),(20,24,30),-1)
    cv2.putText(frame,f'SIMULATED / saved product tracks / camera 2 / {t:.1f}s',(12,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
    writer.write(frame)
    if abs(t-145)<.01: cv2.imwrite(str(media/'poster.jpg'),frame)
writer.release(); cap.release()
subprocess.run(['ffmpeg','-nostdin','-n','-i',str(raw),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(media/'tracking.mp4')],check=True)
html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Floorsight — owner review</title>
<style>body{margin:0;background:#101823;color:#e9eef5;font:17px/1.6 system-ui}main{max-width:1050px;margin:auto;padding:28px 20px}h1{font-size:clamp(32px,6vw,52px);line-height:1.1}h2{font-size:24px}p{max-width:78ch}.tag{color:#8fd6ff}.stats{display:flex;flex-wrap:wrap;gap:12px}.stats span{background:#213145;padding:14px;border-radius:10px}figure{margin:28px 0;padding:16px;background:#192637;border-radius:14px}img,video{width:100%;height:auto;display:block;border-radius:8px}figcaption{padding-top:12px;color:#c6d4e5}strong{color:#fff}.gap{border-left:4px solid #ffc566;padding-left:16px}footer{color:#aabbd0;font-size:14px}</style>
<main><p class="tag">PRIVATE REVIEW · SIMULATED FOOTAGE · 17 OCTOBER 2026</p><h1>Floorsight<br>What is verified, what still fails</h1>
<p>Four camera views become tracked floor positions and evidence-backed safety decisions. Temporal clearance checks now leave the false near-miss unconfirmed. Occluded vehicle positions are marked estimated.</p>
<div class="stats"><span><strong>7 / 7</strong> scripted incidents</span><span><strong>0</strong> false confirmations</span><span><strong>86.9%</strong> tracking recall</span></div>
<p class="gap">Recall is below the 90% target. This is a development demonstration, not a validated warehouse safety system. A rejected alert means insufficient evidence, not proof of safe clearance.</p>
<h2>Watch the current tracking</h2><figure><video controls playsinline preload="metadata" poster="media/poster.jpg" src="media/tracking.mp4"></video><figcaption>16 seconds, camera 2, shift time 145–161s. Saved product tracks from the latest full scored run, projected onto synthetic footage. Blue circles: seen; amber: estimated, lost or unseen. No truth positions enter this clip. Watch the stationary orange worker lose its track.</figcaption></figure>
<h2>The remaining defects are visible</h2>
<figure><img src="media/crossing.jpg" alt="Four camera overlays at 51 seconds show a person track displaced toward a forklift"><figcaption><strong>51 seconds — association defect.</strong> Person track T4 sits near the forklift while the purple worker lacks a matching dot. Green crosses are evaluation truth, not product observations.</figcaption></figure>
<figure><img src="media/stationary.jpg" alt="Four camera overlays at 157 seconds show the stationary orange worker without a track"><figcaption><strong>157 seconds — stationary retention.</strong> The visible orange worker has no dot; the green worker has a lost track. The blue worker is tracked.</figcaption></figure>
<figure><img src="media/uncertainty.jpg" alt="Occluded forklift at 104 seconds is marked estimated"><figcaption><strong>104 seconds — honest uncertainty, unresolved position.</strong> The forklift is now estimated, but its floor position remains displaced. Confidence labeling does not repair geometry.</figcaption></figure>
<h2>Evidence and next repair</h2><p>Full run: 1,800 frames, zero dropped frames or errors. Frozen scorer: floorsight-e2e-legacy8-v1. Supplemental recall audit reproduces 7,777 / 8,949 expected observations. The latest wording change passed 127 tests on GCP, with one missing-fixture skip; those tests are not a new full replay.</p>
<p>Next: inspect pixel detections around crossings and stationary workers, repair association and retention, then repeat the full scored run. Real-camera privacy, degraded-camera behavior, field validation and the judge package remain incomplete. AWS execution is held for owner review.</p>
<footer>2D overlays and metric floor positions over an existing synthetic 3D scene. All media generation and browser verification run on the assigned GCP CPU lease. No public deployment.</footer></main></html>'''
(root/'index.html').write_text(html)
manifest={'version':'opencv-owner-20261017-01','updatedAt':datetime.now(timezone.utc).isoformat(),'summary':'Temporal verification improved; recall remains86.9%. Playable saved-track clip and visible association/stationary defects.','media':[{'path':'media/'+p.name,'type':'video/mp4' if p.suffix=='.mp4' else 'image/jpeg'} for p in sorted(media.iterdir())],'verificationEvidence':{'fullRun':run,'scorer':'floorsight-e2e-legacy8-v1','latestTests':127,'browser':'Pending separate bound GCP receipt'}}
(root/'review-manifest.json').write_text(json.dumps(manifest,indent=2))
