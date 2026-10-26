"""Add source-linked diagnostic evidence without replacing the accepted demo media."""
import os,json,shutil
from pathlib import Path
from datetime import datetime, timezone
assert os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED')=='yes'
root=Path('owner-review')
shutil.copytree(root,Path('eval/reviews/owner-review-v02-preserved-20261017'))
shutil.copyfile('eval/diagnostics/stationary_stage_trace_20261017_01/stages157.jpg',root/'media/stages157.jpg')
shutil.copyfile('eval/diagnostics/stationary_stage_trace_20261017_01/trace.json',root/'media/stage-trace.json')
html=(root/'index.html').read_text()
html=html.replace('latest full scored run','selected accepted full scored run')
section='''<h2>Why the stationary worker disappears</h2><figure><img src="media/stages157.jpg" alt="Six camera-2 diagnostic panels show the orange worker in foreground difference but absent from immediate motion and MOG foreground"><figcaption><strong>157 seconds: two detection paths fail.</strong> The red box contains the worker. Top row: source, foreground difference, immediate motion. Bottom: MOG foreground, held recent motion, stationary fragments. These are pixel-derived diagnostics; no evaluation truth enters the detector.</figcaption></figure><p>The full foreground component has 738 detector pixels, but both immediate-motion and MOG fractions are zero. It fails both moving gates. Held recent motion delays the stationary counter; only fragments of 19 and 29 pixels survive, below the 39-pixel stationary area threshold. No worker blob reaches floor projection. By 159 seconds a 731-pixel stationary component survives. <a href="media/stage-trace.json">Inspect the measured stage trace</a>.</p><p>Disabling ghost healing did not change this failure. A separate recent-motion candidate raised aggregate recall to 88.0% but increased false dots and left this worker missing, so it was rejected. A second candidate using immediate motion for stationary accumulation passed its recovery test but fell to 62.4% recall, 0.426 false dots per update and 6/7 incidents. It was also rejected. The selected demo above remains the accepted 86.9% baseline. A single-camera diagnostic is not field validation or proof that every recall miss has this cause.</p>'''
html=html.replace('<h2>Evidence and next repair</h2>',section+'<h2>Evidence and next repair</h2>')
html=html.replace('Next: inspect pixel detections around crossings and stationary workers, repair association and retention, then repeat the full scored run.', 'Next: coordinate stationary accumulation with observed motion cessation, verify the move-stop boundary and every scripted incident, then investigate remaining association and ghost tracks.')
(root/'index.html').write_text(html)
m=json.loads((root/'review-manifest.json').read_text());m['version']='opencv-owner-20261017-03';m['updatedAt']=datetime.now(timezone.utc).isoformat();m['summary']='Accepted playable baseline plus measured explanation of stationary-worker loss before floor projection.'
m['media'] += [dict(path='media/stages157.jpg',type='image/jpeg'),dict(path='media/stage-trace.json',type='application/json')]
m['verificationEvidence']['stageTrace']='pixel-stage-trace-v1 / 20261017T153544Z-1690c0b6'
(root/'review-manifest.json').write_text(json.dumps(m,indent=2))
