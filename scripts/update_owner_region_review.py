import json,os,shutil
from pathlib import Path
from datetime import datetime,timezone
assert os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED')=='yes'
r=Path('owner-review');shutil.copytree(r,Path('eval/reviews/owner-review-v04-preserved-20261017'))
src=Path('eval/diagnostics/region_history_20261017_01')
for source,dest in [('source-masks-map.webm','region-history.webm'),('t157.0.jpg','region-history.jpg'),('history.json','region-history.json')]:shutil.copyfile(src/source,r/'media'/dest)
p=r/'index.html';s=p.read_text();s=s.replace('scripted incidents','scenario checks');s=s.replace('6/7 incidents','6/7 scenario checks')
s=s.replace('<h2>Why the stationary worker disappears</h2>','''<h2>Watch the detection gap develop</h2><figure><video controls playsinline preload="metadata" poster="media/region-history.jpg" src="media/region-history.webm"></video><figcaption>15.1 seconds, shift time 145–160s. Top: filtered source, foreground difference, static mask. Bottom: accepted saved map and measured region history. Blue map circles are seen tracks; amber circles are estimated, lost or unseen. The red source box selects pixels for diagnosis, not a known identity. Masks come from a single-camera diagnostic replay; the map comes from the accepted full run. No truth positions enter this clip.</figcaption></figure><p>This clip shows loss and reacquisition, not a repaired tracker. <a href="media/region-history.json">Inspect per-frame region overlap and displacement</a>. The baseline passes seven scenario checks, including one correctly rejected false-alarm source; three true near-misses have explicit confirmation evidence. Seven checks do not mean seven confirmed hazards.</p><h2>Why the stationary worker disappears</h2>''')
p.write_text(s)
p=r/'review-manifest.json';m=json.loads(p.read_text());m['version']='opencv-owner-20261017-05';m['updatedAt']=datetime.now(timezone.utc).isoformat();m['summary']='Accepted demo plus synchronized source/masks/map diagnostic and per-frame region history; seven scenario checks distinguished from three true near-misses.'
m['media'] += [dict(path='media/region-history.webm',type='video/webm'),dict(path='media/region-history.jpg',type='image/jpeg'),dict(path='media/region-history.json',type='application/json')]
p.write_text(json.dumps(m,indent=2))
