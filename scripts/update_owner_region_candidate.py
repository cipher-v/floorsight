import os,json,shutil
from pathlib import Path
from datetime import datetime,timezone
assert os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED')=='yes'
r=Path('owner-review');shutil.copytree(r,Path('eval/reviews/owner-review-v05-preserved-20261017'))
run='gcp_region_support_20261017_01'
score=json.loads((Path('eval/results')/(run+'.json')).read_text())
base=json.loads(Path('eval/results/gcp_temporal_clearance_20261017_02.json').read_text())
shutil.copyfile(Path('eval/reviews')/run/'t158.0.jpg',r/'media/region-candidate158.jpg')
summary=dict(run=run,scorer=score['scorer_version'],baseline=base['dots'],candidate=score['dots'],scenario_checks=score['incidents'],status='Experimental; accepted detector restored. No promotion claimed.')
(r/'media/region-candidate.json').write_text(json.dumps(summary,indent=2))
d=score['dots'];b=base['dots'];inc=score['incidents']
section=f'''<h2>Region support: full replay result</h2><p>The candidate adds consecutive shape stability with timestamp resets and the existing foreground edge check. The accepted detector has been restored; this experiment is not a promoted tracking fix.</p><p>Recall: {b['recall']*100:.1f}% baseline → {d['recall']*100:.1f}% candidate. False observed dots per update: {b['false_dots_per_update']:.3f} → {d['false_dots_per_update']:.3f}. Scenario checks: {inc['passed']}/{inc['total']}. Frozen scorer: {score['scorer_version']}.</p><figure><img src="media/region-candidate158.jpg" alt="Four camera evaluation overlays at158 seconds"><figcaption>Candidate at158s. This is a synthetic evaluation overlay, including evaluation reference positions where labeled; those positions are not production detector inputs. Inspect whether the orange worker is tracked, and compare other workers and vehicles. A recovered silhouette alone does not establish reliable tracking.</figcaption></figure><p><a href="media/region-candidate.json">Inspect measured comparison</a>. Ideal controls cover timestamp gaps, resets and a simple departed-object ghost. They do not establish field robustness or calibrated safety.</p>'''
p=r/'index.html';s=p.read_text();s=s.replace('<h2>The remaining defects are visible</h2>',section+'<h2>The remaining defects are visible</h2>');p.write_text(s)
p=r/'review-manifest.json';m=json.loads(p.read_text());m['version']='opencv-owner-20261017-06';m['updatedAt']=datetime.now(timezone.utc).isoformat();m['summary']='Accepted playable demo retained; full consecutive-region candidate comparison, explicit restoration and limitations.'
m['media'] += [dict(path='media/region-candidate158.jpg',type='image/jpeg'),dict(path='media/region-candidate.json',type='application/json')];p.write_text(json.dumps(m,indent=2))
